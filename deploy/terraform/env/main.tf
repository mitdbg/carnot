provider "aws" {
  region = var.aws_region
}

data "terraform_remote_state" "global" {
  backend = "s3"
  config = {
    bucket = "carnot-research"
    key    = "tf/global/terraform.tfstate"
    region = "us-east-1"
  }
}

locals {
  env_name  = terraform.workspace
  full_domain_name = "${var.subdomain}.carnot-research.org"
  alb_sg_id = data.terraform_remote_state.global.outputs.alb_sg_id
  https_listener_arn = data.terraform_remote_state.global.outputs.https_listener_arn
  http_listener_arn = data.terraform_remote_state.global.outputs.http_listener_arn
  alb_dns_name = data.terraform_remote_state.global.outputs.alb_dns_name
  alb_zone_id = data.terraform_remote_state.global.outputs.alb_zone_id
  ssh_key_name = data.terraform_remote_state.global.outputs.global_key_name
}

# -------------------------------
# AMI Used for EC2 Instances
# -------------------------------
data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  owners = ["099720109477"] # Canonical
}

# -----------------------------------
# EC2 Instance for Application Server
# -----------------------------------
resource "aws_instance" "app_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  iam_instance_profile   = aws_iam_instance_profile.instance_profile.name
  key_name               = local.ssh_key_name
  vpc_security_group_ids = [
    aws_security_group.allow_ssh.id,
    aws_security_group.allow_app.id
  ]
  root_block_device {
    volume_size = 40 # desired size in GB
    volume_type = "gp2"
  }

  # mount the EBS volume at /mnt/pg-data and install docker compose
  user_data = <<-EOF
    #!/bin/bash

    # Mount EBS volume and change ownership to UID/GID 999 (default for postgres)
    sudo mkfs -t ext4 /dev/xvdf
    sudo mkdir -p /mnt/pg-data
    sudo mount /dev/xvdf /mnt/pg-data
    echo "/dev/xvdf /mnt/pg-data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
    sudo mkdir -p /mnt/pg-data/data
    sudo chown -R 999:999 /mnt/pg-data

    # Install Docker
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker ubuntu
  EOF

  tags = {
    Name = "${var.instance_name}-${local.env_name}"
  }
}

# -------------------------------
# IAM role per environment
# -------------------------------
resource "aws_iam_role" "instance_role" {
  name = "carnot-${local.env_name}-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "s3_access" {
  role = aws_iam_role.instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
      Effect = "Allow"
      Action = ["s3:ListBucket"]
      Resource = ["arn:aws:s3:::carnot-research"]
    },
    {
      Effect   = "Allow"
      Action   = ["s3:*"]
      Resource = [
        "arn:aws:s3:::carnot-research/${local.env_name}",
        "arn:aws:s3:::carnot-research/${local.env_name}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "instance_profile" {
  name = "carnot-${local.env_name}-instance-profile"
  role = aws_iam_role.instance_role.name
}

# -------------------------------
# Security Groups for SSH and ALB
# -------------------------------
resource "aws_security_group" "allow_app" {
  name   = "allow-app-${local.env_name}"
  vpc_id = var.vpc_id

  ingress {
    description = "Allow ALB access to web service"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    security_groups = [local.alb_sg_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "allow-app-${local.env_name}"
    Environment = local.env_name
  }
}

resource "aws_security_group" "allow_ssh" {
  name        = "allow-ssh-${local.env_name}"
  description = "Allow SSH inbound traffic"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Allows SSH from anywhere. Restrict for production.
    description = "Allow SSH from anywhere"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # Allow all protocols
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name        = "allow-ssh-${local.env_name}"
    Environment = local.env_name
  }
}

# -------------------------------
# EBS Volume for data persistence
# -------------------------------
resource "aws_ebs_volume" "app_pg_data_volume" {
  availability_zone = var.availability_zone
  size              = var.ebs_volume_size
  type              = "gp2"

  tags = {
    Name = "carnot-app-pg-data-volume-${local.env_name}"
  }
}

resource "aws_volume_attachment" "ebs_volume_attachment" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.app_pg_data_volume.id
  instance_id = aws_instance.app_server.id
}

# -------------------------------
# Target Group for ALB
# -------------------------------
resource "aws_lb_target_group" "app_tg" {
  name     = "carnot-${local.env_name}-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  health_check {
    path = "/"
    port = "80"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# -------------------------------
# Attach EC2 instance to TG
# -------------------------------
resource "aws_lb_target_group_attachment" "app" {
  target_group_arn = aws_lb_target_group.app_tg.arn
  target_id        = aws_instance.app_server.id
  port             = 80
}

# -------------------------------
# ALB Listener Rules
# -------------------------------
resource "aws_lb_listener_rule" "app_rule" {
  listener_arn = local.https_listener_arn
  priority     = var.priority_offset + 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app_tg.arn
  }

  condition {
    host_header {
      values = [local.full_domain_name]
    }
  }
}

resource "aws_lb_listener_rule" "http_redirect_rule" {
  listener_arn = local.http_listener_arn
  priority     = var.priority_offset + 20

  action {
    type = "redirect"
    redirect {
      host        = local.full_domain_name
      path        = "/#{path}"
      query       = "#{query}"
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }

  condition {
    host_header {
      values = [local.full_domain_name]
    }
  }
}

# ---------------------------------------------------------
# Route53 Record for per-environment domain
# ---------------------------------------------------------
resource "aws_route53_record" "app_dns" {
  zone_id = var.hosted_zone_id
  name = local.full_domain_name
  type = "A"

  alias {
    name                   = local.alb_dns_name
    zone_id                = local.alb_zone_id
    evaluate_target_health = true
  }
}
