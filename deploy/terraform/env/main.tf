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
# S3 Bucket for application data
# -------------------------------
resource "aws_s3_bucket" "app_data_bucket" {
  bucket = "carnot-research-${local.env_name}"

  tags = {
    Name = "carnot-research-${local.env_name}"
  }
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
    volume_type = "gp3"
  }

  # mount the EBS volume at /mnt/pg-data and install docker compose
  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    exec > /var/log/user-data.log 2>&1

    MOUNT_POINT="/mnt/pg-data"

    # On m5/nitro instances the EBS volume appears as an NVMe device rather
    # than the legacy /dev/xvdf name. Wait for either path, then resolve the
    # real device node using its serial number (which encodes the volume-id).
    echo "Waiting for EBS data volume to attach..."
    DEVICE=""
    for i in $(seq 1 60); do
      # Prefer the legacy symlink when present (older instance types)
      if [ -e /dev/xvdf ]; then
        DEVICE=/dev/xvdf
        break
      fi
      # On Nitro/NVMe instances find the data disk by excluding the root disk
      NVME_DEV=$(lsblk -dpno NAME,TYPE | awk '$2=="disk"{print $1}' | while read d; do
        # root disk always has partitions; the bare data EBS volume does not
        PARTS=$(lsblk -no NAME "$d" | wc -l)
        if [ "$PARTS" -eq 1 ]; then echo "$d"; fi
      done | head -1)
      if [ -n "$NVME_DEV" ]; then
        DEVICE=$NVME_DEV
        break
      fi
      sleep 5
    done

    if [ -z "$DEVICE" ]; then
      echo "ERROR: EBS data volume did not appear after 300s" >&2
      exit 1
    fi
    echo "Using device: $DEVICE"

    ### mount EBS volume and change ownership to UID/GID 999 (default for postgres)
    # check for existing filesystem before formatting
    # use blkid exit code explicitly (set -e would abort on non-zero outside of if)
    BLKID_OUT=$(blkid $DEVICE || true)
    if [ -z "$BLKID_OUT" ]; then
        echo "No filesystem found on $DEVICE. Formatting..."
        mkfs -t ext4 $DEVICE
    else
        echo "Filesystem already exists on $DEVICE. Skipping format."
    fi

    # idempotent mount and fstab entry
    mkdir -p $MOUNT_POINT
    mount $DEVICE $MOUNT_POINT || echo "$DEVICE already mounted or failed"

    if ! grep -q "$MOUNT_POINT" /etc/fstab; then
      echo "$DEVICE $MOUNT_POINT ext4 defaults,nofail 0 2" | tee -a /etc/fstab
    fi

    mkdir -p $MOUNT_POINT/data
    chown -R 999:999 $MOUNT_POINT

    # wait for cloud-init / unattended-upgrades to release the dpkg lock
    echo "Waiting for apt lock..."
    while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
      sleep 5
    done

    # install docker
    apt-get update -y
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    usermod -aG docker ubuntu

    echo "user-data: done"
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
      Resource = ["arn:aws:s3:::carnot-research-${local.env_name}"]
    },
    {
      Effect   = "Allow"
      Action   = ["s3:*"]
      Resource = [
        "arn:aws:s3:::carnot-research-${local.env_name}",
        "arn:aws:s3:::carnot-research-${local.env_name}/*"
      ]
    },
    {
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue",
        "secretsmanager:CreateSecret",
        "secretsmanager:DescribeSecret",
        "secretsmanager:PutSecretValue"
      ]
      Resource = [
        "arn:aws:secretsmanager:${var.aws_region}:422297141788:secret:carnot/${local.env_name}/*"
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
  type              = "gp3"

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
