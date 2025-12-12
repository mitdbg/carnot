provider "aws" {
  region = var.aws_region
}

# -------------------------------
# Global SSH key
# -------------------------------
resource "aws_key_pair" "deployer_key" {
  key_name   = "carnot-deployer-key"
  public_key = var.deployer_public_key
}

# -------------------------------
# Security Group for ALB
# -------------------------------
resource "aws_security_group" "alb_sg" {
  name   = "alb_sg"
  vpc_id = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# -------------------------------
# ALB (shared across all envs)
# -------------------------------
resource "aws_lb" "global_alb" {
  name               = "carnot-global-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = var.public_subnet_ids

  tags = {
    Name = "carnot-global-alb"
  }
}

# -------------------------------
# ALB Listeners
# -------------------------------
resource "aws_lb_listener" "https_listener" {
  load_balancer_arn = aws_lb.global_alb.arn
  port              = 443
  protocol          = "HTTPS"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "Not Found"
      status_code  = "404"
    }
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.global_alb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# ---------------------------------------------------------
# Route53 Record for Auth0 Custom Domain
# ---------------------------------------------------------
resource "aws_route53_record" "auth0_custom_domain" {
  zone_id = var.hosted_zone_id
  name    = var.auth0_custom_domain
  type    = "CNAME"
  ttl     = 300

  records = [var.auth0_cname_target]
}

# -------------------------------
# EC2 Instance for Homepage NGINX
# -------------------------------
data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["*ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

resource "aws_security_group" "homepage_sg" {
  name   = "carnot-homepage-ec2-sg"
  vpc_id = var.vpc_id

  # Allow SSH from anywhere (or restrict to your IP range)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] 
  }

  # Allow traffic on port 8080 ONLY from the ALB's security group
  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "homepage_ec2" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = "t2.micro"
  key_name                    = aws_key_pair.deployer_key.key_name
  vpc_security_group_ids      = [aws_security_group.homepage_sg.id]
  subnet_id                   = var.public_subnet_ids[0]
  associate_public_ip_address = true

  tags = {
    Name = "carnot-homepage-ec2"
  }

  # Script to run on first launch: install docker, pull nginx, and run container
  user_data = <<-EOF
              #!/bin/bash
              # Install Docker
              sudo apt-get update
              sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
              curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
              echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
              sudo apt-get update
              sudo apt-get install -y docker-ce docker-ce-cli containerd.io

              # start and enable docker service
              sudo systemctl start docker
              sudo systemctl enable docker

              # Pull and Run NGINX Container
              sudo docker run -d -p 8080:80 --name carnot-homepage-nginx nginx:latest
              EOF
}

# -------------------------------
# Register Homepage EC2 with its Target Group
# -------------------------------
resource "aws_lb_target_group_attachment" "homepage_ec2_attachment" {
  target_group_arn = aws_lb_target_group.homepage_tg.arn
  target_id        = aws_instance.homepage_ec2.id
  port             = 8080
}

# -------------------------------
# Target Group for Homepage NGINX
# -------------------------------
resource "aws_lb_target_group" "homepage_tg" {
  name     = "carnot-homepage-tg"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  health_check {
    path                = "/"
    port                = "8080"
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

# -------------------------------
# ALB Listener Rule for Root Domain (Homepage) - HTTPS
# -------------------------------
resource "aws_lb_listener_rule" "homepage_rule_https" {
  listener_arn = aws_lb_listener.https_listener.arn
  priority     = 1

  action {
    type = "forward"
    target_group_arn = aws_lb_target_group.homepage_tg.arn
  }

  condition {
    host_header {
      values = ["carnot-research.org"]
    }
  }
}

# -------------------------------
# ALB Listener Rule for Root Domain (Homepage) - HTTP
# -------------------------------
resource "aws_lb_listener_rule" "homepage_rule_http" {
  listener_arn = aws_lb_listener.http_redirect.arn
  priority     = 1

  action {
    type = "redirect"
    redirect {
      host        = "carnot-research.org"
      path        = "/#{path}"
      query       = "#{query}"
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }

  condition {
    host_header {
      values = ["carnot-research.org"]
    }
  }
}

# -------------------------------
# Route53 Record for Root Domain
# -------------------------------
resource "aws_route53_record" "root_domain" {
  zone_id = var.hosted_zone_id
  name    = "carnot-research.org"
  type    = "A"

  alias {
    name                   = aws_lb.global_alb.dns_name
    zone_id                = aws_lb.global_alb.zone_id
    evaluate_target_health = true
  }
}
