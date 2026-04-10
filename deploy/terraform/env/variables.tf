variable "subdomain" {
  description = "The subdomain for the application."
  type    = string
}

variable "priority_offset" {
  description = "The priority offset for the ALB listener rules."
  type    = number
}

variable "aws_region" {
  description = "AWS region for global resources"
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "The VPC ID where the EC2 instance will be launched."
  type        = string
  default     = "vpc-d27da3b7"
}

variable "instance_name" {
  description = "Value of the EC2 instance's Name tag."
  type        = string
  default     = "carnot-web-app"
}

variable "instance_type" {
  description = "The EC2 instance's type."
  type        = string
  default     = "m5.2xlarge"
}

variable "subnet_id" {
  description = "The subnet ID where the EC2 instance will be launched."
  type        = string
  default     = "subnet-27021a61"
}

# NOTE: this must match the AZ of the subnet where the instance is launched
variable "availability_zone" {
  description = "The availability zone for the EBS volume."
  type        = string
  default     = "us-east-1a"
}

variable "ebs_volume_size" {
  description = "The size of the EBS volume in GB."
  type        = number
  default     = 50
}

variable "public_subnet_ids" {
  description = "List of public subnet IDs for the ALB."
  type        = list(string)
  default     = ["subnet-27021a61", "subnet-09ffc721"]
}

variable "hosted_zone_id" {
  description = "The Route 53 Hosted Zone ID for the domain."
  type        = string
  default     = "Z0371749174EVZHL80QS0"
}
