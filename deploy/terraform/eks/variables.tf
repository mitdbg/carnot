variable "aws_region" {
  description = "AWS region for the EKS cluster."
  default     = "us-east-1"
}

variable "account_id" {
  description = "AWS account ID."
  default     = "422297141788"
}

variable "cluster_name" {
  description = "Name of the EKS cluster."
  default     = "carnot-unified-backend"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster."
  default     = "1.35"
}

variable "vpc_id" {
  description = "VPC ID for the EKS cluster."
  default     = "vpc-d27da3b7"
}

variable "subnet_ids" {
  description = "Subnet IDs for the EKS cluster control plane and managed node group (one per AZ)."
  type        = list(string)
  default     = ["subnet-27021a61", "subnet-09ffc721", "subnet-fe946a89"]
}

variable "hosted_zone_id" {
  description = "Route53 hosted zone ID for carnot-research.org."
  default     = "Z0371749174EVZHL80QS0"
}

variable "acm_certificate_arn" {
  description = "ARN of the ACM wildcard certificate covering *.carnot-research.org."
  default     = "arn:aws:acm:us-east-1:422297141788:certificate/10be4bdb-461d-4dec-9ddf-501e2f157fab"
}
