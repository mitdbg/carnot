terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.40"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }

  # Same state bucket as the other stacks (tf/global, tf/env/*, tf/eks); its own key.
  backend "s3" {
    bucket  = "carnot-research"
    key     = "tf/experiments/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}
