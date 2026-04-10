terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    http = {
      source = "hashicorp/http"
    }
  }

  backend "s3" {
    bucket  = "carnot-research"
    key     = "tf/eks/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}
