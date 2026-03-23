provider "aws" {
  region = var.aws_region
}

provider "helm" {
  kubernetes {
    host                   = aws_eks_cluster.carnot.endpoint
    cluster_ca_certificate = base64decode(aws_eks_cluster.carnot.certificate_authority[0].data)

    # Use exec-based auth instead of a static token. The static token from
    # aws_eks_cluster_auth expires after 15 minutes, but cluster + node group
    # creation takes ~20 min — so the token is stale by the time Helm runs.
    # exec fetches a fresh token on each API call.
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", var.cluster_name, "--region", var.aws_region]
    }
  }
}

locals {
  # OIDC issuer without the https:// prefix — used in IRSA trust policy conditions.
  oidc_issuer = trimprefix(aws_eks_cluster.carnot.identity[0].oidc[0].issuer, "https://")
  oidc_arn    = "arn:aws:iam::${var.account_id}:oidc-provider/${local.oidc_issuer}"
}

# -------------------------------
# EKS Cluster
# -------------------------------
resource "aws_eks_cluster" "carnot" {
  name                      = var.cluster_name
  role_arn                  = aws_iam_role.eks_cluster_role.arn
  version                   = var.cluster_version
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = true
  }
}

# -------------------------------
# Managed Node Group
# -------------------------------
resource "aws_eks_node_group" "carnot_nodes" {
  cluster_name    = aws_eks_cluster.carnot.name
  node_group_name = "carnot-nodes"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = var.subnet_ids
  instance_types  = ["t3.large"]
  disk_size       = 100

  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 3
  }

  tags = {
    # Required for Cluster Autoscaler auto-discovery (see addons.tf)
    "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
    "k8s.io/cluster-autoscaler/enabled"             = "true"
  }
}

# -------------------------------
# OIDC Provider for IRSA
# -------------------------------
resource "aws_iam_openid_connect_provider" "carnot" {
  url             = aws_eks_cluster.carnot.identity[0].oidc[0].issuer
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["9e99a48a9960b14926bb7f3b02e22da2b0ab7280"]
}

