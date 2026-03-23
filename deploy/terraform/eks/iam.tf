# -----------------------------------------------------------------
# EKS Control Plane role
# Required by the EKS cluster itself to call EC2/ELB/IAM APIs on
# your behalf when managing the control plane.
# -----------------------------------------------------------------
resource "aws_iam_role" "eks_cluster_role" {
  name = "carnot-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

# -----------------------------------------------------------------
# EKS Node Group role
# Assumed by the EC2 worker nodes. Required for nodes to register
# with the cluster and use the VPC CNI plugin.
# -----------------------------------------------------------------
resource "aws_iam_role" "eks_node_role" {
  name = "carnot-eks-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  role       = aws_iam_role.eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  role       = aws_iam_role.eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

# Required for nodes to pull EKS system images (aws-node, kube-proxy, coredns)
# from ECR. App images use Docker Hub but system components are ECR-hosted.
resource "aws_iam_role_policy_attachment" "eks_ecr_read" {
  role       = aws_iam_role.eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}


# -----------------------------------------------------------------
# EBS CSI Driver — IRSA role
# Grants the EBS CSI controller permission to manage EBS volumes
# on behalf of PersistentVolumeClaim requests.
# -----------------------------------------------------------------
resource "aws_iam_role" "ebs_csi_driver" {
  name = "carnot-ebs-csi-driver-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.oidc_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
          "${local.oidc_issuer}:sub" = "system:serviceaccount:kube-system:ebs-csi-controller-sa"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ebs_csi_driver" {
  role       = aws_iam_role.ebs_csi_driver.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

# -----------------------------------------------------------------
# AWS Load Balancer Controller — IRSA role + policy
# The controller calls ELB/EC2 APIs to manage the shared k8s ALB.
# The policy document is fetched from the upstream AWS repo — same
# source used in bootstrap_cluster.sh.
# -----------------------------------------------------------------
data "http" "alb_controller_policy_doc" {
  url = "https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/main/docs/install/iam_policy.json"
}

resource "aws_iam_policy" "alb_controller" {
  name   = "carnot-alb-controller-policy"
  policy = data.http.alb_controller_policy_doc.response_body
}

resource "aws_iam_role" "alb_controller" {
  name = "carnot-alb-controller-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.oidc_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
          "${local.oidc_issuer}:sub" = "system:serviceaccount:kube-system:aws-load-balancer-controller"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "alb_controller" {
  role       = aws_iam_role.alb_controller.name
  policy_arn = aws_iam_policy.alb_controller.arn
}

# -----------------------------------------------------------------
# External Secrets Operator — IRSA role
# Grants ESO permission to read secrets from Secrets Manager
# under the carnot/* path.
# -----------------------------------------------------------------
resource "aws_iam_role" "external_secrets" {
  name = "carnot-external-secrets-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.oidc_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
          "${local.oidc_issuer}:sub" = "system:serviceaccount:external-secrets:external-secrets"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "external_secrets" {
  role = aws_iam_role.external_secrets.id
  name = "SecretsManagerRead"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
      Resource = "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:carnot/*"
    }]
  })
}

# -----------------------------------------------------------------
# Cluster Autoscaler — IRSA role
# Grants the CA permission to describe and resize the managed node
# group's Auto Scaling Group. The node group must carry the
# auto-discovery tags declared in main.tf for the CA to find it.
# -----------------------------------------------------------------
resource "aws_iam_role" "cluster_autoscaler" {
  name = "carnot-cluster-autoscaler-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.oidc_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
          "${local.oidc_issuer}:sub" = "system:serviceaccount:kube-system:cluster-autoscaler"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "cluster_autoscaler" {
  role = aws_iam_role.cluster_autoscaler.id
  name = "ClusterAutoscalerPolicy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Read-only: CA uses these to discover nodes and make scaling decisions.
        Effect = "Allow"
        Action = [
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances",
          "autoscaling:DescribeLaunchConfigurations",
          "autoscaling:DescribeScalingActivities",
          "autoscaling:DescribeTags",
          "ec2:DescribeImages",
          "ec2:DescribeInstanceTypes",
          "ec2:DescribeLaunchTemplateVersions",
          "ec2:GetInstanceTypesFromInstanceRequirements",
          "eks:DescribeNodegroup",
        ]
        Resource = "*"
      },
      {
        # Write: CA uses these to scale the node group in and out.
        Effect = "Allow"
        Action = [
          "autoscaling:SetDesiredCapacity",
          "autoscaling:TerminateInstanceInAutoScalingGroup",
        ]
        Resource = "*"
      },
    ]
  })
}

# -----------------------------------------------------------------
# Per-environment backend ServiceAccount IRSA roles
# Each environment's backend pod uses a dedicated role scoped to
# its own S3 bucket and Secrets Manager path.
# -----------------------------------------------------------------
locals {
  envs = toset(["dev", "wl-prod", "lm-prod", "at-prod"])
}

resource "aws_iam_role" "backend_sa" {
  for_each = local.envs

  name = "carnot-${each.key}-sa-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.oidc_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
          "${local.oidc_issuer}:sub" = "system:serviceaccount:${each.key}:carnot-backend"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "backend_sa" {
  for_each = local.envs

  role = aws_iam_role.backend_sa[each.key].id
  name = "S3AndSecretsAccess"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = "arn:aws:s3:::carnot-research-${each.key}"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:*"]
        Resource = [
          "arn:aws:s3:::carnot-research-${each.key}",
          "arn:aws:s3:::carnot-research-${each.key}/*",
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
        Resource = "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:carnot/${each.key}/*"
      },
    ]
  })
}
