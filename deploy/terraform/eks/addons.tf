# -----------------------------------------------------------------
# EKS Managed Add-ons
# Managed by AWS — they receive automatic patch-version upgrades
# and are tested for compatibility with the cluster version.
# -----------------------------------------------------------------
resource "aws_eks_addon" "vpc_cni" {
  cluster_name = aws_eks_cluster.carnot.name
  addon_name   = "vpc-cni"
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name = aws_eks_cluster.carnot.name
  addon_name   = "kube-proxy"
}

resource "aws_eks_addon" "coredns" {
  cluster_name = aws_eks_cluster.carnot.name
  addon_name   = "coredns"

  # CoreDNS creates the kube-dns Service, which triggers the ALB controller's
  # mutating webhook. The webhook must have running endpoints or admission is
  # denied and the addon creation fails.
  depends_on = [helm_release.alb_controller]
}

resource "aws_eks_addon" "ebs_csi_driver" {
  cluster_name             = aws_eks_cluster.carnot.name
  addon_name               = "aws-ebs-csi-driver"
  service_account_role_arn = aws_iam_role.ebs_csi_driver.arn
}

# -----------------------------------------------------------------
# AWS Load Balancer Controller — Helm
# Manages the shared k8s ALB (IngressGroup "carnot") that routes
# traffic to all environment namespaces via host-based rules.
# -----------------------------------------------------------------
resource "helm_release" "alb_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"

  set {
    name  = "clusterName"
    value = var.cluster_name
  }

  set {
    name  = "region"
    value = var.aws_region
  }

  set {
    name  = "vpcId"
    value = var.vpc_id
  }

  # Wire up the IRSA role via ServiceAccount annotation so the controller
  # pod inherits AWS credentials without stored secrets.
  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = aws_iam_role.alb_controller.arn
  }

  depends_on = [aws_eks_node_group.carnot_nodes]
}

# -----------------------------------------------------------------
# External Secrets Operator — Helm
# Syncs secrets from AWS Secrets Manager into k8s Secrets at pod
# startup. The ClusterSecretStore is managed by the eks-config/ module (Phase 2).
# -----------------------------------------------------------------
resource "helm_release" "external_secrets" {
  name             = "external-secrets"
  repository       = "https://charts.external-secrets.io"
  chart            = "external-secrets"
  namespace        = "external-secrets"
  create_namespace = true

  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = aws_iam_role.external_secrets.arn
  }

  # Depends on ALB controller (not just node group) because the ALB mutating webhook
  # fires when ESO creates its Service. If the ALB controller pods aren't ready, it fails.
  depends_on = [helm_release.alb_controller]
}

# -----------------------------------------------------------------
# Cluster Autoscaler — Helm
# Scales the managed node group in/out by watching for unschedulable
# pods (scale-up) and underutilized nodes (scale-down).
# Uses auto-discovery via the tags set on the node group in main.tf.
# -----------------------------------------------------------------
resource "helm_release" "cluster_autoscaler" {
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  namespace  = "kube-system"

  set {
    name  = "autoDiscovery.clusterName"
    value = var.cluster_name
  }

  set {
    name  = "awsRegion"
    value = var.aws_region
  }

  set {
    name  = "rbac.serviceAccount.name"
    value = "cluster-autoscaler"
  }

  set {
    name  = "rbac.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = aws_iam_role.cluster_autoscaler.arn
  }

  depends_on = [helm_release.alb_controller]
}

# -----------------------------------------------------------------
# Cluster Config — local Helm chart
# Creates the ClusterSecretStore that tells ESO how to read from
# AWS Secrets Manager. Uses a local Helm chart to avoid the
# kubernetes_manifest provider REST mapper bug with CRDs.
# -----------------------------------------------------------------
resource "helm_release" "cluster_config" {
  name  = "cluster-config"
  chart = "${path.module}/../../helm/cluster-config"

  set {
    name  = "awsRegion"
    value = var.aws_region
  }

  depends_on = [helm_release.external_secrets]
}
