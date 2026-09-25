# -------------------------------
# EKS managed addons
# -------------------------------
resource "aws_eks_addon" "vpc_cni" {
  cluster_name                = aws_eks_cluster.experiments.name
  addon_name                  = "vpc-cni"
  resolve_conflicts_on_create = "OVERWRITE"
  tags                        = local.common_tags
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name                = aws_eks_cluster.experiments.name
  addon_name                  = "kube-proxy"
  resolve_conflicts_on_create = "OVERWRITE"
  tags                        = local.common_tags
}

# coredns needs a node to schedule onto before the addon reports ACTIVE.
resource "aws_eks_addon" "coredns" {
  cluster_name                = aws_eks_cluster.experiments.name
  addon_name                  = "coredns"
  resolve_conflicts_on_create = "OVERWRITE"
  tags                        = local.common_tags

  depends_on = [aws_eks_node_group.system]
}

# No EBS CSI driver: pods keep everything on the node's NVMe emptyDir and in S3; nothing uses a PVC.

# -------------------------------
# Cluster autoscaler (Helm). Auto-discovers both node groups via their tags, scales the experiment
# group from 0 when Job pods are Pending and back to 0 once they are done.
#
# The Job pods MUST carry the annotation
#   cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
# otherwise the autoscaler will happily evict a running cell to compact a half-empty node.
# -------------------------------
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

  # Keep it on the system node group, never on an experiment node that may scale away.
  set {
    name  = "nodeSelector.${replace(local.workload_label, ".", "\\.")}"
    value = "system"
  }

  set {
    name  = "priorityClassName"
    value = "system-cluster-critical"
  }

  # Pick the node group that wastes the least after placing the pending pods (only matters once
  # there is more than one schedulable group), and give finished cells' nodes back quickly.
  set {
    name  = "extraArgs.expander"
    value = "least-waste"
  }

  set {
    name  = "extraArgs.scale-down-unneeded-time"
    value = "5m"
  }

  set {
    name  = "extraArgs.scale-down-delay-after-add"
    value = "5m"
  }

  # A node with one 44 GiB cell on it is "underutilized" by the default 50% threshold; with the
  # safe-to-evict=false annotation on the pods the autoscaler still leaves it alone until the cell
  # ends, this just stops it from logging about it every loop.
  set {
    name  = "extraArgs.scale-down-utilization-threshold"
    value = "0.2"
  }

  set {
    name  = "resources.requests.cpu"
    value = "100m"
  }

  set {
    name  = "resources.requests.memory"
    value = "128Mi"
  }

  set {
    name  = "resources.limits.memory"
    value = "512Mi"
  }

  depends_on = [
    aws_eks_node_group.system,
    aws_eks_addon.coredns,
    aws_iam_role_policy.cluster_autoscaler,
  ]
}
