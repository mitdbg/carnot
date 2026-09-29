# Experiments cluster: a self-contained EKS cluster for running qatfd experiment cells in parallel,
# one pod per cell, each pod with its own Chroma server over a pristine store copy on node NVMe.
#
# Independent of the old app deployment under deploy/ (that cluster has been torn down): different
# cluster name, role names, state key and bucket, so either stack can be applied or destroyed
# without touching the other. Apply is manual (see README.md); nothing here runs from CI.
#
# Layout
#   main.tf      cluster, OIDC provider, system node group, experiment node group + launch template
#   iam.tf       cluster / node roles, cluster-autoscaler IRSA role, experiment-runner IRSA role
#   addons.tf    vpc-cni, kube-proxy, coredns, cluster-autoscaler (Helm)
#   storage.tf   results bucket, ECR repository
#   outputs.tf   what the Kubernetes manifests and the image build need

provider "aws" {
  region = var.aws_region
}

provider "helm" {
  kubernetes {
    host                   = aws_eks_cluster.experiments.endpoint
    cluster_ca_certificate = base64decode(aws_eks_cluster.experiments.certificate_authority[0].data)

    # exec-based auth: a static aws_eks_cluster_auth token expires after 15 min, which is shorter
    # than cluster + node group creation (~20 min); exec fetches a fresh token per API call.
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", var.cluster_name, "--region", var.aws_region]
    }
  }
}

locals {
  oidc_issuer = trimprefix(aws_eks_cluster.experiments.identity[0].oidc[0].issuer, "https://")
  oidc_arn    = "arn:aws:iam::${var.account_id}:oidc-provider/${local.oidc_issuer}"

  workload_label = "qatfd.io/workload"
  workload_value = "experiments"

  common_tags = {
    Project   = "qatfd-experiments"
    ManagedBy = "terraform"
    Stack     = "deploy_exp/terraform"
  }

  # nodeadm NodeConfig for the experiment nodes (AL2023). Managed node groups inject the `cluster`
  # section themselves, so only node-local settings go here.
  #
  # localStorage.strategy = RAID0 stripes the instance-store NVMe disks into one array and moves the
  # containerd and kubelet directories onto it, so image layers and every pod's emptyDir volumes
  # land on NVMe instead of the root EBS volume. That is what gives each pod a fast local copy of
  # its chroma store without any PVC. The instance store is blank on every boot, which is fine:
  # pods pull their data from S3 at start.
  experiment_user_data = <<-EOT
    MIME-Version: 1.0
    Content-Type: multipart/mixed; boundary="==NODEADM=="

    --==NODEADM==
    Content-Type: application/node.eks.aws

    ---
    apiVersion: node.eks.aws/v1alpha1
    kind: NodeConfig
    spec:
      instance:
        localStorage:
          strategy: RAID0
    --==NODEADM==--
  EOT
}

# -------------------------------
# EKS cluster
# -------------------------------
resource "aws_eks_cluster" "experiments" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = var.cluster_version

  # api + audit only: the app cluster logs everything, but here CloudWatch ingest would be pure cost.
  enabled_cluster_log_types = ["api", "audit"]

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  access_config {
    authentication_mode = "API_AND_CONFIG_MAP"
    # The IAM principal running `terraform apply` becomes cluster-admin, so kubectl works right away.
    bootstrap_cluster_creator_admin_permissions = true
  }

  tags = local.common_tags

  depends_on = [aws_iam_role_policy_attachment.cluster_policy]
}

# -------------------------------
# OIDC provider for IRSA
# -------------------------------
resource "aws_iam_openid_connect_provider" "experiments" {
  url             = aws_eks_cluster.experiments.identity[0].oidc[0].issuer
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["9e99a48a9960b14926bb7f3b02e22da2b0ab7280"]

  tags = local.common_tags
}

# -------------------------------
# System node group: coredns + cluster-autoscaler. Small, always on.
# -------------------------------
resource "aws_eks_node_group" "system" {
  cluster_name    = aws_eks_cluster.experiments.name
  node_group_name = "system-nodes"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.subnet_ids
  ami_type        = "AL2023_x86_64_STANDARD"
  instance_types  = [var.system_instance_type]
  capacity_type   = "ON_DEMAND"
  disk_size       = 40

  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 2
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    (local.workload_label) = "system"
  }

  tags = merge(local.common_tags, {
    "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
    "k8s.io/cluster-autoscaler/enabled"             = "true"
  })

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }

  depends_on = [
    aws_iam_role_policy_attachment.node_worker,
    aws_iam_role_policy_attachment.node_cni,
    aws_iam_role_policy_attachment.node_ecr,
  ]
}

# -------------------------------
# Experiment node group: r6id.12xlarge, scales 0 -> experiment_max_nodes.
# -------------------------------

# A launch template is the only way to hand nodeadm the NodeConfig above. With a launch template,
# the root volume must be declared here (disk_size is not allowed on the node group); do NOT set
# an instance profile, security groups or an AMI in it, EKS supplies those for managed node groups.
resource "aws_launch_template" "experiment" {
  name_prefix            = "${var.cluster_name}-experiment-"
  update_default_version = true

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = var.experiment_root_volume_gb
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  user_data = base64encode(local.experiment_user_data)

  tag_specifications {
    resource_type = "instance"
    tags          = merge(local.common_tags, { Name = "${var.cluster_name}-experiment-node" })
  }

  tag_specifications {
    resource_type = "volume"
    tags          = local.common_tags
  }

  tags = local.common_tags
}

resource "aws_eks_node_group" "experiment" {
  cluster_name    = aws_eks_cluster.experiments.name
  node_group_name = "experiment-nodes"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.subnet_ids
  ami_type        = "AL2023_x86_64_STANDARD"
  instance_types  = [var.experiment_instance_type]
  capacity_type   = var.experiment_capacity_type

  launch_template {
    id      = aws_launch_template.experiment.id
    version = aws_launch_template.experiment.latest_version
  }

  # Sits at zero between sweeps. The cluster autoscaler raises desired_size when Job pods that
  # tolerate the taint below are Pending, and lowers it again once they finish.
  scaling_config {
    desired_size = 0
    min_size     = 0
    max_size     = var.experiment_max_nodes
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    (local.workload_label) = local.workload_value
  }

  # Only pods that explicitly tolerate this taint (the experiment Jobs) land on these nodes, so a
  # stray system workload can never pin a 384 GiB node up.
  taint {
    key    = local.workload_label
    value  = local.workload_value
    effect = "NO_SCHEDULE"
  }

  tags = merge(local.common_tags, {
    "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
    "k8s.io/cluster-autoscaler/enabled"             = "true"
    # Scale-from-zero hints (documentation here; the copies that matter are on the ASG, see
    # aws_autoscaling_group_tag.experiment below: EKS does not propagate node group tags to the ASG).
    "k8s.io/cluster-autoscaler/node-template/label/${local.workload_label}" = local.workload_value
    "k8s.io/cluster-autoscaler/node-template/taint/${local.workload_label}" = "${local.workload_value}:NoSchedule"
    "k8s.io/cluster-autoscaler/node-template/resources/ephemeral-storage"   = local.experiment_ephemeral_storage
  })

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }

  depends_on = [
    aws_iam_role_policy_attachment.node_worker,
    aws_iam_role_policy_attachment.node_cni,
    aws_iam_role_policy_attachment.node_ecr,
  ]
}

# The autoscaler builds its scale-from-zero template node for an empty group from the ASG's
# `k8s.io/cluster-autoscaler/node-template/*` tags (labels and taints it can also read through
# eks:DescribeNodegroup; resources it cannot). EKS does not copy node group tags onto the ASG it
# manages, so they are set on the ASG directly here. Without the ephemeral-storage one the template
# node advertises no disk, every pod that requests ephemeral-storage is "unschedulable", and the
# group never leaves zero (2026-09-26: the first smoke test sat Pending on exactly this).
locals {
  # what a node actually has under kubelet: the 2 x 1425 GB instance-store RAID0, minus filesystem and
  # kubelet reserves. Only has to admit the pods' requests; the live node reports its true allocatable.
  experiment_ephemeral_storage = "2500Gi"
  experiment_asg_tags = {
    "k8s.io/cluster-autoscaler/node-template/label/${local.workload_label}" = local.workload_value
    "k8s.io/cluster-autoscaler/node-template/taint/${local.workload_label}" = "${local.workload_value}:NoSchedule"
    "k8s.io/cluster-autoscaler/node-template/resources/ephemeral-storage"   = local.experiment_ephemeral_storage
  }
}

resource "aws_autoscaling_group_tag" "experiment" {
  for_each = local.experiment_asg_tags

  autoscaling_group_name = aws_eks_node_group.experiment.resources[0].autoscaling_groups[0].name

  tag {
    key                 = each.key
    value               = each.value
    propagate_at_launch = false
  }
}

# Suspend AZRebalance on the experiment node group's ASG. With the cluster-autoscaler managing capacity, every
# scale-down that empties an AZ made the ASG launch a node in that AZ ("to balance the group's zones"), overshoot
# desired capacity, and then TERMINATE A BUSY NODE to get back down, killing whatever cells it ran (2026-09-29
# codex-v1r2 sweep: four collection-agent cells restarted from question 0). Rebalancing ignores pods and the
# safe-to-evict annotation, and nothing here needs it: the autoscaler alone decides where nodes go. There is no
# native resource for suspending a process on a managed node group's ASG, so the aws cli does it; it re-runs
# whenever the node group (and hence its ASG) is replaced. Needs the aws cli on the machine running terraform.
resource "terraform_data" "experiment_suspend_az_rebalance" {
  triggers_replace = [aws_eks_node_group.experiment.resources[0].autoscaling_groups[0].name]

  provisioner "local-exec" {
    command = "aws autoscaling suspend-processes --region ${var.aws_region} --auto-scaling-group-name ${aws_eks_node_group.experiment.resources[0].autoscaling_groups[0].name} --scaling-processes AZRebalance"
  }
}
