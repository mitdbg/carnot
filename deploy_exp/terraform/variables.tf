# Inputs for the experiments cluster. Every variable has a default, so `terraform apply` needs no
# tfvars; override on the CLI (-var experiment_max_nodes=2) or with a local *.auto.tfvars.

variable "aws_region" {
  description = "Region for the cluster. Must match the region of data_bucket (carnot-research is in us-east-1)."
  default     = "us-east-1"
}

variable "account_id" {
  description = "AWS account ID."
  default     = "422297141788"
}

variable "cluster_name" {
  description = "EKS cluster name. Distinct from the old app cluster name (carnot-unified-backend) so nothing left over can collide."
  default     = "qatfd-experiments"
}

variable "cluster_version" {
  description = "Kubernetes version. Indexed Jobs with backoffLimitPerIndex and native sidecars need >= 1.33."
  default     = "1.35"
}

variable "vpc_id" {
  description = "VPC for the cluster (the same VPC the app stacks use)."
  default     = "vpc-d27da3b7"
}

variable "subnet_ids" {
  description = "Subnets for the control plane and both node groups. us-east-1a/1c/1d, all of which offer r6id.12xlarge."
  type        = list(string)
  default     = ["subnet-27021a61", "subnet-09ffc721", "subnet-fe946a89"]
}

# --- system node group: hosts coredns + cluster-autoscaler, never scales to zero ---------------

variable "system_instance_type" {
  description = "Instance type for the always-on system node group."
  default     = "t3.medium"
}

# --- experiment node group: one pod per experiment cell, scales 0 -> N on demand ---------------

variable "experiment_instance_type" {
  description = "Instance type for experiment nodes. r6id.12xlarge = 48 vCPU, 384 GiB, 2 x 1425 GB NVMe instance store."
  default     = "r6id.12xlarge"
}

variable "experiment_max_nodes" {
  description = "Upper bound the cluster autoscaler may scale the experiment node group to."
  type        = number
  default     = 4
}

variable "experiment_capacity_type" {
  description = "ON_DEMAND or SPOT. Spot is ~60% cheaper but a reclaimed node kills every cell on it mid-run; only use it once resume-on-retry exists."
  default     = "ON_DEMAND"
}

variable "experiment_root_volume_gb" {
  description = "Root EBS volume of experiment nodes. Images and container layers only; pod data lives on the NVMe RAID0."
  type        = number
  default     = 100
}

# --- data and results -------------------------------------------------------------------------

variable "data_bucket" {
  description = "Bucket that is the source of truth for benchmark data (read-only from pods)."
  default     = "carnot-research"
}

variable "data_prefixes" {
  description = "Prefixes in data_bucket that pods may read (benchmark data + chroma stores)."
  type        = list(string)
  default     = ["officeqa", "browsecomp-plus", "financebench", "freshstack", "qampari", "biogen"]
}

variable "results_bucket" {
  description = "Bucket created by this stack for run results, per-pod store snapshots, and job manifests (read-write from pods)."
  default     = "qatfd-research-experiments"
}

variable "ecr_repository_name" {
  description = "ECR repository for the experiment image (deploy_exp/docker/Dockerfile)."
  default     = "qatfd-experiments"
}

variable "runner_namespace" {
  description = "Kubernetes namespace the experiment Jobs run in."
  default     = "experiments"
}

variable "runner_service_account" {
  description = "ServiceAccount (in runner_namespace) that experiment pods run as; bound to the runner IRSA role."
  default     = "experiment-runner"
}

variable "secrets_prefix" {
  description = "Secrets Manager path prefix pods may read (e.g. qatfd/experiments/llm holding OPENROUTER_API_KEY)."
  default     = "qatfd/experiments"
}
