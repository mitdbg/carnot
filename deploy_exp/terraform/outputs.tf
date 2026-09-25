output "cluster_name" {
  value = aws_eks_cluster.experiments.name
}

output "cluster_endpoint" {
  value = aws_eks_cluster.experiments.endpoint
}

output "region" {
  value = var.aws_region
}

output "kubeconfig_command" {
  description = "Run this once after apply to point kubectl at the cluster."
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${aws_eks_cluster.experiments.name}"
}

output "oidc_issuer" {
  value = local.oidc_issuer
}

output "runner_role_arn" {
  description = "Annotate the experiment-runner ServiceAccount with eks.amazonaws.com/role-arn = this."
  value       = aws_iam_role.runner.arn
}

output "runner_namespace" {
  value = var.runner_namespace
}

output "runner_service_account" {
  value = var.runner_service_account
}

output "ecr_repository_url" {
  description = "Image repository; deploy_exp/docker/build_push.sh pushes <url>:<git-sha> and <url>:latest."
  value       = aws_ecr_repository.experiments.repository_url
}

output "results_bucket" {
  value = aws_s3_bucket.results.bucket
}

output "data_bucket" {
  value = var.data_bucket
}

output "experiment_node_selector" {
  description = "nodeSelector + toleration key/value the Job pods must carry to land on the experiment nodes."
  value = {
    (local.workload_label) = local.workload_value
  }
}
