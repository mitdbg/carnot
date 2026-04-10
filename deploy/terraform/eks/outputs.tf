output "cluster_endpoint" {
  description = "EKS control plane endpoint."
  value       = aws_eks_cluster.carnot.endpoint
}

output "cluster_ca_data" {
  description = "Base64-encoded certificate authority data for the cluster."
  value       = aws_eks_cluster.carnot.certificate_authority[0].data
}

output "oidc_issuer" {
  description = "OIDC issuer URL (without https://) — used in IRSA trust policy conditions."
  value       = local.oidc_issuer
}

output "backend_sa_role_arns" {
  description = "Map of environment name to backend ServiceAccount IAM role ARN."
  value       = { for env, role in aws_iam_role.backend_sa : env => role.arn }
}
