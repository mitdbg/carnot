#!/usr/bin/env bash
# bootstrap_cluster.sh
#
# One-time setup script for the Carnot EKS cluster.
# Run this once after a new EKS cluster is created 
# It installs all cluster-level add-ons and creates the IAM roles needed
# for them to authenticate to AWS APIs via IRSA.
#
# Prerequisites:
#   - aws CLI configured with credentials that have IAM + EKS permissions
#   - kubectl connected to the target cluster (aws eks update-kubeconfig ...)
#   - helm installed (brew install helm)
#
# Usage:
#   ./bootstrap_cluster.sh <cluster-name> <region> <aws-account-id>
#
# Example:
#   ./bootstrap_cluster.sh carnot-unified-backend us-east-1 422297141788

set -euo pipefail

CLUSTER_NAME="${1:?Usage: $0 <cluster-name> <region> <aws-account-id>}"
REGION="${2:?Usage: $0 <cluster-name> <region> <aws-account-id>}"
ACCOUNT_ID="${3:?Usage: $0 <cluster-name> <region> <aws-account-id>}"

echo "==> Bootstrapping cluster: $CLUSTER_NAME (region: $REGION, account: $ACCOUNT_ID)"

# ---------------------------------------------------------------------------
# Step 1: Connect kubectl to the cluster
# ---------------------------------------------------------------------------
echo "==> Connecting kubectl..."
aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER_NAME"

# ---------------------------------------------------------------------------
# Step 2: Install core Kubernetes add-ons (managed by EKS)
#
# These are required for basic cluster operation:
#   - vpc-cni:    AWS VPC CNI plugin — assigns VPC IPs to pods. Without this,
#                 nodes join the cluster but stay NotReady (NetworkPluginNotReady).
#   - kube-proxy: Maintains iptables rules for Service routing on each node.
#   - coredns:    In-cluster DNS — required for pods to resolve service names.
# ---------------------------------------------------------------------------
echo "==> Installing core add-ons (vpc-cni, kube-proxy, coredns)..."
aws eks create-addon --cluster-name "$CLUSTER_NAME" --addon-name vpc-cni --region "$REGION" || echo "vpc-cni already exists, skipping"
aws eks create-addon --cluster-name "$CLUSTER_NAME" --addon-name kube-proxy --region "$REGION" || echo "kube-proxy already exists, skipping"
aws eks create-addon --cluster-name "$CLUSTER_NAME" --addon-name coredns --region "$REGION" || echo "coredns already exists, skipping"

echo "==> Waiting for core add-on pods to be ready..."
kubectl wait --for=condition=Ready pods -l k8s-app=aws-node -n kube-system --timeout=120s
kubectl wait --for=condition=Ready pods -l k8s-app=kube-dns -n kube-system --timeout=120s

# ---------------------------------------------------------------------------
# Step 3: Register the cluster's OIDC provider with IAM
#
# IRSA (IAM Roles for Service Accounts) lets pods assume IAM roles without
# storing credentials on disk. It works by federating the cluster's built-in
# OIDC provider with AWS IAM. This step registers that OIDC provider.
# ---------------------------------------------------------------------------
echo "==> Registering OIDC provider with IAM..."
OIDC_URL=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
  --query "cluster.identity.oidc.issuer" --output text)
OIDC_ID=$(echo "$OIDC_URL" | awk -F'/' '{print $NF}')
echo "    OIDC URL: $OIDC_URL"
echo "    OIDC ID:  $OIDC_ID"

aws iam create-open-id-connect-provider \
  --url "$OIDC_URL" \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 9e99a48a9960b14926bb7f3b02e22da2b0ab7280 \
  || echo "OIDC provider already exists, skipping"

# ---------------------------------------------------------------------------
# Step 4: EBS CSI Driver
#
# Enables Kubernetes PersistentVolumeClaims backed by EBS volumes.
# Required for the PostgreSQL StatefulSet (each environment gets a 50Gi gp3 PVC).
#
# The driver's controller pod calls EC2 APIs (CreateVolume, AttachVolume, etc.)
# so it needs an IAM role. We create an IRSA role scoped to the
# ebs-csi-controller-sa ServiceAccount in kube-system.
# ---------------------------------------------------------------------------
echo "==> Setting up EBS CSI Driver..."

# Create IRSA trust policy for the EBS CSI controller service account
python3 -c "
import json
policy = {
  'Version': '2012-10-17',
  'Statement': [{
    'Effect': 'Allow',
    'Principal': {'Federated': 'arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}'},
    'Action': 'sts:AssumeRoleWithWebIdentity',
    'Condition': {'StringEquals': {
      'oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}:aud': 'sts.amazonaws.com',
      'oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}:sub': 'system:serviceaccount:kube-system:ebs-csi-controller-sa'
    }}
  }]
}
print(json.dumps(policy))
" > /tmp/ebs-csi-trust-policy.json

aws iam create-role \
  --role-name carnot-ebs-csi-driver-role \
  --assume-role-policy-document file:///tmp/ebs-csi-trust-policy.json \
  || echo "Role already exists, skipping"

aws iam attach-role-policy \
  --role-name carnot-ebs-csi-driver-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy

# Install the add-on with the IRSA role ARN so EKS wires up the service account
# annotation before the controller pod starts — avoids a race where the pod
# launches without AWS credentials and needs a manual restart.
aws eks create-addon \
  --cluster-name "$CLUSTER_NAME" \
  --addon-name aws-ebs-csi-driver \
  --region "$REGION" \
  --service-account-role-arn "arn:aws:iam::${ACCOUNT_ID}:role/carnot-ebs-csi-driver-role" \
  || echo "aws-ebs-csi-driver already exists, skipping"

kubectl wait --for=condition=Ready pods -l app=ebs-csi-controller -n kube-system --timeout=120s

# ---------------------------------------------------------------------------
# Step 5: AWS Load Balancer Controller
#
# Manages the shared ALB that routes traffic to all environment namespaces.
# Uses host-based routing (api-{env}.carnot-research.org → namespace/{env}).
#
# Like the EBS CSI driver, it needs an IAM role to call ELB/EC2 APIs.
# The controller's IAM policy is large (managing ALBs, target groups, etc.)
# so we download the AWS-maintained policy document rather than writing it by hand.
# ---------------------------------------------------------------------------
echo "==> Setting up AWS Load Balancer Controller..."

# Create IRSA trust policy for the ALB controller service account
python3 -c "
import json
policy = {
  'Version': '2012-10-17',
  'Statement': [{
    'Effect': 'Allow',
    'Principal': {'Federated': 'arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}'},
    'Action': 'sts:AssumeRoleWithWebIdentity',
    'Condition': {'StringEquals': {
      'oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}:aud': 'sts.amazonaws.com',
      'oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}:sub': 'system:serviceaccount:kube-system:aws-load-balancer-controller'
    }}
  }]
}
print(json.dumps(policy))
" > /tmp/alb-trust-policy.json

aws iam create-role \
  --role-name carnot-alb-controller-role \
  --assume-role-policy-document file:///tmp/alb-trust-policy.json \
  || echo "Role already exists, skipping"

# Download the AWS-maintained ALB controller IAM policy and attach it
curl -sf -o /tmp/alb-iam-policy.json \
  https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/main/docs/install/iam_policy.json

aws iam create-policy \
  --policy-name carnot-alb-controller-policy \
  --policy-document file:///tmp/alb-iam-policy.json \
  || echo "Policy already exists, skipping"

aws iam attach-role-policy \
  --role-name carnot-alb-controller-role \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/carnot-alb-controller-policy

# Get the VPC ID — the ALB controller needs this explicitly because it runs
# in a pod (not on EC2) and cannot auto-detect it from instance metadata.
VPC_ID=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
  --query "cluster.resourcesVpcConfig.vpcId" --output text)
echo "    VPC ID: $VPC_ID"

helm repo add eks https://aws.github.io/eks-charts 2>/dev/null || true
helm repo update

helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName="$CLUSTER_NAME" \
  --set region="$REGION" \
  --set vpcId="$VPC_ID"

kubectl annotate serviceaccount aws-load-balancer-controller -n kube-system \
  eks.amazonaws.com/role-arn=arn:aws:iam::${ACCOUNT_ID}:role/carnot-alb-controller-role \
  --overwrite

kubectl rollout restart deployment aws-load-balancer-controller -n kube-system
kubectl wait --for=condition=Ready pods -l app.kubernetes.io/name=aws-load-balancer-controller -n kube-system --timeout=120s

# ---------------------------------------------------------------------------
# Step 6: External Secrets Operator
#
# Syncs secrets from AWS Secrets Manager into Kubernetes Secrets at pod startup.
# Each environment's ExternalSecret pulls from carnot/{env}/db and carnot/{env}/app,
# materialising them as a Kubernetes Secret mounted at /run/secrets/ in the
# backend pod (the path database.py already reads from).
# ---------------------------------------------------------------------------
echo "==> Installing External Secrets Operator..."
helm repo add external-secrets https://charts.external-secrets.io 2>/dev/null || true
helm repo update

helm upgrade --install external-secrets external-secrets/external-secrets \
  --namespace external-secrets \
  --create-namespace

kubectl wait --for=condition=Ready pods -l app.kubernetes.io/name=external-secrets -n external-secrets --timeout=120s

# ---------------------------------------------------------------------------
# Step 7: IRSA role for External Secrets Operator
#
# The ESO controller pod must call secretsmanager:GetSecretValue to sync secrets.
# We create an IAM role scoped to the ESO ServiceAccount via IRSA, then annotate
# the ServiceAccount so the pod inherits the credentials automatically.
# ---------------------------------------------------------------------------
echo "==> Setting up IRSA role for External Secrets Operator..."

python3 -c "
import json
policy = {
  'Version': '2012-10-17',
  'Statement': [{
    'Effect': 'Allow',
    'Principal': {'Federated': 'arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}'},
    'Action': 'sts:AssumeRoleWithWebIdentity',
    'Condition': {'StringEquals': {
      'oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}:aud': 'sts.amazonaws.com',
      'oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}:sub': 'system:serviceaccount:external-secrets:external-secrets'
    }}
  }]
}
print(json.dumps(policy))
" > /tmp/eso-trust-policy.json

aws iam create-role \
  --role-name carnot-external-secrets-role \
  --assume-role-policy-document file:///tmp/eso-trust-policy.json \
  || echo "Role already exists, skipping"

aws iam put-role-policy \
  --role-name carnot-external-secrets-role \
  --policy-name SecretsManagerRead \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"secretsmanager:GetSecretValue\",\"secretsmanager:DescribeSecret\"],\"Resource\":\"arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:carnot/*\"}]}"

kubectl annotate serviceaccount external-secrets -n external-secrets \
  eks.amazonaws.com/role-arn=arn:aws:iam::${ACCOUNT_ID}:role/carnot-external-secrets-role \
  --overwrite

kubectl rollout restart deployment external-secrets -n external-secrets
kubectl wait --for=condition=Ready pods -l app.kubernetes.io/name=external-secrets -n external-secrets --timeout=120s

# ---------------------------------------------------------------------------
# Step 8: ClusterSecretStore
#
# The ExternalSecret resources in each Helm release reference a ClusterSecretStore
# named 'aws-secrets-manager'. This store must exist before any Carnot environment
# is deployed, otherwise ExternalSecret sync will fail with "store not found".
#
# Authentication uses the ESO ServiceAccount's IRSA role (set up in Step 7).
# ---------------------------------------------------------------------------
echo "==> Creating ClusterSecretStore..."

kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: ${REGION}
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets
            namespace: external-secrets
EOF

echo "==> Waiting for ClusterSecretStore to become ready..."
kubectl wait clustersecretstore/aws-secrets-manager \
  --for=condition=Ready \
  --timeout=60s

# ---------------------------------------------------------------------------
# Step 9: Backend ServiceAccount IAM roles (one per environment)
#
# Each Carnot environment's backend pod uses a dedicated IAM role (IRSA) to
# access its own S3 bucket and Secrets Manager path. The role ARN is referenced
# in each values-{env}.yaml and annotated onto the carnot-backend ServiceAccount
# by the Helm chart.
#
# Environments: dev, wl-prod, lm-prod, at-prod
# ---------------------------------------------------------------------------
echo "==> Creating per-environment backend ServiceAccount IAM roles..."

for ENV in dev wl-prod lm-prod at-prod; do
  echo "    -> carnot-${ENV}-sa-role"

  python3 -c "
import json
policy = {
  'Version': '2012-10-17',
  'Statement': [{
    'Effect': 'Allow',
    'Principal': {'Federated': 'arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}'},
    'Action': 'sts:AssumeRoleWithWebIdentity',
    'Condition': {'StringEquals': {
      'oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}:aud': 'sts.amazonaws.com',
      'oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}:sub': 'system:serviceaccount:${ENV}:carnot-backend'
    }}
  }]
}
print(json.dumps(policy))
" > /tmp/backend-sa-trust-policy.json

  aws iam create-role \
    --role-name "carnot-${ENV}-sa-role" \
    --assume-role-policy-document file:///tmp/backend-sa-trust-policy.json \
    || echo "    Role already exists, skipping"

  aws iam put-role-policy \
    --role-name "carnot-${ENV}-sa-role" \
    --policy-name S3AndSecretsAccess \
    --policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [
        {
          \"Effect\": \"Allow\",
          \"Action\": [\"s3:ListBucket\"],
          \"Resource\": [\"arn:aws:s3:::carnot-research-${ENV}\"]
        },
        {
          \"Effect\": \"Allow\",
          \"Action\": [\"s3:*\"],
          \"Resource\": [
            \"arn:aws:s3:::carnot-research-${ENV}\",
            \"arn:aws:s3:::carnot-research-${ENV}/*\"
          ]
        },
        {
          \"Effect\": \"Allow\",
          \"Action\": [
            \"secretsmanager:GetSecretValue\",
            \"secretsmanager:DescribeSecret\"
          ],
          \"Resource\": [
            \"arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:carnot/${ENV}/*\"
          ]
        }
      ]
    }"
done

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "==> Cluster bootstrap complete. Summary:"
echo "    kubectl get nodes"
kubectl get nodes
echo ""
echo "    kubectl get pods -n kube-system"
kubectl get pods -n kube-system
echo ""
echo "Next step: deploy Carnot environments via Helm."
echo "  cd deploy && helm upgrade --install carnot helm/carnot --namespace dev --create-namespace --values helm/carnot/values-dev.yaml ..."
