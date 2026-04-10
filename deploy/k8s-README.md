# Carnot EKS Deployment

This document covers the Kubernetes deployment path for Carnot. The EKS stack runs
alongside the existing EC2 stack during migration; each environment is a separate
namespace in one shared cluster.

For the overall migration plan and current status, see `deploy/migration.md`.

---

## Architecture

```
Terraform (manual, single apply)
  deploy/terraform/eks/
    → EKS cluster, node group, OIDC, IAM/IRSA roles
    → Helm releases: ALB Controller, ESO, Cluster Autoscaler, ClusterSecretStore

Push to k8s branch
  → GitHub Actions (deploy-k8s.yaml)
      → Build & push images to Docker Hub
      → helm upgrade --install (per environment namespace)
          → AWS Load Balancer Controller provisions shared ALB

Route53 (manual)
  → Operator creates A alias records pointing at the ALB

Browser
  → Route53 (A alias)
  → k8s ALB (shared across all environments via IngressGroup "carnot")
  → frontend Service → NGINX pod (React SPA)
  → backend Service  → FastAPI pod
                          → PostgreSQL StatefulSet (EBS-backed PVC)
                          → S3 (file storage, via IRSA)
                          → Secrets Manager (credentials, via ESO)
```

### Key components

| Component | Managed by | Notes |
|---|---|---|
| EKS cluster + node group | Terraform (`eks/`) | Manual apply only |
| OIDC provider + IRSA roles | Terraform (`eks/`) | Manual apply only |
| EKS managed addons | Terraform (`eks/`) | Manual apply only |
| ALB controller, ESO, Cluster Autoscaler | Terraform (`eks/addons.tf`) | Manual apply only |
| ClusterSecretStore | Terraform (`eks/addons.tf`) | Via local Helm chart (`helm/cluster-config/`) |
| Carnot app (per namespace) | Helm (`helm/carnot/`) | Automated via CI |
| Route53 DNS records | Manual (AWS console or CLI) | After first Helm deploy creates ALB |

---

## Relevant files

```
deploy/
  terraform/
    eks/                    # Cluster infrastructure — manual terraform
      backend.tf            # S3 remote state at tf/eks/terraform.tfstate
      main.tf               # EKS cluster, node group, OIDC
      iam.tf                # All IAM roles (cluster, node, IRSA roles per component)
      addons.tf             # EKS managed addons + Helm releases (ALB, ESO, CA, ClusterSecretStore)
      variables.tf          # All inputs (all have defaults — no tfvars needed)
      outputs.tf            # Cluster endpoint, IRSA role ARNs

  helm/
    cluster-config/         # Cluster-level config (ClusterSecretStore) — deployed by Terraform
    carnot/                 # Application Helm chart
    values.yaml             # Default values (image tags set by CI at deploy time)
    values-dev.yaml         # dev environment overrides
    values-wl-prod.yaml     # wl-prod environment overrides
    values-lm-prod.yaml     # lm-prod environment overrides
    values-at-prod.yaml     # at-prod environment overrides
    templates/              # Kubernetes manifests (Deployment, Service, Ingress, etc.)

  bootstrap_cluster.sh      # SUPERSEDED — kept for reference only.
                            # All steps are now covered by the terraform module.

.github/workflows/
  deploy-k8s.yaml           # CI: build → Docker Hub push → Helm deploy
```

---

## Initial cluster setup (run once)

These steps are performed by an operator. They are **not** automated in CI.

### Prerequisites

- `aws` CLI configured with credentials that can create EKS, IAM, ECR resources
- `terraform` >= 1.5
- `helm` >= 3
- `kubectl`

### 1. Complete any pending AWS cleanup

Before first apply, ensure the old cluster's resources are gone:

```bash
# Delete the old cluster OIDC provider (replace OIDC_ID with your value)
aws iam delete-open-id-connect-provider \
  --open-id-connect-provider-arn \
  arn:aws:iam::422297141788:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/<OIDC_ID>

# Remove stale Route53 records pointing at the old k8s ALB
# Look up current values first, then delete:
aws route53 list-resource-record-sets --hosted-zone-id Z0371749174EVZHL80QS0 \
  --query "ResourceRecordSets[?contains(Name, 'k8s-dev')]"
```

### 2. Run Terraform

```bash
cd deploy/terraform/eks
terraform init
terraform plan    # review — should be all creates, no updates or destroys
terraform apply   # single pass, ~25 minutes
```

This creates (in dependency order):
- IAM roles for the control plane and node group
- EKS cluster (`carnot-unified-backend`, Kubernetes 1.35)
- Managed node group (`carnot-nodes`, t3.large, 1–3 nodes)
- OIDC provider for IRSA
- EKS managed addons (vpc-cni, kube-proxy, coredns, aws-ebs-csi-driver)
- Helm releases: AWS Load Balancer Controller, External Secrets Operator, Cluster Autoscaler
- ClusterSecretStore (tells ESO how to read from Secrets Manager)
- IRSA roles for all components and per-environment backend ServiceAccounts

### 3. Connect kubectl

```bash
aws eks update-kubeconfig --region us-east-1 --name carnot-unified-backend
kubectl get nodes   # should show carnot-nodes instances as Ready
```

### 4. Verify Secrets Manager secrets exist

The following secrets must exist before deploying the app. They are created manually
and are intentionally outside Terraform (they contain passwords and encryption keys).

```
carnot/dev/db       → { db_user, db_password }
carnot/dev/app      → { settings_encryption_key }
carnot/dockerhub    → { username, token }
```

Create any missing ones:
```bash
aws secretsmanager create-secret --name carnot/dev/db \
  --secret-string '{"db_user":"carnot","db_password":"<password>"}'
```

### 5. Push to the k8s branch

```bash
git push origin k8s
```

This triggers `deploy-k8s.yaml`, which:
1. Builds backend and frontend images and pushes to Docker Hub
2. Runs `helm upgrade --install` for the `dev` namespace
   - The ALB Controller sees the new Ingress and provisions the shared k8s ALB

### 6. Create Route53 records (manual)

After the first push, the ALB Controller creates the shared ALB. Get its DNS name:

```bash
kubectl get ingress -n dev -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}'
```

Then create A alias records in the Route53 console (or CLI) for:
- `k8s-dev.carnot-research.org` → ALB
- `api-k8s-dev.carnot-research.org` → ALB

After DNS propagates, the application will be accessible at those URLs.

---

## Day-to-day: deploying application changes

Push to the `k8s` branch. The `deploy-k8s.yaml` workflow builds images, pushes to
Docker Hub, and deploys via Helm.

**No manual steps required for application deployments.**

---

## Adding a new environment (e.g. wl-prod)

1. **Create Secrets Manager secrets** for the new environment:
   ```bash
   aws secretsmanager create-secret --name carnot/wl-prod/db \
     --secret-string '{"db_user":"carnot","db_password":"<password>"}'
   aws secretsmanager create-secret --name carnot/wl-prod/app \
     --secret-string '{"settings_encryption_key":"<key>"}'
   ```

2. **Verify the Helm values file** (`deploy/helm/carnot/values-wl-prod.yaml`) has
   correct hostnames, ACM cert ARN, and IRSA role ARN.
   The IRSA role ARN is output by Terraform:
   ```bash
   cd deploy/terraform/eks && terraform output backend_sa_role_arns
   ```

3. **Add a deploy job** to `deploy-k8s.yaml` for the new environment, mirroring the
   existing `dev` job.

4. **Create Route53 records** manually for the new environment hostnames, pointing
   at the same shared ALB.

---

## Modifying cluster infrastructure

Any change to `deploy/terraform/eks/` (node group size, Kubernetes version, new IRSA
role, etc.) must be applied manually by an operator:

```bash
cd deploy/terraform/eks
terraform plan    # always review before applying
terraform apply
```

**Do not add `terraform apply` to CI.** Cluster-level changes carry significant risk
and must be reviewed before execution.

---

## Cluster teardown and recreation

If you need to destroy and recreate the cluster (e.g. to pick up a new Kubernetes
version or make breaking infrastructure changes):

1. Delete the app Helm releases first — this lets the ALB Controller clean up the ALB:
   ```bash
   helm uninstall carnot -n dev
   # Wait for ALB to be deleted before proceeding
   ```

2. Delete PVCs to trigger EBS volume cleanup:
   ```bash
   kubectl delete pvc --all -n dev
   ```

3. Destroy the cluster:
   ```bash
   cd deploy/terraform/eks && terraform destroy
   ```
   This tears down the cluster, node group, addons, IRSA roles, and Helm releases
   in the correct dependency order.

4. Delete the Route53 records manually (they point at the now-deleted ALB).

5. Follow the **Initial cluster setup** steps above to rebuild.