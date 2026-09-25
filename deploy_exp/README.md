# deploy_exp — parallel experiments on EKS

Self-contained deployment for running qatfd experiment cells in parallel: one Kubernetes pod per
cell, each pod with its own Chroma server over a pristine store copy on node-local NVMe, results
synced to S3. Independent of the old app deployment under `deploy/`, which can be deleted once this
is in use.

```
deploy_exp/
  terraform/        EKS cluster + node groups + IAM + results bucket + ECR   (manual apply, see below)
  docker/           the experiment image, its pinned requirements, build/push script, local compose smoke test
  helm/experiments/ the sweep chart: one release = one sweep = one Indexed Job, one pod per cell
```

The cluster-autoscaler is the one thing terraform installs into the cluster, because
scale-from-zero of the experiment node group depends on it; everything else in the cluster comes
from the Helm chart below.

## Cluster (terraform/)

What it creates, in dependency order: IAM roles, the EKS cluster `qatfd-experiments` (us-east-1,
same VPC and subnets as before), the OIDC provider, a 1-node `system-nodes` group (t3.medium, for
coredns and the autoscaler), the `experiment-nodes` group (r6id.12xlarge, 0–4 nodes, tainted
`qatfd.io/workload=experiments:NoSchedule`, NVMe RAID0 under kubelet via a nodeadm launch
template), the managed addons, the cluster-autoscaler Helm release, the `qatfd-research-experiments`
bucket, the `qatfd-experiments` ECR repository, and the `qatfd-experiments-runner-role` IRSA role
that pods use to read `carnot-research/<benchmark>/*`, read/write the results bucket, and read
Secrets Manager under `qatfd/experiments/*`.

```bash
export AWS_PROFILE=mit AWS_REGION=us-east-1
cd deploy_exp/terraform
terraform init
terraform plan          # all creates on the first run
terraform apply         # ~20 min; the cluster-autoscaler release waits on the system node group
$(terraform output -raw kubeconfig_command)
kubectl get nodes       # one system node Ready; experiment group sits at 0 until Jobs are submitted
```

Secrets are created by hand, outside terraform (same convention as before):

```bash
aws secretsmanager create-secret --name qatfd/experiments/llm \
  --secret-string '{"OPENROUTER_API_KEY":"sk-or-..."}'
```

Things the Job manifests must carry, or the cluster misbehaves:

- `nodeSelector` and a toleration for `qatfd.io/workload=experiments` (output `experiment_node_selector`).
- `serviceAccountName: experiment-runner` in namespace `experiments`, the ServiceAccount annotated
  `eks.amazonaws.com/role-arn: <output runner_role_arn>`; the names are terraform variables if you
  want different ones.
- The pod annotation `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"`, otherwise the
  autoscaler evicts running cells to compact half-empty nodes.
- Memory limits per container; the runner and the chroma sidecar each get their own so a leak in
  one cannot take the other down.

Node sizing (r6id.12xlarge: 48 vCPU, 384 GiB, 2 x 1425 GB NVMe) is set per benchmark in the
chart's `benchmarks:` profiles (`helm/experiments/values.yaml`): officeqa cells get a 56 Gi chroma
limit and a 24 Gi runner limit, so 4 cells fit per node and the default parallelism of 16 fills
4 nodes. The experiment group scales back to zero a few minutes after the last cell finishes.
Tear everything down with `terraform destroy` (delete any Jobs first so the autoscaler is not
fighting the node group deletion).

## Sweeps (helm/experiments/)

A release is a sweep. Its values carry the cell list (`cells:`), and the chart renders a ConfigMap
holding it as `cells.json` plus an Indexed Job with `completions = len(cells)`. Each pod reads its
completion index, looks up its cell, and runs three containers in order: `fetch-data` pulls the
cell's chroma store and benchmark files from `carnot-research` onto the NVMe-backed `/data`
emptyDir; `chroma` is a native sidecar serving that store on `127.0.0.1:8001` with its own memory
limit; `runner` waits for the collection to answer, runs the cell's argv from `/app/qatfd` with
`results_root=/results`, mirrors `/results` to the results bucket every minute and at exit, and
writes a status json under `jobs/<release>/`. The in-pod choreography lives in `files/*.sh` and is
shipped as a ConfigMap, so it can change without an image rebuild. Failed cells are retried once in
a fresh pod (`backoffLimitPerIndex`); node disruptions do not count against that budget.

One-time per namespace, then the smoke test:

```bash
kubectl create namespace experiments
kubectl -n experiments create secret generic llm-keys --from-literal=OPENROUTER_API_KEY=sk-or-...
helm upgrade --install smoke deploy_exp/helm/experiments -n experiments \
    -f deploy_exp/helm/experiments/values-smoke.yaml \
    --set serviceAccount.roleArn=$(cd deploy_exp/terraform && terraform output -raw runner_role_arn) \
    --set image.tag=<git sha from build_push.sh>
kubectl -n experiments get pods -w        # Pending until the autoscaler adds an r6id node (~3 min), then Init -> Running
```

The ServiceAccount is created by the first release in the namespace; later releases in the same
namespace set `serviceAccount.create=false`. `values.yaml` documents every knob and the cell
schema.

Real sweeps are not hand-written values files: `python -m qatfd.k8s` (run from `qatfd/`, with
aws, helm and kubectl on PATH) generates the cells from a sweep module under
`qatfd/qatfd/k8s/sweeps/`, skips cells whose run dir on S3 is already complete, writes the values
file to `qatfd/results/.k8s/<release>.values.yaml`, runs `helm upgrade --install`, and can then
follow the Job while pulling finished run dirs into `qatfd/results/`:

```bash
python -m qatfd.k8s plan   --sweep bootstrap_enrich_ub --xs 1 5 10 --seeds 0 1 2     # what would run
python -m qatfd.k8s submit ub-full --sweep bootstrap_enrich_ub --image-tag <sha> --watch
python -m qatfd.k8s watch  ub-full                                                  # re-attach later
python -m qatfd.k8s pull   ub-full                                                  # just the results
```

`submit --dry-run` writes the values file and prints the helm command without touching the
cluster. A release runs one benchmark (its sizing profile); `--parallelism` overrides the profile.

## Image (docker/)

`Dockerfile` installs skunk and qatfd (editable, into `/app/skunk/venv` so the existing scripts'
`./venv/bin/...` paths keep working), the chroma CLI, and the aws cli. Requirements are pinned in
`requirements.lock`, frozen from the dev box venv minus torch/CUDA. Data is not baked in: pods pull
the chroma store into `/data/chromadb` and benchmark files into `/data/benchmarks` from S3.

```bash
deploy_exp/docker/build_push.sh            # build from the repo root, push :<sha> and :latest to ECR
PUSH=0 deploy_exp/docker/build_push.sh     # build only
```

`docker-compose.yaml` runs the pod shape locally (chroma + runner) against a store copy for a
one-question smoke test before anything touches the cluster; see its header for the knobs.

Not in the image: the Codex CLI (the `codex` system) and anything GPU-side (vLLM). Add them when
those systems move to the cluster.
