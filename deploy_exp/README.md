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

The experiment node group's ASG has **AZRebalance suspended** (`terraform_data.experiment_suspend_az_rebalance`,
via the aws cli during apply). Left on, a scale-down that empties an AZ makes the ASG launch a node there, overshoot,
and terminate a *busy* node to compensate, restarting every cell on it from scratch (it ignores pods and the
safe-to-evict annotation). Check it with:

```bash
aws autoscaling describe-auto-scaling-groups --query \
  "AutoScalingGroups[?contains(AutoScalingGroupName,'experiment-nodes')].SuspendedProcesses[].ProcessName"
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
# the runner ServiceAccount lives OUTSIDE any Helm release (releases share it, and Helm refuses to install a
# release whose manifest contains an object another release owns). `python -m qatfd.k8s submit` applies this
# same object before every install, so this step is only needed for hand-run helm installs:
kubectl -n experiments create serviceaccount experiment-runner --dry-run=client -o yaml \
  | kubectl annotate --local -f - eks.amazonaws.com/role-arn=$(cd deploy_exp/terraform && terraform output -raw runner_role_arn) -o yaml \
  | kubectl apply -f -
helm upgrade --install smoke deploy_exp/helm/experiments -n experiments \
    -f deploy_exp/helm/experiments/values-smoke.yaml \
    --set image.tag=<git sha from build_push.sh>
kubectl -n experiments get pods -w        # Pending until the autoscaler adds an r6id node (~3 min), then Init -> Running
```

`values.yaml` documents every knob and the cell schema.

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

The Codex CLI (the `codex` system) is in the image: `build_push.sh` stages the dev box's own standalone
release (`~/.codex/packages/standalone/current`, checked to be 0.150.1) into `deploy_exp/docker/codex/`
(git-ignored) and the Dockerfile copies it to `/opt/codex`. Not in the image: anything GPU-side (vLLM).

## Codex cells

`python -m qatfd.k8s ... --sweep codex_ablation` (qatfd/qatfd/k8s/sweeps/codex_ablation.py) runs the codex
system, one pod per (scenario, seed). Two extra keys in the `llm-keys` Secret:

```bash
kubectl -n experiments patch secret llm-keys --type merge \
  -p '{"stringData":{"OPENROUTER_CODEX_API_KEY":"sk-or-...","OPENROUTER_MGMT_API_KEY":"sk-or-..."}}'
```

**The sandbox.** A shell-enabled codex once answered questions by reading the gold-answer CSV and earlier runs'
results off the disk. In a codex pod (values `codexSandbox.enabled`, which the sweep sets) that is not possible:

- codex runs as the unprivileged `codex` user (uid 10001, `systems.sandbox_uid`); `run.sh` makes `/data`
  (benchmark files incl. the gold answers, the raw corpus pages and PDFs, the chroma store) and `/results` root-only,
  and codex gets `/codex/home` + `/codex/work` of its own (filed into the run dir as `codex_home/` + `codex_scratch/`
  when the cell ends). The answer schema it needs is copied into its home.
- codex's environment is a short allowlist (`_CODEX_ENV_PASSTHROUGH` in qatfd/qatfd/systems/codex.py) plus its own
  OpenRouter key: no runner key, no management key, no AWS variables.
- the `lockdown` init container (NET_ADMIN, `files/lockdown.sh`) installs iptables rules in the pod's network
  namespace: the codex uid may open TCP connections to 127.0.0.1 on the MCP port and the egress-proxy port only
  (no DNS, chroma, S3/STS, instance metadata, Kubernetes API, internet; no IPv6 at all). `files/egress_proxy.py`
  tunnels `CONNECT` to `codexSandbox.allowHosts` (openrouter.ai) on :443 and logs every refusal. The service-account
  tokens are readable but unusable: nothing that accepts them is reachable.
- `run.sh` proves the fence from the codex uid's side (`files/sandbox_check.py`) and refuses to run the cell
  (exit 78) if any check fails; the lockdown container likewise fails the pod rather than start unfenced.

The codex shell tool still runs in codex's own sandbox (bubblewrap from the release dir) on top of this. The
trace scan for gold / results paths (`traces/*.codex.jsonl`) is still worth running as a tripwire.
