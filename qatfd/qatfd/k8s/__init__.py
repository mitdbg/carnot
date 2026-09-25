"""Run qatfd sweeps on the experiments EKS cluster (see deploy_exp/): one Helm release per sweep, one pod
per cell, results synced back into qatfd/results/ so the existing plot / eval scripts run unchanged.

    python -m qatfd.k8s plan   --sweep bootstrap_enrich_ub --xs 1 5 --seeds 0
    python -m qatfd.k8s submit ub_x1x5 --sweep bootstrap_enrich_ub --xs 1 5 --seeds 0 --image-tag <sha> --watch
    python -m qatfd.k8s watch  ub_x1x5
    python -m qatfd.k8s pull   ub_x1x5

`sweeps/` holds one module per sweep (the cell tables, ported from the scripts/run_*.sh drivers);
`benchmarks.py` says what each benchmark's pods pull from S3; `cells.py` is the cell record + the S3
completeness check; `__main__.py` is the CLI (it shells out to helm, kubectl and the aws cli)."""
