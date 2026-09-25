# syntax=docker/dockerfile:1
# Experiment image: skunk + qatfd + the chroma CLI + aws cli, for one experiment cell per pod.
#
# Build from the REPO ROOT (the context must contain skunk/ and qatfd/):
#   docker build -f deploy_exp/docker/Dockerfile -t qatfd-experiments:local .
# or, to build and push to ECR:  deploy_exp/docker/build_push.sh
#
# Layout inside the image mirrors the dev box so every existing script works unchanged:
#   /app/skunk           skunk checkout, editable-installed into /app/skunk/venv
#   /app/skunk/venv      the ONE interpreter (skunk/scripts/run_chroma_server.sh hardcodes ./venv/bin/{python,chroma})
#   /app/qatfd           qatfd checkout, editable-installed (hydra resolves configs/ relative to the package)
# Data is NOT in the image. Pods pull it from S3 into these mount points at start:
#   /data/chromadb       a pristine chroma store copy (SKUNK_CHROMADB_DIR)
#   /data/benchmarks     benchmark files (QATFD_BENCHMARKS_DIR), i.e. what qatfd/benchmarks/ holds on the dev box
#   /results             run directories (results_root=/results), synced back to S3
#
# Python 3.14 to match the dev venv the lock was frozen from (qatfd carries a 3.14 shim for hydra).
# torch and the CUDA wheels are deliberately left out: nothing on the runtime path imports them.

# ---------------------------------------------------------------------------------------------
FROM python:3.14-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# build-essential: a few pinned packages have no 3.14 wheel and compile at install time
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /app/skunk/venv
ENV PATH=/app/skunk/venv/bin:$PATH

# dependencies first so source edits do not invalidate this (slow) layer
COPY deploy_exp/docker/requirements.lock deploy_exp/docker/requirements-extra.txt /tmp/req/
RUN pip install -r /tmp/req/requirements.lock -r /tmp/req/requirements-extra.txt

# the two packages, editable so relative resources (prompts/*.yaml, configs/, scripts/) resolve
COPY skunk/ /app/skunk/
COPY qatfd/ /app/qatfd/
RUN pip install --no-deps -e /app/skunk -e /app/qatfd

# ---------------------------------------------------------------------------------------------
FROM python:3.14-slim

# curl: chroma heartbeat probes; zstd + tar: store tarballs; procps: ps/pgrep for the chroma scripts;
# aws cli v2: S3 pulls/syncs and Secrets Manager reads (the zip installer, same as on the dev box)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl unzip zstd tar procps \
 && rm -rf /var/lib/apt/lists/* \
 && curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
 && unzip -q /tmp/awscliv2.zip -d /tmp \
 && /tmp/aws/install \
 && rm -rf /tmp/aws /tmp/awscliv2.zip

COPY --from=builder /app /app

ENV PATH=/app/skunk/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    QATFD_BENCHMARKS_DIR=/data/benchmarks \
    SKUNK_CHROMADB_DIR=/data/chromadb \
    SKUNK_CHROMA_SERVER_HOST=127.0.0.1 \
    SKUNK_CHROMA_SERVER_PORT=8001

RUN mkdir -p /data/chromadb /data/benchmarks /results

# qatfd is run from its own dir on the dev box (hydra's results/.hydra dir is cwd-relative)
WORKDIR /app/qatfd

# The Kubernetes Job and the compose file set the real command; the default just proves the install.
CMD ["python", "-c", "import qatfd, skunk, chromadb; print('qatfd-experiments image ok: chromadb', chromadb.__version__)"]
