#!/usr/bin/env bash
# Build the experiment image from the repo root and push it to the ECR repository created by
# deploy_exp/terraform (output `ecr_repository_url`). Tags: the short git sha, plus `latest`.
#
#   deploy_exp/docker/build_push.sh                 # build + push
#   PUSH=0 deploy_exp/docker/build_push.sh          # build only (tags qatfd-experiments:local too)
#   AWS_PROFILE=mit deploy_exp/docker/build_push.sh
#
# Knobs (env): AWS_REGION (default us-east-1), ECR_REPO (default qatfd-experiments), TAG (default
# short git sha), PUSH (default 1), PLATFORM (default linux/amd64: the nodes are x86_64, so an
# arm64 laptop must cross-build; docker buildx handles it).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-qatfd-experiments}"
TAG="${TAG:-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"
PUSH="${PUSH:-1}"
PLATFORM="${PLATFORM:-linux/amd64}"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- skunk/src qatfd/qatfd qatfd/configs qatfd/scripts 2>/dev/null)" ]]; then
    echo "WARNING: uncommitted changes under skunk/src, qatfd/qatfd, qatfd/configs or qatfd/scripts; tag $TAG will not reproduce this image" >&2
fi

# stage the Codex CLI release into the build context (the Dockerfile COPYs deploy_exp/docker/codex/): the dev box's
# own standalone install, so the image runs the same codex version as local runs. CODEX_PACKAGE_DIR overrides it.
CODEX_PACKAGE_DIR="${CODEX_PACKAGE_DIR:-$(readlink -f "$HOME/.codex/packages/standalone/current")}"
CODEX_VERSION="${CODEX_VERSION:-0.150.1}"
python3 - "$CODEX_PACKAGE_DIR/codex-package.json" "$CODEX_VERSION" <<'PY' || exit 1
import json, sys
pkg = json.load(open(sys.argv[1]))
if pkg.get("version") != sys.argv[2] or pkg.get("target") != "x86_64-unknown-linux-musl":
    sys.exit(f"codex package {sys.argv[1]} is {pkg.get('version')} / {pkg.get('target')}; want {sys.argv[2]} / x86_64-unknown-linux-musl "
             "(set CODEX_PACKAGE_DIR, or CODEX_VERSION to build a different version on purpose)")
PY
rm -rf "$REPO_ROOT/deploy_exp/docker/codex"
cp -a "$CODEX_PACKAGE_DIR/." "$REPO_ROOT/deploy_exp/docker/codex/"
echo "staged codex $CODEX_VERSION from $CODEX_PACKAGE_DIR"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${ECR_REPO}"

echo "building ${IMAGE}:${TAG} (${PLATFORM}) from ${REPO_ROOT}"
docker buildx build \
    --platform "$PLATFORM" \
    --file "$REPO_ROOT/deploy_exp/docker/Dockerfile" \
    --tag "${IMAGE}:${TAG}" \
    --tag "${IMAGE}:latest" \
    --tag "qatfd-experiments:local" \
    --load \
    "$REPO_ROOT"

if [[ "$PUSH" == "1" ]]; then
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
    docker push "${IMAGE}:${TAG}"
    docker push "${IMAGE}:latest"
    echo "pushed ${IMAGE}:${TAG} and :latest"
else
    echo "PUSH=0: not pushing; local tags ${IMAGE}:${TAG}, qatfd-experiments:local"
fi
