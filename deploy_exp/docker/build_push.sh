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
