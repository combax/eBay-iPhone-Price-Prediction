#!/usr/bin/env bash
# Build both serving images and push them to ECR (repositories are created on
# first run). Requires the AWS CLI configured and Docker running.
#
#   AWS_REGION=us-east-1 ./deploy/aws/push-ecr.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

aws ecr get-login-password --region "$REGION" |
    docker login --username AWS --password-stdin "$REGISTRY"
for repo in ebay-price-api ebay-price-go; do
    aws ecr describe-repositories --repository-names "$repo" --region "$REGION" >/dev/null 2>&1 ||
        aws ecr create-repository --repository-name "$repo" --region "$REGION" >/dev/null
done

docker build --target api -t "$REGISTRY/ebay-price-api:latest" .
docker build -f deploy/go/Dockerfile -t "$REGISTRY/ebay-price-go:latest" .
docker push "$REGISTRY/ebay-price-api:latest"
docker push "$REGISTRY/ebay-price-go:latest"

echo "pushed:"
echo "  $REGISTRY/ebay-price-api:latest   (FastAPI; deploy with apprunner.sh)"
echo "  $REGISTRY/ebay-price-go:latest    (Go + ONNX; deploy with kubectl apply -f deploy/k8s/)"
