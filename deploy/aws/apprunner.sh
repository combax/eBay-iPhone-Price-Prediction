#!/usr/bin/env bash
# Create (or update) an AWS App Runner service running the FastAPI image —
# the simplest managed way to put the python API on the internet.
# Run push-ecr.sh first, and set APPRUNNER_ECR_ROLE to an IAM role ARN that
# has the AWSAppRunnerServicePolicyForECRAccess managed policy.
#
#   AWS_REGION=us-east-1 APPRUNNER_ECR_ROLE=arn:aws:iam::...:role/apprunner-ecr \
#       ./deploy/aws/apprunner.sh
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
SERVICE="${SERVICE_NAME:-ebay-price-api}"
ROLE="${APPRUNNER_ECR_ROLE:?set APPRUNNER_ECR_ROLE (IAM role with ECR access for App Runner)}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ebay-price-api:latest"

SOURCE=$(cat <<JSON
{
  "ImageRepository": {
    "ImageIdentifier": "$IMAGE",
    "ImageRepositoryType": "ECR",
    "ImageConfiguration": { "Port": "8000" }
  },
  "AuthenticationConfiguration": { "AccessRoleArn": "$ROLE" },
  "AutoDeploymentsEnabled": false
}
JSON
)

ARN=$(aws apprunner list-services --region "$REGION" \
    --query "ServiceSummaryList[?ServiceName=='$SERVICE'].ServiceArn" --output text)
if [ -z "$ARN" ]; then
    aws apprunner create-service --region "$REGION" --service-name "$SERVICE" \
        --source-configuration "$SOURCE" \
        --instance-configuration '{"Cpu": "1024", "Memory": "2048"}' \
        --health-check-configuration '{"Protocol": "HTTP", "Path": "/health"}' >/dev/null
else
    aws apprunner update-service --region "$REGION" --service-arn "$ARN" \
        --source-configuration "$SOURCE" >/dev/null
fi

echo "service URL (may take a few minutes to go RUNNING):"
aws apprunner list-services --region "$REGION" \
    --query "ServiceSummaryList[?ServiceName=='$SERVICE'].ServiceUrl" --output text
