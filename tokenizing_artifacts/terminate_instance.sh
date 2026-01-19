#!/bin/bash
# Terminate the spot instance
# Usage: ./terminate_instance.sh <instance-id>
#    or: ./terminate_instance.sh  (finds by name tag)

set -euo pipefail

REGION="us-west-2"
INSTANCE_NAME="cs336-tokenizer"

if [[ -n "${1:-}" ]]; then
    INSTANCE_ID="$1"
else
    echo "Finding instance by name '$INSTANCE_NAME'..."
    INSTANCE_ID=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[0].Instances[0].InstanceId" \
        --output text \
        --region "$REGION")
    
    if [[ "$INSTANCE_ID" == "None" ]] || [[ -z "$INSTANCE_ID" ]]; then
        echo "No running instance found with name '$INSTANCE_NAME'"
        exit 1
    fi
fi

echo "Terminating instance: $INSTANCE_ID"
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Instance termination initiated."
echo "It may take a minute for the instance to fully terminate."
