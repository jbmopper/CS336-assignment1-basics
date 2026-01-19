#!/bin/bash
# Launch a spot instance for BPE tokenizer training
# Usage: ./launch_spot.sh [--dry-run] [--use-ssh]

set -euo pipefail

# Configuration
INSTANCE_TYPE="x8g.2xlarge"
REGION="us-west-2"
AZ="us-west-2b"
AMI="ami-08b6b732b0e8f8f41"  # Amazon Linux 2023 ARM64
MAX_SPOT_PRICE="0.25"  # Safety margin above current ~$0.16
EBS_SIZE_GB="${EBS_SIZE_GB:-200}"  # For data + working space
INSTANCE_NAME="cs336-tokenizer"
IAM_ROLE_NAME="cs336-ssm-role"
INSTANCE_PROFILE_NAME="cs336-ssm-profile"
S3_BUCKET="${S3_BUCKET:-}"  # For code/data transfer

# SSH config (only used with --use-ssh flag)
KEY_NAME="${AWS_KEY_NAME:-cs336-spot-key}"
SSH_CIDR="${SSH_CIDR:-}"

DRY_RUN=""
USE_SSH=""
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
        --use-ssh) USE_SSH="true" ;;
    esac
done

if [[ -n "$DRY_RUN" ]]; then
    echo "=== DRY RUN MODE ==="
fi

echo "Region: $REGION"
echo "Instance: $INSTANCE_TYPE (128GB RAM, 8 vCPU ARM64)"
echo "Max spot price: \$$MAX_SPOT_PRICE/hr"
echo "Access method: ${USE_SSH:+SSH}${USE_SSH:-SSM Session Manager (more secure)}"

# Set up S3 bucket for file transfer
if [[ -z "$S3_BUCKET" ]]; then
    # Generate a unique bucket name using account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    S3_BUCKET="cs336-spot-${ACCOUNT_ID}-${REGION}"
fi

echo ""
echo "S3 bucket: $S3_BUCKET"

if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
    echo "Creating S3 bucket..."
    aws s3api create-bucket \
        --bucket "$S3_BUCKET" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" > /dev/null
    echo "Created bucket: $S3_BUCKET"
else
    echo "Bucket already exists"
fi

# Set up IAM role for SSM + S3
echo ""
echo "Checking IAM role for SSM + S3..."

S3_POLICY_NAME="cs336-s3-access"

if ! aws iam get-role --role-name "$IAM_ROLE_NAME" &>/dev/null; then
    echo "Creating IAM role '$IAM_ROLE_NAME'..."
    aws iam create-role \
        --role-name "$IAM_ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' > /dev/null
    
    # SSM permissions
    aws iam attach-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    
    echo "Created IAM role with SSM permissions"
else
    echo "IAM role '$IAM_ROLE_NAME' already exists"
fi

# Add/update S3 policy for the specific bucket
echo "Updating S3 permissions for bucket $S3_BUCKET..."
S3_POLICY_DOC=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::${S3_BUCKET}",
                "arn:aws:s3:::${S3_BUCKET}/*"
            ]
        }
    ]
}
EOF
)

# Create or update inline policy
aws iam put-role-policy \
    --role-name "$IAM_ROLE_NAME" \
    --policy-name "$S3_POLICY_NAME" \
    --policy-document "$S3_POLICY_DOC"

# Create instance profile if needed
if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" &>/dev/null; then
    echo "Creating instance profile..."
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" > /dev/null
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME" \
        --role-name "$IAM_ROLE_NAME"
    echo "Waiting for instance profile to propagate..."
    sleep 10
else
    echo "Instance profile '$INSTANCE_PROFILE_NAME' already exists"
fi

# Get or create security group
SG_NAME="cs336-spot-sg"
echo ""
echo "Checking security group '$SG_NAME'..."

SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query "SecurityGroups[0].GroupId" \
    --output text \
    --region "$REGION" 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" ]] || [[ -z "$SG_ID" ]]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "CS336 spot instance access" \
        --region "$REGION" \
        --query 'GroupId' \
        --output text)
    echo "Created security group: $SG_ID (no inbound rules - SSM doesn't need any)"
else
    echo "Using existing security group: $SG_ID"
fi

# SSH setup (only if --use-ssh)
if [[ -n "$USE_SSH" ]]; then
    echo ""
    echo "Setting up SSH access..."
    
    if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
        echo "Creating new key pair '$KEY_NAME'..."
        aws ec2 create-key-pair \
            --key-name "$KEY_NAME" \
            --region "$REGION" \
            --query 'KeyMaterial' \
            --output text > "${KEY_NAME}.pem"
        chmod 600 "${KEY_NAME}.pem"
        echo "Key saved to ${KEY_NAME}.pem - keep this safe!"
    else
        echo "Key pair '$KEY_NAME' already exists"
    fi
    
    # Add SSH ingress rule if not present
    if [[ -z "$SSH_CIDR" ]]; then
        SSH_CIDR="0.0.0.0/0"
        echo "Warning: SSH_CIDR not set, allowing 0.0.0.0/0"
    fi
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr "$SSH_CIDR" \
        --region "$REGION" 2>/dev/null || true
fi

# User data script - runs on instance startup
USER_DATA=$(cat <<USERDATA
#!/bin/bash
set -ex

# Log everything
exec > >(tee /var/log/user-data.log) 2>&1

echo "=== Starting setup at \$(date) ==="

# Install dependencies
dnf install -y docker git wget awscli
systemctl enable docker
systemctl start docker

# Add ec2-user to docker group
usermod -aG docker ec2-user

# Create working directory
mkdir -p /data
chown ec2-user:ec2-user /data

# Save S3 bucket name for later use
echo "export S3_BUCKET=$S3_BUCKET" >> /etc/profile.d/cs336.sh
echo "export AWS_DEFAULT_REGION=$REGION" >> /etc/profile.d/cs336.sh

echo "=== Setup complete at \$(date) ==="
USERDATA
)

# Encode user data
USER_DATA_B64=$(echo "$USER_DATA" | base64)

echo ""
echo "Requesting spot instance..."

# Build the run-instances command
RUN_ARGS=(
    --image-id "$AMI"
    --instance-type "$INSTANCE_TYPE"
    --security-group-ids "$SG_ID"
    --iam-instance-profile "Name=$INSTANCE_PROFILE_NAME"
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time"}}'
    --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$EBS_SIZE_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
    --user-data "$USER_DATA_B64"
    --placement "AvailabilityZone=$AZ"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]"
    --region "$REGION"
    --output json
)

if [[ -n "$USE_SSH" ]]; then
    RUN_ARGS+=(--key-name "$KEY_NAME")
fi

# Launch spot instance
RESULT=$(aws ec2 run-instances $DRY_RUN "${RUN_ARGS[@]}" 2>&1) || {
    if [[ "$RESULT" == *"DryRunOperation"* ]]; then
        echo "Dry run successful - spot request would work!"
        exit 0
    else
        echo "Error: $RESULT"
        exit 1
    fi
}

if [[ -n "$DRY_RUN" ]]; then
    echo "Dry run successful!"
    exit 0
fi

INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")
echo "Launched instance: $INSTANCE_ID"

echo ""
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text \
    --region "$REGION")

echo ""
echo "============================================"
echo "Instance ready!"
echo "============================================"
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo ""

if [[ -n "$USE_SSH" ]]; then
    echo "Connect with SSH:"
    echo "  ssh -i ${KEY_NAME}.pem ec2-user@$PUBLIC_IP"
    echo ""
    echo "Upload your code:"
    echo "  scp -r -i ${KEY_NAME}.pem $(dirname "$0")/.. ec2-user@$PUBLIC_IP:/data/assignment1-basics"
else
    echo "Connect with SSM (no SSH key needed, may take 1-2 min to be ready):"
    echo "  aws ssm start-session --target $INSTANCE_ID --region $REGION"
    echo ""
    echo "Upload code to S3 (run this locally):"
    echo "  aws s3 sync $(dirname "$0")/.. s3://$S3_BUCKET/assignment1-basics --exclude '.git/*'"
    echo ""
    echo "Download code on instance (run after connecting via SSM):"
    echo "  aws s3 sync s3://$S3_BUCKET/assignment1-basics /data/assignment1-basics"
    echo ""
    echo "Download results when done (run locally):"
    echo "  aws s3 sync s3://$S3_BUCKET/assignment1-basics/tokenizers ./tokenizers-from-aws"
fi

echo ""
echo "Current spot price: ~\$0.16/hr"
echo ""
echo "IMPORTANT: Terminate when done to stop charges:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo "  # Or use: ./terminate_instance.sh"
echo ""
