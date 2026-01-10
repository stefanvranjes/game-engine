#!/bin/bash

# AWS Deployment Script for Distributed Physics

set -e

# Configuration
REGION="us-west-2"
AMI_ID="ami-xxxxxxxxx"  # Replace with your custom AMI
INSTANCE_TYPE="p3.2xlarge"
KEY_NAME="your-key-pair"
SECURITY_GROUP="sg-xxxxxxxxx"
SUBNET_ID="subnet-xxxxxxxxx"

echo "=== AWS Distributed Physics Deployment ==="

# Launch Master Instance
echo "Launching master instance..."
MASTER_INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --subnet-id $SUBNET_ID \
    --user-data file://scripts/master-userdata.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=physics-master},{Key=Role,Value=master}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Master instance launched: $MASTER_INSTANCE_ID"

# Wait for master to be running
echo "Waiting for master instance to be running..."
aws ec2 wait instance-running --instance-ids $MASTER_INSTANCE_ID

# Get master IP
MASTER_IP=$(aws ec2 describe-instances \
    --instance-ids $MASTER_INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Master IP: $MASTER_IP"

# Create Launch Template for Workers
echo "Creating launch template for workers..."
aws ec2 create-launch-template \
    --launch-template-name physics-worker-template \
    --version-description "Physics worker template" \
    --launch-template-data "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SecurityGroupIds\": [\"$SECURITY_GROUP\"],
        \"UserData\": \"$(base64 -w 0 scripts/worker-userdata.sh | sed 's/MASTER_IP_PLACEHOLDER/'$MASTER_IP'/g')\",
        \"TagSpecifications\": [{
            \"ResourceType\": \"instance\",
            \"Tags\": [{\"Key\": \"Name\", \"Value\": \"physics-worker\"}, {\"Key\": \"Role\", \"Value\": \"worker\"}]
        }]
    }"

# Create Auto Scaling Group
echo "Creating auto scaling group..."
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name physics-workers \
    --launch-template LaunchTemplateName=physics-worker-template,Version='$Latest' \
    --min-size 2 \
    --max-size 20 \
    --desired-capacity 5 \
    --vpc-zone-identifier $SUBNET_ID \
    --tags "Key=Name,Value=physics-worker,PropagateAtLaunch=true" "Key=MasterIP,Value=$MASTER_IP,PropagateAtLaunch=true"

echo "Auto scaling group created"

# Create scaling policies
echo "Creating scaling policies..."

# Scale up policy
aws autoscaling put-scaling-policy \
    --auto-scaling-group-name physics-workers \
    --policy-name scale-up \
    --scaling-adjustment 2 \
    --adjustment-type ChangeInCapacity \
    --cooldown 300

# Scale down policy
aws autoscaling put-scaling-policy \
    --auto-scaling-group-name physics-workers \
    --policy-name scale-down \
    --scaling-adjustment -1 \
    --adjustment-type ChangeInCapacity \
    --cooldown 300

echo "=== Deployment Complete ==="
echo "Master IP: $MASTER_IP"
echo "Access master: ssh -i $KEY_NAME.pem ubuntu@$MASTER_IP"
echo ""
echo "To monitor workers:"
echo "aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names physics-workers"
