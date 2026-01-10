# Cloud Deployment Guide

## Quick Start

### Docker (Local Testing)

```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build

# Scale workers
docker-compose up --scale worker=5
```

### Kubernetes

```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/master-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml

# Check status
kubectl get pods -l app=physics

# Scale workers
kubectl scale deployment physics-workers --replicas=10
```

### AWS

```bash
# Deploy to AWS
cd scripts
chmod +x deploy-aws.sh
./deploy-aws.sh

# Monitor
aws autoscaling describe-auto-scaling-groups \
    --auto-scaling-group-names physics-workers
```

---

## Docker

### Build Images

```bash
# Master
docker build -f docker/Dockerfile.master -t physics-master:latest .

# Worker
docker build -f docker/Dockerfile.worker -t physics-worker:latest .
```

### Run Manually

```bash
# Master
docker run --gpus all -p 8080:8080 physics-master:latest

# Worker
docker run --gpus all -e MASTER_ADDRESS=192.168.1.100 physics-worker:latest
```

---

## Kubernetes

### Prerequisites

- Kubernetes cluster with GPU support
- NVIDIA device plugin installed

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
```

### Deploy

```bash
# Deploy master
kubectl apply -f k8s/master-deployment.yaml

# Wait for master
kubectl wait --for=condition=ready pod -l role=master

# Deploy workers
kubectl apply -f k8s/worker-deployment.yaml
```

### Auto-Scaling

Workers auto-scale based on CPU/memory:
- Min: 2 replicas
- Max: 20 replicas
- Target CPU: 70%

---

## AWS

### Prerequisites

1. AWS CLI configured
2. Custom AMI with CUDA + dependencies
3. VPC, subnet, security group configured

### Deploy

```bash
./scripts/deploy-aws.sh
```

### Instance Types

| Type | GPUs | vCPUs | RAM | Cost/Hour |
|------|------|-------|-----|-----------|
| p3.2xlarge | 1x V100 | 8 | 61 GB | $3.06 |
| p3.8xlarge | 4x V100 | 32 | 244 GB | $12.24 |
| p4d.24xlarge | 8x A100 | 96 | 1152 GB | $32.77 |

### Auto-Scaling

- Min: 2 instances
- Max: 20 instances
- Desired: 5 instances
- Scale up: +2 instances when CPU > 70%
- Scale down: -1 instance when CPU < 30%

---

## Azure

### Deploy with ARM Template

```bash
az deployment group create \
    --resource-group physics-rg \
    --template-file azure/template.json \
    --parameters @azure/parameters.json
```

### Instance Types

| Type | GPUs | vCPUs | RAM | Cost/Hour |
|------|------|-------|-----|-----------|
| Standard_NC6 | 1x K80 | 6 | 56 GB | $0.90 |
| Standard_NC24 | 4x K80 | 24 | 224 GB | $3.60 |

---

## GCP

### Deploy with Deployment Manager

```bash
gcloud deployment-manager deployments create physics-cluster \
    --config gcp/deployment.yaml
```

### Instance Types

| Type | GPUs | vCPUs | RAM | Cost/Hour |
|------|------|-------|-----|-----------|
| n1-standard-8 + K80 | 1x K80 | 8 | 30 GB | $0.70 |
| n1-standard-16 + V100 | 1x V100 | 16 | 60 GB | $2.48 |

---

## Monitoring

### Docker

```bash
docker ps
docker logs physics-master
docker stats
```

### Kubernetes

```bash
kubectl get pods
kubectl logs -l app=physics
kubectl top pods
```

### AWS

```bash
# CloudWatch metrics
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=AutoScalingGroupName,Value=physics-workers
```

---

## Cost Estimation

### Example: 10 workers, 8 hours/day

**AWS (p3.2xlarge):**
- 10 workers × $3.06/hour × 8 hours = $244.80/day
- Monthly: ~$7,344

**With auto-scaling (avg 5 workers):**
- 5 workers × $3.06/hour × 8 hours = $122.40/day
- Monthly: ~$3,672
- **Savings: 50%**

---

## Troubleshooting

### Workers not connecting

```bash
# Check master IP
kubectl get svc physics-master

# Check worker logs
kubectl logs -l role=worker
```

### High costs

- Reduce max replicas
- Use smaller instance types
- Enable auto-scaling
- Set cost alerts

---

## Security

### Network

- Use private IPs for worker-master communication
- Expose only master port (8080) publicly
- Use VPC/VNet for isolation

### Authentication

- Use IAM roles (AWS)
- Use managed identities (Azure)
- Use service accounts (GCP)

---

## Best Practices

1. **Use auto-scaling** for cost optimization
2. **Monitor metrics** (CPU, GPU, network)
3. **Set cost limits** to prevent overruns
4. **Use spot instances** for non-critical workloads
5. **Enable logging** for debugging
6. **Regular backups** of master state
