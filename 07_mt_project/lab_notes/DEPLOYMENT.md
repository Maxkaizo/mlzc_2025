# AWS Deployment Guide for Mushroom Classifier

## Option 1: AWS Elastic Beanstalk (Recommended for beginners)

### Prerequisites
```bash
# Install AWS CLI
pip install awscli

# Install EB CLI
pip install awsebcli

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, region (us-east-1), and output format (json)
```

### Deploy Steps

#### 1. Initialize Elastic Beanstalk Application
```bash
# From project root directory
eb init -p python-3.12 mushroom-classifier --region us-east-1
```

#### 2. Create Environment
```bash
# Create and deploy in one command
eb create mushroom-prod --instance-type t3.micro --scale 1
```

This will:
- Create an EC2 instance (t3.micro - within free tier)
- Set up Auto Scaling Group
- Configure Load Balancer
- Deploy your application
- Train the model automatically

#### 3. Verify Deployment
```bash
# Open in browser
eb open

# Check status
eb status

# View logs
eb logs

# View health
eb health
```

#### 4. Test the API
```bash
# Get the URL
eb open --print-url

# Test health endpoint
curl $(eb open --print-url)/health

# Make a prediction
curl -X POST "$(eb open --print-url)/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cap-diameter": 8.5,
    "stem-height": 7.2,
    "stem-width": 6.5,
    "cap-shape": "x",
    "cap-color": "n",
    "gill-attachment": "f",
    "gill-color": "k",
    "stem-color": "w",
    "stem-surface": "s",
    "habitat": "d",
    "odor": "p",
    "veil-color": "w",
    "ring-number": "o",
    "ring-type": "p",
    "bruises": "f",
    "season": "s",
    "has-ring": "t",
    "spore-print-color": "k"
  }'
```

#### 5. Update Deployment
```bash
# Make changes to code
# Redeploy
eb deploy
```

#### 6. Cleanup (Stop charges)
```bash
# Terminate environment (IMPORTANT: stops charges)
eb terminate mushroom-prod
```

---

## Option 2: AWS Lambda (Serverless - Most Cost-Effective)

### Prerequisites
```bash
pip install mangum zappa
```

### Deploy Steps

#### 1. Create Lambda Handler
Create `lambda_handler.py`:

```python
from mangum import Mangum
from predict import app

handler = Mangum(app, lifespan="off")
```

#### 2. Deploy with Zappa
```bash
# Initialize Zappa
zappa init

# When prompted:
# - Choose AWS region: us-east-1
# - Should we remove environment variables: y
# - Django settings module: leave blank

# Deploy
zappa deploy dev

# Update after changes
zappa update dev
```

#### 3. Get API Endpoint
```bash
# After deployment, Zappa will print the endpoint URL
# It will look like: https://xxxxxxxx.execute-api.us-east-1.amazonaws.com/dev/

# Test the endpoint
curl https://xxxxxxxx.execute-api.us-east-1.amazonaws.com/dev/health
```

#### 4. Cleanup
```bash
zappa undeploy dev
```

---

## Option 3: Docker on AWS ECR + Fargate

### Prerequisites
```bash
# Install Docker
# (See https://docs.docker.com/get-docker/)

# AWS CLI installed and configured
```

### Deploy Steps

#### 1. Build Docker Image
```bash
# Build locally first to test
docker build -t mushroom-classifier:latest .

# Run locally
docker run -p 8000:8000 mushroom-classifier:latest

# Test at http://localhost:8000/docs
```

#### 2. Push to ECR
```bash
# Create ECR repository
aws ecr create-repository --repository-name mushroom-classifier --region us-east-1

# Get login credentials
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag image for ECR
docker tag mushroom-classifier:latest \
  ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/mushroom-classifier:latest

# Push to ECR
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/mushroom-classifier:latest
```

#### 3. Deploy to Fargate
```bash
# Create Fargate cluster, task definition, and service using AWS Console
# Or use CloudFormation/Terraform for IaC

# Via AWS CLI:
# 1. Create Fargate cluster
aws ecs create-cluster --cluster-name mushroom-prod --region us-east-1

# 2. Register task definition (see task-definition.json below)
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json \
  --region us-east-1

# 3. Create service
aws ecs create-service \
  --cluster mushroom-prod \
  --service-name mushroom-api \
  --task-definition mushroom-classifier \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:ACCOUNT_ID:targetgroup/mushroom/xxx,containerName=mushroom,containerPort=8000 \
  --region us-east-1
```

---

## Cost Estimation

### Elastic Beanstalk
- **t3.micro**: ~$5/month (free tier eligible for first 12 months)
- **Total**: $0-5/month

### Lambda
- **Requests**: $0.0000002 per request
- **Duration**: $0.0000166667 per GB-second
- **Example**: 1M predictions/month = ~$0.20-0.50

### Fargate
- **vCPU**: ~$0.04272/hour (0.25 vCPU)
- **Memory**: ~$0.004688/hour (0.5GB)
- **Total**: ~$31/month (24/7 running)

**Recommendation**: Use Lambda for serverless, or Elastic Beanstalk on t3.micro for always-on service.

---

## Environment Variables

Set these in Elastic Beanstalk console or via CLI:

```bash
eb setenv MODEL_PATH=/var/app/current/models/model.pkl
```

---

## Monitoring & Logs

### Elastic Beanstalk
```bash
# View logs
eb logs

# Tail logs in real-time
eb logs -a

# View environment health
eb health
```

### CloudWatch
```bash
# View logs in CloudWatch
aws logs tail /aws/elasticbeanstalk/mushroom-prod/var/log/eb-activity.log --follow
```

---

## Security Best Practices

1. **API Keys** (Optional)
```python
# Add to predict.py
from fastapi import Header, HTTPException

@app.post("/predict")
async def predict(features: MushroomFeatures, x_token: str = Header(...)):
    if x_token != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of function
```

2. **CORS** (if needed)
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

3. **HTTPS** - Automatic with AWS services

4. **Model Protection** - Store model.pkl in S3 with restricted access

---

## Scaling Configuration

### Auto-scaling Policy (Elastic Beanstalk)
```bash
# In EB console:
# Configuration > Capacity > Auto Scaling Group
# - Min instances: 1
# - Max instances: 5
# - Trigger: Average CPU Utilization > 70%
```

---

## Troubleshooting

### "Model not found" error
```bash
# SSH into instance
eb ssh

# Train model manually
python train.py
```

### Port already in use
```bash
# Elastic Beanstalk uses port 8000 automatically
# Make sure predict.py uses 0.0.0.0:8000
```

### Deployment fails
```bash
# Check logs
eb logs

# Validate configuration
eb config validate

# Rebuild environment
eb rebuild
```

---

## Resources

- [AWS Elastic Beanstalk Docs](https://docs.aws.amazon.com/elasticbeanstalk/)
- [AWS Lambda + FastAPI](https://mangum.io/)
- [Zappa Documentation](https://github.com/zappa/Zappa)
- [AWS ECR Guide](https://docs.aws.amazon.com/ecr/)
- [Fargate Pricing](https://aws.amazon.com/fargate/pricing/)

---

**Last Updated**: November 2025
