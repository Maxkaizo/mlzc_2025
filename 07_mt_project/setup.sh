#!/bin/bash
# Mushroom Classifier - Setup & Run Script
# Execute this to set up and run the entire project

set -e

echo "🍄 Mushroom Classifier Setup Script"
echo "===================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Python version
echo -e "${BLUE}📌 Step 1: Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Step 2: Create virtual environment
echo -e "${BLUE}📌 Step 2: Setting up virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Step 3: Install dependencies
echo -e "${BLUE}📌 Step 3: Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Step 4: Train model
echo -e "${BLUE}📌 Step 4: Training model...${NC}"
if [ ! -f "models/model.pkl" ]; then
    python train.py
    echo "✅ Model trained and saved"
else
    echo "⚠️  Model already exists. Skipping training."
    read -p "Train new model anyway? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python train.py
    fi
fi

# Step 5: Test API locally
echo -e "${BLUE}📌 Step 5: Ready to start API...${NC}"
echo "Run the following command to start the API:"
echo ""
echo -e "${GREEN}python predict.py${NC}"
echo ""
echo "Then in another terminal, test with:"
echo -e "${GREEN}python test_api.py${NC}"
echo ""

# Optional: Docker
echo -e "${YELLOW}Optional - Build Docker image:${NC}"
echo -e "${GREEN}docker build -t mushroom-classifier:latest .${NC}"
echo ""

# Optional: AWS
echo -e "${YELLOW}Optional - Deploy to AWS Elastic Beanstalk:${NC}"
echo -e "${GREEN}pip install awsebcli${NC}"
echo -e "${GREEN}aws configure${NC}"
echo -e "${GREEN}eb init -p python-3.12 mushroom-classifier${NC}"
echo -e "${GREEN}eb create mushroom-prod${NC}"
echo ""

echo -e "${GREEN}✨ Setup complete! Ready to deploy.${NC}"
