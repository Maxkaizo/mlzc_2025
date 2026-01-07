# 🍄 Mushroom Classification Project - ML Zoomcamp Capstone
![banner](imgs/banner.png)

## Problem Description

**Objective**: Predict whether a mushroom is edible or poisonous based on its morphological characteristics.

### Why This Problem Matters
Misidentifying mushrooms can have severe consequences. This model provides a data-driven approach to assist classification and serves as a reproducible end-to-end ML project (training + API + deployment).

### Dataset Overview
- **Source**: [UCI Machine Learning Repository - Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)
- **Samples**: 61,069 observations
- **Features**: 20 raw columns (plus target)
- **Target**: Binary classification - Edible (`e`) vs Poisonous (`p`)
- **Class Distribution**: `p` = 33,888, `e` = 27,181

### Raw Feature Schema
`class`, `cap-diameter`, `cap-shape`, `cap-surface`, `cap-color`, `does-bruise-or-bleed`, `gill-attachment`, `gill-spacing`, `gill-color`, `stem-height`, `stem-width`, `stem-root`, `stem-surface`, `stem-color`, `veil-type`, `veil-color`, `has-ring`, `ring-type`, `spore-print-color`, `habitat`, `season`.

### Features Used (after cleaning)
**Numerical Features** (4):
- `cap-diameter` (cm)
- `stem-height` (cm)
- `stem-width` (mm)
- `spore_print_color_present` (derived indicator)

**Categorical Features** (13):
- `cap-shape`, `cap-surface`, `cap-color`
- `does-bruise-or-bleed`
- `gill-attachment`, `gill-spacing`, `gill-color`
- `stem-surface`, `stem-color`
- `has-ring`, `ring-type`
- `habitat`, `season`

**Dropped Columns** (>80% missing): `stem-root`, `veil-type`, `veil-color`, `spore-print-color`.

### Solution Application
This project demonstrates:
- A reproducible ML training pipeline
- A FastAPI service for real-time predictions
- Docker-based packaging and cloud-ready deployment

---

## Project Structure

```
07_mt_project/
├── README.md                 # Project documentation
├── notebook.ipynb            # EDA, feature analysis, model selection
├── train.py                  # Training script to generate final model
├── predict.py                # FastAPI service for predictions
├── test_api.py               # API test suite
├── Dockerfile                # Container configuration
├── pyproject.toml            # Python dependencies and project config
├── requirements.txt          # pip dependencies
├── setup.sh                  # Helper script
├── data/
│   └── mushroom.csv          # Dataset (semicolon-separated)
└── models/
    └── model.pkl             # Trained model (generated after training)
```

---

## Quick Start

### 1. Setup Environment

#### Option A: Using uv (Recommended)
```bash
# Create virtual environment
uv venv

# Activate
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

#### Option B: Using pip + venv
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Explore the Data & Models
```bash
jupyter notebook notebook.ipynb
```
The notebook includes:
- Exploratory data analysis (EDA)
- Missing value analysis and cleaning decisions
- Multiple model training and tuning
- Feature importance analysis

### 3. Train the Final Model
```bash
# With uv
uv run python train.py

# Or with active venv
python3 train.py
```
Output: `models/model.pkl`

### 4. Start the Prediction Service
```bash
# With uv
uv run python predict.py

# Or with active venv
python3 predict.py
```
Service starts at: `http://localhost:8000`

### 5. Make Predictions

#### Interactive API Documentation
```
http://localhost:8000/docs    # Swagger UI
http://localhost:8000/redoc   # ReDoc
```

#### Example: Using curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cap-diameter": 8.5,
    "stem-height": 7.2,
    "stem-width": 6.5,
    "cap-shape": "x",
    "cap-surface": "s",
    "cap-color": "n",
    "does-bruise-or-bleed": "f",
    "gill-attachment": "f",
    "gill-spacing": "c",
    "gill-color": "k",
    "stem-surface": "s",
    "stem-color": "w",
    "has-ring": "t",
    "ring-type": "p",
    "habitat": "d",
    "season": "s"
  }'
```

---

## Deployment

### Local Docker Deployment
```bash
# Build Docker image
docker build -t mushroom-classifier:latest .

# Run container
docker run -p 8000:8000 mushroom-classifier:latest
```

### Cloud Deployment
Detailed guides are in `lab_notes/DEPLOYMENT.md`:
- AWS Elastic Beanstalk
- AWS Lambda (serverless)
- ECR + Fargate

---

## Data Quality & Preprocessing
- Removed duplicate rows
- Dropped columns with >80% missing values
- Imputed categorical missing values with `Unknown`
- Imputed numeric missing values with median
- Created `spore_print_color_present` to capture missingness signal
- OneHotEncoded categorical features

---

## Model Workflow
```
Raw Data → Duplicate Removal → Missing Value Handling →
Feature Engineering → Train/Test Split →
Model Training + Tuning → Evaluation → Model Export
```

---

## Features & Technical Stack

**Core Libraries**:
- `pandas`, `numpy`, `scikit-learn`

**API & Deployment**:
- `fastapi`, `uvicorn`, `pydantic`

**Containerization**:
- `Docker`

**Data Exploration**:
- `jupyter`, `matplotlib`, `seaborn`

---

## Reproducibility

### Steps to Reproduce Results
1. Install dependencies (see Quick Start)
2. Train model: `python3 train.py`
3. Run service: `python3 predict.py`
4. Verify health: `curl http://localhost:8000/health`
5. Run API tests: `python3 test_api.py`

### Dataset Availability
- Dataset included in repo: `data/mushroom.csv`
- Source link: [UCI ML Repository](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)

---

## Troubleshooting

### Issue: "Module not found" error
```bash
uv sync
# or
pip install -r requirements.txt
```

### Issue: Port 8000 already in use
```bash
python -m uvicorn predict:app --port 8001
```

### Issue: Model file not found
```bash
python3 train.py
```

---

## References

- [UCI Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [AWS Elastic Beanstalk Docs](https://docs.aws.amazon.com/elasticbeanstalk/)

---

## License

This project is part of ML Zoomcamp 2025 Capstone Project.

## Author

**Max Kaizo** - ML Zoomcamp Participant

---

**Last Updated**: November 2025
