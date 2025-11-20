# Multi-stage Docker build for Mushroom Classifier API

# Stage 1: Build stage
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY data/ ./data/
COPY train.py predict.py ./

# Install uv for fast dependency installation
RUN pip install --no-cache-dir uv

# Install Python dependencies using uv
RUN uv pip install --python 3.12 --system-site-packages -r requirements.txt 2>/dev/null || \
    pip install --no-cache-dir pandas numpy scikit-learn fastapi uvicorn pydantic

# Train the model (optional - can be done separately)
# Uncomment if you want model included in image
# RUN python train.py

# Stage 2: Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /build .

# Create models directory
RUN mkdir -p /app/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the FastAPI application
CMD ["python", "predict.py"]
