# Mushroom Classifier API - Single Stage Build

FROM python:3.12-slim

WORKDIR /app

# Install only necessary runtime dependencies for sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY train.py predict.py test_api.py ./
COPY data/ ./data/
COPY models/ ./models/

# Install uv for fast dependency installation
RUN pip install --no-cache-dir uv

# Sync Python dependencies using uv (creates virtual environment)
RUN uv sync

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the FastAPI application
CMD ["uv", "run", "python", "predict.py"]
