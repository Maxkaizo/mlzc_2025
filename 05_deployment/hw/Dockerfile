# Use the official Python 3.13.5 slim image (Debian Bookworm)
FROM python:3.13.5-slim-bookworm

# Copy 'uv' and 'uvx' (fast Python package installer and env manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container
WORKDIR /code

# Add the virtual environment’s bin directory to the PATH
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency configuration files
COPY pyproject.toml uv.lock ./

# Install dependencies exactly as locked in uv.lock (reproducible build)
RUN uv sync --frozen

# Copy your FastAPI app and model file
COPY predict.py pipeline_v1.bin ./

# Expose port 9696 for API access
EXPOSE 9696

# Start the FastAPI app with uvicorn
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
