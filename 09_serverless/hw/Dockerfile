# This image already includes hair_classifier_empty.onnx and hair_classifier_empty.onnx.data
FROM agrigorev/model-2025-hairstyle:v1

# Lambda images use /var/task as the code directory by default
WORKDIR /var/task

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Lambda code
COPY lambda_function.py .

# Set the handler (file.function)
CMD ["lambda_function.lambda_handler"]
