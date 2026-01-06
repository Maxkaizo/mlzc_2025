import numpy as np
from PIL import Image
import requests
from io import BytesIO
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# IMPORTANT: hair_classifier_v1.onnx and hair_classifier_v1.onnx.data must be together
session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def preprocess_image_from_url(url, size=(200, 200)):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    img = img.resize(size)  # (optional) .resize(size, Image.BILINEAR)

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))   # CHW
    x = np.expand_dims(x, axis=0)    # NCHW

    return x.astype(np.float32)


def predict_single(image_url: str) -> float:
    X = preprocess_image_from_url(image_url)
    result = session.run([output_name], {input_name: X})
    return float(result[0][0][0])


def lambda_handler(event, context):
    # supports either {"url": "..."} or {"body": {"url":"..."}} (API Gateway proxy)
    url = event.get("url")
    if url is None and isinstance(event.get("body"), dict):
        url = event["body"].get("url")

    if not url:
        return {"error": "Missing 'url' in event"}

    score = predict_single(url)
    return {"score": score}
