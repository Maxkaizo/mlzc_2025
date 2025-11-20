"""
Mushroom Classification Prediction Service
FastAPI + Uvicorn for serving predictions
"""

import pickle
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="🍄 Mushroom Classifier API",
    description="ML model for predicting mushroom edibility based on morphological features",
    version="1.0.0",
)

# Global model dictionary
model_dict = None


class MushroomFeatures(BaseModel):
    """Input schema for mushroom prediction"""

    cap_diameter: float = Field(
        ..., gt=0, description="Cap diameter in cm", alias="cap-diameter"
    )
    stem_height: float = Field(
        ..., gt=0, description="Stem height in cm", alias="stem-height"
    )
    stem_width: float = Field(
        ..., gt=0, description="Stem width in mm", alias="stem-width"
    )
    cap_shape: str = Field(..., description="Cap shape (x, b, s, p, o, c, etc)", alias="cap-shape")
    cap_color: str = Field(..., description="Cap color (n, y, w, g, p, b, u, e, o, r, l, k, c, etc)", alias="cap-color")
    gill_attachment: str = Field(..., description="Gill attachment (a, d, f, n)", alias="gill-attachment")
    gill_color: str = Field(
        ..., description="Gill color (k, w, g, p, o, n, u, y, b, r, etc)", alias="gill-color"
    )
    stem_color: str = Field(..., description="Stem color (w, p, g, o, n, b, y, e, c, r, k, etc)", alias="stem-color")
    stem_surface: str = Field(..., description="Stem surface (s, f, y, k)", alias="stem-surface")
    habitat: str = Field(..., description="Habitat (u, g, m, d, w, p, l, c)", alias="habitat")
    odor: str = Field(..., description="Odor (a, l, f, n, s, p, m, c, y, o, u, e, w, r, k, etc)", alias="odor")
    veil_color: str = Field(..., description="Veil color (w, n, o, y, p, u, r, g, b, e, k, l, c)", alias="veil-color")
    ring_number: str = Field(..., description="Ring number (o, t, n)", alias="ring-number")
    ring_type: str = Field(..., description="Ring type (p, e, l, f, n, g, s, z, r, c, etc)", alias="ring-type")
    bruises: str = Field(..., description="Bruises (t, f)", alias="bruises")
    season: str = Field(..., description="Season (s, u, a, w)", alias="season")
    has_ring: str = Field(..., description="Has ring (t, f)", alias="has-ring")
    spore_print_color: str = Field(
        ..., description="Spore print color (k, w, n, b, u, o, y, r, p, e, g, l, c, etc)", alias="spore-print-color"
    )

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Output schema for prediction response"""

    prediction: str = Field(description="Predicted class: 'edible' or 'poisonous'")
    probability: float = Field(description="Prediction confidence (0-1)")
    confidence_percent: str = Field(description="Confidence as percentage")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool


def load_model():
    """Load trained model and encoders"""
    global model_dict

    model_path = "models/model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run 'python train.py' first."
        )

    try:
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        print(f"✅ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    success = load_model()
    if not success:
        raise RuntimeError("Failed to load model on startup")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="ok", model_loaded=model_dict is not None)


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "🍄 Mushroom Classifier API",
        "version": "1.0.0",
        "description": "Predict mushroom edibility using machine learning",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "predict": "/predict",
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(features: MushroomFeatures):
    """
    Predict mushroom edibility

    **Example request:**
    ```json
    {
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
    }
    ```
    """

    if model_dict is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Extract model components
        model = model_dict["model"]
        label_encoders = model_dict["label_encoders"]
        le_target = model_dict["le_target"]

        # Prepare feature vector
        feature_dict = features.model_dump(by_alias=True)

        # Get feature names in order
        feature_names = [
            "cap-diameter",
            "stem-height",
            "stem-width",
            "cap-shape",
            "cap-color",
            "gill-attachment",
            "gill-color",
            "stem-color",
            "stem-surface",
            "habitat",
            "odor",
            "veil-color",
            "ring-number",
            "ring-type",
            "bruises",
            "season",
            "has-ring",
            "spore-print-color",
        ]

        # Create feature array
        feature_array = []

        for feat_name in feature_names:
            feat_value = feature_dict[feat_name]

            # Check if categorical (encoded) or numerical
            if feat_name in label_encoders:
                # Encode categorical feature
                le = label_encoders[feat_name]
                try:
                    encoded_value = le.transform([feat_value])[0]
                except ValueError:
                    # Unknown category
                    print(f"⚠️  Unknown value '{feat_value}' for {feat_name}, using 'Unknown'")
                    encoded_value = le.transform(["Unknown"])[0]
            else:
                # Numerical feature - use as is
                encoded_value = float(feat_value)

            feature_array.append(encoded_value)

        # Make prediction
        X = np.array([feature_array])
        prediction_proba = model.predict_proba(X)[0]
        prediction_class = model.predict(X)[0]

        # Map prediction to class name
        predicted_class_name = le_target.inverse_transform([prediction_class])[0]
        confidence = float(prediction_proba[prediction_class])

        # Determine human-readable label
        prediction_label = "edible" if predicted_class_name == "e" else "poisonous"

        return PredictionResponse(
            prediction=prediction_label,
            probability=round(confidence, 4),
            confidence_percent=f"{confidence * 100:.2f}%",
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Prediction error: {str(e)}"
        )


@app.post("/batch_predict", tags=["Predictions"])
async def batch_predict(features_list: list[MushroomFeatures]):
    """
    Batch prediction for multiple mushrooms

    Returns a list of predictions
    """

    predictions = []

    for features in features_list:
        result = await predict(features)
        predictions.append(result)

    return {"count": len(predictions), "predictions": predictions}


if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists("models/model.pkl"):
        print("❌ Model not found!")
        print("Please run: python train.py")
        exit(1)

    print("\n🍄 Starting Mushroom Classifier API")
    print("=" * 60)
    print("📚 Interactive Docs: http://localhost:8000/docs")
    print("📖 ReDoc Docs: http://localhost:8000/redoc")
    print("🏥 Health Check: http://localhost:8000/health")
    print("=" * 60 + "\n")

    # Run Uvicorn server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
