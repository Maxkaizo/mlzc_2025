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
import pandas as pd
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
    cap_shape: str = Field(..., description="Cap shape", alias="cap-shape")
    cap_surface: str = Field(..., description="Cap surface", alias="cap-surface")
    cap_color: str = Field(..., description="Cap color", alias="cap-color")
    does_bruise_or_bleed: str = Field(..., description="Does bruise or bleed (t, f)", alias="does-bruise-or-bleed")
    gill_attachment: str = Field(..., description="Gill attachment", alias="gill-attachment")
    gill_spacing: str = Field(..., description="Gill spacing", alias="gill-spacing")
    gill_color: str = Field(..., description="Gill color", alias="gill-color")
    stem_surface: str = Field(..., description="Stem surface", alias="stem-surface")
    stem_color: str = Field(..., description="Stem color", alias="stem-color")
    has_ring: str = Field(..., description="Has ring (t, f)", alias="has-ring")
    ring_type: str = Field(..., description="Ring type", alias="ring-type")
    habitat: str = Field(..., description="Habitat", alias="habitat")
    season: str = Field(..., description="Season (s, u, a, w)", alias="season")

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
        "cap-surface": "s",
        "cap-color": "n",
        "does-bruise-or-bleed": "t",
        "gill-attachment": "f",
        "gill-spacing": "c",
        "gill-color": "k",
        "stem-color": "w",
        "stem-surface": "s",
        "habitat": "d",
        "ring-type": "p",
        "season": "s",
        "has-ring": "t"
    }
    ```
    """

    if model_dict is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Extract model components
        model = model_dict["model"]
        ohe = model_dict["ohe"]
        le_target = model_dict["le_target"]
        feature_names = model_dict["feature_names"]

        # Prepare feature vector from input
        feature_dict = features.model_dump(by_alias=True)

        # Get categorical and numerical feature names
        categorical_cols = [
            "cap-shape",
            "cap-surface",
            "cap-color",
            "does-bruise-or-bleed",
            "gill-attachment",
            "gill-spacing",
            "gill-color",
            "stem-surface",
            "stem-color",
            "has-ring",
            "ring-type",
            "habitat",
            "season",
        ]

        numerical_cols = [
            "cap-diameter",
            "stem-height",
            "stem-width",
            "spore_print_color_present",
        ]

        # Create DataFrame for preprocessing (matching training pipeline)
        input_data = {}

        for col in categorical_cols:
            input_data[col] = [feature_dict[col]]

        for col in numerical_cols:
            if col == "spore_print_color_present":
                # Not provided in input, default to 0 (missing)
                input_data[col] = [0]
            else:
                input_data[col] = [feature_dict[col]]

        df_input = pd.DataFrame(input_data)

        # Apply OneHotEncoding to categorical features
        X_cat_encoded = ohe.transform(df_input[categorical_cols])

        # Combine with numerical features
        X_encoded = np.hstack([X_cat_encoded, df_input[numerical_cols].values])

        # Make prediction
        prediction_proba = model.predict_proba(X_encoded)[0]
        prediction_class = model.predict(X_encoded)[0]

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
