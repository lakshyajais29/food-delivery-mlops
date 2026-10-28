from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Initialize FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Predict delivery time using a trained Linear Regression model.",
    version="1.0.0"
)


# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "food_delivery_model.joblib"
SCALER_PATH = MODEL_DIR / "food_delivery_scaler.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# --------------------------------------------------
# Validate model and scaler
# --------------------------------------------------

if model.n_features_in_ != scaler.n_features_in_:
    raise RuntimeError(
        f"Model expects {model.n_features_in_} features, "
        f"but scaler expects {scaler.n_features_in_} features."
    )

EXPECTED_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Type_of_order",
    "Type_of_vehicle",
    "rating_group",
    "Age_group"
]

if scaler.n_features_in_ != len(EXPECTED_FEATURES):
    raise RuntimeError(
        f"Expected {len(EXPECTED_FEATURES)} features, "
        f"but scaler expects {scaler.n_features_in_}."
    )


# --------------------------------------------------
# Define request schema
# --------------------------------------------------

class DeliveryInput(BaseModel):
    delivery_person_age: float = Field(..., ge=18, le=100)
    delivery_person_ratings: float = Field(..., ge=0, le=5)

    restaurant_latitude: float
    restaurant_longitude: float

    delivery_location_latitude: float
    delivery_location_longitude: float

    type_of_order: int = Field(..., ge=0, le=3)
    type_of_vehicle: int = Field(..., ge=0, le=2)

    # Required because the exact training transformations
    # for these two engineered features are not confirmed.
    rating_group: float
    age_group: float


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Food Delivery Time Prediction API is running"
    }


# --------------------------------------------------
# Health-check endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "expected_features": scaler.n_features_in_
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: DeliveryInput):

    # Arrange all 10 features in the training order.
    features = np.array([[
        data.delivery_person_age,
        data.delivery_person_ratings,
        data.restaurant_latitude,
        data.restaurant_longitude,
        data.delivery_location_latitude,
        data.delivery_location_longitude,
        data.type_of_order,
        data.type_of_vehicle,
        data.rating_group,
        data.age_group
    ]], dtype=float)

    # Validate feature count.
    if features.shape[1] != scaler.n_features_in_:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Feature mismatch: scaler expects "
                f"{scaler.n_features_in_} features, but received "
                f"{features.shape[1]}."
            )
        )

    # Scale using the saved scaler.
    scaled_features = scaler.transform(features)

    # Generate prediction.
    prediction = float(model.predict(scaled_features)[0])

    return {
        "predicted_delivery_time_minutes": round(prediction, 2)
    }