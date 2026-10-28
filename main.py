from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Initialize FastAPI application
app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Predict delivery time using a trained Linear Regression model.",
    version="1.0.0"
)


# Define paths for the trained model and scaler
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "food_delivery_model.joblib"
SCALER_PATH = MODEL_DIR / "food_delivery_scaler.joblib"


# Load the trained model and fitted scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# Define the input features
class DeliveryInput(BaseModel):
    delivery_person_age: float = Field(..., ge=18, le=100)
    delivery_person_ratings: float = Field(..., ge=0, le=5)
    restaurant_latitude: float
    restaurant_longitude: float
    delivery_location_latitude: float
    delivery_location_longitude: float
    type_of_order: int = Field(..., ge=0, le=3)
    type_of_vehicle: int = Field(..., ge=0, le=2)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Food Delivery Time Prediction API is running"
    }


# Health-check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: DeliveryInput):

    # Arrange input features in the same order as model training
    features = np.array([[
        data.delivery_person_age,
        data.delivery_person_ratings,
        data.restaurant_latitude,
        data.restaurant_longitude,
        data.delivery_location_latitude,
        data.delivery_location_longitude,
        data.type_of_order,
        data.type_of_vehicle
    ]])

    # Scale the input using the fitted scaler
    scaled_features = scaler.transform(features)

    # Predict delivery time
    prediction = model.predict(scaled_features)[0]

    # Return predicted time in minutes
    return {
        "predicted_delivery_time_minutes": round(float(prediction), 2)
    }
