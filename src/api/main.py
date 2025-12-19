# src/api/main.py
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import mlflow.sklearn

from src.api.pydantic_models import CustomerData, PredictionResponse

MODEL_PATH = "notebooks/models/best_model"

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0"
)

model = None
feature_names = None
model_source = None


@app.on_event("startup")
def load_model():
    global model, feature_names, model_source

    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        feature_names = list(model.feature_names_in_)
        model_source = "Local Model"
        print(f"✅ Loaded local model from {MODEL_PATH}")
    except Exception as e:
        print("❌ Failed to load local model:", e)
        model = None
        feature_names = None
        model_source = None


@app.get("/")
def health_check():
    return {
        "message": "Credit Risk API is running",
        "model_loaded": model_source
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert request to DataFrame
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])

        # Ensure correct column order
        df = df[feature_names]

        # Predict
        risk_probability = float(model.predict_proba(df)[0, 1])
        prediction_class = int(risk_probability >= 0.5)

        return PredictionResponse(
            prediction_class=prediction_class,
            risk_probability=round(risk_probability, 4),
            model_source=model_source
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )
