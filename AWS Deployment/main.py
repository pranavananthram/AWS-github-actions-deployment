from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

# Import trained model and scaler from your linear_regression.py
from linearregression import trained_model

# Extract needed items
w_out = trained_model["w_out"]
b_out = trained_model["b_out"]
scaler = trained_model["scaler"]

app = FastAPI(title="Boston Housing Linear Regression API")

# Define schema for incoming data
class HouseFeatures(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

@app.get("/")
def home():
    return {"message": "Boston Housing Price Prediction API. Use POST /predict with housing features."}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Convert input to a DataFrame
    input_data = pd.DataFrame([features.dict()])

    # Normalize using same scaler as training
    X_input_norm = scaler.transform(input_data)

    # Compute prediction
    y_pred = np.dot(X_input_norm, w_out) + b_out

    return {"prediction": float(y_pred[0])}
