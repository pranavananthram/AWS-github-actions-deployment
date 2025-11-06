from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
from mlflow import sklearn as mlflow_sklearn

app = FastAPI(title="Boston Housing LASSO Regression API (MLflow)")

# Load model from MLflow Model Registry
model_name = "Boston-Lasso-Regression-Model"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow_sklearn.load_model(model_uri)

# Define request schema
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

@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_data = pd.DataFrame([features.dict()])
    y_pred = model.predict(input_data)
    return {"prediction": float(y_pred[0])}
