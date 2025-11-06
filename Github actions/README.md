## MLflow Integration Summary

I modified the main.py and linearregression.py used in assignment A0 to use a FastAPI application that uses the trained model that is integrated into MLFlow to track and register the model automatically.

---

#### **linearregression.py**

This code is used to train the LASSO-based Linear Regression model and integrates with MLflow to track, log, and register the trained model. 

**Changes:**
- Added `mlflow` imports and initialized an MLflow run using:
  ```python
  with mlflow.start_run() as run:
- The following code segment was added to log model parameters, metrics and artifacts and then register the model to MLflow:
  ```python
  mlflow.log_param(), mlflow.log_metric(), mlflow.log_artifact()
  mlflow.sklearn.log_model(
    sk_model=model_object,
    registered_model_name="Boston-Lasso-Regression-Model

And then the trained model was exported to trained_model.pkl for later inference.

#### **main.py**

This code defines a FastAPI web service that loads the latest registered MLflow model and predictions.

**Changes:** 
- Configured MLflow model loading from the registry:
```python
model_name = "Boston-Lasso-Regression-Model"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow_sklearn.load_model(model_uri)
```
- Extracted trained model parameters (w_out, b_out, and scaler) for inference and added a /predict endpoint that uses Pydantic to perform data validation.

Acknowledgements: I used the provided tutorials and AI tools to help write code and understand how to use MLFlow and register models. 
