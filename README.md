# ML Model Deployment & Tracking Suite

This repository showcases a complete machine learning deployment pipeline—from training a Lasso-based Linear Regression model to deploying it using **FastAPI**, **Docker**, **AWS ECS**, and **GitHub Actions**, and finally integrating **MLflow** for model tracking and registry.

**Live App URL:** [http://3.145.84.64/docs](http://3.145.84.64/docs)

---

## Project Structure

```
.
├── A0/                     # Linear Regression training and visualization
├── A1/                     # FastAPI app and Docker deployment
├── mlflow_integration/    # MLflow tracking and model registry
├── Dockerfile
├── main.py
├── linearregression.py
└── README.md
```

---

## A0: Linear Regression Model

### Overview

Implemented a **Lasso-based Linear Regression** model using `scikit-learn`. The model was trained, evaluated, and visualized using `matplotlib`.

### Key Steps

1. **Dataset Split**
   - 80% training / 20% testing using `train_test_split`.

2. **Gradient Descent & Learning Rate**
   - Explored the effect of `alpha` on convergence.
   - Avoided instability by tuning `alpha` based on tutorial guidance.

3. **Lasso Cost Function**

```python
# Predictions
y_hat = X @ w + b

# Squared Error
SSE = sum((y_hat[i] - y[i])**2 for i in range(m))

# L1 Penalty
L1_penalty = sum(abs(w[j]) for j in range(n))

# Total Cost
Cost = SSE + lambda * L1_penalty
```

4. **Visualization**

```python
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue')
plt.plot(X, y_hat, color='red')
plt.show()
```

5. **Challenges**
   - IDE setup issues (VSCode, IntelliJ).
   - Jupyter Notebook proved most stable for experimentation.

6. **Resources**
   - [Linear Regression – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/linear-regression-python-implementation/)
   - [Lasso Regression from Scratch – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/implementation-of-lasso-regression-from-scratch-using-python/)

---

## FastAPI + Docker + AWS Deployment

### Overview

Deployed the trained model as a REST API using **FastAPI**, containerized it with **Docker**, and deployed it to **AWS ECS** using **CloudShell** and **GitHub Actions**.

### Local Testing

- Created a FastAPI app with `/predict` endpoint.
- Verified functionality using Swagger UI at `/docs`.

### Dockerization

- Built Docker image and pushed to ECR repository: `my-fastapi-app`.

### AWS Deployment (via CloudShell)

1. Configured AWS region (`us-east-2`) and credentials.
2. Uploaded files and built Docker image in CloudShell.
3. Created ECS cluster: `mlops-cluster`.
4. Defined task: `mlops-task2` with correct VPC and networking.
5. Deployed successfully.

#### AWS Deployment Challenges

- Couldn’t use AWS CLI locally; relied on CloudShell.
- Docker image required Excel file during build (due to model training).
- Resolved by uploading Excel file and rebuilding image.
- Created a new repo and task definition to bypass earlier errors.

### GitHub Actions CI/CD

- Used **Deploy to Amazon ECS** workflow.
- Added secrets: AWS region, cluster name, task definition, access keys.
- Configured workflow to deploy on every commit.

#### GitHub Deployment Challenges

- Faced error with `enableFaultInjection` in task definition JSON.
- Removed the field, but ECS referenced old version.
- Fixed by committing changes and letting ECS update to `mlops_task2:2`.
- Dockerfile was in `A1/`, so added `cd A1` in workflow to locate it.

---

## MLflow Integration

### Overview

Integrated **MLflow** to track, log, and register the trained model. Modified `linearregression.py` and `main.py` to support MLflow-based model management.

### `linearregression.py`

- Trains the Lasso Regression model.
- Logs parameters, metrics, and artifacts to MLflow.
- Registers model under: `"Boston-Lasso-Regression-Model"`.

```python
with mlflow.start_run() as run:
    mlflow.log_param("alpha", alpha)
    mlflow.log_metric("mse", mse)
    mlflow.log_artifact("trained_model.pkl")
    mlflow.sklearn.log_model(
        sk_model=model_object,
        registered_model_name="Boston-Lasso-Regression-Model"
    )
```

### `main.py`

- Loads the latest registered model from MLflow Model Registry.
- Extracts `w_out`, `b_out`, and `scaler` for inference.
- Defines `/predict` endpoint using Pydantic for input validation.

```python
model_name = "Boston-Lasso-Regression-Model"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow_sklearn.load_model(model_uri)
```

---

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/your-repo.git
cd A1

# Build Docker image
docker build -t my-fastapi-app .

# Run locally
docker run -p 8000:8000 my-fastapi-app

# Access the app
http://localhost:8000/docs
```

---

## Secrets & Configuration

- AWS credentials stored as GitHub secrets:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
- ECS parameters configured in workflow YAML:
  - Cluster name
  - Task definition
  - Region

---

## Contact

If the FastAPI app URL becomes inactive, feel free to reach out so I can reactivate it.

---
