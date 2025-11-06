import pandas as pd
import copy
import math
import mlflow
import joblib  # for saving Python objects

# Load data
data = pd.read_excel("Boston_Housing.xlsx")

data.shape
# (506, 14)
# Split up predictors and target
y = data['MEDV']
X = data.drop(columns=['MEDV'])
import numpy as np
import matplotlib.pyplot as plt

# Distribution of predictors and relationship with target
#for col in X.columns:
 #   fig, ax = plt.subplots(1, 2, figsize=(6,2))
  #  ax[0].hist(X[col])
   # ax[1].scatter(X[col], y)
    #fig.suptitle(col)
    #plt.show()

def compute_cost(X, y, w, b): 
    m = X.shape[0] 

    f_wb = np.dot(X, w) + b
    cost = np.sum(np.power(f_wb - y, 2))

    total_cost = 1 / (2 * m) * cost

    return total_cost
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    err = (np.dot(X, w) + b) - y
    dj_dw = np.dot(X.T, err)    # dimension: (n,m)*(m,1)=(n,1)
    dj_db = np.sum(err)

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = cost_function(X, y, w, b)
        J_history.append(cost)

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
iterations = 1000
alpha = 1.0e-6

w_init = np.zeros(X_train.shape[1],) # shape = (13,) for Boston Housing
b_init = 0.0

w_out, b_out, J_hist = gradient_descent(X_train, y_train, w_init, b_init, compute_cost, compute_gradient, alpha, iterations)
def plot_cost(data, cost_type):
    plt.figure(figsize=(4,2))
    plt.plot(data)
    plt.xlabel("Iteration Step")
    plt.ylabel(cost_type)
    plt.title("Cost vs. Iteration")
    plt.show() 
#plot_cost(J_hist, "Cost")
def predict(X, w, b):
    p = np.dot(X, w) + b
    return p
y_pred = predict(X_test, w_out, b_out)
#print(y_pred)
def compute_mse(y1, y2):
    return np.mean(np.power((y1 - y2),2))
mse = compute_mse(y_test, y_pred)
#print(mse)
def plot_pred_actual(y_actual, y_pred):
    x_ul = int(math.ceil(max(y_actual.max(), y_pred.max()) / 10.0)) * 10
    y_ul = x_ul

    plt.figure(figsize=(4,4))
    plt.scatter(y_actual, y_pred)
    plt.xlim(0, x_ul)
    plt.ylim(0, y_ul)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Predicted vs Actual values")
    plt.show()
# Generate predictions
y_pred = np.dot(X_test, w_out) + b_out

# Plot predicted vs actual values
#plot_pred_actual(y_test, y_pred)

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
X_train_norm = standard_scaler.fit_transform(X_train)
X_test_norm = standard_scaler.transform(X_test)

iterations = 1000
alpha = 1.0e-2

w_out, b_out, J_hist = gradient_descent(X_train_norm, y_train, w_init, b_init, compute_cost, compute_gradient, alpha, iterations)

#print(f"Training result: w = {w_out}, b = {b_out}")
#print(f"Training MSE = {J_hist[-1]}")

#plot_cost(J_hist, "Cost")

y_pred = np.dot(X_test_norm, w_out) + b_out
#plot_pred_actual(y_test, y_pred)

mse = compute_mse(y_test, y_pred)
#print(f"Test MSE = {mse}")

def compute_cost_ridge(X, y, w, b, lambda_ = 1): 
    m = X.shape[0] 

    f_wb = np.dot(X, w) + b
    cost = np.sum(np.power(f_wb - y, 2))    

    reg_cost = np.sum(np.power(w, 2))

    total_cost = 1 / (2 * m) * cost + (lambda_ / (2 * m)) * reg_cost

    return total_cost
def compute_gradient_ridge(X, y, w, b, lambda_):
    m = X.shape[0]

    err = np.dot(X, w) + b - y
    dj_dw = np.dot(X.T, err) / m + (lambda_ / m) * w
    dj_db = np.sum(err) / m

    return dj_db, dj_dw
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, lambda_=0.7, num_iters=1000):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = cost_function(X, y, w, b, lambda_)
        J_history.append(cost)

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history

iterations = 1000
alpha = 1.0e-2
lambda_ = 1

w_out, b_out, J_hist = gradient_descent(X_train_norm, y_train, w_init, b_init, compute_cost_ridge, compute_gradient_ridge, alpha, lambda_, iterations)

#print(f"Training result: w = {w_out}, b = {b_out}")
#print(f"Training MSE = {J_hist[-1]}")

#plot_cost(J_hist, "Cost")

y_pred = np.dot(X_test_norm, w_out) + b_out
#plot_pred_actual(y_test, y_pred)

def soft_threshold(rho, lamda_):
    if rho < - lamda_:
        return (rho + lamda_)
    elif rho >  lamda_:
        return (rho - lamda_)
    else: 
        return 0
def compute_residuals(X, y, w, b):
    return y - (np.dot(X, w) + b)

def compute_rho_j(X, y, w, b, j):
    X_k = np.delete(X, j, axis=1)    # remove the jth element
    w_k = np.delete(w, j)    # remove the jth element

    err = compute_residuals(X_k, y, w_k, b)

    X_j = X[:,j]
    rho_j = np.dot(X_j, err)

    return rho_j
def coordinate_descent_lasso(X, y, w_in, b_in, cost_function, lambda_, num_iters=1000, tolerance=1e-4):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    n = X.shape[1]

    for i in range(num_iters):
        # Update weights
        for j in range(n):
            X_j = X[:,j]
            rho_j = compute_rho_j(X, y, w, b, j)
            w[j] = soft_threshold(rho_j, lambda_) / np.sum(X_j ** 2)

        # Update bias
        b = np.mean(y - np.dot(X, w))
        err = compute_residuals(X, y, w, b)

        # Calculate total cost
        cost = cost_function(X, y, w, b, lambda_)
        J_history.append(cost)

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

        # Check convergence
        if np.max(np.abs(err)) < tolerance:
            break

    return w, b, J_history

def compute_cost_lasso(X, y, weights, bias, lambda_ = 0.7):
    # Number of training examples
    num_examples = X.shape[0]

    # Make predictions using current weights and bias
    predictions = np.dot(X, weights) + bias

    # Calculate squared error between predictions and actual values
    squared_error = np.sum((predictions - y) ** 2)

    # L1 regularization: sum of absolute values of weights
    l1_penalty = np.sum(np.abs(weights))

    # Combine error and regularization into total cost
    total_cost = (1 / (2 * num_examples)) * squared_error + (lambda_ / (2 * num_examples)) * l1_penalty

    return total_cost
iterations = 1000
lambda_ = 1e-2
tolerance = 1

w_out, b_out, J_hist = coordinate_descent_lasso(X_train_norm, y_train, w_init, b_init, compute_cost_lasso, lambda_, iterations, tolerance)

#plot_cost(J_hist, "Cost")

#y_pred = np.dot(X_test_norm, w_out) + b_out
#plot_pred_actual(y_test, y_pred)

#print(f"Training result: w = {w_out}, b = {b_out}")

#print(standard_scaler.scale_)

#per_capita_crimerate = -0.91848853/7.2662

#print(f"If the per capita crime rate increases by 1 percentage point, the median housing price of that location drops by ${1000 * per_capita_crimerate:.2f}")
# Export trained model for FastAPI
trained_model = {
    "w_out": w_out,
    "b_out": b_out,
    "scaler": standard_scaler
}

# Save the trained model locally first
joblib.dump(trained_model, "trained_model.pkl")
class LassoModel:
    def __init__(self, w_out, b_out, scaler):
        self.w_out = w_out
        self.b_out = b_out
        self.scaler = scaler

    def predict(self, X):
        X_norm = self.scaler.transform(X)
        return np.dot(X_norm, self.w_out) + self.b_out

# === MLflow logging ===
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("lambda", lambda_)
    mlflow.log_param("iterations", iterations)
    mlflow.log_param("tolerance", tolerance)
    
    # Example: log last cost
    final_cost = J_hist[-1] if len(J_hist) > 0 else None
    if final_cost:
        mlflow.log_metric("final_cost", final_cost)
    
    # Log the model file as an artifact
    mlflow.log_artifact("trained_model.pkl", artifact_path="model")
    # Create an MLflow-compatible model instance
    lasso_model = LassoModel(w_out, b_out, standard_scaler)
    
    mlflow.sklearn.log_model(
        sk_model=lasso_model,
        name="lasso-model",
        registered_model_name="Boston-Lasso-Regression-Model"
    )

    print("Model has been logged and registered successfully with MLflow.")