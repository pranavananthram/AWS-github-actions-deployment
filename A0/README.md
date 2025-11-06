# A0: Building a Linear Regression Model

In this tutorial, I implemented a **Linear Regression** model using the `scikit-learn` library (`sklearn`). The workflow and key observations are summarized below.

---

## 1. Dataset Split

I split the dataset into training and testing sets:

- **Test size:** 20% of the data  
- **Train size:** 80% of the data  

I chose a 20% test size after multiple iterations because it balances training and testing data, ensuring enough samples to reliably estimate model performance.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## 2. Gradient Descent and Learning Rate (Alpha)

The learning rate `alpha` controls the step size in gradient descent. Key observations:

- **Too small alpha**: Training barely progresses.  
- **Too large alpha**: Cost can diverge to infinity in later iterations.

I followed the tutorial and kept `alpha` at the recommended value to avoid instability.

---

## 3. Lasso Regression Cost Function

The `compute_cost_lasso` function calculates the total cost for **Lasso Regression**, which combines the standard squared error with an L1 penalty for regularization.

```python
# Predictions
y_hat = X @ w + b

# Squared Error (SSE)
SSE = sum((y_hat[i] - y[i])**2 for i in range(m))

# L1 Penalty
L1_penalty = sum(abs(w[j]) for j in range(n))

# Total Cost
Cost = SSE + lambda * L1_penalty
```

---

## 4. Visualization

Visualizing model predictions was straightforward. I used Jupyter Notebook due to setup challenges in VSCode and IntelliJ/PyCharm. This allowed easy plotting and testing using matplotlib.

```python
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue')
plt.plot(X, y_hat, color='red')
plt.show()
```

---

## 5. Resources

- [Linear Regression (Python Implementation) – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/linear-regression-python-implementation/)  
- [Implementation of Lasso Regression from Scratch (Python) – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/implementation-of-lasso-regression-from-scratch-using-python/)

---
