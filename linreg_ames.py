from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
ames = fetch_openml(name="house_prices", as_frame=True)
X = ames.data["GrLivArea"].to_numpy(dtype=float)  # Living area (sq ft)
y = ames.target.to_numpy(dtype=float)              # Sale price

# --- Normalise for stable gradient descent ---
X_norm = (X - X.mean()) / X.std()
y_norm = (y - y.mean()) / y.std()

# --- Train: linear regression via gradient descent ---
W, b, lr, n = 0.0, 0.0, 0.1, len(X)

for _ in range(1000):
    y_pred = W * X_norm + b
    error  = y_norm - y_pred
    W -= lr * -(2/n) * np.sum(X_norm * error)
    b -= lr * -(2/n) * np.sum(error)

# --- Denormalise to original scale ---
slope     = W * y.std() / X.std()
intercept = b * y.std() + y.mean() - slope * X.mean()

print(f"slope:     {slope:.4f}")
print(f"intercept: {intercept:.4f}")