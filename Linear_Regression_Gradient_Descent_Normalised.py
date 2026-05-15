import numpy as np

# Data: y = 2x + 50
n = 30
x = np.arange(n)
y = 2 * x + 50

# Normalise inputs for stable gradient descent
x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()

# Parameters and hyperparameters
w, b = 0, 0
lr, epochs = 0.01, 1000

# Gradient descent
for _ in range(epochs):
    pred = w * x_norm + b
    w -= lr * -(2/n) * np.sum(x_norm * (y_norm - pred))  # ∂L/∂w
    b -= lr * -(2/n) * np.sum(y_norm - pred)              # ∂L/∂b

# Denormalise back to original scale
slope     = w * y.std() / x.std()
intercept = b * y.std() + y.mean() - w * y.std() * x.mean() / x.std()

print(f"Slope:     {slope:.4f}  (expected 2)")
print(f"Intercept: {intercept:.4f}  (expected 50)")