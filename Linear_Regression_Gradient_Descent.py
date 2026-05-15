import numpy as np

# Data: 30 points following y = 2x + 50
n = 30
x = np.arange(n)
y = 2 * x + 50

# Initialise parameters and learning rate
w, b, lr = 0, 0, 0.0001

for _ in range(50000):
    y_hat = w * x + b                           # Forward pass: predicted y values
    dw = -(2/n) * np.sum(x * (y - y_hat))      # Gradient w.r.t. weight
    db = -(2/n) * np.sum(y - y_hat)            # Gradient w.r.t. bias
    w -= lr * dw                                # Update weight
    b -= lr * db                                # Update bias

print(f"weight: {w}\nbias: {b}")