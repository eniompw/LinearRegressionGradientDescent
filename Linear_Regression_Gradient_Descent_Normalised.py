import numpy as np

# Data: 30 points following y = 2x + 50
n = 30
x = np.arange(n)
y = 2 * x + 50

# Normalise inputs for stable gradient descent
x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()

# Initialise parameters and learning rate
w, b, lr = 0, 0, 0.01

for _ in range(1000):
    y_hat = w * x_norm + b                          # Forward pass: predicted y values
    dw = -(2/n) * np.sum(x_norm * (y_norm - y_hat)) # Gradient w.r.t. weight
    db = -(2/n) * np.sum(y_norm - y_hat)            # Gradient w.r.t. bias
    w -= lr * dw                                    # Update weight
    b -= lr * db                                    # Update bias

# Denormalise back to original scale
slope = w * y.std() / x.std()
intercept = b * y.std() + y.mean() - slope * x.mean()

print(f"weight: {slope}\nbias: {intercept}")