import numpy as np

# Data: 30 points following y = 2x + 50
n_samples = 30
X = np.arange(n_samples)
y = 2 * X + 50

# Normalise inputs for stable gradient descent
X_norm = (X - X.mean()) / X.std()
y_norm = (y - y.mean()) / y.std()

# Initialise parameters and learning rate
W, b, learning_rate = 0, 0, 0.1

for _ in range(20):
    y_pred = W * X_norm + b                                       # Forward pass: predicted y values
    dW = -(2/n_samples) * np.sum(X_norm * (y_norm - y_pred))     # Gradient w.r.t. weight
    db = -(2/n_samples) * np.sum(y_norm - y_pred)                 # Gradient w.r.t. bias
    W -= learning_rate * dW                                       # Update weight
    b -= learning_rate * db                                       # Update bias

# Denormalise back to original scale
slope = W * y.std() / X.std()
intercept = b * y.std() + y.mean() - slope * X.mean()

print(f"weight: {slope}\nbias: {intercept}")