import numpy as np

# Data: 30 points following y = 2x + 50
n_samples = 30
X = np.arange(n_samples)          # features
y = 2 * X + 50                    # targets

# Parameters and learning rate
W = 0.0                           # weight
b = 0.0                           # bias
learning_rate = 0.0001

for _ in range(50_000):
    y_pred = W * X + b             # forward pass
    dW = -(2 / n_samples) * np.sum(X * (y - y_pred))
    db = -(2 / n_samples) * np.sum(y - y_pred)
    W -= learning_rate * dW        # update weight
    b -= learning_rate * db        # update bias

print(f"[Linear]  W: {W:.3f}, b: {b:.3f}")