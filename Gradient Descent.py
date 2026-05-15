import numpy as np

n = 30              # number of data points
x = np.arange(n)    # Create x values: [0, 1, 2, ..., 29]
y = 2 * x + 50      # True relationship: y = 2x + 50

w = 0.1   # weight (slope) - initialized to 0.1
b = 0.01  # bias (intercept) - initialized to 0.01

lr = 0.0001  # learning rate

for i in range(50000):
  # Calculate predictions with current parameters
  pred = (w * x) + b
  
  # Calculate gradients (partial derivatives of loss with respect to w and b)
  dw = -(2/n) * np.sum(x * (y - pred))
  db = -(2/n) * np.sum(y - pred)
  
  # Update parameters in the direction that reduces loss
  w = w - lr * dw
  b = b - lr * db

print("weight:" , w)
print("bias:" , b)