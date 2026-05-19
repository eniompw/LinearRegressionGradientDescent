# Linear Regression with Gradient Descent from Scratch

This document explains a simple implementation of linear regression using gradient descent optimization.

## Overview

The code implements a basic machine learning model that learns to fit a linear relationship (`y = 2x + 50`) using gradient descent, an optimization algorithm that iteratively adjusts model parameters to minimize error.

## Code Breakdown

### 1. Data Generation

```python
import numpy as np

n = 30              # number of data points
x = np.arange(n)    # Create x values: [0, 1, 2, ..., 29]
y = 2 * x + 50      # True relationship: y = 2x + 50
y = y + np.random.normal(0, 5, n)  # Add random noise to y values
```

- **`x`**: Independent variable values (0 to 29)
- **`y`**: Dependent variable following the linear equation `y = 2x + 50`
- **Noise**: Gaussian noise (mean=0, std=5) is added to simulate real-world data imperfections
- Note: n=30 is used because larger values cause overflow during training

### 2. Model Initialization

```python
w = 0.1   # weight (slope) - initialized to 0.1
b = 0.01  # bias (intercept) - initialized to 0.01
```

- **`w`**: Weight parameter that multiplies x (similar to slope in `y = mx + b`)
- **`b`**: Bias parameter (similar to intercept)
- These start with arbitrary small values and will be optimized through training

### 3. Training Configuration

```python
lr = 0.0001  # learning rate
```

- **Learning rate**: Controls the step size during each parameter update
- Smaller values (like 0.0001) lead to slower but more stable convergence
- Larger values risk overshooting the optimal solution

### 4. Gradient Descent Training Loop

```python
for i in range(50000):
  # Calculate predictions with current parameters
  pred = (w * x) + b
  
  # Calculate gradients (partial derivatives of loss with respect to w and b)
  dw = -(2/n) * np.sum(x * (y - pred))
  db = -(2/n) * np.sum(y - pred)
  
  # Update parameters in the direction that reduces loss
  w = w - lr * dw
  b = b - lr * db
```

**What happens in each iteration:**

1. **Prediction**: `pred = (w * x) + b` generates predictions for all x values using current parameters
2. **Error Calculation**: `(y - pred)` computes the residual (difference between actual and predicted values)
3. **Gradient Calculation**:
   - `dw`: Gradient with respect to weight (how much w needs to change)
   - `db`: Gradient with respect to bias (how much b needs to change)
   - The formula `-(2/n)` represents the derivative of Mean Squared Error (MSE) loss
4. **Parameter Update**: Subtract (learning rate × gradient) from each parameter to move toward lower loss

### 5. Results

```python
print("weight:" , w)
print("bias:" , b)
```

After 50,000 iterations, the trained parameters should approximate:
- **weight ≈ 2** (close to the true slope)
- **bias ≈ 50** (close to the true intercept)

## How Gradient Descent Works

Gradient descent is an optimization algorithm that:
1. Computes the gradient (slope) of the loss function at current parameters
2. Takes a step proportional to the negative gradient
3. Repeats until convergence (minimal loss)

Think of it like walking down a hill in the fog—you can only feel the ground beneath your feet. The gradient tells you which direction is downhill, and you take small steps (controlled by learning rate) in that direction.

## Expected Behavior

- **Early iterations**: Large changes in w and b as the model adjusts from random initialization
- **Later iterations**: Smaller changes as the model converges toward optimal values
- **Final result**: w ≈ 2 and b ≈ 50, successfully learning the underlying linear relationship despite noise

## Potential Improvements

1. **Normalize inputs**: Could improve numerical stability
2. **Adaptive learning rate**: Adjust learning rate during training for better convergence
3. **Track loss**: Store MSE at each iteration to monitor training progress
4. **Add regularization**: Prevent overfitting on larger datasets
5. **Early stopping**: Stop training when loss plateaus to save computation
