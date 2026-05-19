# Linear Regression with Gradient Descent

A collection of notebooks and scripts demonstrating the fundamentals of a Simple Single-Neuron Neural Network, focusing on Linear Regression and Gradient Descent from scratch.

No deep learning frameworks — just NumPy and a clean implementation of forward pass and gradient descent.

> Followed on by [MLP Digits Classifier](https://github.com/eniompw/MLP-Digits-Classifier), extending gradient descent from a single-output linear model to a multi-class neural network.

## 📚 Contents

* [Gradient Descent from Scratch](./linreg_gd.ipynb) - Jupyter Notebook exploring the implementation step-by-step.
* [Linear Regression Explanation](./linreg_notes.md) - Detailed explanation of the underlying gradient descent code and math.
* [Linear Regression Gradient Descent](./linreg_gd.py) - Basic Python implementation using raw inputs.
* [Linear Regression Normalised](./linreg_gd_norm.py) - Implementation with normalized inputs for more stable and faster convergence.

## How It Works

1. **Data** — Generates 30 points following `y = 2x + 50` using `np.arange`.
2. **Initialisation** — Weight `w` and bias `b` start at 0; learning rate `lr = 0.0001`.
3. **Forward pass** — Computes predictions: `y_hat = w * x + b`.
4. **Gradients** — Calculates partial derivatives of Mean Squared Error (MSE) loss with respect to `w` and `b`.
5. **Update** — Subtracts `lr * gradient` from each parameter, nudging them toward lower loss.
6. **Result** — After 50,000 epochs, the model recovers `w ≈ 2` and `b ≈ 50`.

See [linreg_notes.md](./linreg_notes.md) for a detailed breakdown of the math and code.

> **Note 1:** Removing the learning rate (`lr`) causes `dw` and `db` to explode, resulting in `y_hat` becoming `NaN` and training failing completely.
>
> **Note 2:** Normalising the inputs reduces the required epochs from 50,000 to 1,000, making training 50x faster.

## 🛠️ Key Libraries

* `matplotlib` - For histogram and scatter plot visualizations.

## 🔗 References

* [Regression from Scratch Example](https://web.archive.org/web/20240301022512/https://www.kaggle.com/code/aakashns/pytorch-basics-linear-regression-from-scratch) (Kaggle)
* [Gradient Descent in Python](https://www.geeksforgeeks.org/how-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum) (GeeksforGeeks)
* [Gradient Derivative Calculation](https://web.archive.org/web/20220419231100/https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc?gi=2fc7792409a4) (Towards Data Science)

