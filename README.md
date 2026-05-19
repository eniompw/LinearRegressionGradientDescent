# Linear Regression with Gradient Descent

A collection of notebooks and scripts demonstrating the fundamentals of a Simple Single-Neuron Neural Network, focusing on Linear Regression and Gradient Descent from scratch.

## 📚 Contents

* [Gradient Descent from Scratch](./linreg_gd.ipynb) - Jupyter Notebook exploring the implementation step-by-step.
* [Linear Regression Explanation](./linreg_notes.md) - Detailed explanation of the underlying gradient descent code and math.
* [Linear Regression Gradient Descent](./linreg_gd.py) - Basic Python implementation using raw inputs.
* [Linear Regression Normalised](./linreg_gd_norm.py) - Implementation with normalized inputs for more stable and faster convergence.

> **Note 1:** Removing the learning rate (`lr`) causes `dw` and `db` to explode, resulting in `y_hat` becoming `NaN` and training failing completely.
>
> **Note 2:** Normalising the inputs reduces the required epochs from 50,000 to 1,000, making training 50x faster.

## 🛠️ Key Libraries

* `matplotlib` - For histogram and scatter plot visualizations.

## 🔗 Related Repositories

* [MLP Digits Classifier](https://github.com/eniompw/MLP-Digits-Classifier) - Builds on this project with a Multi-Layer Perceptron for digit classification.

## 🔗 References

* [Regression from Scratch Example](https://web.archive.org/web/20240301022512/https://www.kaggle.com/code/aakashns/pytorch-basics-linear-regression-from-scratch) (Kaggle)
* [Gradient Descent in Python](https://www.geeksforgeeks.org/how-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum) (GeeksforGeeks)
* [Gradient Derivative Calculation](https://web.archive.org/web/20220419231100/https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc?gi=2fc7792409a4) (Towards Data Science)

