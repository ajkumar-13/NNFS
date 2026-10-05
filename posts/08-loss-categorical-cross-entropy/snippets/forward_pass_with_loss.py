"""Post 08, sections 7 and 8: the two loss classes, then post 07's forward pass with a loss and an accuracy.

Run from the series root:
    python posts/08-loss-categorical-cross-entropy/snippets/forward_pass_with_loss.py

Needs NumPy and the nnfs helper package (pip install nnfs).
Layer_Dense, Activation_ReLU and Activation_Softmax are the classes of posts 04 and 06, unchanged.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        shifted       = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values    = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss     = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples        = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Integer labels:
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # One-hot labels:
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1      = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2      = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_fn     = Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_fn.calculate(activation2.output, y)
print(f"Loss: {loss:.4f}")

predictions = np.argmax(activation2.output, axis=1)
accuracy    = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.3f}")

# What stands behind the two numbers.
sample_losses = loss_fn.forward(activation2.output, y)
print("per-sample losses:", sample_losses.shape, "from", sample_losses.min(), "to", sample_losses.max())
print("log(3):", np.log(3))
print("loss with one-hot labels:", loss_fn.calculate(activation2.output, np.eye(3)[y]))
print("correct:", np.sum(predictions == y), "of", len(y))
print("predictions per class:", np.bincount(predictions, minlength=3))
