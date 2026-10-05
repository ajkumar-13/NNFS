"""Post 06, section 5: the first complete forward pass, Dense -> ReLU -> Dense -> Softmax.

Run from the series root:
    python posts/06-activation-functions-relu-and-softmax/snippets/forward_pass.py

Needs NumPy and the nnfs helper package (pip install nnfs).
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Small random weights; zero biases.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:

    def forward(self, inputs):
        # Subtract the per-row max for stability.
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        # Exponentiate and normalise per row.
        exp_values    = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities


nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1      = Layer_Dense(2, 3)            # 2 inputs, 3 hidden neurons
activation1 = Activation_ReLU()

dense2      = Layer_Dense(3, 3)            # 3 inputs, 3 output neurons (1 per class)
activation2 = Activation_Softmax()

dense1.forward(X)                          # linear
activation1.forward(dense1.output)         # ReLU
dense2.forward(activation1.output)         # linear: the logits
activation2.forward(dense2.output)         # softmax: logits to probabilities

print(activation2.output[:5])

# The shape and type of every stage.
print()
for name, array in [("X", X), ("dense1.output", dense1.output), ("activation1.output", activation1.output),
                    ("dense2.output", dense2.output), ("activation2.output", activation2.output)]:
    print(f"{name:<19}{str(array.shape):<10}{array.dtype}")

# What ReLU did to the hidden layer.
print()
print("entries of dense1.output: negative", np.sum(dense1.output < 0),
      " exactly zero", np.sum(dense1.output == 0), " positive", np.sum(dense1.output > 0))
print("entries of activation1.output equal to zero:", np.sum(activation1.output == 0), "of", activation1.output.size)
print("rows of X at the origin:", np.flatnonzero(~X.any(axis=1)))
print("coordinates of X that are negative:", np.sum(X < 0), "of", X.size)

# How close to uniform the untrained output is.
print()
print("largest |logit|:      ", np.max(np.abs(dense2.output)))
print("largest |p - 1/3|:    ", np.max(np.abs(activation2.output - 1 / 3)),
      " (first five rows:", np.max(np.abs(activation2.output[:5] - 1 / 3)), ")")
row = np.argmax(np.max(np.abs(activation2.output - 1 / 3), axis=1))
print("least uniform row:     row", row, activation2.output[row])
sums = np.sum(activation2.output, axis=1)
print("row sums: smallest", np.min(sums), " largest", np.max(sums))
print("rows whose sum is exactly 1.0:", np.sum(sums == 1.0), "of", len(sums))
print("all rows sum to 1 within 1e-6:", np.allclose(sums, 1.0, atol=1e-6))
