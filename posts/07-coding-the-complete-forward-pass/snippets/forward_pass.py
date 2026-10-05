"""Post 07, section 3: the complete forward pass, spiral data through Dense, ReLU, Dense, Softmax.

Run from the series root:
    python posts/07-coding-the-complete-forward-pass/snippets/forward_pass.py

Needs NumPy and the nnfs helper package (pip install nnfs).
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# ============================ Classes (built in posts 04 and 06) ============================

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


# ============================ Data ============================

X, y = spiral_data(samples=100, classes=3)

# ============================ Build the network ============================

dense1      = Layer_Dense(2, 3)         # 2 input features, 3 hidden neurons
activation1 = Activation_ReLU()

dense2      = Layer_Dense(3, 3)         # 3 inputs (hidden), 3 outputs (one per class)
activation2 = Activation_Softmax()

# ============================ Forward pass ============================

dense1.forward(X)                       # step 1: linear
activation1.forward(dense1.output)      # step 2: ReLU
dense2.forward(activation1.output)      # step 3: linear, gives the logits
activation2.forward(dense2.output)      # step 4: softmax, gives the probabilities

# ============================ Inspect ============================

print(activation2.output[:5])
