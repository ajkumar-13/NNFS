"""Post 04, sections 5 and 6: the Layer_Dense class, one layer on the spiral data, then two.

Run from the series root:
    python posts/04-dense-layer-class-and-spiral-data/snippets/dense_layer.py

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


nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)   # 2 input features, 3 neurons
dense1.forward(X)

print(dense1.output[:5])

# Shape trace of section 6.1.
print("X              ", X.shape)
print("dense1.weights ", dense1.weights.shape)
print("dense1.biases  ", dense1.biases.shape)
print("dense1.output  ", dense1.output.shape)
print("dense1 parameters:", dense1.weights.size + dense1.biases.size)
print("largest |entry| of dense1.output:", np.max(np.abs(dense1.output)))

dense2 = Layer_Dense(3, 3)    # 3 inputs (the three outputs of dense1), 3 neurons
dense2.forward(dense1.output)

print(dense2.output[:5])

print("dense2.weights ", dense2.weights.shape)
print("dense2.output  ", dense2.output.shape)
print("dense2 parameters:", dense2.weights.size + dense2.biases.size)
print("largest |entry| of dense2.output:", np.max(np.abs(dense2.output)))
print("dense1 and dense2 share a weight array:", dense1.weights is dense2.weights)
