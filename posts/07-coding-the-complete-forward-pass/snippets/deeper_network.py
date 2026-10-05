"""Post 07, section 7: the same three classes wired into a deeper, wider network.

Run from the series root:
    python posts/07-coding-the-complete-forward-pass/snippets/deeper_network.py

Needs NumPy and the nnfs helper package (pip install nnfs).
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


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


X, y = spiral_data(samples=100, classes=3)

dense1      = Layer_Dense(2, 64)
activation1 = Activation_ReLU()

dense2      = Layer_Dense(64, 64)
activation2 = Activation_ReLU()

dense3      = Layer_Dense(64, 3)
activation3 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

print(activation3.output[:5])

print("shapes")
arrays = [("X", X), ("dense1.output", dense1.output), ("activation1.output", activation1.output),
          ("dense2.output", dense2.output), ("activation2.output", activation2.output),
          ("dense3.output", dense3.output), ("activation3.output", activation3.output)]
for name, array in arrays:
    print(f"  {name:<19} {str(array.shape):<10} largest |entry| {np.max(np.abs(array)):.1e}")

print("parameters")
total = 0
for name, layer in (("dense1", dense1), ("dense2", dense2), ("dense3", dense3)):
    count = layer.weights.size + layer.biases.size
    total += count
    print(f"  {name}: weights {layer.weights.shape}, biases {layer.biases.shape}, {count:,} numbers")
print(f"  total: {total:,}")

probabilities = activation3.output
predictions = np.argmax(probabilities, axis=1)
print(f"farthest any probability is from 1/3: {np.max(np.abs(probabilities - 1 / 3)):.1e}")
print(f"largest |row sum - 1|: {np.max(np.abs(np.sum(probabilities, axis=1) - 1)):.1e}")
print("predictions per class:", np.bincount(predictions, minlength=3))
print(f"correct: {int(np.sum(predictions == y))} of {len(y)}, accuracy {np.mean(predictions == y):.2f}")
