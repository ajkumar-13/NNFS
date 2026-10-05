"""Post 07, section 4: the shape of every array the forward pass creates.

Run from the series root:
    python posts/07-coding-the-complete-forward-pass/snippets/shape_audit.py

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

dense1      = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2      = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# The audit: one line per array, in the order the forward pass creates them.
steps = [
    ("X",                  X,                  "the input batch"),
    ("dense1.output",      dense1.output,      "X . W1 + b1"),
    ("activation1.output", activation1.output, "max(0, .) on every entry"),
    ("dense2.output",      dense2.output,      "A1 . W2 + b2, the logits"),
    ("activation2.output", activation2.output, "softmax along axis 1"),
]
for number, (name, array, operation) in enumerate(steps):
    print(f"step {number}  {name:<19} {str(array.shape):<9} {operation}")

print("parameters")
for name, layer in (("dense1", dense1), ("dense2", dense2)):
    count = layer.weights.size + layer.biases.size
    print(f"  {name}: weights {layer.weights.shape}, biases {layer.biases.shape}, {count} numbers")
print("  total:", dense1.weights.size + dense1.biases.size + dense2.weights.size + dense2.biases.size)

rows = [array.shape[0] for _, array, _ in steps]
print("rows at every step:", rows)
print("labels y:", y.shape, "one class index per row")

# The same audit on a batch of a different size: only the first number moves.
dense1.forward(X[:7])
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print("a batch of 7 rows:", X[:7].shape, "->", dense1.output.shape, "->", activation1.output.shape,
      "->", dense2.output.shape, "->", activation2.output.shape)
