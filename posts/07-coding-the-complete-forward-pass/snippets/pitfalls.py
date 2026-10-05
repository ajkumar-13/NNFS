"""Post 07, section 9: each failure the post warns about, triggered on purpose.

Run from the series root:
    python posts/07-coding-the-complete-forward-pass/snippets/pitfalls.py

Every block prints the error message or the wrong result that the section quotes.
Needs NumPy and the nnfs helper package (pip install nnfs).
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


def build_and_run():
    """The script of section 3 from the data line on; returns the data and the four objects."""
    X, y = spiral_data(samples=100, classes=3)
    dense1      = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    dense2      = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    return X, y, dense1, activation1, dense2, activation2


# Before nnfs.init(): the same seed in plain NumPy. Kept for block 7.
np.random.seed(0)
plain = build_and_run()

nnfs.init()
X, y, dense1, activation1, dense2, activation2 = build_and_run()
correct_logits = dense2.output.copy()
correct_probabilities = activation2.output.copy()

print("1. dense2 fed dense1.output: the ReLU is skipped and nothing complains")
dense2.forward(dense1.output)
skipped = dense2.output.copy()
one_layer = np.dot(X, np.dot(dense1.weights, dense2.weights)) + dense2.biases
print("   output shape:", skipped.shape)
print(f"   largest difference from the single layer X . (W1 . W2): {np.max(np.abs(skipped - one_layer)):.1e}")
print(f"   largest difference from the correct logits: {np.max(np.abs(skipped - correct_logits)):.1e}")
activation2.forward(skipped)
print(f"   its probabilities, farthest from 1/3: {np.max(np.abs(activation2.output - 1 / 3)):.1e}",
      "| rows sum to 1:", np.allclose(np.sum(activation2.output, axis=1), 1))
dense2.forward(activation1.output)              # restore the correct logits
activation2.forward(dense2.output)              # and the correct probabilities

print("2. dense2 sized from the wrong neighbour")
try:
    wrong = Layer_Dense(2, 3)                   # sized for X, but it receives the 3 hidden values
    wrong.forward(activation1.output)
except ValueError as error:
    print("   ValueError:", error)

print("3. reading the return value of forward")
result = activation2.forward(dense2.output)
print("   forward returned:", result)
try:
    result.shape
except AttributeError as error:
    print("   AttributeError:", error)

print("4. softmax along the wrong axis")
shifted = dense2.output - np.max(dense2.output, axis=0, keepdims=True)
exp_values = np.exp(shifted)
by_column = exp_values / np.sum(exp_values, axis=0, keepdims=True)
print("   output shape:", by_column.shape)
print("   first three row sums:   ", np.sum(by_column, axis=1)[:3])
print("   the three column sums:  ", np.sum(by_column, axis=0))
print("   np.allclose(row sums, 1):", np.allclose(np.sum(by_column, axis=1), 1))

print("5. softmax used as the hidden activation")
hidden_softmax = Activation_Softmax()
hidden_softmax.forward(dense1.output)
print("   hidden rows sum to 1:", np.allclose(np.sum(hidden_softmax.output, axis=1), 1))
print(f"   hidden values run from {hidden_softmax.output.min():.3f} to {hidden_softmax.output.max():.3f}")

print("6. np.argmax without axis=1")
print("   np.argmax(activation2.output):", np.argmax(activation2.output))
print("   np.unravel_index(299, (300, 3)):", tuple(int(i) for i in np.unravel_index(299, (300, 3))))
per_row = np.argmax(activation2.output, axis=1)
print("   np.argmax(activation2.output, axis=1): shape", per_row.shape, "first five", per_row[:5])
print("   np.argmax of a three-way tie [1/3, 1/3, 1/3]:", np.argmax(np.array([1 / 3, 1 / 3, 1 / 3])))

print("7. the same seed without nnfs.init()")
plain_probabilities = plain[5].output
print("   without it:", plain[0].dtype, "data,", plain_probabilities.dtype, "probabilities, second row", plain_probabilities[1])
print("   with it:   ", X.dtype, "data,", correct_probabilities.dtype, "probabilities, second row", correct_probabilities[1])
print(f"   largest difference between the two outputs: {np.max(np.abs(plain_probabilities - correct_probabilities)):.1e}")

print("8. the layers created before the data")
np.random.seed(0)
early1 = Layer_Dense(2, 3)                      # takes the first six random numbers
early2 = Layer_Dense(3, 3)
X_late, y_late = spiral_data(samples=100, classes=3)
early1.forward(X_late)
relu = Activation_ReLU()
relu.forward(early1.output)
early2.forward(relu.output)
softmax = Activation_Softmax()
softmax.forward(early2.output)
early_predictions = np.argmax(softmax.output, axis=1)
print("   first weight row of dense1, as in section 3:", dense1.weights[0])
print("   first weight row of dense1, created first:  ", early1.weights[0])
print("   predictions per class, as in section 3:", np.bincount(per_row, minlength=3),
      "| created first:", np.bincount(early_predictions, minlength=3))
print(f"   accuracy, created first: {np.mean(early_predictions == y_late):.2f}")
