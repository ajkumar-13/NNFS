"""Post 07, section 5: why the untrained output is uniform, and what that baseline is worth.

Run from the series root:
    python posts/07-coding-the-complete-forward-pass/snippets/uniform_baseline.py

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


def forward_pass(X):
    """Build the 2 -> 3 -> 3 network of section 3 and run X through it."""
    dense1      = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    dense2      = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    return dense1, activation1, dense2, activation2


X, y = spiral_data(samples=100, classes=3)
dense1, activation1, dense2, activation2 = forward_pass(X)
probabilities = activation2.output

# The two checks of section 5, on all 300 rows.
row_sums    = np.sum(probabilities, axis=1)         # (300,): one sum per row
predictions = np.argmax(probabilities, axis=1)      # (300,): one class index per row
accuracy    = np.mean(predictions == y)             # fraction of rows predicted correctly

print("1. the numbers shrink at every layer")
stages = [("X", X), ("dense1.output", dense1.output), ("activation1.output", activation1.output),
          ("dense2.output (logits)", dense2.output)]
for name, array in stages:
    print(f"   {name:<23} largest |entry| {np.max(np.abs(array)):.5f}   mean |entry| {np.mean(np.abs(array)):.6f}")
zeros = int(np.sum(activation1.output == 0))
print(f"   ReLU output: {zeros} of {activation1.output.size} entries are exactly 0")

print("2. softmax of logits this small")
print(f"   smallest probability {probabilities.min():.6f}, largest {probabilities.max():.6f}")
print(f"   farthest any probability is from 1/3: {np.max(np.abs(probabilities - 1 / 3)):.1e}")
row = int(np.argmax(np.max(probabilities, axis=1)))
print(f"   row {row}, the least uniform: logits {dense2.output[row]}")
print(f"   row {row} probabilities: {probabilities[row]}")
gap = float(dense2.output[row].max() - dense2.output[row].min())
ratio = float(probabilities[row].max() / probabilities[row].min())
print(f"   row {row}: largest logit gap {gap:.7f}, exp of that gap {np.exp(gap):.6f}")
print(f"   row {row}: largest probability over smallest {ratio:.6f}")

print("3. every row is a probability distribution")
print(f"   row sums run from {row_sums.min():.7f} to {row_sums.max():.7f}")
print(f"   rows that sum to exactly 1: {int(np.sum(row_sums == 1))} of {len(row_sums)}")
print(f"   largest |row sum - 1|: {np.max(np.abs(row_sums - 1)):.1e}")
print("   np.allclose(row_sums, 1):", np.allclose(row_sums, 1))
print("   every probability positive:", bool(np.all(probabilities > 0)))

print("4. predictions and accuracy")
print("   predictions per class:", np.bincount(predictions, minlength=3))
print(f"   correct: {int(np.sum(predictions == y))} of {len(y)}, accuracy {accuracy:.2f}")
for k in range(3):
    print(f"   always answering class {k}: {int(np.sum(y == k))} of {len(y)} correct")

print("5. why class 2 collects the predictions")
print("   dense2.weights, one row per hidden neuron, one column per class:")
for weights_row in dense2.weights:
    print("    ", weights_row)
print("   column of the largest weight in each row:", np.argmax(dense2.weights, axis=1))
tied = np.all(dense2.output == 0, axis=1)
print(f"   rows whose three logits are all exactly 0: {int(tied.sum())}, at indices {np.where(tied)[0]}")
print("   predictions on those rows:", predictions[tied], "| true classes:", y[tied])
print("   predictions on the other", int((~tied).sum()), "rows:", np.unique(predictions[~tied]))

print("6. the same network under seeds 0 to 199")
accuracies, largest_group = [], []
for seed in range(200):
    np.random.seed(seed)
    X_s, y_s = spiral_data(samples=100, classes=3)
    predictions_s = np.argmax(forward_pass(X_s)[3].output, axis=1)
    accuracies.append(np.mean(predictions_s == y_s))
    largest_group.append(np.bincount(predictions_s, minlength=3).max())
print(f"   accuracy: mean {np.mean(accuracies):.3f}, lowest {np.min(accuracies):.2f}, highest {np.max(accuracies):.2f}")
print(f"   rows given to the most-predicted class: mean {np.mean(largest_group):.0f} of 300, fewest {np.min(largest_group)}")
