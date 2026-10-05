"""Post 06, section 7: each way of getting an activation wrong, triggered on purpose.

Run from the series root:
    python posts/06-activation-functions-relu-and-softmax/snippets/pitfalls.py

Needs NumPy and the nnfs helper package (pip install nnfs), which supplies the spiral data for
case 8. nnfs.init() is not called, so arrays are float64 unless a line says float32.
"""
import warnings

import numpy as np
from nnfs.datasets import spiral_data


def softmax(logits):
    """The stable softmax of the post, as a function."""
    exp_values = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def naive_softmax(logits):
    """The definition with no shift."""
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def softmax_max_without_keepdims(logits):
    """keepdims=True left off the max only."""
    exp_values = np.exp(logits - np.max(logits, axis=1))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def softmax_without_keepdims(logits):
    """keepdims=True left off both reductions."""
    exp_values = np.exp(logits - np.max(logits, axis=1))
    return exp_values / np.sum(exp_values, axis=1)


def softmax_without_axis(logits):
    """Both reductions without an axis: they run over the whole batch."""
    exp_values = np.exp(logits - np.max(logits))
    return exp_values / np.sum(exp_values)


np.set_printoptions(precision=3, suppress=True)

# 1. np.max where np.maximum was meant.
inputs = np.array([1, -2, 3, -0.5, 0])
print("1. np.maximum(0, inputs):", np.maximum(0, inputs))
print("   np.max(inputs):", np.max(inputs))
try:
    np.max(0, inputs)
except TypeError as error:
    print("   np.max(0, inputs) raises TypeError:", error)

# 2. Softmax without keepdims=True, on a batch that is not square and on one that is.
tall = np.array([[1.0, 2.0, 3.0],
                 [3.0, 2.0, 1.0],
                 [0.0, 0.0, 5.0],
                 [2.0, 2.0, 2.0]])
try:
    softmax_max_without_keepdims(tall)
except ValueError as error:
    print("2. no keepdims, 4 rows and 3 columns, raises ValueError:", str(error).strip())
square = tall[:3]
print("   the correct softmax of the first 3 rows:")
print(softmax(square))
wrong = softmax_max_without_keepdims(square)
print("   no keepdims on the max, 3 rows and 3 columns, runs:")
print(wrong)
print("   its row sums:", np.sum(wrong, axis=1))
wrong = softmax_without_keepdims(square)
print("   no keepdims on the max or the sum, 3 rows and 3 columns, runs:")
print(wrong)
print("   its row sums:", np.sum(wrong, axis=1))

# 3. Softmax without an axis: the whole batch is normalised as if it were one row.
whole = softmax_without_axis(tall)
print("3. no axis: sum of all 12 entries =", np.round(np.sum(whole), 12), " row sums =", np.sum(whole, axis=1))

# 4. The naive softmax in float32, the type nnfs.init() gives the arrays of this series.
logits32 = np.array([[90.0, 91.0, 89.0]], dtype=np.float32)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("4. float32 logits", logits32[0], ": naive", naive_softmax(logits32)[0], " stable", softmax(logits32)[0])

# 5. A ReLU between the last dense layer and the softmax.
logits = np.array([[-2.0, -1.0, -3.0],
                   [ 2.0, -1.0, -3.0]])
print("5. softmax of the logits:")
print(softmax(logits))
print("   softmax after a ReLU on the logits:")
print(softmax(np.maximum(0, logits)))

# 6. A probability can underflow to exactly zero.
far = np.array([[0.0, -800.0]])
print("6. softmax of", far[0], "=", softmax(far)[0], " second entry exactly zero:", softmax(far)[0, 1] == 0.0)

# 7. Softmax applied to input features: only the differences inside a row survive.
points = np.array([[1.0, 2.0], [3.0, 4.0], [-5.0, -4.0]])
print("7. three different input points, one softmax output:")
print(softmax(points))

# 8. A dead ReLU neuron: its weighted sum is negative for every sample of the spiral data.
np.random.seed(0)
X, y = spiral_data(samples=100, classes=3)
weights = np.array([[1.0], [1.0]])          # one neuron, 2 inputs
bias = -3.0
z = np.dot(X, weights) + bias
print("8. largest weighted sum over the 300 samples:", np.round(np.max(z), 3),
      " non-zero ReLU outputs:", np.count_nonzero(np.maximum(0, z)), "of", z.size)

# 9. The softmax needs a batch: a single sample passed as a 1-D array has no axis 1.
try:
    softmax(np.array([1.0, 2.0, 3.0]))
except Exception as error:
    print("9. a 1-D input raises", type(error).__name__ + ":", error)
print("   the same sample as one row of shape (1, 3):", softmax(np.array([[1.0, 2.0, 3.0]])))
