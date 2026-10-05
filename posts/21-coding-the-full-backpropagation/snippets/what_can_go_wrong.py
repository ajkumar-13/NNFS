"""Post 21, section 11: four ways in which the script or its gradient check misleads.

Run from the series root:
    python posts/21-coding-the-full-backpropagation/snippets/what_can_go_wrong.py

1. The check run on the script's own starting point, the 0.01 initialisation with zero biases:
   correct code, and the check fails.
2. A backward pass without the division by the number of samples: every shape is right, and the
   check fails on all four arrays.
3. The backward calls in the wrong order on fresh objects.
4. The check run after nnfs.init(), in float32.

The classes and the check are imported from gradient_check.py in the same directory.
Needs NumPy and the nnfs helper package (pip install nnfs). Item 4 is last because nnfs.init()
changes NumPy for the rest of the process.
"""
import sys

import numpy as np
import nnfs

sys.dont_write_bytecode = True
from gradient_check import (Activation_Softmax_Loss_CategoricalCrossentropy, build,
                            check_network, report)

h = 1e-5

print("== 1. The check on the 0.01 initialisation with zero biases: seed 0, float64, h = 1e-05")
dense1, activation1, dense2, loss_activation, X, y = build(seed=0, redraw=False)
loss, rows = check_network(dense1, activation1, dense2, loss_activation, X, y, h)
report(loss, rows, dense1, h)
near = np.any(np.abs(dense1.output) < h, axis=1)
print(f"largest |Z1| {np.max(np.abs(dense1.output)):.1e}; samples with an entry of Z1 within h of 0: "
      f"{int(np.sum(near))}, of which at the origin: {int(np.sum(np.all(X == 0, axis=1)))}")
print("the same check without those samples:")
loss, rows = check_network(dense1, activation1, dense2, loss_activation, X[~near], y[~near], h)
report(loss, rows, dense1, h)
failed = 0
for seed in range(10):
    dense1, activation1, dense2, loss_activation, X, y = build(seed, redraw=False)
    loss, rows = check_network(dense1, activation1, dense2, loss_activation, X, y, h)
    failed += max(row[3] for row in rows) >= 1e-7
print(f"seeds 0 to 9 on this initialisation: the check fails on {failed} of 10")
dense1, activation1, dense2, loss_activation, X, y = build(seed=0, redraw=False)
check_network(dense1, activation1, dense2, loss_activation, X, y, h)
origin = np.all(X == 0, axis=1)
others = np.any(np.abs(dense1.output) < h, axis=1) & ~origin
distance = np.linalg.norm(X[others], axis=1)
origin_rows = np.sum(dense2.dinputs[origin], axis=0)
loss, rows = check_network(dense1, activation1, dense2, loss_activation, X[~others], y[~others], h)
print(f"seed 0: near-corner samples away from the origin: {int(np.sum(others))}, at distances "
      f"{distance.min():.2f} to {distance.max():.2f}; sum of the origin rows of dense2.dinputs: "
      f"{np.max(np.abs(origin_rows)):.0e}")
print(f"with only those {int(np.sum(others))} left out the largest absolute gap is {max(row[4] for row in rows):.1e}")


class Loss_Without_Division(Activation_Softmax_Loss_CategoricalCrossentropy):
    """A deliberately wrong backward: the last line of the combined class is missing."""

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1


print()
print("== 2. The division by the number of samples left out: seed 0, float64, parameters redrawn")
dense1, activation1, dense2, loss_activation, X, y = build(seed=0, loss_class=Loss_Without_Division)
loss, rows = check_network(dense1, activation1, dense2, loss_activation, X, y, h)
same = [layer.dweights.shape == layer.weights.shape and layer.dbiases.shape == layer.biases.shape
        for layer in (dense1, dense2)]
print(f"gradient shapes equal parameter shapes: {all(same)}; "
      f"largest |row sum| of loss_activation.dinputs: {np.max(np.abs(loss_activation.dinputs.sum(axis=1))):.1e}")
report(loss, rows, dense1, h)
print(f"relative error of dense2.dweights to four decimals: {rows[0][3]:.4f}; 1 - 1/N = {1 - 1 / len(X):.4f}")

print()
print("== 3. dense2.backward called before loss_activation.backward, on fresh objects")
dense1, activation1, dense2, loss_activation, X, y = build(seed=0)
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss_activation.forward(dense2.output, y)
try:
    dense2.backward(loss_activation.dinputs)
except AttributeError as error:
    print(f"AttributeError: {error}")

print()
print("== 4. The same check after nnfs.init(): seed 0, parameters redrawn")
nnfs.init()                                     # float32 arrays and an np.dot that returns float32


def worst(rows):
    """The largest relative error over the four arrays, and the array it occurs in."""
    name, _, _, error, _ = max(rows, key=lambda row: row[3])
    return f"largest relative error {error:.1e} in {name}"


dense1, activation1, dense2, loss_activation, X, y = build(seed=0)
loss, rows = check_network(dense1, activation1, dense2, loss_activation, X, y, h)
print(f"X {X.dtype}, dense1.weights {dense1.weights.dtype}, loss {type(loss).__name__}")
print(f"float32, h = 1e-05:              {worst(rows)}")

loss, rows = check_network(dense1, activation1, dense2, loss_activation, X, y, 1e-2)
print(f"float32, h = 1e-02:              {worst(rows)}")

X = X.astype(np.float64)
for layer in (dense1, dense2):
    layer.weights = layer.weights.astype(np.float64)
    layer.biases = layer.biases.astype(np.float64)
loss, rows = check_network(dense1, activation1, dense2, loss_activation, X, y, h)
print(f"arrays cast to float64, h = 1e-05: {worst(rows)}")
print(f"X {X.dtype}, dense1.weights {dense1.weights.dtype}, but np.dot(X, dense1.weights) "
      f"{np.dot(X, dense1.weights).dtype}")
