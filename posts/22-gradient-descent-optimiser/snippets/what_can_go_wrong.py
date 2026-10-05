"""Post 22, section 8: three ways the training loop goes wrong.

Run from the series root:
    python posts/22-gradient-descent-optimiser/snippets/what_can_go_wrong.py

Contents: the optimiser called where it has nothing to read; one layer left out of the update;
and the gradient added instead of subtracted. The setup is that of train_sgd.py, with the seeds of
seed_spread.py.

Needs NumPy and the nnfs helper package (pip install nnfs).
"""
import warnings

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from optimizer_sgd import (Layer_Dense, Activation_ReLU,
                           Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD)
from seed_spread import train

nnfs.init()
SEEDS = range(5)

print("== 1. Nothing to read")
X, y = spiral_data(samples=100, classes=3)
dense1, activation1 = Layer_Dense(2, 64), Activation_ReLU()
optimizer = Optimizer_SGD(learning_rate=1.0)
dense1.forward(X)
activation1.forward(dense1.output)
for label, layer in (("update_params(dense1) before any backward pass", dense1),
                     ("update_params(activation1)", activation1)):
    try:
        optimizer.update_params(layer)
    except AttributeError as error:
        print(f"{label}: AttributeError: {error}")

print("== 2. dense1 left out of the update, 10,001 epochs")


def update_dense2_only(optimizer, dense1, dense2):
    optimizer.update_params(dense2)


print("seed  final loss  final acc  loss rose on")
finals, accuracies_ = [], []
for seed in SEEDS:
    losses, accuracies, dead = train(seed, update=update_dense2_only)
    finals.append(losses[-1])
    accuracies_.append(accuracies[-1])
    print(f"{seed:<4}  {losses[-1]:<10.4f}  {accuracies[-1]:<9.4f}  {int(np.sum(np.diff(losses) > 0))}", flush=True)
print(f"final loss {min(finals):.4f} to {max(finals):.4f}, final accuracy {min(accuracies_):.4f} to {max(accuracies_):.4f}")

print("== 3. The gradient added instead of subtracted, 1,001 epochs")


def update_uphill(optimizer, dense1, dense2):
    for layer in (dense1, dense2):
        layer.weights += optimizer.learning_rate * layer.dweights
        layer.biases += optimizer.learning_rate * layer.dbiases


print("seed  loss at epochs 0, 10, 50, 100       first nan at epoch  accuracy at the end")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)     # the overflow that produces the nan
    for seed in SEEDS:
        losses, accuracies, dead = train(seed, epochs=1001, update=update_uphill)
        shown = "  ".join(f"{losses[epoch]:.4f}" for epoch in (0, 10, 50, 100))
        print(f"{seed:<4}  {shown:<34}  {int(np.argmax(np.isnan(losses))):<18}  {accuracies[-1]:.4f}")
print(f"-ln(1e-7) = {-np.log(1e-7):.4f}; two thirds of it = {-2 * np.log(1e-7) / 3:.4f}")
