"""Post 27, section 12: four mistakes with Optimizer_Adam, each run.

Run from the series root:
    python posts/27-adam-optimiser/snippets/what_can_go_wrong.py

Contents: the step counter read without the + 1; the learning rate of Optimizer_SGD given to Adam;
post_update_params left out of the loop; and a layer that an earlier optimiser has already touched.

Needs NumPy and the nnfs package. The second part trains five times for 1,001 epochs,
not the 10,001 of the documented run. Takes about 5 seconds.
"""
import warnings

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from adam import (Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Layer_Dense,
                  Optimizer_Adam, train)

nnfs.init()


def one_backward_pass():
    """The shared network after one forward and one backward pass, seed 0."""
    np.random.seed(0)
    X, y = spiral_data(samples=100, classes=3)
    dense1, activation1 = Layer_Dense(2, 64), Activation_ReLU()
    dense2, loss_activation = Layer_Dense(64, 3), Activation_Softmax_Loss_CategoricalCrossentropy()
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_activation.forward(dense2.output, y)
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    return dense1, dense2


print("== 1. t = self.iterations instead of self.iterations + 1")
dense1, dense2 = one_backward_pass()
optimizer = Optimizer_Adam(learning_rate=0.02)
optimizer.iterations = -1                 # makes iterations + 1 equal 0, as the wrong line would
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    optimizer.update_params(dense1)
print("warnings:", sorted({str(w.message) for w in caught}))
print(f"1 - beta_1 ** 0 = {1 - 0.9 ** 0}   NaN weights in dense1: {int(np.isnan(dense1.weights).sum())} of {dense1.weights.size}")

print()
print("== 2. Optimizer_Adam(learning_rate=1.0), the rate of post 22: five seeds, 1,001 epochs")
print("seed  loss@1  loss@1000  accuracy@1000  largest loss")
for seed in range(5):
    r = train(Optimizer_Adam(learning_rate=1.0), seed=seed, epochs=1001)
    print(f"{seed:4d}  {r['losses'][1]:6.3f}  {r['losses'][1000]:9.4f}  {r['accuracies'][1000]:13.4f}  {r['losses'].max():12.3f}")


class Stub:
    """Stands in for a dense layer: one weight, one bias, a constant gradient of 1."""

    def __init__(self):
        self.weights, self.biases = np.zeros((1, 1)), np.zeros((1, 1))
        self.dweights, self.dbiases = np.ones((1, 1)), np.ones((1, 1))


print()
print("== 3. post_update_params never called: constant gradient, learning_rate=0.02, decay=1e-3")
print("update  step, loop complete  rate    step, call missing  rate")
right, wrong = Optimizer_Adam(learning_rate=0.02, decay=1e-3), Optimizer_Adam(learning_rate=0.02, decay=1e-3)
layer_right, layer_wrong = Stub(), Stub()
for update in range(1, 10001):
    before_right, before_wrong = layer_right.weights.item(), layer_wrong.weights.item()
    right.pre_update_params()
    right.update_params(layer_right)
    right.post_update_params()
    wrong.pre_update_params()
    wrong.update_params(layer_wrong)                      # the third call is missing
    if update in (1, 10, 100, 1000, 10000):
        print(f"{update:6d}  {before_right - layer_right.weights.item():19.5f}  {right.current_learning_rate:.4f}"
              f"  {before_wrong - layer_wrong.weights.item():18.5f}  {wrong.current_learning_rate:.4f}")
print(f"iterations: {right.iterations} with the call, {wrong.iterations} without")

print()
print("== 4. a layer that another optimiser has already updated")
dense1, dense2 = one_backward_pass()
dense1.weight_cache = np.zeros_like(dense1.weights)      # what Optimizer_Adagrad or Optimizer_RMSprop leaves
dense1.bias_cache = np.zeros_like(dense1.biases)
try:
    Optimizer_Adam().update_params(dense1)
except AttributeError as error:
    print("AttributeError:", error)
