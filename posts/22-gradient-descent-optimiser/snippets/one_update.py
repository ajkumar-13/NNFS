"""Post 22, section 2: one call of Optimizer_SGD.update_params, twice.

Run from the series root:
    python posts/22-gradient-descent-optimiser/snippets/one_update.py

Contents: one update on numbers small enough to check by hand, made on an object that is not a
layer; and the single update of post 20, section 5, repeated with the class on the same batch and
the same seed.

Needs only NumPy. The only random numbers are the initial weights, seeded with np.random.seed(0).
"""
from types import SimpleNamespace

import numpy as np

from optimizer_sgd import (Layer_Dense, Activation_ReLU,
                           Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD)

print("== One update, checked by hand")
layer = SimpleNamespace(weights=np.array([[0.5, -1.0], [2.0, 0.0]]),
                        biases=np.array([[0.1, -0.2]]),
                        dweights=np.array([[1.0, -2.0], [0.0, 4.0]]),
                        dbiases=np.array([[-1.0, 0.5]]))
array_before = layer.weights

optimizer = Optimizer_SGD(learning_rate=0.1)
optimizer.update_params(layer)

print("weights", layer.weights.tolist())
print("biases ", layer.biases.tolist())
print("same array object as before the update:", layer.weights is array_before)

print("== The update of post 20, section 5, made by the class")
np.random.seed(0)
X = np.array([[ 1.0,  2.0],
              [-1.5,  0.5],
              [ 0.5, -2.0],
              [ 2.0,  1.0]])
y = np.array([0, 1, 2, 1])
dense1, activation1 = Layer_Dense(2, 3), Activation_ReLU()
dense2, loss_activation = Layer_Dense(3, 3), Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.01)

losses = []
for step in range(2):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    losses.append(loss_activation.forward(dense2.output, y))
    if step == 0:
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        by_hand = [dense1.weights - 0.01 * dense1.dweights, dense1.biases - 0.01 * dense1.dbiases,
                   dense2.weights - 0.01 * dense2.dweights, dense2.biases - 0.01 * dense2.dbiases]
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        by_class = [dense1.weights, dense1.biases, dense2.weights, dense2.biases]
        gap = max(np.abs(a - b).max() for a, b in zip(by_hand, by_class))

print(f"loss before {losses[0]:.6f}   after {losses[1]:.6f}")
print(f"largest difference from the four subtractions written out: {gap:.1e}")
