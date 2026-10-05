"""Post 22, sections 3 and 4: the training loop, and the documented run of Part VI.

Run from the series root:
    python posts/22-gradient-descent-optimiser/snippets/train_sgd.py

Contents: the 2 -> 64 -> 3 spiral classifier trained for 10,001 epochs of full-batch gradient
descent with Optimizer_SGD(learning_rate=1.0); the log every 1,000 epochs; and what the loss and
the accuracy did between the logged epochs.

Needs NumPy and the nnfs helper package (pip install nnfs). nnfs.init() seeds NumPy with 0.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from optimizer_sgd import (Layer_Dense, Activation_ReLU,
                           Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD)

nnfs.init()                                   # seed 0, float32 arrays, and a patched np.dot
X, y = spiral_data(samples=100, classes=3)    # X (300, 2), y (300,): three classes of 100 points

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(learning_rate=1.0)

losses, accuracies = [], []

for epoch in range(10001):
    # Forward pass.
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Accuracy: the share of samples whose largest output is the true class.
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Backward pass.
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update: one call for each layer that has parameters.
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    # Log.
    losses.append(loss)
    accuracies.append(accuracy)
    if epoch % 1000 == 0:
        print(f"epoch {epoch:5d}  loss {loss:.4f}  acc {accuracy:.4f}")

losses, accuracies = np.array(losses), np.array(accuracies)
parameters = sum(a.size for a in (dense1.weights, dense1.biases, dense2.weights, dense2.biases))
print("== Setup")
print(f"data {X.shape} {X.dtype}, labels {y.shape}, classes {len(set(y.tolist()))}; "
      f"parameters {parameters}, dtype {dense1.weights.dtype}")
print("nnfs.init() seed 0; weights 0.01 * randn, biases 0; learning rate 1.0; 10,001 full-batch epochs")

print("== Between the logged epochs")
steps = np.diff(losses)
print(f"after 100 updates: loss {losses[100]:.4f}  acc {accuracies[100]:.4f}")
print(f"the loss rose on {int(np.sum(steps > 0)):,} of {len(steps):,} updates;"
      f" largest single rise {steps.max():.4f}, at epoch {int(steps.argmax()) + 1:,}")
print(f"lowest loss {losses.min():.4f}, at epoch {int(losses.argmin()):,};"
      f" highest accuracy {accuracies.max():.4f}, at epoch {int(accuracies.argmax()):,}")
last = slice(9001, 10001)
print(f"epochs 9,001 to 10,000: loss {losses[last].min():.4f} to {losses[last].max():.4f},"
      f" accuracy {accuracies[last].min():.4f} to {accuracies[last].max():.4f}")
for first in (0, 1000):
    window = steps[first:first + 1000]
    print(f"updates {first + 1:,} to {first + 1000:,}: the loss rose on {int(np.sum(window > 0))}")
