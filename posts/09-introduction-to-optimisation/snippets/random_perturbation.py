"""Post 09, sections 3 and 4: random perturbation, on easy data and on the spiral.

Run from the series root:
    python posts/09-introduction-to-optimisation/snippets/random_perturbation.py
    python posts/09-introduction-to-optimisation/snippets/random_perturbation.py long
    python posts/09-introduction-to-optimisation/snippets/random_perturbation.py seeds

Every iteration adds a small random amount to all parameters, keeps the change if the loss is
lower than the best so far (starting from the loss of the initial parameters) and undoes it
otherwise. Three runs of 10,000 iterations each: the 21-parameter network on three side-by-side
clusters (vertical_data), the same network on the spiral, and a 387-parameter network (64 hidden
neurons) on the spiral. With the argument long, one run of 100,000 iterations of the 21-parameter
network on the spiral instead (about 15 seconds on a CPU). With the argument seeds, the two
spiral runs are repeated from the same start with 20 other streams of nudges and the spread of the
final loss is printed (about a minute and a half).

Needs NumPy and the nnfs helper package (pip install nnfs). Layer_Dense, Activation_ReLU,
Activation_Softmax and the two loss classes are the classes of posts 04, 06 and 08, unchanged.
"""
import contextlib
import io
import sys

import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data


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


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss     = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples        = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Integer labels:
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # One-hot labels:
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


def random_perturbation(X, y, n_hidden, iterations=10_000, nudge_seed=None):
    dense1      = Layer_Dense(2, n_hidden)
    activation1 = Activation_ReLU()
    dense2      = Layer_Dense(n_hidden, 3)
    activation2 = Activation_Softmax()
    loss_fn     = Loss_CategoricalCrossentropy()

    parameters = dense1.weights.size + dense1.biases.size + dense2.weights.size + dense2.biases.size
    print(f"  {parameters} parameters")
    kept, last_kept = 0, 0
    if nudge_seed is not None:
        np.random.seed(nudge_seed)      # same starting parameters, another stream of nudges

    # The loss to beat is the loss of the starting parameters.
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    best_loss = loss_fn.calculate(activation2.output, y)
    best_correct = int(np.sum(np.argmax(activation2.output, axis=1) == y))
    print(f"  start: loss {best_loss:.4f}, {best_correct} of {len(y)} correct")

    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases  = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases  = dense2.biases.copy()

    for iteration in range(iterations):
        # Nudge every parameter by a small random amount.
        dense1.weights += 0.05 * np.random.randn(2, n_hidden)
        dense1.biases  += 0.05 * np.random.randn(1, n_hidden)
        dense2.weights += 0.05 * np.random.randn(n_hidden, 3)
        dense2.biases  += 0.05 * np.random.randn(1, 3)

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        loss = loss_fn.calculate(activation2.output, y)

        if loss < best_loss:
            # Keep the change: the nudged parameters become the new best.
            best_loss = loss
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases  = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases  = dense2.biases.copy()
            best_correct = int(np.sum(np.argmax(activation2.output, axis=1) == y))
            kept, last_kept = kept + 1, iteration + 1
        else:
            # Revert: go back to the best parameters seen so far.
            dense1.weights = best_dense1_weights.copy()
            dense1.biases  = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases  = best_dense2_biases.copy()

        if iteration + 1 in (100, 1_000, 10_000, 100_000):
            print(f"  after {iteration + 1:>7,} iterations: loss {best_loss:.4f}, {best_correct} of {len(y)} correct")

    print(f"  kept {kept} of {iterations:,} nudges, the last at iteration {last_kept:,}")
    print("  every rejected nudge is undone, so the final loss is also the lowest loss seen")
    return best_loss, best_correct


nnfs.init()

if len(sys.argv) > 1 and sys.argv[1] == "long":
    # The long run of section 4: ten times the budget on the 21-parameter network.
    X, y = spiral_data(samples=100, classes=3)
    print("spiral, 3 hidden neurons, 100,000 iterations")
    random_perturbation(X, y, n_hidden=3, iterations=100_000)
    sys.exit()

if len(sys.argv) > 1 and sys.argv[1] == "seeds":
    # The seed spread of section 4: the same data and starting parameters, 20 other streams of nudges.
    # The gradient-descent losses are the final losses printed by gradient_descent_preview.py.
    gradient_descent = {3: {"1": 1.0776, "0.1": 1.0797}, 64: {"1": 0.8737, "0.1": 1.0226}}
    for n_hidden in (3, 64):
        finals = []
        for nudge_seed in range(500, 520):
            np.random.seed(0)
            X, y = spiral_data(samples=100, classes=3)
            with contextlib.redirect_stdout(io.StringIO()):
                finals.append(float(random_perturbation(X, y, n_hidden=n_hidden, nudge_seed=nudge_seed)[0]))
        print(f"spiral, {n_hidden} hidden neurons, 20 streams of nudges, 10,000 iterations each")
        print(f"  final loss: lowest {min(finals):.4f}, median {np.median(finals):.4f}, highest {max(finals):.4f}")
        for rate, loss in gradient_descent[n_hidden].items():
            below = sum(final < loss for final in finals)
            print(f"  below gradient descent at learning rate {rate} ({loss:.4f}): {below} of 20 streams")
    sys.exit()

print("three side-by-side clusters (vertical_data), 3 hidden neurons")
X, y = vertical_data(samples=100, classes=3)
cuts = int(np.sum(np.digitize(X[:, 0], [1 / 6, 1 / 2]) == y))
print(f"  two vertical cuts halfway between the cluster centres: {cuts} of {len(y)} correct")
random_perturbation(X, y, n_hidden=3)

for n_hidden in (3, 64):
    np.random.seed(0)          # replay the stream: the same spiral and the same start as the other scripts
    X, y = spiral_data(samples=100, classes=3)
    print(f"spiral, {n_hidden} hidden neurons")
    random_perturbation(X, y, n_hidden=n_hidden)
