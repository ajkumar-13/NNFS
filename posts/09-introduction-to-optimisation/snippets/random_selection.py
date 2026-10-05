"""Post 09, section 2: random selection on the 21-parameter spiral classifier.

Run from the series root:
    python posts/09-introduction-to-optimisation/snippets/random_selection.py
    python posts/09-introduction-to-optimisation/snippets/random_selection.py 100000

Every iteration throws the 21 parameters away, draws 21 new ones, and keeps the set with the
lowest loss seen so far. The default is 10,000 draws at a scale of 0.05; a number on the command
line sets another budget. The script then makes 10,000 draws each at scales 1.0 and 10.0, to check
that wider draws do not rescue the method.

Needs NumPy and the nnfs helper package (pip install nnfs). Layer_Dense, Activation_ReLU,
Activation_Softmax and the two loss classes are the classes of posts 04, 06 and 08, unchanged.
"""
import sys

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


draws = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1      = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2      = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_fn     = Loss_CategoricalCrossentropy()

# The starting point: the untrained network of post 08.
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_fn.calculate(activation2.output, y)
correct = int(np.sum(np.argmax(activation2.output, axis=1) == y))
parameters = dense1.weights.size + dense1.biases.size + dense2.weights.size + dense2.biases.size
print(f"parameters: {parameters}")
print(f"untrained network: loss {loss:.4f}, {correct} of {len(y)} correct")
print(f"uniform guess: loss log(3) = {np.log(3):.4f}")

improvements, last_improvement, below_uniform = 0, 0, 0
lowest, highest = float('inf'), float('-inf')

best_loss = float('inf')

for iteration in range(draws):
    # Throw the 21 parameters away and draw 21 new ones.
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases  = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases  = 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_fn.calculate(activation2.output, y)

    if loss < best_loss:
        best_loss = loss
        # Snapshot the winning parameters.
        best_parameters = [dense1.weights.copy(), dense1.biases.copy(),
                           dense2.weights.copy(), dense2.biases.copy()]
        best_correct = int(np.sum(np.argmax(activation2.output, axis=1) == y))
        improvements += 1
        last_improvement = iteration + 1

    lowest, highest = min(lowest, loss), max(highest, loss)
    below_uniform += int(loss < np.log(3))
    if iteration + 1 in (10, 100, 1_000, 10_000, 100_000, 1_000_000):
        print(f"after {iteration + 1:>7,} draws: best loss {best_loss:.4f}, {best_correct} of {len(y)} correct")

print(f"the best set changed {improvements} times, the last time at draw {last_improvement:,}")
print(f"losses of all {draws:,} draws: from {lowest:.4f} to {highest:.4f}, {below_uniform:,} of them below log(3)")


def best_of_random_draws(scale, draws):
    """The same search with every parameter drawn as scale * randn."""
    best_loss, best_correct, below_uniform = float('inf'), 0, 0
    for _ in range(draws):
        dense1.weights = scale * np.random.randn(2, 3)
        dense1.biases  = scale * np.random.randn(1, 3)
        dense2.weights = scale * np.random.randn(3, 3)
        dense2.biases  = scale * np.random.randn(1, 3)
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        loss = loss_fn.calculate(activation2.output, y)
        below_uniform += int(loss < np.log(3))
        if loss < best_loss:
            best_loss = loss
            best_correct = int(np.sum(np.argmax(activation2.output, axis=1) == y))
    return best_loss, best_correct, below_uniform


print("wider draws, 10,000 each:")
np.random.seed(0)          # a fixed stream, so this check does not depend on the budget above
for scale in (1.0, 10.0):
    best_loss, best_correct, below_uniform = best_of_random_draws(scale, 10_000)
    print(f"  scale {scale:>4}: best loss {best_loss:.4f}, {best_correct} of {len(y)} correct, "
          f"{below_uniform:,} of 10,000 draws below log(3)")
