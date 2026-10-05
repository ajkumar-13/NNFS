"""Post 09, sections 4 and 5: what the update rule does with the same budget as the random walk.

Run from the series root:
    python posts/09-introduction-to-optimisation/snippets/gradient_descent_preview.py
    python posts/09-introduction-to-optimisation/snippets/gradient_descent_preview.py long

Gradient descent on the spiral for 10,000 steps, from the same start as random_perturbation.py:
on the 21-parameter network and on the 387-parameter one, each with a learning rate of 1 and of
0.1. With the argument long, one run of 100,000 steps of the 21-parameter network at a learning
rate of 0.1 instead (about 30 seconds on a CPU).

For every run the script prints the loss at the last step, the lowest loss seen on the way, and
on how many steps the loss went up, so that the runs can be compared with the random walk, whose
final loss is always its lowest.

The gradient is computed by a backward pass that this post has not derived. Its lines are
borrowed from later posts (the loss and softmax gradient from post 19, the dense-layer gradients
from post 16, the ReLU gradient from post 17); post 22 writes the same loop with classes.

Needs NumPy and the nnfs helper package (pip install nnfs).
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


def gradient_descent(X, y, n_hidden, learning_rate, steps=10_000):
    dense1      = Layer_Dense(2, n_hidden)
    activation1 = Activation_ReLU()
    dense2      = Layer_Dense(n_hidden, 3)
    activation2 = Activation_Softmax()
    loss_fn     = Loss_CategoricalCrossentropy()

    samples = len(X)
    previous_loss, rises = None, 0
    lowest_loss, lowest_step = float('inf'), 0

    for step in range(steps + 1):
        # Forward pass: the same four calls as in the two random strategies.
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        loss = loss_fn.calculate(activation2.output, y)

        if previous_loss is not None and loss > previous_loss:
            rises += 1
        if loss < lowest_loss:
            lowest_loss, lowest_step = loss, step
        previous_loss = loss

        if step in (0, 100, 1_000, 10_000, 100_000):
            correct = int(np.sum(np.argmax(activation2.output, axis=1) == y))
            print(f"  after {step:>7,} steps: loss {loss:.4f}, {correct} of {samples} correct")
        if step == steps:
            break

        # Backward pass: the slope of the loss with respect to every parameter.
        dlogits = activation2.output.copy()                      # post 19: (y_hat - y) / N
        dlogits[range(samples), y] -= 1
        dlogits = dlogits / samples
        dweights2 = np.dot(activation1.output.T, dlogits)        # post 16
        dbiases2  = np.sum(dlogits, axis=0, keepdims=True)
        dhidden   = np.dot(dlogits, dense2.weights.T)
        dhidden[dense1.output <= 0] = 0                          # post 17: ReLU
        dweights1 = np.dot(X.T, dhidden)
        dbiases1  = np.sum(dhidden, axis=0, keepdims=True)

        # The update rule of section 5: every parameter steps against its own slope.
        dense1.weights -= learning_rate * dweights1
        dense1.biases  -= learning_rate * dbiases1
        dense2.weights -= learning_rate * dweights2
        dense2.biases  -= learning_rate * dbiases2

    print(f"  lowest loss on the way: {lowest_loss:.4f}, at step {lowest_step:,}")
    print(f"  the loss went up on {rises:,} of {steps:,} steps")
    return loss, lowest_loss, rises


nnfs.init()

if len(sys.argv) > 1 and sys.argv[1] == "long":
    # The long run of section 4: ten times the budget on the 21-parameter network.
    X, y = spiral_data(samples=100, classes=3)
    print("spiral, 3 hidden neurons (21 parameters), learning rate 0.1, 100,000 steps")
    gradient_descent(X, y, n_hidden=3, learning_rate=0.1, steps=100_000)
    sys.exit()

for n_hidden, parameters in ((3, 21), (64, 387)):
    for learning_rate in (1.0, 0.1):
        np.random.seed(0)      # replay the stream: the same spiral and the same start as random_perturbation.py
        X, y = spiral_data(samples=100, classes=3)
        print(f"spiral, {n_hidden} hidden neurons ({parameters} parameters), learning rate {learning_rate}")
        gradient_descent(X, y, n_hidden=n_hidden, learning_rate=learning_rate)
