"""Post 04, section 2.1: how many spiral points can a straight-line model get right?

Run from the series root:
    python posts/04-dense-layer-class-and-spiral-data/snippets/linear_baseline.py

A single Layer_Dense(2, 3) with nothing after it but softmax is a linear classifier. It is
trained here to convergence with tools that later posts build (softmax in post 06, the loss in
post 08, the gradient in post 19, the update in post 22). The training loop is not the subject
of post 04; it is here only to put a number on "hard". A least-squares fit, a second and
unrelated way to fit a linear model, is the cross-check.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Small random weights; zero biases.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


nnfs.init()

X, y = spiral_data(samples=100, classes=3)
N = len(X)
rows = np.arange(N)

largest_class = int(np.max(np.bincount(y)))
print(f"always answering one class: {largest_class} of {N} correct, {largest_class / N:.1%}")

# 1. One dense layer plus softmax, trained by gradient descent with a learning rate of 1.
dense = Layer_Dense(2, 3)
for step in range(1001):
    dense.forward(X)
    exp = np.exp(dense.output - np.max(dense.output, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    if step in (0, 10, 100, 1000):
        loss = -np.mean(np.log(probs[rows, y]))
        correct = int(np.sum(np.argmax(probs, axis=1) == y))
        print(f"step {step:4d}: loss {loss:.4f}, {correct} of {N} correct, {correct / N:.1%}")
    dvalues = probs.copy()
    dvalues[rows, y] -= 1
    dvalues = dvalues / N
    dense.weights = dense.weights - np.dot(X.T, dvalues)
    dense.biases = dense.biases - np.sum(dvalues, axis=0, keepdims=True)

# 2. Least squares: the best straight-line fit to one-hot targets, no training loop.
A = np.c_[X, np.ones(N)]                   # a column of ones gives the fit a bias
targets = np.eye(3)[y]                     # row i has a 1 in the column of sample i's class
coefficients = np.linalg.lstsq(A, targets, rcond=None)[0]
predicted = np.argmax(np.dot(A, coefficients), axis=1)
correct = int(np.sum(predicted == y))
print(f"least squares: {correct} of {N} correct, {correct / N:.1%}")
print("least-squares predictions per class:", np.bincount(predicted, minlength=3))
