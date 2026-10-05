"""Post 16, section 8: the ways a hand-written backward method fails.

Run from the series root:
    python posts/16-coding-backpropagation/snippets/what_can_go_wrong.py

Contents: backward called before forward; a ReLU backward that aliases dvalues instead of
copying it; four ways to write the bias sum and what each does to the biases; a cache
overwritten by a second forward call; and the gradient at an input of exactly zero.

Needs only NumPy. Seeded with np.random.seed(0).
"""
import numpy as np


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs                                    # cached for backward
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)          # shape of weights
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)   # shape of biases
        self.dinputs = np.dot(dvalues, self.weights.T)          # shape of inputs


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs                                    # cached for backward
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()                           # the caller's array stays intact
        self.dinputs[self.inputs <= 0] = 0                      # closed gates pass nothing back


class ReLU_Alias(Activation_ReLU):
    """The same backward without .copy(): self.dinputs and the caller's dvalues are one array."""

    def backward(self, dvalues):
        self.dinputs = dvalues
        self.dinputs[self.inputs <= 0] = 0


np.random.seed(0)
X = np.array([[ 1.0,  2.0,  3.0,  2.5],
              [ 2.0,  5.0, -1.0,  2.0],
              [-1.5,  2.7,  3.3, -0.8]])
dvalues = np.array([[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0]])

print("== 1. backward before forward")
layer = Layer_Dense(4, 3)
try:
    layer.backward(dvalues)
except AttributeError as error:
    print("AttributeError:", error)

print()
print("== 2. alias or copy in the ReLU backward")
for relu in (Activation_ReLU(), ReLU_Alias()):
    relu.forward(np.array([[1.0, -2.0, 3.0]]))
    upstream = np.array([[5.0, 6.0, 7.0]])          # the caller's array
    relu.backward(upstream)
    print(f"{type(relu).__name__:<16} dinputs {relu.dinputs}   caller's array afterwards {upstream}"
          f"   same array: {relu.dinputs is upstream}")

print()
print("== 3. four ways to write the bias sum")
biases = np.array([[0.1, 0.2, 0.3]])
ragged = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])                # (2, 3): two samples, three neurons
candidates = (("np.sum(d, axis=0, keepdims=True)", np.sum(ragged, axis=0, keepdims=True)),
              ("np.sum(d, axis=0)", np.sum(ragged, axis=0)),
              ("np.sum(d)", np.sum(ragged)),
              ("np.sum(d, axis=1, keepdims=True)", np.sum(ragged, axis=1, keepdims=True)))
for text, dbiases in candidates:
    updated = biases - 0.1 * dbiases
    print(f"{text:<34} shape {str(np.shape(dbiases)):<7} biases - 0.1 * dbiases has shape "
          f"{str(updated.shape):<7} {np.round(updated, 2).tolist()}")

print()
print("== 4. a second forward call overwrites the cache")
layer = Layer_Dense(4, 3)
layer.forward(X)
layer.backward(dvalues)
right = layer.dweights.copy()
layer.forward(X)                    # the training batch
layer.forward(X[::-1] * 2.0)        # another batch of the same shape, for example a validation batch
layer.backward(dvalues)             # meant for the training batch
print("dweights, first column, from the cached training batch:        ", right[:, 0])
print("dweights, first column, after another batch went through forward:", layer.dweights[:, 0])
print("same shape:", layer.dweights.shape == right.shape, "  same numbers:", np.allclose(layer.dweights, right))

print()
print("== 5. an input of exactly zero")
relu = Activation_ReLU()
relu.forward(np.array([[-1.0, 0.0, 1.0]]))
relu.backward(np.array([[5.0, 6.0, 7.0]]))
print("inputs ", relu.inputs, "  dinputs", relu.dinputs)
