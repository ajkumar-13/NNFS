"""Post 23, section 10: five ways the decay code goes wrong, four of them without raising anything.

Run from the series root:
    python posts/23-learning-rate-decay/snippets/what_can_go_wrong.py

Contents: the counter incremented once per layer; a hook that is never called; an update that reads
the base rate; an optimiser object used for a second run; and a negative decay.

Needs only NumPy. Seeded with np.random.seed(0). Nothing is trained: the two layers are given
gradients of ones, so that the size of every step can be read off the weights.
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


class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases  -= self.current_learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1


class Optimizer_SGD_CounterPerLayer(Optimizer_SGD):
    """Mistake 1: the counter is incremented in update_params, which runs once per layer."""

    def update_params(self, layer):
        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases  -= self.current_learning_rate * layer.dbiases
        self.iterations += 1

    def post_update_params(self):
        pass


class Optimizer_SGD_BaseRate(Optimizer_SGD):
    """Mistake 3: update_params reads the base rate, not the decayed one."""

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases  -= self.learning_rate * layer.dbiases


def two_layers():
    """Two dense layers whose gradients are all ones, so that a step of rate r moves every weight by r."""
    np.random.seed(0)
    layers = [Layer_Dense(2, 3), Layer_Dense(3, 3)]
    for layer in layers:
        layer.dweights = np.ones_like(layer.weights)
        layer.dbiases = np.ones_like(layer.biases)
    return layers


def run(optimizer, layers, updates, pre=True, post=True):
    """The update part of the training loop, `updates` times; returns the total distance one weight moved."""
    start = layers[0].weights[0, 0]
    for _ in range(updates):
        if pre:
            optimizer.pre_update_params()
        for layer in layers:
            optimizer.update_params(layer)
        if post:
            optimizer.post_update_params()
    return start - layers[0].weights[0, 0]


UPDATES = 10001

print("== Reference: the class of section 4, decay=1e-3, two layers, 10,001 updates")
reference = Optimizer_SGD(learning_rate=1.0, decay=1e-3)
moved = run(reference, two_layers(), UPDATES)
print(f"iterations {reference.iterations}   last rate {reference.current_learning_rate:.4f}   distance moved {moved:.1f}")

print()
print("== 1. The counter incremented in update_params")
wrong = Optimizer_SGD_CounterPerLayer(learning_rate=1.0, decay=1e-3)
moved = run(wrong, two_layers(), UPDATES)
print(f"iterations {wrong.iterations}   last rate {wrong.current_learning_rate:.4f}   distance moved {moved:.1f}")
print(f"1 / (1 + 2e-3 * 10000) = {1 / (1 + 2e-3 * 10000):.4f}")

print()
print("== 2. A hook that is never called")
for name, pre, post in (("no pre_update_params ", False, True), ("no post_update_params", True, False)):
    wrong = Optimizer_SGD(learning_rate=1.0, decay=1e-3)
    moved = run(wrong, two_layers(), UPDATES, pre=pre, post=post)
    print(f"{name}: iterations {wrong.iterations:5d}   last rate {wrong.current_learning_rate:.4f}   distance moved {moved:.1f}")

print()
print("== 3. update_params reads learning_rate instead of current_learning_rate")
wrong = Optimizer_SGD_BaseRate(learning_rate=1.0, decay=1e-3)
moved = run(wrong, two_layers(), UPDATES)
print(f"iterations {wrong.iterations}   last rate {wrong.current_learning_rate:.4f}   distance moved {moved:.1f}")

print()
print("== 4. One optimiser object used for a second training run")
optimizer = Optimizer_SGD(learning_rate=1.0, decay=1e-3)
run(optimizer, two_layers(), UPDATES)
optimizer.pre_update_params()
print(f"first rate of the second run: {optimizer.current_learning_rate:.4f}   iterations {optimizer.iterations}")
optimizer.iterations = 0
optimizer.pre_update_params()
print(f"after iterations = 0:         {optimizer.current_learning_rate:.4f}")

print()
print("== 5. A negative decay")
wrong = Optimizer_SGD(learning_rate=1.0, decay=-1e-3)
rates = []
try:
    for update in range(UPDATES):
        wrong.pre_update_params()
        rates.append(wrong.current_learning_rate)
        wrong.post_update_params()
except ZeroDivisionError as error:
    print(f"rate at updates 0, 500, 900, 999: {rates[0]:.1f}, {rates[500]:.1f}, {rates[900]:.1f}, {rates[999]:.1f}")
    print(f"update {wrong.iterations}: ZeroDivisionError: {error}")
