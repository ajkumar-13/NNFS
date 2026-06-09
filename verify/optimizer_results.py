"""Reproducible proof for the optimiser-comparison numbers in Parts 22-27.

Every class below is copied verbatim from `cumulative_notebook.ipynb`, and every
optimiser config matches the exact line used in the corresponding post. Each run
calls `nnfs.init()` first, which reseeds NumPy to 0, so every optimiser sees the
identical spiral dataset and identical initial weights -- a fair comparison.

Run:  python verify/optimizer_results.py
Deps: numpy, nnfs   (pip install nnfs)

The printed table is the source of truth for the accuracy/loss figures quoted in
posts 22-27, the optimiser dashboard, and the cheatsheets. If a post disagrees
with this table, the post is wrong.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


# --- classes copied verbatim from cumulative_notebook.ipynb ---------------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def forward(self, inputs, y_true):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        clipped = np.clip(self.output, 1e-7, 1 - 1e-7)
        return np.mean(-np.log(clipped[range(len(inputs)), y_true]))

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        layer.weights -= self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
        layer.weights -= self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights -= self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# --- experiment runner ----------------------------------------------------
def run(make_optimizer, epochs=10001):
    nnfs.init()  # reseeds NumPy to 0 -> identical data + initial weights every run
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2, 64)
    act1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_act = Activation_Softmax_Loss_CategoricalCrossentropy()
    opt = make_optimizer()
    best_acc = 0.0
    for epoch in range(epochs):
        dense1.forward(X)
        act1.forward(dense1.output)
        dense2.forward(act1.output)
        loss = loss_act.forward(dense2.output, y)
        acc = np.mean(np.argmax(loss_act.output, axis=1) == y)
        best_acc = max(best_acc, acc)
        loss_act.backward(loss_act.output, y)
        dense2.backward(loss_act.dinputs)
        act1.backward(dense2.dinputs)
        dense1.backward(act1.dinputs)
        opt.pre_update_params()
        opt.update_params(dense1)
        opt.update_params(dense2)
        opt.post_update_params()
    return loss, acc, best_acc


CONFIGS = [
    ("Part 22  SGD (fixed lr=1.0)",        lambda: Optimizer_SGD(learning_rate=1.0)),
    ("Part 23  SGD + decay=1e-3",          lambda: Optimizer_SGD(learning_rate=1.0, decay=1e-3)),
    ("Part 23  SGD + decay=1e-2",          lambda: Optimizer_SGD(learning_rate=1.0, decay=1e-2)),
    ("Part 23  SGD + decay=1e-4",          lambda: Optimizer_SGD(learning_rate=1.0, decay=1e-4)),
    ("Part 24  SGD + decay=1e-3 + mom=0.9", lambda: Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)),
    ("Part 25  AdaGrad lr=1.0 decay=1e-4", lambda: Optimizer_Adagrad(learning_rate=1.0, decay=1e-4)),
    ("Part 26  RMSProp lr=0.02 d=1e-5 rho=0.999", lambda: Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)),
    ("Part 27  Adam lr=0.02 decay=1e-5",   lambda: Optimizer_Adam(learning_rate=0.02, decay=1e-5)),
]

if __name__ == "__main__":
    print(f"{'Optimiser config':46s} {'final loss':>10s} {'final acc':>10s} {'best acc':>9s}")
    print("-" * 78)
    for name, make in CONFIGS:
        loss, acc, best = run(make)
        print(f"{name:46s} {loss:10.4f} {acc*100:8.1f}% {best*100:8.1f}%")
