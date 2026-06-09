"""Reproducible proof for the regularisation numbers (Parts 30-31 + the
Regularization dashboard). Classes copied verbatim from cumulative_notebook.ipynb.
Each run reseeds via nnfs.init() so every config trains on identical data + init.

Run:  python verify/regularization_results.py
Deps: numpy, nnfs

Matches Parts 30/31 EXACTLY: Adam(lr=0.05, decay=1e-5); regularisation on dense1
only, applied to BOTH weights and biases (weight_regularizer + bias_regularizer);
network 2 -> 64 -> 3; 10 000 epochs.

Three accuracy measurements are reported, because the posts use different ones:
  - train(drop) : training accuracy WITH dropout active (the handicapped number the
                  training loop prints; this is what Part 31 quotes -> negative gap)
  - train(full) : training accuracy with dropout OFF (the true training fit)
  - test        : a fresh spiral draw, dropout OFF
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:
    def __init__(self, n_in, n_out, weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_in, n_out)
        self.biases = np.zeros((1, n_out))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights); dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases); dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy(); return
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def forward(self, inputs, y_true):
        e = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = e / np.sum(e, axis=1, keepdims=True)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        clipped = np.clip(self.output, 1e-7, 1 - 1e-7)
        return np.mean(-np.log(clipped[range(len(inputs)), y_true]))

    def backward(self, dvalues, y_true):
        n = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(n), y_true] -= 1
        self.dinputs /= n


class Optimizer_Adam:
    def __init__(self, learning_rate=0.05, decay=1e-5, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate; self.current_learning_rate = learning_rate
        self.decay = decay; self.iterations = 0; self.epsilon = epsilon
        self.beta_1 = beta_1; self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights); layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases); layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        wmc = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bmc = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        wcc = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bcc = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights -= self.current_learning_rate * wmc / (np.sqrt(wcc) + self.epsilon)
        layer.biases -= self.current_learning_rate * bmc / (np.sqrt(bcc) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


def run(l1=0.0, l2=0.0, dropout=0.0, samples=100, epochs=10001):
    nnfs.init()
    X, y = spiral_data(samples=samples, classes=3)
    Xt, yt = spiral_data(samples=samples, classes=3)  # fresh test set
    # reg on dense1 only, both weights AND biases (matches Parts 30/31)
    dense1 = Layer_Dense(2, 64, weight_regularizer_l1=l1, weight_regularizer_l2=l2,
                         bias_regularizer_l1=l1, bias_regularizer_l2=l2)
    act1 = Activation_ReLU()
    drop1 = Layer_Dropout(dropout) if dropout else None
    dense2 = Layer_Dense(64, 3)
    loss_act = Activation_Softmax_Loss_CategoricalCrossentropy()
    opt = Optimizer_Adam(learning_rate=0.05, decay=1e-5)
    for epoch in range(epochs):
        dense1.forward(X); act1.forward(dense1.output)
        if drop1:
            drop1.forward(act1.output, training=True); dense2.forward(drop1.output)
        else:
            dense2.forward(act1.output)
        loss_act.forward(dense2.output, y)
        loss_act.backward(loss_act.output, y); dense2.backward(loss_act.dinputs)
        if drop1:
            drop1.backward(dense2.dinputs); act1.backward(drop1.dinputs)
        else:
            act1.backward(dense2.dinputs)
        dense1.backward(act1.dinputs)
        opt.pre_update_params(); opt.update_params(dense1); opt.update_params(dense2); opt.post_update_params()

    def acc_full(Xe, ye):  # dropout OFF
        dense1.forward(Xe); act1.forward(dense1.output); dense2.forward(act1.output)
        return np.mean(np.argmax(dense2.output, axis=1) == ye)

    def acc_handi(Xe, ye):  # dropout ON (handicapped), averaged over 5 masks for stability
        if not drop1:
            return acc_full(Xe, ye)
        accs = []
        for _ in range(5):
            dense1.forward(Xe); act1.forward(dense1.output)
            drop1.forward(act1.output, training=True); dense2.forward(drop1.output)
            accs.append(np.mean(np.argmax(dense2.output, axis=1) == ye))
        return float(np.mean(accs))

    return acc_handi(X, y), acc_full(X, y), acc_full(Xt, yt)


if __name__ == "__main__":
    print("== 100 samples/class (dashboard configs) ==")
    print(f"{'Config':18s} {'train(drop)':>11s} {'train(full)':>11s} {'test':>7s}")
    for name, kw in [("No reg", {}), ("L1 5e-4", {'l1':5e-4}), ("L2 5e-4", {'l2':5e-4}),
                     ("L1+L2 5e-4", {'l1':5e-4,'l2':5e-4}), ("Dropout 0.1", {'dropout':0.1}),
                     ("L2+Dropout0.1", {'l2':5e-4,'dropout':0.1})]:
        h, f, t = run(samples=100, **kw)
        print(f"{name:18s} {h*100:10.1f}% {f*100:10.1f}% {t*100:6.1f}%")
    print("\n== 1000 samples/class (Parts 30/31 rows) ==")
    print(f"{'Config':18s} {'train(drop)':>11s} {'train(full)':>11s} {'test':>7s}")
    for name, kw in [("No reg", {}), ("L2 5e-4", {'l2':5e-4}), ("L2+Dropout0.1", {'l2':5e-4,'dropout':0.1})]:
        h, f, t = run(samples=1000, **kw)
        print(f"{name:18s} {h*100:10.1f}% {f*100:10.1f}% {t*100:6.1f}%")
