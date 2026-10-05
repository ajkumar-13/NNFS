"""Post 21, section 9: the backward pass of the whole network checked against central differences.

Run from the series root:
    python posts/21-coding-the-full-backpropagation/snippets/gradient_check.py

The same classes, the same spiral data and the same eight calls as forward_backward.py, in float64:
this file does not call nnfs.init(), it seeds NumPy with np.random.seed. Every one of the 21
parameters is moved by +h and -h with h = 1e-5, the loss is recomputed, and the measured slope is
compared with the gradient that the backward pass stored. The parameters are redrawn at scale 1
before the check; what_can_go_wrong.py runs it on the 0.01 initialisation and shows why.

Needs NumPy and the nnfs helper package (pip install nnfs), which supplies the spiral data.
"""
import numpy as np
from nnfs.datasets import spiral_data


class Activation_Softmax:
    def forward(self, inputs):
        shifted       = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values    = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        # One Jacobian and one product for every sample of the batch.
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            jacobian = np.diagflat(single_output) - np.outer(single_output, single_output)
            self.dinputs[index] = single_dvalues @ jacobian


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

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels  = len(dvalues[0])

        # Integer labels become one-hot rows.
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # The gradient of each sample's loss, then the 1/N of the batch mean.
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


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


class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot, convert to indices.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Three lines: copy, subtract 1 at the true class, normalise.
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


def numerical_gradient(loss_fn, array, h=1e-5):
    """Central difference on every entry of array, which is changed in place and restored."""
    grad = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        saved = array[index]
        array[index] = saved + h
        plus = loss_fn()
        array[index] = saved - h
        minus = loss_fn()
        array[index] = saved
        grad[index] = (plus - minus) / (2 * h)
    return grad


def relative_error(analytic, numeric):
    """Largest entry-wise |a - n| / max(|a|, |n|); an entry where both are exactly 0 counts as 0."""
    scale = np.maximum(np.abs(analytic), np.abs(numeric))
    gap = np.abs(analytic - numeric)
    return np.max(np.divide(gap, scale, out=np.zeros_like(gap), where=scale > 0))


def check_network(dense1, activation1, dense2, loss_activation, X, y, h=1e-5):
    """Run the eight calls, then measure all four parameter gradients. Returns the loss and one row per array."""

    def loss_fn():
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        return loss_activation.forward(dense2.output, y)

    loss = loss_fn()
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    rows = []
    for name, parameter, analytic in (("dense2.dweights", dense2.weights, dense2.dweights),
                                      ("dense2.dbiases", dense2.biases, dense2.dbiases),
                                      ("dense1.dweights", dense1.weights, dense1.dweights),
                                      ("dense1.dbiases", dense1.biases, dense1.dbiases)):
        numeric = numerical_gradient(loss_fn, parameter, h)
        rows.append((name, analytic.shape, np.max(np.abs(analytic)),
                     relative_error(analytic, numeric), np.max(np.abs(analytic - numeric))))
    loss_fn()                                       # leave the caches as the unmoved forward pass made them
    return loss, rows


def build(seed, redraw=True, loss_class=Activation_Softmax_Loss_CategoricalCrossentropy):
    """The data and the network of forward_backward.py from a seed, in whatever precision NumPy is set to."""
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)
    loss_activation = loss_class()
    if redraw:                                      # weights and biases of ordinary size
        dense1.weights = np.random.randn(2, 3)
        dense1.biases = np.random.randn(1, 3)
        dense2.weights = np.random.randn(3, 3)
        dense2.biases = np.random.randn(1, 3)
    return dense1, activation1, dense2, loss_activation, X, y


def report(loss, rows, dense1, h=1e-5, threshold=1e-7):
    """Print one check: the state of the ReLU gates, then one line per gradient array."""
    Z1 = dense1.output
    print(f"loss {loss:.7f}; closed ReLU gates {int(np.sum(Z1 <= 0))} of {Z1.size}; "
          f"entries of Z1 exactly 0: {int(np.sum(Z1 == 0))}; within h of 0: {int(np.sum(np.abs(Z1) < h))}")
    print("gradient         shape    largest |entry|   largest relative error   largest absolute gap")
    for name, shape, size, error, gap in rows:
        verdict = "pass" if error < threshold else "FAIL"
        print(f"{name:<16} {str(shape):<8} {size:<17.1e} {error:<24.1e} {gap:<8.1e}  {verdict}")


if __name__ == "__main__":
    h = 1e-5

    print("== Seed 0, float64, h = 1e-05, weights and biases redrawn at scale 1")
    dense1, activation1, dense2, loss_activation, X, y = build(seed=0)
    loss, rows = check_network(dense1, activation1, dense2, loss_activation, X, y, h)
    print(f"X {X.dtype}, dense1.weights {dense1.weights.dtype}, loss {type(loss).__name__}")
    report(loss, rows, dense1, h)
    parameters = sum(p.size for p in (dense1.weights, dense1.biases, dense2.weights, dense2.biases))
    print(f"forward passes for the numerical side: {2 * parameters}; backward calls: 4")

    print()
    print("== Seeds 0 to 9: the largest relative error over the four arrays, and the array it occurs in")
    print("seed   closed gates   smallest |Z1|   largest relative error   in                largest absolute gap")
    for seed in range(10):
        dense1, activation1, dense2, loss_activation, X, y = build(seed)
        loss, rows = check_network(dense1, activation1, dense2, loss_activation, X, y, h)
        name, _, _, error, _ = max(rows, key=lambda row: row[3])
        gap = max(row[4] for row in rows)
        print(f"{seed:>4}   {int(np.sum(dense1.output <= 0)):<14} {np.min(np.abs(dense1.output)):<15.1e} "
              f"{error:<24.1e} {name:<17} {gap:.1e}")

    print()
    print("== Seeds 10 to 49: how often the same correct code is above the pass mark of 1e-07")
    above, worst, worst_seed, largest_gap = [], 0.0, None, 0.0
    for seed in range(10, 50):
        dense1, activation1, dense2, loss_activation, X, y = build(seed)
        loss, rows = check_network(dense1, activation1, dense2, loss_activation, X, y, h)
        error = max(row[3] for row in rows)
        largest_gap = max(largest_gap, max(row[4] for row in rows))
        if error >= 1e-7:
            above.append(seed)
        if error > worst:
            worst, worst_seed = error, seed
    print(f"above the mark on {len(above)} of 40 seeds: {above}")
    print(f"largest relative error {worst:.1e} on seed {worst_seed}; largest absolute gap {largest_gap:.1e}")
