"""Post 31, section 6: the backward pass through a dropout layer, checked with the mask held fixed.

Run from the series root:
    python posts/31-dropout/snippets/gradient_check.py

A network with a dropout layer is a random function, so a central difference across two forward
passes would compare two different masks. The check draws one mask, stores it, and differences the
loss of that one thinned network. This file does not call nnfs.init(), so np.dot is NumPy's own
and every array is float64. Network: 2 inputs, 4 ReLU neurons, Layer_Dropout(0.5), 3 classes,
parameters redrawn at scale 1, on the spiral data; h = 1e-5.

Needs NumPy and the nnfs package (spiral_data only). Takes about a second.
"""
import numpy as np
from nnfs.datasets import spiral_data

from network import (Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Layer_Dense,
                     Layer_Dropout)

H = 1e-5
PASS_MARK = 1e-7


def numerical_gradient(loss_fn, array, h=H):
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


def build(seed, rate=0.5):
    """Spiral data and a 2-4-3 network with a dropout layer after the ReLU."""
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2, 4)
    activation1 = Activation_ReLU()
    dropout1 = Layer_Dropout(rate)
    dense2 = Layer_Dense(4, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    dense1.weights = np.random.randn(2, 4)          # weights and biases of ordinary size
    dense1.biases = np.random.randn(1, 4)
    dense2.weights = np.random.randn(4, 3)
    dense2.biases = np.random.randn(1, 3)
    return X, y, dense1, activation1, dropout1, dense2, loss_activation


def check(X, y, dense1, activation1, dropout1, dense2, loss_activation, dropout_backward=None):
    """Largest relative error and largest absolute gap of each of the four parameter gradients.

    dropout_backward(dropout1, dvalues), if given, replaces the layer's own backward.
    """
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output, training=True)         # the one mask of this check
    mask = dropout1.binary_mask

    def loss_fn():
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output * mask)               # the stored mask, not a new one
        return loss_activation.forward(dense2.output, y)

    loss_fn()
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    if dropout_backward is None:
        dropout1.backward(dense2.dinputs)
    else:
        dropout_backward(dropout1, dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    rows = []
    for name, parameter, analytic in (("dense2.dweights", dense2.weights, dense2.dweights),
                                      ("dense2.dbiases", dense2.biases, dense2.dbiases),
                                      ("dense1.dweights", dense1.weights, dense1.dweights),
                                      ("dense1.dbiases", dense1.biases, dense1.dbiases)):
        numeric = numerical_gradient(loss_fn, parameter)
        rows.append((name, relative_error(analytic, numeric), np.max(np.abs(analytic - numeric))))
    return rows


if __name__ == "__main__":
    network = build(0)
    print(f"h = {H:g}, pass mark {PASS_MARK:g}; X {network[0].dtype}, weights {network[2].weights.dtype}, "
          f"np.dot returns {np.dot(network[0], network[2].weights).dtype}")

    print()
    print("== Seed 0, Layer_Dropout(0.5), one mask held fixed")
    print("gradient         largest relative error   largest absolute gap")
    for name, error, gap in check(*network):
        print(f"{name:<16} {error:<24.1e} {gap:<8.1e}  {'pass' if error < PASS_MARK else 'FAIL'}")
    mask = network[4].binary_mask
    print(f"mask values {np.unique(mask)}, entries dropped {int(np.sum(mask == 0))} of {mask.size}")

    print()
    print("== Seeds 0 to 9: the largest relative error over the four arrays")
    results = [check(*build(seed)) for seed in range(10)]
    errors = [max(row[1] for row in rows) for rows in results]
    gap = max(row[2] for rows in results for row in rows)
    print(f"largest relative error {max(errors):.1e}, largest absolute gap {gap:.1e}, "
          f"seeds above the pass mark: {[seed for seed, error in enumerate(errors) if error >= PASS_MARK]}")
