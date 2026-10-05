"""Post 30, section 6: the penalty gradients checked against central differences, in float64.

Run from the series root:
    python posts/30-l1-and-l2-regularisation/snippets/gradient_check.py

The classes are imported from network.py. This file does not call nnfs.init(), so np.dot is
NumPy's own and every array is float64. The network is the small one of post 21's check (2 inputs,
3 ReLU neurons, 3 classes, parameters redrawn at scale 1) on the spiral data; both dense layers
carry penalties on weights and biases. Every one of the 21 parameters is moved by +h and -h with
h = 1e-5, the loss with its penalty is recomputed, and the slope is compared with the gradient
the backward pass stored.

Needs NumPy and the nnfs package (spiral_data only). Takes about a second.
"""
import numpy as np
from nnfs.datasets import spiral_data

from network import Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Layer_Dense

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


def build(seed, l1, l2):
    """Spiral data and a 2-3-3 network with the same strengths on every parameter array."""
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)
    strengths = dict(weight_regularizer_l1=l1, weight_regularizer_l2=l2,
                     bias_regularizer_l1=l1, bias_regularizer_l2=l2)
    dense1 = Layer_Dense(2, 3, **strengths)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3, **strengths)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    dense1.weights = np.random.randn(2, 3)          # weights and biases of ordinary size
    dense1.biases = np.random.randn(1, 3)
    dense2.weights = np.random.randn(3, 3)
    dense2.biases = np.random.randn(1, 3)
    return X, y, dense1, activation1, dense2, loss_activation


def check(X, y, dense1, activation1, dense2, loss_activation):
    """Largest relative error and largest absolute gap of each of the four parameter gradients."""

    def loss_fn():
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        data_loss = loss_activation.forward(dense2.output, y)
        return (data_loss + loss_activation.loss.regularization_loss(dense1)
                + loss_activation.loss.regularization_loss(dense2))

    loss_fn()
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
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
    X, y, dense1, activation1, dense2, loss_activation = build(0, 0.01, 0.01)
    product = np.dot(X, dense1.weights)
    print(f"h = {H:g}, pass mark {PASS_MARK:g}; X {X.dtype}, weights {dense1.weights.dtype}, np.dot returns {product.dtype}")

    print()
    print("== Seed 0, both penalties at 0.01 on all four arrays")
    print("gradient         largest relative error   largest absolute gap")
    for name, error, gap in check(X, y, dense1, activation1, dense2, loss_activation):
        print(f"{name:<16} {error:<24.1e} {gap:<8.1e}  {'pass' if error < PASS_MARK else 'FAIL'}")

    print()
    print("== Seeds 0 to 9: the largest relative error over the four arrays")
    print("penalties             largest relative error   largest absolute gap   seeds above the pass mark")
    for label, l1, l2 in (("L2 = 0.01", 0.0, 0.01), ("L1 = 0.01", 0.01, 0.0), ("L1 = L2 = 0.01", 0.01, 0.01)):
        results = [check(*build(seed, l1, l2)) for seed in range(10)]
        errors = [max(row[1] for row in rows) for rows in results]
        gap = max(row[2] for rows in results for row in rows)
        above = [seed for seed, error in enumerate(errors) if error >= PASS_MARK]
        print(f"{label:<21} {max(errors):<24.1e} {gap:<22.1e} {above}")

    print()
    print("== A weight of exactly zero under L1 = 0.01 (seed 0, dense2.weights[0, 0] set to 0)")
    X, y, dense1, activation1, dense2, loss_activation = build(0, 0.01, 0.0)
    dense2.weights[0, 0] = 0.0
    rows = check(X, y, dense1, activation1, dense2, loss_activation)
    print(f"dense2.dweights against the loss with the penalty: largest absolute gap {rows[0][2]:.6f}")
    print(f"the other three arrays: largest relative error {max(row[1] for row in rows[1:]):.1e}")
    print(f"the code's sign at w = 0: +1, so it adds {dense2.weight_regularizer_l1:g}; "
          f"central difference of 0.01 * |w| at 0: {0.01 * (abs(H) - abs(-H)) / (2 * H):g}")
