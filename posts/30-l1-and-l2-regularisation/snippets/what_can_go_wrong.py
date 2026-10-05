"""Post 30, section 11: three mistakes with the penalty, each measured.

Run from the series root:
    python posts/30-l1-and-l2-regularisation/snippets/what_can_go_wrong.py

The network, the data and the check are those of gradient_check.py (float64, no nnfs.init()),
with L2 = 0.01 on all four parameter arrays, seed 0. The first two mistakes are subclasses of
Layer_Dense with one thing wrong in backward; the third is a strength with the wrong sign.

Needs NumPy and the nnfs package (spiral_data only). Takes about a second.
"""
import numpy as np

import gradient_check
from gradient_check import build, check
from network import Layer_Dense, Loss_CategoricalCrossentropy


class Layer_Dense_NoTerm(Layer_Dense):
    """Mistake 1: the penalty is in the loss, and backward is still the one of post 16."""

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dense_NoTwo(Layer_Dense):
    """Mistake 2: the L2 term without its factor of 2."""

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.weight_regularizer_l2 > 0:
            self.dweights += self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases += self.bias_regularizer_l2 * self.biases
        self.dinputs = np.dot(dvalues, self.weights.T)


def checked_with(layer_class):
    """The four rows of the gradient check with layer_class standing in for Layer_Dense."""
    gradient_check.Layer_Dense = layer_class
    try:
        return check(*build(0, 0.0, 0.01))
    finally:
        gradient_check.Layer_Dense = Layer_Dense


if __name__ == "__main__":
    print("== L2 = 0.01 on every array, seed 0: largest relative error (largest absolute gap)")
    print("gradient         correct              no term in backward   term without the 2")
    columns = [checked_with(c) for c in (Layer_Dense, Layer_Dense_NoTerm, Layer_Dense_NoTwo)]
    for correct, no_term, no_two in zip(*columns):
        print(f"{correct[0]:<16} {correct[1]:.1e} ({correct[2]:.1e})    "
              f"{no_term[1]:.1e} ({no_term[2]:.1e})     {no_two[1]:.1e} ({no_two[2]:.1e})")
    X, y, dense1, activation1, dense2, loss_activation = build(0, 0.0, 0.01)
    print(f"largest |w| in dense2: {np.abs(dense2.weights).max():.4f}; "
          f"2 * 0.01 * that: {0.02 * np.abs(dense2.weights).max():.4f}; 0.01 * that: {0.01 * np.abs(dense2.weights).max():.4f}")

    print()
    print("== A strength with the wrong sign")
    np.random.seed(0)
    layer = Layer_Dense(2, 3, weight_regularizer_l2=-0.01)
    layer.forward(np.zeros((1, 2)))
    layer.backward(np.zeros((1, 3)))
    print("weight_regularizer_l2 = -0.01: regularization_loss",
          Loss_CategoricalCrossentropy().regularization_loss(layer),
          " largest |dweights|", float(np.abs(layer.dweights).max()))
