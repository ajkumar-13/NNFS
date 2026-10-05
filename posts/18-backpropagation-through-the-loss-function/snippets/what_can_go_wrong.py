"""Post 18, section 8: the failures of the cross-entropy backward method, measured.

Run from the series root:
    python posts/18-backpropagation-through-the-loss-function/snippets/what_can_go_wrong.py

Needs only NumPy. Everything is float64.
"""
import warnings

import numpy as np


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


def clipped_backward(dvalues, y_true):
    """The same backward with the clip of forward applied to the denominator."""
    y_true = np.eye(len(dvalues[0]))[y_true]
    dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
    return -y_true / dvalues_clipped / len(dvalues)


loss_fn = Loss_CategoricalCrossentropy()

print("1. exact zeros in the predictions")
y_pred = np.array([[0.7, 0.2, 0.1],
                   [0.0, 0.0, 1.0],      # zeros on the two wrong classes
                   [0.0, 0.6, 0.4]])     # a zero on the true class
y_true = np.array([0, 2, 0])
print("   forward, per sample:", loss_fn.forward(y_pred, y_true))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    loss_fn.backward(y_pred, y_true)
for line in str(loss_fn.dinputs).splitlines():
    print("   backward:", line)
for w in caught:
    print("   raised:  ", w.category.__name__ + ":", w.message)
with np.printoptions(suppress=True, precision=4):
    for line in str(clipped_backward(y_pred, y_true) + 0.0).splitlines():
        print("   clipped: ", line)

print("2. what the clip does to the gradient: one sample, true class 0")
for p in (0.3, 1e-9):
    h = min(1e-5, p / 10)
    pred  = np.array([[p, 1 - p]])
    label = np.array([0])
    plus, minus = pred.copy(), pred.copy()
    plus[0, 0]  += h
    minus[0, 0] -= h
    slope = (loss_fn.calculate(plus, label) - loss_fn.calculate(minus, label)) / (2 * h)
    loss_fn.backward(pred, label)
    print(f"   prediction {p:g}: loss {loss_fn.calculate(pred, label):.4f}, its slope by central difference {slope + 0.0:.6g},"
          f" backward {loss_fn.dinputs[0, 0]:.6g}, clipped backward {clipped_backward(pred, label)[0, 0]:.6g}")

print("3. a soft label")
soft = np.array([[0.9, 0.05, 0.05]])
pred = np.array([[0.7, 0.1, 0.2]])


def cross_entropy(p):
    return float(np.mean(-np.sum(soft * np.log(p), axis=1)))


def slopes(f, p, h=1e-5):
    out = np.zeros_like(p)
    for k in range(p.shape[1]):
        plus, minus = p.copy(), p.copy()
        plus[0, k]  += h
        minus[0, k] -= h
        out[0, k] = (f(plus) - f(minus)) / (2 * h)
    return out


loss_fn.backward(pred, soft)
print("   backward:                      ", np.round(loss_fn.dinputs, 4))
print("   slopes of the cross-entropy:   ", np.round(slopes(cross_entropy, pred), 4),
      f" value {cross_entropy(pred):.4f}")
print("   slopes of what forward returns:", np.round(slopes(lambda p: loss_fn.calculate(p, soft), pred), 4),
      f" value {loss_fn.calculate(pred, soft):.4f}")

print("4. integer labels as a column of shape (N, 1)")
y_pred = np.array([[0.7, 0.2, 0.1],
                   [0.1, 0.6, 0.3],
                   [0.2, 0.3, 0.5]])
column = np.array([[0], [1], [2]])
loss_fn.backward(y_pred, column)
for line in str(np.round(loss_fn.dinputs, 3) + 0.0).splitlines():
    print("   column:", line)
loss_fn.backward(y_pred, column.ravel())
for line in str(np.round(loss_fn.dinputs, 3) + 0.0).splitlines():
    print("   ravel: ", line)
