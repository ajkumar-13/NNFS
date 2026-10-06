"""Post 34, sections 4 and 6: the combined gradient against a central difference, and the two-step route.

Run from the series root:
    python posts/34-sigmoid-and-binary-cross-entropy/snippets/gradient_check.py

Central difference with h = 1e-5 in float64, relative error with a pass mark of 1e-7 (posts 10, 16
and 21). The script does not call nnfs.init(). Needs only NumPy. Takes about a second.
"""
import warnings

import numpy as np

from binary_classes import Activation_Sigmoid, Activation_Sigmoid_Loss_BinaryCrossentropy
from network import Layer_Dense

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


def two_step(logits, y, clip=False):
    """The separate route: the loss gradient with respect to y_hat, then the sigmoid backward."""
    activation = Activation_Sigmoid()
    activation.forward(logits)
    y_hat = np.clip(activation.output, 1e-7, 1 - 1e-7) if clip else activation.output
    y = y.reshape(-1, 1)
    dvalues = -(y / y_hat - (1 - y) / (1 - y_hat)) / len(logits)
    activation.backward(dvalues)
    return activation.dinputs


def combined(logits, y):
    loss_activation = Activation_Sigmoid_Loss_BinaryCrossentropy()
    loss_activation.forward(logits, y)
    loss_activation.backward(loss_activation.output, y)
    return loss_activation.dinputs


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, floatmode="fixed")
    loss_activation = Activation_Sigmoid_Loss_BinaryCrossentropy()
    print(f"h = {H:g}, pass mark {PASS_MARK:g}, float64, no nnfs.init()")

    print()
    print("== The worked batch: dinputs against the measured slope of the loss")
    logits = np.array([[2.0], [-1.0], [0.5], [-3.0]])
    y = np.array([1, 0, 0, 1])
    analytic = combined(logits, y)
    numeric = numerical_gradient(lambda: loss_activation.forward(logits, y), logits)
    print("dinputs           ", analytic[:, 0])
    print("central difference", numeric[:, 0])
    print(f"largest relative error {relative_error(analytic, numeric):.1e}, largest absolute gap {np.abs(analytic - numeric).max():.1e}")
    print(f"two-step route, largest |two-step - combined|: {np.abs(two_step(logits, y) - analytic).max():.1e}")

    print()
    print("== Seeds 0 to 9: 8 logits drawn as 3 * randn, labels drawn 0 or 1")
    worst_error, worst_gap, above = 0.0, 0.0, []
    for seed in range(10):
        np.random.seed(seed)
        logits = 3 * np.random.randn(8, 1)
        y = np.random.randint(0, 2, size=8)
        analytic = combined(logits, y)
        numeric = numerical_gradient(lambda: loss_activation.forward(logits, y), logits)
        error = relative_error(analytic, numeric)
        worst_error, worst_gap = max(worst_error, error), max(worst_gap, np.abs(analytic - numeric).max())
        if error > PASS_MARK:
            entry = np.argmax(np.abs(analytic - numeric) / np.abs(analytic))
            above.append(f"seed {seed}: z = {logits[entry, 0]:.2f}, y = {y[entry]}, dinputs {analytic[entry, 0]:.1e}, "
                         f"absolute gap {abs(analytic[entry, 0] - numeric[entry, 0]):.1e}")
    print(f"dinputs: largest relative error {worst_error:.1e}, largest absolute gap {worst_gap:.1e}")
    print("checks above the pass mark:", above)

    print()
    print("== Seeds 0 to 9: one Layer_Dense(2, 1) in front of the head, 8 samples")
    worst = {"dweights": [0.0, 0.0], "dbiases": [0.0, 0.0]}
    above = []
    for seed in range(10):
        np.random.seed(seed)
        X = np.random.randn(8, 2)
        y = np.random.randint(0, 2, size=8)
        dense1 = Layer_Dense(2, 1, init="he")
        dense1.biases = 0.1 * np.random.randn(1, 1)

        def loss_fn():
            dense1.forward(X)
            return loss_activation.forward(dense1.output, y)

        loss_fn()
        loss_activation.backward(loss_activation.output, y)
        dense1.backward(loss_activation.dinputs)
        for name, array, analytic in (("dweights", dense1.weights, dense1.dweights), ("dbiases", dense1.biases, dense1.dbiases)):
            numeric = numerical_gradient(loss_fn, array)
            error = relative_error(analytic, numeric)
            worst[name][0] = max(worst[name][0], error)
            worst[name][1] = max(worst[name][1], np.abs(analytic - numeric).max())
            if error > PASS_MARK:
                above.append((seed, name))
    for name, (error, gap) in worst.items():
        print(f"{name:9} largest relative error {error:.1e}, largest absolute gap {gap:.1e}")
    print("checks above the pass mark:", above)

    print()
    print("== Saturated logits: the two routes")
    logits = np.array([[40.0], [40.0], [-40.0], [-800.0]])
    y = np.array([0, 1, 1, 1])
    print("z                       ", logits[:, 0])
    print("y                       ", y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plain = two_step(logits, y)[:, 0]
        clipped = two_step(logits, y, clip=True)[:, 0]
    print("two-step                ", plain)
    print("two-step, y_hat clipped ", clipped)
    print("combined                ", combined(logits, y)[:, 0])
    print("exact (sigma(z) - y) / 4", np.array([1.0, 0.0, -1.0, -1.0]) / 4)
    print("warnings of the two-step route:", sorted({str(w.message) for w in caught}))

    print()
    print("== Where the check cannot pass: the clipped loss is flat, the gradient is not")
    for z, label in ((10.0, 0), (20.0, 0), (-20.0, 1)):
        logits = np.array([[z]])
        y = np.array([label])
        analytic = combined(logits, y)[0, 0]
        numeric = numerical_gradient(lambda: loss_activation.forward(logits, y), logits)[0, 0]
        print(f"z = {z:>5g}, y = {label}: loss {loss_activation.forward(logits, y):.4f}  dinputs {analytic:+.6f}  central difference {numeric:+.6f}")

    print()
    print("== A confidently wrong output: the slope each loss sends to the logit, one sample, y = 1")
    print("z        sigma(z)      cross-entropy: sigma - y    squared error: 2 (sigma - y) sigma (1 - sigma)")
    for z in (0.0, -5.0, -10.0):
        s = 1.0 / (1.0 + np.exp(-z))
        print(f"{z:<8g} {s:<13.6f} {s - 1:<27.6f} {2 * (s - 1) * s * (1 - s):.6f}")
