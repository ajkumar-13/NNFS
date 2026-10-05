"""Post 16, section 7: the two backward methods checked against central differences.

Run from the series root:
    python posts/16-coding-backpropagation/snippets/gradient_check.py

A Dense, ReLU, Dense chain built from the classes of the post, in float64, with the central
difference of post 10 at h = 1e-5. Every parameter is moved by +h and -h, the forward pass is
rerun, and the measured slope is compared with what backward stored. The check is then repeated
with a ReLU whose backward forgets the mask, to show what a failure looks like.

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


class ReLU_Without_Mask(Activation_ReLU):
    """A deliberately wrong backward: every shape is right and every closed gate leaks."""

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()


def forward_loss(X, dense1, activation1, dense2):
    """Forward pass and a stand-in loss: the batch mean of each sample's squared output sum."""
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    Y = np.sum(dense2.output, axis=1, keepdims=True)
    return np.mean(Y ** 2), Y


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


def check(activation1, seed=0, title=None):
    """Build the chain from a seed, run backward, and compare five gradients with central differences."""
    np.random.seed(seed)
    X = np.random.randn(5, 4)                       # five samples, four inputs
    dense1 = Layer_Dense(4, 3)
    dense2 = Layer_Dense(3, 2)
    # Weights and biases of ordinary size, so that no gradient is tiny and no bias is zero.
    dense1.weights = np.random.randn(4, 3)
    dense1.biases = np.random.randn(1, 3)
    dense2.weights = np.random.randn(3, 2)
    dense2.biases = np.random.randn(1, 2)

    def loss_fn():
        return forward_loss(X, dense1, activation1, dense2)[0]

    # Analytic gradients: one forward pass, then the backward chain.
    loss, Y = forward_loss(X, dense1, activation1, dense2)
    dloss = 2 * Y / len(X) * np.ones_like(dense2.output)
    dense2.backward(dloss)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    if title:
        print(f"== {title}")
        closed = int(np.sum(dense1.output <= 0))
        print(f"loss {loss:.4f}; closed ReLU gates {closed} of {dense1.output.size}; "
              f"smallest |Z1| {np.min(np.abs(dense1.output)):.3f}")
    errors, worst, evaluations = [], None, 0
    for label, array, analytic in (("dense2.dweights", dense2.weights, dense2.dweights),
                                   ("dense2.dbiases", dense2.biases, dense2.dbiases),
                                   ("dense1.dweights", dense1.weights, dense1.dweights),
                                   ("dense1.dbiases", dense1.biases, dense1.dbiases),
                                   ("dense1.dinputs", X, dense1.dinputs)):
        numeric = numerical_gradient(loss_fn, array)
        evaluations += 2 * array.size
        errors.append(relative_error(analytic, numeric))
        if errors[-1] == max(errors):               # remember the entry with the largest relative error
            scale = np.maximum(np.abs(analytic), np.abs(numeric))
            gap = np.abs(analytic - numeric)
            at = np.unravel_index(np.argmax(np.divide(gap, scale, out=np.zeros_like(gap), where=scale > 0)),
                                  gap.shape)
            worst = (label, analytic[at], gap[at])
        if title:
            verdict = "pass" if errors[-1] < 1e-7 else "FAIL"
            print(f"{label:<16} {str(analytic.shape):<7} largest relative error {errors[-1]:.1e}  {verdict}")
    if title:
        print(f"loss evaluations for the numerical side: {evaluations}; backward calls: 3")
    return max(errors), worst


check(Activation_ReLU(), title="Activation_ReLU as written, seed 0")
print()
check(ReLU_Without_Mask(), title="ReLU backward without the mask, seed 0")
print()
print("== Seeds 0 to 9: largest relative error over the five gradients, and the entry it occurs at")
print("seed   without the mask   as written   at an entry of     analytic value   absolute gap")
for seed in range(10):
    bad, _ = check(ReLU_Without_Mask(), seed)
    good, (label, value, gap) = check(Activation_ReLU(), seed)
    print(f"{seed:>4}   {bad:>16.1e}   {good:>10.1e}   {label:<16}   {value:>14.2e}   {gap:>12.1e}")
