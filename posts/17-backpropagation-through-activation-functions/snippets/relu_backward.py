"""ReLU's backward pass: the worked example of section 2 and three checks on it.

Run from the series root:
    python posts/17-backpropagation-through-activation-functions/snippets/relu_backward.py
"""
import numpy as np


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs                    # cached: backward needs the sign of every input
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()           # a copy, so the caller's array is left alone
        self.dinputs[self.inputs <= 0] = 0      # no gradient where the input was not positive


def weighted_sum_loss(Z, dvalues):
    """A stand-in loss whose gradient with respect to A = ReLU(Z) is exactly dvalues."""
    return float(np.sum(dvalues * np.maximum(0, Z)))


print("== Section 2.1: the worked example")
Z = np.array([[1.0, -2.0, 3.0]])
dvalues = np.array([[5.0, 6.0, 7.0]])
relu = Activation_ReLU()
relu.forward(Z)
relu.backward(dvalues)
print("inputs  Z       :", Z)
print("output  A       :", relu.output)
print("dvalues dL/dA   :", dvalues)
print("dinputs dL/dZ   :", relu.dinputs)
print("dvalues after the call, unchanged:", dvalues)

print()
print("== Section 2.3: testing the input or testing the output")
np.random.seed(0)
Z = np.round(np.random.randn(4, 5), 1)
Z[0, 0] = 0.0                                   # one input exactly on the corner
Z[2, 3] = 0.0                                   # and another
relu.forward(Z)
mask_from_inputs = relu.inputs <= 0
mask_from_output = relu.output <= 0
print("entries:", Z.size, " exactly zero:", int(np.sum(Z == 0)), " negative:", int(np.sum(Z < 0)))
print("entries zeroed when the input is tested :", int(mask_from_inputs.sum()))
print("entries zeroed when the output is tested:", int(mask_from_output.sum()))
print("positions where the two masks differ    :", int(np.sum(mask_from_inputs != mask_from_output)))

print()
print("== Section 6: the backward pass against a central difference, h = 1e-5")
np.random.seed(0)
Z = np.random.randn(3, 4)                       # float64, and no entry within h of the corner
dvalues = np.random.randn(3, 4)
relu.forward(Z)
relu.backward(dvalues)
h = 1e-5
numerical = np.zeros_like(Z)
for index in np.ndindex(*Z.shape):
    step = np.zeros_like(Z)
    step[index] = h
    numerical[index] = (weighted_sum_loss(Z + step, dvalues) - weighted_sum_loss(Z - step, dvalues)) / (2 * h)
print("smallest |z| in the batch:", f"{np.abs(Z).min():.4f}")
print("entries passed:", int(np.sum(Z > 0)), " entries zeroed:", int(np.sum(Z <= 0)))
print("largest gap between dinputs and the central difference:", f"{np.abs(relu.dinputs - numerical).max():.1e}")
