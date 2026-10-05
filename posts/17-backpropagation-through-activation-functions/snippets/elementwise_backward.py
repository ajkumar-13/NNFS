"""The one-line element-wise backward for ReLU, sigmoid, and tanh, each checked numerically.

Run from the series root:
    python posts/17-backpropagation-through-activation-functions/snippets/elementwise_backward.py
"""
import numpy as np


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # f'(z) = sigma(z) * (1 - sigma(z)), read from the cached output
        self.dinputs = dvalues * self.output * (1 - self.output)


class Activation_Tanh:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        # f'(z) = 1 - tanh(z)^2, read from the cached output
        self.dinputs = dvalues * (1 - self.output ** 2)


def numerical_dinputs(activation, Z, dvalues, h=1e-5):
    """Central difference of L = sum(dvalues * f(Z)) with respect to every entry of Z."""
    def loss(Z_moved):
        activation.forward(Z_moved)
        return float(np.sum(dvalues * activation.output))

    grad = np.zeros_like(Z)
    for index in np.ndindex(*Z.shape):
        step = np.zeros_like(Z)
        step[index] = h
        grad[index] = (loss(Z + step) - loss(Z - step)) / (2 * h)
    return grad


activations = [("ReLU", Activation_ReLU()), ("sigmoid", Activation_Sigmoid()), ("tanh", Activation_Tanh())]

print("== Section 3: one upstream gradient through three activations")
Z = np.array([[1.0, -2.0, 3.0]])
dvalues = np.array([[5.0, 6.0, 7.0]])
print("Z       =", Z[0])
print("dvalues =", dvalues[0])
print(f"{'':8} {'local slope f_prime(Z)':>31}   {'dinputs = dvalues * f_prime(Z)':>31}")
for name, activation in activations:
    activation.forward(Z)
    activation.backward(dvalues)
    slope = activation.dinputs[0] / dvalues[0]
    print(f"{name:8} {np.array2string(slope, precision=6, floatmode='fixed'):>31}   "
          f"{np.array2string(activation.dinputs[0], precision=5, floatmode='fixed'):>31}")

print()
print("== Section 3.1: the two slopes at their peak and far from it")
sigmoid, tanh = Activation_Sigmoid(), Activation_Tanh()
points = np.array([[0.0, 2.0, 5.0, 10.0]])
sigmoid.forward(points)
tanh.forward(points)
sigmoid.backward(np.ones_like(points))
tanh.backward(np.ones_like(points))
print("z             ", "".join(f"{z:>12.0f}" for z in points[0]))
print("sigmoid slope ", "".join(f"{s:>12.2e}" for s in sigmoid.dinputs[0]))
print("tanh slope    ", "".join(f"{s:>12.2e}" for s in tanh.dinputs[0]))

print()
print("== Section 3.2: the Jacobian of one sample is diagonal")
sigmoid.forward(Z)
slopes = sigmoid.output[0] * (1 - sigmoid.output[0])
jacobian = np.diagflat(slopes)                  # entry (k, j) is d a_k / d z_j
print("sigmoid Jacobian at Z:")
print(np.array2string(jacobian, precision=6, floatmode="fixed"))
print("non-zero entries:", int(np.count_nonzero(jacobian)), "of", jacobian.size)
print("dvalues @ jacobian      =", np.array2string(dvalues[0] @ jacobian, precision=5, floatmode="fixed"))
print("dvalues * f_prime(Z)    =", np.array2string(dvalues[0] * slopes, precision=5, floatmode="fixed"))

print()
print("== Section 6: each backward against a central difference, h = 1e-5, float64")
np.random.seed(0)
Z = np.random.randn(3, 4)
dvalues = np.random.randn(3, 4)
for name, activation in activations:
    numerical = numerical_dinputs(activation, Z, dvalues)
    activation.forward(Z)
    activation.backward(dvalues)
    gap = np.abs(activation.dinputs - numerical).max()
    print(f"{name:8} largest gap between dinputs and the central difference: {gap:.1e}")
