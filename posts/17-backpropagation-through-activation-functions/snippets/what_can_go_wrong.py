"""Three ways an element-wise backward pass goes wrong, each reproduced on a small array.

Run from the series root:
    python posts/17-backpropagation-through-activation-functions/snippets/what_can_go_wrong.py
"""
import numpy as np


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_ReLU_NoCopy(Activation_ReLU):

    def backward(self, dvalues):
        self.dinputs = dvalues                  # BUG: a second name for the caller's array
        self.dinputs[self.inputs <= 0] = 0


print("== 1. The copy is left out")
Z = np.array([[1.0, -2.0, 3.0]])
for label, relu in (("with .copy()   ", Activation_ReLU()), ("without .copy()", Activation_ReLU_NoCopy())):
    dvalues = np.array([[5.0, 6.0, 7.0]])
    relu.forward(Z)
    relu.backward(dvalues)
    print(f"{label}: dinputs = {relu.dinputs[0]}, the caller's dvalues afterwards = {dvalues[0]}")

print()
print("== 2. The sigmoid slope is computed from the input instead of the cached output")
np.random.seed(0)
Z = np.random.randn(3, 4)
dvalues = np.random.randn(3, 4)
output = 1 / (1 + np.exp(-Z))
correct = dvalues * output * (1 - output)
wrong = dvalues * Z * (1 - Z)                   # BUG: sigma * (1 - sigma) needs the output, not Z
h = 1e-5
numerical = np.zeros_like(Z)
for index in np.ndindex(*Z.shape):
    step = np.zeros_like(Z)
    step[index] = h
    plus = np.sum(dvalues / (1 + np.exp(-(Z + step))))
    minus = np.sum(dvalues / (1 + np.exp(-(Z - step))))
    numerical[index] = (plus - minus) / (2 * h)
print("largest gap to the central difference, slope from the output:", f"{np.abs(correct - numerical).max():.1e}")
print("largest gap to the central difference, slope from the input :", f"{np.abs(wrong - numerical).max():.2f}")
print("shapes of the two results:", correct.shape, wrong.shape)

print()
print("== 3. A dead neuron: one column of Z is negative for every sample")
X = np.array([[0.5, 1.0],
              [1.5, 0.2],
              [0.3, 0.8],
              [1.1, 0.6]])
Z = np.array([[0.4, -0.3, 1.2],
              [-0.7, -1.1, 0.5],
              [0.9, -0.2, -0.6],
              [0.2, -2.0, 0.8]])
dvalues = np.array([[0.3, -0.5, 0.1],
                    [-0.2, 0.4, 0.6],
                    [0.7, 0.9, -0.3],
                    [-0.4, -0.8, 0.2]])
relu = Activation_ReLU()
relu.forward(Z)
relu.backward(dvalues)
print("dinputs of the activation:")
print(relu.dinputs)
dweights = np.dot(X.T, relu.dinputs)            # the dense layer's weight gradient, as in post 16
dbiases = np.sum(relu.dinputs, axis=0, keepdims=True)
print("dweights of the dense layer before it:")
print(np.array2string(dweights, precision=3, floatmode="fixed"))
print("dbiases:", np.array2string(dbiases, precision=3, floatmode="fixed"))
