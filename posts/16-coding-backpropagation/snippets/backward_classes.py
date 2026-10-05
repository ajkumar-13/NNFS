"""Post 16, sections 2 to 6: Layer_Dense and Activation_ReLU with a backward method.

Run from the series root:
    python posts/16-coding-backpropagation/snippets/backward_classes.py

Contents: the two classes; the batch of post 14 pushed through Layer_Dense.backward and compared
with the numbers of posts 14 and 15; the three-element ReLU example; and a Dense, ReLU, Dense
chain walked forward and then backward with a stand-in loss.

Needs only NumPy. The only random numbers are the initial weights, seeded with np.random.seed(0).
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


np.random.seed(0)

print("== Section 3: the batch of post 14 through Layer_Dense.backward")
X = np.array([[ 1.0,  2.0,  3.0,  2.5],
              [ 2.0,  5.0, -1.0,  2.0],
              [-1.5,  2.7,  3.3, -0.8]])        # (3, 4): three samples, four inputs

dvalues = np.array([[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0]])           # (3, 3): one row per sample, one column per neuron

layer = Layer_Dense(4, 3)
layer.weights = np.array([[0.1, 0.5, 0.9],
                          [0.2, 0.6, 1.0],
                          [0.3, 0.7, 1.1],
                          [0.4, 0.8, 1.2]])     # (4, 3): the layer of post 13, one column per neuron
layer.biases = np.array([[0.1, 0.2, 0.3]])

layer.forward(X)
layer.backward(dvalues)

print("dweights", layer.dweights.shape)
print(layer.dweights)
print("dbiases", layer.dbiases.shape)
print(layer.dbiases)
print("dinputs", layer.dinputs.shape)
print(layer.dinputs)

post14_rows = np.array([[0.5, 20.1, 10.9, 4.1]] * 3)     # post 14, section 6.1: one row per neuron
column_sums = np.array([1.5, 1.8, 2.1, 2.4])             # post 15, section 4
print("dweights is the transpose of post 14's batch result:", np.allclose(layer.dweights, post14_rows.T))
print("dinputs rows are 1, 2 and 3 times post 15's [1.5 1.8 2.1 2.4]:",
      np.allclose(layer.dinputs, np.outer([1.0, 2.0, 3.0], column_sums)))
for name in ("weights", "biases", "inputs"):
    print(f"d{name} has the shape of {name}:",
          getattr(layer, "d" + name).shape == getattr(layer, name).shape)

print()
print("== Section 4: the masked copy")
relu = Activation_ReLU()
relu.forward(np.array([[1.0, -2.0, 3.0]]))
upstream = np.array([[5.0, 6.0, 7.0]])
relu.backward(upstream)
print("inputs  ", relu.inputs)
print("dvalues ", upstream)
print("dinputs ", relu.dinputs)
print("dinputs is a separate array:", relu.dinputs is not upstream)

print()
print("== Section 6: a Dense, ReLU, Dense chain, forward and then backward")
dense1 = Layer_Dense(4, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 2)

# Forward pass, left to right.
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

# A stand-in for the softmax and loss of posts 18 and 19: L = mean over the batch of
# (the sum of each sample's outputs) squared, the loss of posts 14 and 15.
Y = np.sum(dense2.output, axis=1, keepdims=True)                # (3, 1)
loss = np.mean(Y ** 2)
dloss = 2 * Y / len(X) * np.ones_like(dense2.output)            # (3, 2): dL/d(dense2.output)

# Backward pass, right to left: each dinputs is the next call's dvalues.
dense2.backward(dloss)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(f"loss {loss:.3e}")
print(f"{'array':<22}{'shape':<10}{'gradient':<22}shape")
for label, obj in (("dense1", dense1), ("dense2", dense2)):
    for name in ("weights", "biases"):
        print(f"{label + '.' + name:<22}{str(getattr(obj, name).shape):<10}"
              f"{label + '.d' + name:<22}{getattr(obj, 'd' + name).shape}")
print(f"{'dloss':<22}{str(dloss.shape):<10}-> dense2.backward")
print(f"{'dense2.dinputs':<22}{str(dense2.dinputs.shape):<10}-> activation1.backward")
print(f"{'activation1.dinputs':<22}{str(activation1.dinputs.shape):<10}-> dense1.backward")
print(f"{'dense1.dinputs':<22}{str(dense1.dinputs.shape):<10}the gradient at the data; nothing reads it")
closed = int(np.sum(dense1.output <= 0))
print(f"closed ReLU gates: {closed} of {dense1.output.size}; "
      f"zeros in activation1.dinputs: {int(np.sum(activation1.dinputs == 0))}")
