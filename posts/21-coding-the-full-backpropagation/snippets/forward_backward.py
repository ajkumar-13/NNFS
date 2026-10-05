"""Post 21, sections 2 to 8: one forward and one backward pass on the spiral data, as one script.

Run from the series root:
    python posts/21-coding-the-full-backpropagation/snippets/forward_backward.py

Contents: the classes of posts 16, 18 and 19, unchanged; the fifteen-line script (twelve
statements under three comments) on a 2 -> 3 -> 3 network; and prints of the loss, the
predictions, the four gradient arrays and their shapes.

Needs NumPy and the nnfs helper package (pip install nnfs). nnfs.init() seeds NumPy with 0 and
makes the arrays float32. Nothing is updated and nothing is trained: the script stops at the gradients.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Activation_Softmax:
    def forward(self, inputs):
        shifted       = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values    = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        # One Jacobian and one product for every sample of the batch.
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            jacobian = np.diagflat(single_output) - np.outer(single_output, single_output)
            self.dinputs[index] = single_dvalues @ jacobian


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


class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot, convert to indices.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Three lines: copy, subtract 1 at the true class, normalise.
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


nnfs.init()                                     # seed 0, float32 arrays, a patched np.dot

X, y = spiral_data(samples=100, classes=3)      # X: (300, 2), y: (300,)

# Network.
dense1 = Layer_Dense(2, 3)                            # 2 inputs, 3 hidden neurons
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)                            # 3 hidden neurons, 3 classes
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Forward.
dense1.forward(X)                                     # (300, 2) -> (300, 3)
activation1.forward(dense1.output)                    # (300, 3), shape unchanged
dense2.forward(activation1.output)                    # (300, 3) -> (300, 3)
loss = loss_activation.forward(dense2.output, y)      # logits and labels -> one number

# Backward.
loss_activation.backward(loss_activation.output, y)   # predictions and labels -> (300, 3)
dense2.backward(loss_activation.dinputs)              # stores (3, 3), (1, 3), (300, 3)
activation1.backward(dense2.dinputs)                  # stores (300, 3), masked
dense1.backward(activation1.dinputs)                  # stores (2, 3), (1, 3), (300, 2)

print("== Sections 2 and 4: the data and the network")
print(f"X {X.shape} {X.dtype}   y {y.shape} {y.dtype}   samples per class {np.bincount(y)}")
print(f"points at the origin: {int(np.sum(np.all(X == 0, axis=1)))}   largest distance from it: "
      f"{np.max(np.linalg.norm(X, axis=1)):.1f}")

print("== Section 5: the forward pass")
np.set_printoptions(precision=7, suppress=True, floatmode="fixed")
print(f"loss {loss:.7f}   ln 3 = {np.log(3):.7f}")
print("loss_activation.output, first five rows:")
print(loss_activation.output[:5])
print(f"farthest any of the {loss_activation.output.size} probabilities is from 1/3: "
      f"{np.max(np.abs(loss_activation.output - 1 / 3)):.1e}")

print("== Section 6: the backward pass")
print("gradient                 shape      read by")
for name, array, reader in [
        ("loss_activation.dinputs", loss_activation.dinputs, "dense2.backward"),
        ("dense2.dinputs", dense2.dinputs, "activation1.backward"),
        ("activation1.dinputs", activation1.dinputs, "dense1.backward"),
        ("dense1.dinputs", dense1.dinputs, "nothing")]:
    print(f"{name:<24} {str(array.shape):<10} {reader}")
closed = int(np.sum(dense1.output <= 0))
print(f"closed ReLU gates: {closed} of {dense1.output.size};"
      f" zeros in activation1.dinputs: {int(np.sum(activation1.dinputs == 0))}")

print("== Section 7: the four gradient arrays")
np.set_printoptions(precision=3, suppress=False, floatmode="fixed")
print("dense1.dweights:")
print(dense1.dweights)
print("dense1.dbiases:", dense1.dbiases)
print("dense2.dweights:")
print(dense2.dweights)
print("dense2.dbiases:", dense2.dbiases)

print("parameter        shape    gradient          shape    same   largest |entry|   exact zeros")
count = 0
for layer_name, layer in (("dense1", dense1), ("dense2", dense2)):
    for parameter, gradient in ((layer.weights, layer.dweights), (layer.biases, layer.dbiases)):
        kind = "weights" if parameter is layer.weights else "biases"
        count += gradient.size
        print(f"{layer_name + '.' + kind:<16} {str(parameter.shape):<8} {layer_name + '.d' + kind:<17} "
              f"{str(gradient.shape):<8} {str(parameter.shape == gradient.shape):<6} "
              f"{np.max(np.abs(gradient)):<17.1e} {int(np.sum(gradient == 0))}")
print(f"gradient entries in all: {count}")

row_sums = np.sum(loss_activation.dinputs, axis=1)
print(f"largest |row sum| of loss_activation.dinputs: {np.max(np.abs(row_sums)):.1e}")
print(f"largest |entry| of loss_activation.dinputs: {np.max(np.abs(loss_activation.dinputs)):.2e}"
      f"   2/(3N) = {2 / (3 * len(X)):.2e}")
