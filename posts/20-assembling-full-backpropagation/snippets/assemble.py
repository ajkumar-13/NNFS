"""Post 20, sections 2 to 6: the three classes wired into one forward pass, one backward pass and one update.

Run from the series root:
    python posts/20-assembling-full-backpropagation/snippets/assemble.py

Contents: the classes of posts 16, 18 and 19, unchanged; a 2 -> 3 -> 3 network on a batch of four samples walked
forward and then backward; the same backward pass replayed on loose arrays; one gradient-descent
update of the four parameter arrays; and the same step for ten seeds.

Needs only NumPy. The only random numbers are the initial weights, seeded with np.random.seed.
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


np.set_printoptions(precision=6, suppress=True)
np.random.seed(0)

X = np.array([[ 1.0,  2.0],
              [-1.5,  0.5],
              [ 0.5, -2.0],
              [ 2.0,  1.0]])                    # (4, 2): four samples, two inputs
y = np.array([0, 1, 2, 1])                      # one integer label per sample

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Forward pass, left to right: each call reads the output of the call before it.
dense1.forward(X)                                   # X  -> Z1
activation1.forward(dense1.output)                  # Z1 -> A1
dense2.forward(activation1.output)                  # A1 -> Z2
loss = loss_activation.forward(dense2.output, y)    # Z2 -> predictions -> loss

print("== Sections 2 and 3: forward, then backward")
print(f"loss {loss:.6f}   ln 3 = {np.log(3):.6f}")
print("array                    shape")
for name, array in [("X", X), ("dense1.output", dense1.output),
                    ("activation1.output", activation1.output),
                    ("dense2.output", dense2.output),
                    ("loss_activation.output", loss_activation.output)]:
    print(f"{name:<24} {array.shape}")

# Backward pass, right to left: each call reads the dinputs of the call before it.
loss_activation.backward(loss_activation.output, y)   # predictions, labels -> dL/dZ2
dense2.backward(loss_activation.dinputs)              # dL/dZ2 -> dweights, dbiases, dL/dA1
activation1.backward(dense2.dinputs)                  # dL/dA1 -> dL/dZ1
dense1.backward(activation1.dinputs)                  # dL/dZ1 -> dweights, dbiases, dL/dX

print("gradient                 shape     read by")
for name, array, reader in [
        ("loss_activation.dinputs", loss_activation.dinputs, "dense2.backward"),
        ("dense2.dinputs", dense2.dinputs, "activation1.backward"),
        ("activation1.dinputs", activation1.dinputs, "dense1.backward"),
        ("dense1.dinputs", dense1.dinputs, "nothing"),
        ("dense2.dweights", dense2.dweights, "the update"),
        ("dense2.dbiases", dense2.dbiases, "the update"),
        ("dense1.dweights", dense1.dweights, "the update"),
        ("dense1.dbiases", dense1.dbiases, "the update")]:
    print(f"{name:<24} {str(array.shape):<9} {reader}")
print("loss_activation.dinputs")
print(loss_activation.dinputs)
print("dense2.dbiases", dense2.dbiases)
closed = int(np.sum(dense1.output <= 0))
print(f"closed ReLU gates: {closed} of {dense1.output.size};"
      f" zeros in activation1.dinputs: {int(np.sum(activation1.dinputs == 0))}")

# Section 4: the same backward pass on loose arrays, one formula per line.
dZ2 = loss_activation.output.copy()
dZ2[range(len(y)), y] -= 1
dZ2 /= len(y)                                         # (predictions - one-hot labels) / N
dW2 = np.dot(activation1.output.T, dZ2)
db2 = np.sum(dZ2, axis=0, keepdims=True)
dA1 = np.dot(dZ2, dense2.weights.T)                   # what dense2 hands back
dZ1 = dA1 * (dense1.output > 0)                       # what the ReLU hands back
dW1 = np.dot(X.T, dZ1)
db1 = np.sum(dZ1, axis=0, keepdims=True)

print("== Section 4: the chain of objects against the formulas on loose arrays")
gap = max(np.abs(a - b).max() for a, b in [(dW2, dense2.dweights), (db2, dense2.dbiases),
                                           (dW1, dense1.dweights), (db1, dense1.dbiases)])
print(f"largest difference over the four parameter gradients: {gap:.1e}")

# Section 5: one gradient-descent update of the four parameter arrays.
learning_rate = 0.01
squared_norm = sum(np.sum(g ** 2) for g in (dense1.dweights, dense1.dbiases,
                                            dense2.dweights, dense2.dbiases))

dense1.weights -= learning_rate * dense1.dweights
dense1.biases -= learning_rate * dense1.dbiases
dense2.weights -= learning_rate * dense2.dweights
dense2.biases -= learning_rate * dense2.dbiases

# A second forward pass shows what the step did to the loss.
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
new_loss = loss_activation.forward(dense2.output, y)

print("== Section 5: one update with learning rate 0.01")
print(f"loss before {loss:.6f}   after {new_loss:.6f}   change {new_loss - loss:.3e}")
print(f"sum of squared gradients {squared_norm:.6f}   -learning_rate * sum {-learning_rate * squared_norm:.3e}")
print("dense2.biases after the update", dense2.biases)


def one_step(seed, learning_rate=0.01, X=X, y=y):
    """Build the network from a seed, run forward, backward, update, forward; return both losses."""
    np.random.seed(seed)
    dense1, activation1 = Layer_Dense(X.shape[1], 3), Activation_ReLU()
    dense2, loss_activation = Layer_Dense(3, 3), Activation_Softmax_Loss_CategoricalCrossentropy()
    losses = []
    for step in range(2):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        losses.append(loss_activation.forward(dense2.output, y))
        if step == 0:
            loss_activation.backward(loss_activation.output, y)
            dense2.backward(loss_activation.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)
            for layer in (dense1, dense2):
                layer.weights -= learning_rate * layer.dweights
                layer.biases -= learning_rate * layer.dbiases
    return losses


falls = [before - after for before, after in (one_step(seed) for seed in range(10))]
print("== Section 5: the same step for seeds 0 to 9")
print(f"fall in loss: smallest {min(falls):.3e}, largest {max(falls):.3e}")

# A batch with every class equally often: the last layer's bias gradient nearly vanishes.
X_balanced = np.vstack([X, [[-1.0, -1.0], [0.0, 1.5]]])
y_balanced = np.array([0, 1, 2, 1, 0, 2])
falls = [before - after for before, after in
         (one_step(seed, X=X_balanced, y=y_balanced) for seed in range(10))]
print(f"balanced batch of six: smallest {min(falls):.3e}, largest {max(falls):.3e}")
