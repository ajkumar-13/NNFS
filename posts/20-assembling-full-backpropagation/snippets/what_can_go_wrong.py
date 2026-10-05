"""Post 20, section 7: the ways the assembled pass goes wrong.

Run from the series root:
    python posts/20-assembling-full-backpropagation/snippets/what_can_go_wrong.py

Contents: the logits handed to the first backward call; the ReLU left out of the backward pass;
the gradient added instead of subtracted; the loss handed to a dense layer; predictions without
labels; and an update order that looks wrong and is not.

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

X = np.array([[ 1.0,  2.0],
              [-1.5,  0.5],
              [ 0.5, -2.0],
              [ 2.0,  1.0]])
y = np.array([0, 1, 2, 1])
learning_rate = 0.01


def build(seed=0):
    np.random.seed(seed)
    return (Layer_Dense(2, 3), Activation_ReLU(), Layer_Dense(3, 3),
            Activation_Softmax_Loss_CategoricalCrossentropy())


def forward(network):
    dense1, activation1, dense2, loss_activation = network
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    return loss_activation.forward(dense2.output, y)


def backward(network):
    dense1, activation1, dense2, loss_activation = network
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


def update(network, sign=-1.0):
    for layer in (network[0], network[2]):
        layer.weights += sign * learning_rate * layer.dweights
        layer.biases += sign * learning_rate * layer.dbiases


def largest_gap(network, reference):
    """The largest absolute difference between two networks' parameter gradients, per array."""
    pairs = [("dense1.dweights", network[0].dweights, reference[0].dweights),
             ("dense1.dbiases", network[0].dbiases, reference[0].dbiases),
             ("dense2.dweights", network[2].dweights, reference[2].dweights),
             ("dense2.dbiases", network[2].dbiases, reference[2].dbiases)]
    return "   ".join(f"{name} {np.abs(a - b).max():.1e}" for name, a, b in pairs)


correct = build()
loss = forward(correct)
backward(correct)

print("== 1. the logits, not the predictions, handed to the first backward call")
network = build()
forward(network)
dense1, activation1, dense2, loss_activation = network
loss_activation.backward(dense2.output, y)              # wrong: dense2.output holds the logits
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
print("first row of dinputs, correct:", correct[3].dinputs[0])
print("first row of dinputs, wrong:  ", loss_activation.dinputs[0])
print("row sums, correct:", correct[3].dinputs.sum(axis=1).round(12) + 0.0,
      "  wrong:", loss_activation.dinputs.sum(axis=1).round(4))
print("gap to the correct gradients:", largest_gap(network, correct))

print("== 2. the ReLU left out of the backward pass")
network = build()
forward(network)
dense1, activation1, dense2, loss_activation = network
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
dense1.backward(dense2.dinputs)                         # wrong: activation1.backward is skipped
wrong, right = dense1.dweights, correct[0].dweights
print("dense1.dweights, correct")
print(right)
print("dense1.dweights, ReLU skipped")
print(wrong)
print(f"entries that differ: {int(np.sum(~np.isclose(wrong, right)))} of {wrong.size}")

print("== 3. the gradient added instead of subtracted")
for name, sign in [("-=", -1.0), ("+=", +1.0)]:
    network = build()
    before = forward(network)
    backward(network)
    update(network, sign)
    after = forward(network)
    print(f"{name}   loss before {before:.6f}   after {after:.6f}   change {after - before:+.3e}")

print("== 4. the loss handed to dense2.backward")
network = build()
loss = forward(network)
dense1, activation1, dense2, loss_activation = network
dense2.backward(loss)                                   # wrong: a scalar, and no error
print("dense2.dweights", dense2.dweights.shape, "  dense2.dbiases", np.shape(dense2.dbiases),
      "  dense2.dinputs", dense2.dinputs.shape, "  activation1.inputs", activation1.inputs.shape)
try:
    activation1.backward(dense2.dinputs)                # the next call of the pass
except IndexError as error:
    print(f"IndexError: {error}")
try:
    dense2.weights -= learning_rate * dense2.dweights   # the update would fail as well
except ValueError as error:
    print(f"ValueError: {error}")

print("== 5. predictions without labels")
network = build()
try:
    network[3].forward(np.zeros((1, 3)))
except TypeError as error:
    print(f"TypeError: {error}")
network[3].activation.forward(np.array([[2.0, 1.0, 0.0]]))
print("loss_activation.activation.output", network[3].activation.output)

print("== 6. not a mistake: each layer updated straight after its own backward call")
network = build()
forward(network)
dense1, activation1, dense2, loss_activation = network
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
dense2.weights -= learning_rate * dense2.dweights       # dense2.dinputs already exists
dense2.biases -= learning_rate * dense2.dbiases
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
dense1.weights -= learning_rate * dense1.dweights
dense1.biases -= learning_rate * dense1.dbiases
update(correct)
same = all(np.array_equal(a, b) for a, b in [(dense1.weights, correct[0].weights),
                                              (dense1.biases, correct[0].biases),
                                              (dense2.weights, correct[2].weights),
                                              (dense2.biases, correct[2].biases)])
print("same parameters as four backward calls followed by four subtractions:", same)
