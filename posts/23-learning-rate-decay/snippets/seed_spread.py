"""Post 23, sections 6 and 7: a constant rate against decay=1e-3, for five seeds.

Run from the series root:
    python posts/23-learning-rate-decay/snippets/seed_spread.py

Contents: the classes of posts 16 and 19, unchanged; Optimizer_SGD with decay; the training loop of
train_with_decay.py as a function; and ten runs of 10,001 epochs, Optimizer_SGD(learning_rate=1.0)
(the run of post 22) and Optimizer_SGD(learning_rate=1.0, decay=1e-3), each for seeds 0 to 4.

Needs NumPy and the nnfs package. nnfs.init() is called once; every run then calls
np.random.seed(seed) and draws its own spiral data and its own weights. Seed 0 is the documented
run. Takes about a minute.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


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


class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases  -= self.current_learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1


def build(seed):
    """The spiral data and a fresh 2 -> 64 -> 3 network, drawn in the order of the documented run."""
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)
    network = (Layer_Dense(2, 64), Activation_ReLU(), Layer_Dense(64, 3),
               Activation_Softmax_Loss_CategoricalCrossentropy())
    return X, y, network


def train(X, y, network, optimizer, epochs=10001):
    """The loop of train_with_decay.py. Returns the loss and the accuracy of every epoch."""
    dense1, activation1, dense2, loss_activation = network
    losses, accuracies = np.empty(epochs), np.empty(epochs)
    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy    = np.mean(predictions == y)
        losses[epoch], accuracies[epoch] = loss, accuracy

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    return losses, accuracies


def gradient_norm(network):
    """The length of the whole gradient: the root of the sum of squares over the four arrays."""
    dense1, _, dense2, _ = network
    arrays = (dense1.dweights, dense1.dbiases, dense2.dweights, dense2.dbiases)
    return float(np.sqrt(sum(np.sum(np.square(a, dtype=np.float64)) for a in arrays)))


def describe(losses, accuracies):
    """One row of the table: where the run ended and how it moved on the way."""
    steps = np.diff(losses)                         # 10,000 changes of the loss between epochs
    return (f"{losses[-1]:.4f}  {accuracies[-1]:.4f}  {int(np.sum(steps > 0)):5d}  {max(0.0, steps.max()):7.4f}  "
            f"{losses[-1000:].min():.3f} to {losses[-1000:].max():.3f}")


nnfs.init()
SEEDS = [0, 1, 2, 3, 4]
SETTINGS = [("constant", 0.0), ("decay 1e-3", 1e-3)]

print("setup: nnfs.init() once (float32), then per run np.random.seed(seed), spiral_data(samples=100, classes=3),")
print("       Layer_Dense(2, 64), ReLU, Layer_Dense(64, 3), softmax with cross-entropy, weights 0.01 * randn,")
print("       10,001 full-batch epochs, Optimizer_SGD(learning_rate=1.0) or Optimizer_SGD(learning_rate=1.0, decay=1e-3)")
print()
print("seed  setting     loss    acc     rises  largest  loss in the last 1,000 epochs   |gradient|  mean loss  mean acc")

final = {name: {"loss": [], "acc": [], "rise": [], "band": [], "mean_loss": [], "mean_acc": []} for name, _ in SETTINGS}
for seed in SEEDS:
    for name, decay in SETTINGS:
        X, y, network = build(seed)
        optimizer = Optimizer_SGD(learning_rate=1.0, decay=decay)
        losses, accuracies = train(X, y, network, optimizer)
        print(f"{seed:4d}  {name:<10}  {describe(losses, accuracies)}                  {gradient_norm(network):.4f}"
              f"      {losses[-1000:].mean():.4f}     {accuracies[-1000:].mean():.4f}")
        final[name]["loss"].append(losses[-1])
        final[name]["acc"].append(accuracies[-1])
        final[name]["rise"].append(max(0.0, np.diff(losses).max()))
        final[name]["band"].append(losses[-1000:].max() - losses[-1000:].min())
        final[name]["mean_loss"].append(losses[-1000:].mean())
        final[name]["mean_acc"].append(accuracies[-1000:].mean())

print()
for name, _ in SETTINGS:
    f = final[name]
    print(f"{name:<10}  final loss {min(f['loss']):.4f} to {max(f['loss']):.4f}   "
          f"final accuracy {min(f['acc']):.4f} to {max(f['acc']):.4f}   "
          f"largest rise {min(f['rise']):.4f} to {max(f['rise']):.4f}   "
          f"width of the last band {min(f['band']):.3f} to {max(f['band']):.3f}")

constant, decayed = final["constant"], final["decay 1e-3"]
lower = [s for s, a, b in zip(SEEDS, decayed["loss"], constant["loss"]) if a < b]
higher = [s for s, a, b in zip(SEEDS, decayed["acc"], constant["acc"]) if a > b]
equal = [s for s, a, b in zip(SEEDS, decayed["acc"], constant["acc"]) if a == b]
print(f"decay ends with the lower loss on seeds {lower}, with the higher accuracy on seeds {higher}, "
      f"with the same accuracy on seeds {equal}")

print()
print("the same comparison on the means of the last 1,000 epochs, which do not depend on the stopping epoch")
for name, _ in SETTINGS:
    f = final[name]
    print(f"{name:<10}  mean loss {min(f['mean_loss']):.4f} to {max(f['mean_loss']):.4f}   "
          f"mean accuracy {min(f['mean_acc']):.4f} to {max(f['mean_acc']):.4f}")
lower_mean = [s for s, a, b in zip(SEEDS, decayed["mean_loss"], constant["mean_loss"]) if a < b]
higher_mean = [s for s, a, b in zip(SEEDS, decayed["mean_acc"], constant["mean_acc"]) if a > b]
print(f"decay has the lower mean loss on seeds {lower_mean}, the higher mean accuracy on seeds {higher_mean}")
