"""Post 26, sections 5 to 7: Optimizer_RMSprop, and the documented run beside post 25's AdaGrad run.

Run from the series root:
    python posts/26-rmsprop/snippets/rmsprop.py

Contents: the layer, activation and loss classes of posts 16 and 19, unchanged; Optimizer_Adagrad
of post 25, unchanged; Optimizer_RMSprop; the shared training run of Part VI as one function; and
the two documented runs with their caches read at seven epochs.

Needs NumPy and the nnfs package (spiral data, and nnfs.init(): float32, np.dot replaced, seed 0).
Importing this file trains nothing; the five seeds_ scripts and what_can_go_wrong.py import the
classes, train() and seed_table() from it.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()     # once per process: every further call wraps np.dot again and slows each run down


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


class Optimizer_Adagrad:

    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.epsilon               = epsilon
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        # Lazy cache creation on first call.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        # Accumulate squared gradients.
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache   += layer.dbiases ** 2

        # Per-parameter update.
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:

    def __init__(self, learning_rate=0.02, decay=0.0,
                 epsilon=1e-7, rho=0.9):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.epsilon               = epsilon
        self.rho                   = rho
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        # Lazy cache creation on first call.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        # Moving average of squared gradients: the two statements that differ from AdaGrad.
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache   = self.rho * layer.bias_cache + \
                             (1 - self.rho) * layer.dbiases ** 2

        # Per-parameter update, as in AdaGrad.
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


def train(optimizer, seed=0, epochs=10001, watch=None):
    """The shared run of Part VI with the given optimiser; returns every epoch's loss and accuracy.

    watch, when given, is called after each update as watch(epoch, dense1, dense2).
    """
    np.random.seed(seed)                           # nnfs.init() seeds with 0, so seed 0 is the documented run
    X, y = spiral_data(samples=100, classes=3)     # (300, 2), (300,)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    losses, accuracies = [], []
    for epoch in range(epochs):
        # Forward pass, accuracy and backward pass: the loop of post 22, unchanged.
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update: the three-call contract.
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

        losses.append(loss)
        accuracies.append(accuracy)
        if watch is not None:
            watch(epoch, dense1, dense2)
    return np.array(losses), np.array(accuracies)


def all_parameters(dense1, dense2, name):
    """One flat array over the 387 parameters: name is 'cache' or 'd' (the gradients)."""
    weight, bias = ("weight_cache", "bias_cache") if name == "cache" else ("dweights", "dbiases")
    return np.concatenate([getattr(layer, attribute).ravel()
                           for layer in (dense1, dense2) for attribute in (weight, bias)])


def seed_table(description, make_optimizer, seeds):
    """Full runs of train() for a list of seeds: the final epoch and the last 1,000 epochs of each."""
    print("setup: nnfs.init(), then np.random.seed(seed) before the data and the weights are drawn; "
          "10,001 full-batch epochs")
    print(description)
    print("seed   final loss   final accuracy   last 1,000 epochs: mean accuracy   lowest accuracy   highest loss")
    rows = []
    for seed in seeds:
        losses, accuracies = train(make_optimizer(), seed=seed)
        rows.append((losses[-1], accuracies[-1], accuracies[-1000:].mean(), accuracies[-1000:].min(),
                     losses[-1000:].max()))
        print(f"{seed:4d}   {rows[-1][0]:.4f}       {rows[-1][1]:.4f}                              "
              f"{rows[-1][2]:.4f}          {rows[-1][3]:.4f}            {rows[-1][4]:.4f}", flush=True)
    rows = np.array(rows)
    names = ("final loss", "final accuracy", "mean accuracy of the last 1,000 epochs",
             "lowest accuracy of the last 1,000 epochs", "highest loss of the last 1,000 epochs")
    for column, name in enumerate(names):
        print(f"{name}: {rows[:, column].min():.4f} to {rows[:, column].max():.4f}")


def documented_run(label, optimizer, watched=(0, 1, 10, 100, 1000, 5000, 10000)):
    """One full run on seed 0, with the caches of all 387 parameters read at the watched epochs."""
    caches, scales, gradients, dead = {}, {}, {}, {}
    rows = []

    def watch(epoch, dense1, dense2):
        if epoch not in watched:
            return
        cache = all_parameters(dense1, dense2, "cache")
        gradient = all_parameters(dense1, dense2, "d")
        scale = optimizer.current_learning_rate / (np.sqrt(cache) + optimizer.epsilon)
        caches[epoch], scales[epoch], gradients[epoch] = cache, scale, gradient
        gone = np.max(dense1.output, axis=0) <= 0       # hidden neurons at zero on all 300 samples: dead
        dead[epoch] = np.concatenate([np.tile(gone, 2), gone, np.repeat(gone, 3), np.zeros(3, dtype=bool)])
        rows.append((epoch, optimizer.current_learning_rate, np.median(cache), cache.max(),
                     np.median(scale), np.mean(np.abs(scale * gradient))))

    losses, accuracies = train(optimizer, watch=watch)

    print()
    print(f"== {label}: {type(optimizer).__name__}, learning_rate {optimizer.learning_rate}, "
          f"decay {optimizer.decay}" + (f", rho {optimizer.rho}" if hasattr(optimizer, "rho") else ""))
    print(" epoch   loss     accuracy   rate      median cache   largest cache   median scale   mean |update|")
    for epoch, rate, median, largest, scale, update in rows:
        print(f"{epoch:6d}   {losses[epoch]:.4f}   {accuracies[epoch]:.4f}     {rate:.5f}   {median:.3e}      "
              f"{largest:.3e}       {scale:10.3f}     {update:.2e}")
    smaller = caches[10000] < caches[1000]
    print(f"caches smaller at epoch 10,000 than at epoch 1,000: {int(smaller.sum())} of {smaller.size}")
    theirs, live = dead[10000], ~dead[10000]       # the six parameters of every dead neuron, and the rest
    same = "the same" if np.array_equal(dead[1000], theirs) else "not the same"
    print(f"dead hidden neurons at epoch 10,000: {int(theirs.sum()) // 6} of 64 ({same} as at epoch 1,000); their "
          f"{int(theirs.sum())} parameters: {int(np.sum(gradients[10000][theirs] == 0))} zero gradients, "
          f"{int(smaller[theirs].sum())} caches smaller")
    print(f"the other {int(live.sum())} parameters: {int(smaller[live].sum())} caches smaller; median scale "
          f"{np.median(scales[1000][live]):.3f} at epoch 1,000 and {np.median(scales[10000][live]):.3f} at epoch 10,000")
    late_loss, late_accuracy = losses[-1000:], accuracies[-1000:]
    print(f"final: loss {losses[-1]:.4f}, accuracy {accuracies[-1]:.4f}; best accuracy of the run {accuracies.max():.4f}")
    print(f"last 1,000 epochs: accuracy {late_accuracy.min():.4f} to {late_accuracy.max():.4f}, "
          f"mean {late_accuracy.mean():.4f}; loss {late_loss.min():.4f} to {late_loss.max():.4f}, "
          f"median {np.median(late_loss):.4f}")
    print(f"last 1,000 epochs: the loss rose on {int(np.sum(np.diff(late_loss) > 0))} epochs and was above "
          f"twice its median on {int(np.sum(late_loss > 2 * np.median(late_loss)))}")
    peak = len(losses) - 1000 + int(np.argmax(late_loss))
    print(f"around the highest late loss, epoch {peak}:")
    for epoch in range(peak - 15, peak + 10, 3):
        print(f"{epoch:6d}   {losses[epoch]:.4f}   {accuracies[epoch]:.4f}")


if __name__ == "__main__":
    print("setup: nnfs.init() (float32, seed 0), spiral_data(samples=100, classes=3), Layer_Dense(2, 64), ReLU,")
    print("       Layer_Dense(64, 3), combined softmax and loss, 10,001 full-batch epochs, 387 parameters")
    print("scale = current_learning_rate / (sqrt(cache) + epsilon); |update| = scale * |gradient|")

    documented_run("AdaGrad", Optimizer_Adagrad(learning_rate=1.0, decay=1e-4))
    documented_run("RMSProp", Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999))
