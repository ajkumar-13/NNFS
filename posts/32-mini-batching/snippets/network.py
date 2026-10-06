"""Post 32: the network of posts 16, 19 and 27, the mini-batch training loop, and the shared setup.

Run from the series root:
    python posts/32-mini-batching/snippets/network.py

Contents: Layer_Dense and Activation_ReLU (post 16), Activation_Softmax, Loss,
Loss_CategoricalCrossentropy and the combined class (post 19) and Optimizer_Adam (post 27), all
unchanged; build(), which draws the training data, the weights and the test data; train(), the
two-loop mini-batch loop this post adds; evaluate(), the forward-only pass of post 28; and
spread() and spread_checkpoints(), which the batch-size scripts call.

Run on its own, the script trains the documented run of section 4 (seed 0, batches of 32,
1,000 epochs), prints its log, and compares the two ways of averaging batch losses.
Needs NumPy and the nnfs package (spiral_data and nnfs.init). Takes about 3 seconds.
"""
import sys

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


class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0.0,
                 epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.epsilon               = epsilon
        self.beta_1                = beta_1
        self.beta_2                = beta_2
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        # Lazy buffer creation: one momentum and one cache per parameter array.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache     = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)
            layer.bias_cache       = np.zeros_like(layer.biases)

        # 1) First moment: moving average of the gradient.
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums   = self.beta_1 * layer.bias_momentums + \
                                 (1 - self.beta_1) * layer.dbiases

        # 2) Bias correction of the first moment. The step counter is
        #    incremented in post_update_params, after this call, hence the + 1.
        t = self.iterations + 1
        weight_m_hat = layer.weight_momentums / (1 - self.beta_1 ** t)
        bias_m_hat   = layer.bias_momentums   / (1 - self.beta_1 ** t)

        # 3) Second moment: moving average of the squared gradient.
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache   = self.beta_2 * layer.bias_cache + \
                             (1 - self.beta_2) * layer.dbiases ** 2

        # 4) Bias correction of the second moment.
        weight_v_hat = layer.weight_cache / (1 - self.beta_2 ** t)
        bias_v_hat   = layer.bias_cache   / (1 - self.beta_2 ** t)

        # 5) Parameter update.
        layer.weights -= self.current_learning_rate * weight_m_hat / \
                         (np.sqrt(weight_v_hat) + self.epsilon)
        layer.biases  -= self.current_learning_rate * bias_m_hat / \
                         (np.sqrt(bias_v_hat)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


SETUP = ("setup: nnfs.init() once (seed 0, float32, float32 np.dot), then per run np.random.seed(s), "
         "X, y = spiral_data(samples=100, classes=3), Layer_Dense(2, 64), ReLU, Layer_Dense(64, 3), "
         "softmax and cross-entropy, 0.01 * randn weights, X_test, y_test = spiral_data(samples=100, classes=3), "
         "Optimizer_Adam(learning_rate=0.02, decay=1e-5); the shuffles are drawn after the test data")
SEEDS = (0, 1, 2, 3, 4)


def build(seed, learning_rate=0.02, decay=1e-5):
    """Training data, the four objects, the test data and the optimiser. nnfs.init() must have been called.

    The draws from the global stream, in order: the training points, the weights of dense1, the
    weights of dense2, the test points. That is the order of post 28, whose full-batch loop draws
    nothing, so a seed gives the data, the weights and the test set of post 28.
    """
    np.random.seed(seed)                          # seed 0 is the seed nnfs.init() itself sets
    X, y = spiral_data(samples=100, classes=3)    # (300, 2), (300,), sorted by class

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    X_test, y_test = spiral_data(samples=100, classes=3)
    optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=decay)
    return X, y, X_test, y_test, dense1, activation1, dense2, loss_activation, optimizer


def train(X, y, dense1, activation1, dense2, loss_activation, optimizer,
          epochs, batch_size, order=np.random.permutation, log=None):
    """Mini-batch training: a new order every epoch, one forward, backward and update per batch.

    order(n) returns the row order of an epoch: np.random.permutation by default, np.arange for
    no shuffling. log(epoch, loss, accuracy, last_batch_loss), if given, is called at the end of
    every epoch with the two epoch figures and the loss of the epoch's last batch.
    Returns the lists of the epoch losses and epoch accuracies.
    """
    n_samples = len(X)
    losses, accuracies = [], []

    for epoch in range(epochs):
        # 1. A new order of the rows for every epoch.
        idx = order(n_samples)
        X_shuf = X[idx]
        y_shuf = y[idx]

        epoch_loss = 0.0
        n_correct = 0
        n_batches = 0

        # 2. One pass over the data, batch_size rows at a time; the last batch may be shorter.
        for start in range(0, n_samples, batch_size):
            X_batch = X_shuf[start:start + batch_size]
            y_batch = y_shuf[start:start + batch_size]

            # 3. Forward, on this batch only.
            dense1.forward(X_batch)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            loss = loss_activation.forward(dense2.output, y_batch)

            predictions = np.argmax(loss_activation.output, axis=1)
            n_correct += int(np.sum(predictions == y_batch))

            # 4. Backward: the gradient of this batch's mean loss.
            loss_activation.backward(loss_activation.output, y_batch)
            dense2.backward(loss_activation.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            # 5. One update per batch.
            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.post_update_params()

            epoch_loss += float(loss)
            n_batches += 1

        losses.append(epoch_loss / n_batches)       # mean of the batch losses, every batch counted once
        accuracies.append(n_correct / n_samples)    # share of the rows predicted correctly when they were seen
        if log is not None:
            log(epoch, losses[-1], accuracies[-1], float(loss))

    return losses, accuracies


def evaluate(X, y, dense1, activation1, dense2, loss_activation):
    """Loss and accuracy on (X, y): four forward calls, no backward call, no optimiser."""
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    return float(loss), float(accuracy)


def run_checkpoints(seed, batch_size, checkpoints, order=np.random.permutation, learning_rate=0.02, decay=1e-5):
    """Train one network and measure it forward-only each time the epoch count reaches a checkpoint.

    Training goes on from one checkpoint to the next, so the figures at a checkpoint are those of
    a run that stops there. Returns one dict per checkpoint.
    """
    X, y, X_test, y_test, dense1, activation1, dense2, loss_activation, optimizer = build(seed, learning_rate, decay)
    network = (dense1, activation1, dense2, loss_activation)
    measured, done = [], 0
    for epochs in checkpoints:
        losses, accuracies = train(X, y, *network, optimizer, epochs - done, batch_size, order)
        done = epochs
        train_loss, train_accuracy = evaluate(X, y, *network)
        dead = int(np.sum(activation1.output.max(axis=0) <= 0))    # hidden neurons at zero on every training row
        test_loss, test_accuracy = evaluate(X_test, y_test, *network)
        measured.append(dict(updates=optimizer.iterations, rate=optimizer.current_learning_rate, dead=dead,
                             epoch_loss=losses[-1], epoch_accuracy=accuracies[-1],
                             train_loss=train_loss, train_accuracy=train_accuracy,
                             test_loss=test_loss, test_accuracy=test_accuracy))
    return measured


def run(seed, epochs, batch_size, order=np.random.permutation, learning_rate=0.02, decay=1e-5):
    """Train one network and measure it forward-only after the last update."""
    return run_checkpoints(seed, batch_size, (epochs,), order, learning_rate, decay)[0]


HEADER = "seed  updates  last rate  train loss  train acc  test loss  test acc  dead"


def print_row(seed, r):
    print(f"{seed:4d}  {r['updates']:7d}  {r['rate']:9.6f}  {r['train_loss']:10.4f}  {r['train_accuracy']:9.4f}  "
          f"{r['test_loss']:9.4f}  {r['test_accuracy']:8.4f}  {r['dead']:4d}", flush=True)


def print_ranges(rows):
    train_loss = [r["train_loss"] for r in rows]
    train_acc = [100 * r["train_accuracy"] for r in rows]
    test_acc = [100 * r["test_accuracy"] for r in rows]
    print(f"train loss {min(train_loss):.4f} to {max(train_loss):.4f} (mean {np.mean(train_loss):.4f}), "
          f"train accuracy {min(train_acc):.2f} to {max(train_acc):.2f} (mean {np.mean(train_acc):.2f}) percent, "
          f"test accuracy {min(test_acc):.2f} to {max(test_acc):.2f} (mean {np.mean(test_acc):.2f}) percent")
    dead = [r["dead"] for r in rows]
    print(f"dead hidden neurons (zero on every training row): {min(dead)} to {max(dead)} of 64")


def command_line_seeds():
    """The integers on the command line, or else SEEDS."""
    return tuple(int(argument) for argument in sys.argv[1:]) or SEEDS


def spread(epochs, batch_size, order=np.random.permutation, learning_rate=0.02, decay=1e-5, seeds=None):
    """One row per seed and the ranges. Every figure but the last rate is forward-only, after the last update.

    The last column counts the dead neurons of the hidden layer: those whose output is zero on
    every one of the training rows (post 22).

    The seeds are the argument, or else the integers on the command line, or else SEEDS.
    """
    if seeds is None:
        seeds = command_line_seeds()
    print(f"batch size {batch_size}, {epochs} epochs, learning_rate {learning_rate:g}, decay {decay:g}, "
          f"order {order.__name__}, seeds {seeds}")
    print(HEADER)
    rows = []
    for seed in seeds:
        r = run(seed, epochs, batch_size, order, learning_rate, decay)
        rows.append(r)
        print_row(seed, r)
    print_ranges(rows)
    return rows


def spread_checkpoints(batch_size, checkpoints, seeds=None):
    """The tables of spread() for several epoch counts, from one training run per seed.

    Returns one list of rows per checkpoint.
    """
    if seeds is None:
        seeds = command_line_seeds()
    per_seed = [run_checkpoints(seed, batch_size, checkpoints) for seed in seeds]
    tables = []
    for position, epochs in enumerate(checkpoints):
        rows = [measured[position] for measured in per_seed]
        tables.append(rows)
        print()
        print(f"== Batch size {batch_size}, {epochs:,} epochs, seeds {seeds}")
        print(HEADER)
        for seed, r in zip(seeds, rows):
            print_row(seed, r)
        print_ranges(rows)
    return tables


if __name__ == "__main__":
    nnfs.init()
    print(SETUP)

    BATCH_SIZE = 32
    EPOCHS = 1000
    X, y, X_test, y_test, dense1, activation1, dense2, loss_activation, optimizer = build(seed=0)
    network = (dense1, activation1, dense2, loss_activation)
    n_samples = len(X)
    print(f"seed 0, {n_samples} rows, BATCH_SIZE = {BATCH_SIZE}, EPOCHS = {EPOCHS}: "
          f"{int(np.ceil(n_samples / BATCH_SIZE))} batches an epoch, "
          f"the last of {n_samples - BATCH_SIZE * (n_samples // BATCH_SIZE)} rows")

    last_batch_losses = []

    def log(epoch, loss, accuracy, last_batch_loss):
        last_batch_losses.append(last_batch_loss)
        if epoch in (0, 1, 9, 99, 499, 998, 999):
            print(f"epoch {epoch:3d} | loss {loss:.4f} | acc {accuracy:.4f} | "
                  f"updates {optimizer.iterations:5d} | lr {optimizer.current_learning_rate:.6f}")

    losses, accuracies = train(X, y, *network, optimizer, EPOCHS, BATCH_SIZE, log=log)
    train_loss, train_accuracy = evaluate(X, y, *network)
    test_loss, test_accuracy = evaluate(X_test, y_test, *network)
    print(f"forward-only after the last update: training loss {train_loss:.4f}, accuracy {train_accuracy:.4f}; "
          f"test loss {test_loss:.4f}, accuracy {test_accuracy:.4f}")
    print(f"epoch loss over the last 100 epochs: {min(losses[-100:]):.4f} to {max(losses[-100:]):.4f}; "
          f"epoch accuracy: {min(accuracies[-100:]):.4f} to {max(accuracies[-100:]):.4f}")
    print(f"loss of the last batch (12 rows) over the same 100 epochs: "
          f"{min(last_batch_losses[-100:]):.4f} to {max(last_batch_losses[-100:]):.4f}")

    print()
    print("== The two means of the batch losses, on the trained network, rows in their stored order, no update")
    batch_losses, batch_sizes = [], []
    for start in range(0, n_samples, BATCH_SIZE):
        loss, _ = evaluate(X[start:start + BATCH_SIZE], y[start:start + BATCH_SIZE], *network)
        batch_losses.append(loss)
        batch_sizes.append(len(X[start:start + BATCH_SIZE]))
    print("batch sizes:", batch_sizes)
    print("batch losses:", " ".join(f"{loss:.4f}" for loss in batch_losses))
    print(f"mean of the batch losses {np.mean(batch_losses):.4f}; "
          f"weighted by batch size {np.average(batch_losses, weights=batch_sizes):.4f}; "
          f"loss of all {n_samples} rows at once {train_loss:.4f}")
