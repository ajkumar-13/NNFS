"""Post 31: Layer_Dropout, the network it joins, the training loop and the shared setup.

Run from the series root:
    python posts/31-dropout/snippets/network.py

Contents: Layer_Dropout, new in this post, and No_Dropout for the baseline; the network classes
of posts 16 and 19 and Optimizer_Adam of post 27, all unchanged, as in post 28's network.py;
build(), train(), evaluate(), masked_accuracy(), run(), spread() and paired(), which the other
scripts import.

Run on its own, the script builds the network of seed 0 with a drop rate of 0.1 and prints one
training-mode and one evaluation-mode forward pass, without training.
Needs NumPy and the nnfs package (spiral_data and nnfs.init). Takes about a second.
"""
import sys

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dropout:

    def __init__(self, rate):
        # 'rate' is the drop probability p. The arithmetic needs the keep
        # probability 1 - p, so that is what the layer stores.
        self.rate = 1 - rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()                         # evaluation: identity, no draw
            return
        # Inverted dropout: a fresh mask of 0 and 1 / (1 - p) on every call.
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask               # the mask of the last training forward


class No_Dropout:
    """Stands where the dropout layer stands in the baseline runs: passes both ways, untouched."""

    def forward(self, inputs, training=True):
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues


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
         "X, y = spiral_data(samples=100, classes=3), Layer_Dense(2, 64), ReLU, Layer_Dropout(rate), "
         "Layer_Dense(64, 3), softmax and cross-entropy, 0.01 * randn weights, "
         "X_test, y_test = spiral_data(samples=100, classes=3), "
         "Optimizer_Adam(learning_rate=0.02, decay=1e-5), 10,001 full-batch epochs")
SEEDS = (0, 1, 2, 3, 4)
MASKED_PASSES = 100
PROBE_RATE = 0.1


def build(seed, rate, samples=100, n_neurons=64, learning_rate=0.02):
    """Training data, the five objects, test data and the optimiser. nnfs.init() must have been called.

    rate=None puts No_Dropout where the dropout layer stands: the network of posts 27 and 28.
    The draws from the global stream, in order: the training points, the weights of dense1, the
    weights of dense2, the test points. This is the order of post 28, whose test points are drawn
    after a loop that draws nothing; here the loop draws masks, so the test points come first.
    """
    np.random.seed(seed)
    X, y = spiral_data(samples=samples, classes=3)

    dense1 = Layer_Dense(2, n_neurons)
    activation1 = Activation_ReLU()
    dropout1 = Layer_Dropout(rate) if rate is not None else No_Dropout()
    dense2 = Layer_Dense(n_neurons, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    X_test, y_test = spiral_data(samples=samples, classes=3)
    optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=1e-5)
    return X, y, X_test, y_test, dense1, activation1, dropout1, dense2, loss_activation, optimizer


def train(X, y, dense1, activation1, dropout1, dense2, loss_activation, optimizer, epochs=10001):
    """The full-batch loop of post 28 with the dropout layer in it. Draws one mask per epoch.

    Returns the loss and the accuracy of the last epoch, both measured through that epoch's
    mask and before its update.
    """
    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dropout1.forward(activation1.output, training=True)
        dense2.forward(dropout1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    return float(loss), float(accuracy)


def evaluate(X, y, dense1, activation1, dropout1, dense2, loss_activation, training=False):
    """Loss and accuracy on (X, y), forward only. The mask is off unless training is set."""
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output, training=training)
    dense2.forward(dropout1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    return float(loss), float(accuracy)


def masked_accuracy(X, y, network, passes=MASKED_PASSES):
    """Accuracies of `passes` forward passes through fresh masks, the parameters held fixed."""
    return [evaluate(X, y, *network, training=True)[1] for _ in range(passes)]


def run(seed, rate, samples=100, n_neurons=64, learning_rate=0.02, epochs=10001):
    """Train one network, then measure it forward-only: mask off, and through fresh masks."""
    X, y, X_test, y_test, dense1, activation1, dropout1, dense2, loss_activation, optimizer = build(
        seed, rate, samples, n_neurons, learning_rate)
    network = (dense1, activation1, dropout1, dense2, loss_activation)
    last_loss, last_accuracy = train(X, y, *network, optimizer, epochs=epochs)
    train_loss, train_accuracy = evaluate(X, y, *network)
    dead = int(np.sum(np.all(activation1.output == 0, axis=0)))    # no output on any training point
    test_loss, test_accuracy = evaluate(X_test, y_test, *network)
    masked = masked_accuracy(X, y, network)
    probe = None
    if rate is None:        # the network trained without dropout, scored through a mask it never saw
        probed = (dense1, activation1, Layer_Dropout(PROBE_RATE), dense2, loss_activation)
        passes = [evaluate(X, y, *probed, training=True) for _ in range(MASKED_PASSES)]
        probe = (float(np.mean([loss for loss, _ in passes])), float(np.mean([accuracy for _, accuracy in passes])))
    return dict(last_accuracy=last_accuracy, train_loss=train_loss, train_accuracy=train_accuracy,
                test_loss=test_loss, test_accuracy=test_accuracy, dead=dead, probe=probe,
                masked_mean=float(np.mean(masked)), masked_min=float(np.min(masked)),
                masked_max=float(np.max(masked)))


def spread(rate, samples=100, n_neurons=64, learning_rate=0.02, seeds=None):
    """One row per seed and the ranges. Accuracies are fractions in the rows, percent in the ranges.

    The seeds are the argument, or else the integers on the command line, or else SEEDS.
    """
    if seeds is None:
        seeds = tuple(int(argument) for argument in sys.argv[1:]) or SEEDS
    layer = "no dropout layer" if rate is None else f"Layer_Dropout({rate:g})"
    print(f"{layer}; {n_neurons} neurons; {samples} samples per class; "
          f"learning rate {learning_rate:g}; seeds {seeds}")
    print(f"seed  train, mask on (mean of {MASKED_PASSES}: min to max)  train, mask off  test, mask off  "
          "off - test  on - test  test loss  dead")
    rows = []
    for seed in seeds:
        r = run(seed, rate, samples, n_neurons, learning_rate)
        rows.append(r)
        print(f"{seed:4d}  {r['masked_mean']:17.4f} ({r['masked_min']:.4f} to {r['masked_max']:.4f})  "
              f"{r['train_accuracy']:15.4f}  {r['test_accuracy']:14.4f}  "
              f"{100 * (r['train_accuracy'] - r['test_accuracy']):10.2f}  "
              f"{100 * (r['masked_mean'] - r['test_accuracy']):9.2f}  {r['test_loss']:9.4f}  {r['dead']:4d}",
              flush=True)
    on = [100 * r["masked_mean"] for r in rows]
    off = [100 * r["train_accuracy"] for r in rows]
    test = [100 * r["test_accuracy"] for r in rows]
    gaps = [a - b for a, b in zip(off, test)]
    on_gaps = [a - b for a, b in zip(on, test)]
    losses = [r["test_loss"] for r in rows]
    print(f"train accuracy, mask on {min(on):.2f} to {max(on):.2f}, mask off {min(off):.2f} to {max(off):.2f} percent; "
          f"test accuracy {min(test):.2f} to {max(test):.2f} (mean {np.mean(test):.2f}) percent")
    print(f"mask off minus test {min(gaps):.2f} to {max(gaps):.2f} (mean {np.mean(gaps):.2f}) points; "
          f"mask on minus test {min(on_gaps):.2f} to {max(on_gaps):.2f} (mean {np.mean(on_gaps):.2f}) points; "
          f"test loss {min(losses):.3f} to {max(losses):.3f}")
    print(f"runs whose test accuracy is above the mask-on training accuracy: "
          f"{sum(g < 0 for g in on_gaps)} of {len(rows)}; above the mask-off training accuracy: "
          f"{sum(g < 0 for g in gaps)} of {len(rows)}")
    if rate is None:
        probe_accuracy = [100 * r["probe"][1] for r in rows]
        fall = [a - b for a, b in zip(off, probe_accuracy)]
        print(f"the same weights through a Layer_Dropout({PROBE_RATE:g}) they were not trained with (mean of "
              f"{MASKED_PASSES} passes): training accuracy {min(probe_accuracy):.2f} to {max(probe_accuracy):.2f} percent, "
              f"{min(fall):.2f} to {max(fall):.2f} points below mask off; training loss "
              f"{min(r['probe'][0] for r in rows):.2f} to {max(r['probe'][0] for r in rows):.2f} "
              f"against {min(r['train_loss'] for r in rows):.2f} to {max(r['train_loss'] for r in rows):.2f}")
    return rows


def paired(base, rows, label):
    """Seed-by-seed comparison of two results of spread() over the same seeds: rows against base, in points."""
    def tally(differences):
        lower = sum(d < -1e-9 for d in differences)
        higher = sum(d > 1e-9 for d in differences)
        return (f"lower on {lower}, equal on {len(differences) - lower - higher}, higher on {higher} of "
                f"{len(differences)} seeds, difference {min(differences):+.2f} to {max(differences):+.2f} points")
    test = [100 * (r["test_accuracy"] - b["test_accuracy"]) for b, r in zip(base, rows)]
    off = [100 * (r["train_accuracy"] - b["train_accuracy"]) for b, r in zip(base, rows)]
    gap = [100 * ((r["train_accuracy"] - r["test_accuracy"]) - (b["train_accuracy"] - b["test_accuracy"]))
           for b, r in zip(base, rows)]
    mask = [100 * (r["train_accuracy"] - r["masked_mean"]) for r in rows]
    print(f"{label} against no dropout layer, seed by seed:")
    print(f"  test accuracy {tally(test)}")
    print(f"  training accuracy, mask off, {tally(off)}")
    print(f"  gap, mask off minus test, narrower on {sum(g < -1e-9 for g in gap)} of {len(rows)} seeds; "
          f"test loss lower on {sum(r['test_loss'] < b['test_loss'] for b, r in zip(base, rows))} of {len(rows)}")
    print(f"  training accuracy through the mask below mask off on {sum(m > 1e-9 for m in mask)} of {len(rows)} seeds, "
          f"mask off minus mask on {min(mask):.2f} to {max(mask):.2f} points")
    print(f"  test accuracy above the mask-on training accuracy on "
          f"{sum(r['test_accuracy'] > r['masked_mean'] for r in rows)} of {len(rows)} seeds, above the mask-off "
          f"training accuracy on {sum(r['test_accuracy'] > r['train_accuracy'] for r in rows)}")
    print(f"  dead neurons {min(b['dead'] for b in base)} to {max(b['dead'] for b in base)} without the layer, "
          f"{min(r['dead'] for r in rows)} to {max(r['dead'] for r in rows)} with it")


if __name__ == "__main__":
    nnfs.init()
    print(SETUP)
    X, y, X_test, y_test, dense1, activation1, dropout1, dense2, loss_activation, optimizer = build(seed=0, rate=0.1)
    network = (dense1, activation1, dropout1, dense2, loss_activation)

    loss, accuracy = evaluate(X, y, *network, training=True)
    mask = dropout1.binary_mask
    print(f"training mode:   output {dropout1.output.shape}, mask values {np.unique(mask)}, "
          f"entries dropped {int(np.sum(mask == 0))} of {mask.size} ({np.mean(mask == 0):.4f}), loss {loss:.4f}")
    print(f"rows of the mask that differ from row 0: {int(np.sum(np.any(mask != mask[0], axis=1)))} of {len(mask) - 1}")

    loss, accuracy = evaluate(X, y, *network)
    print(f"evaluation mode: output equals input {np.array_equal(dropout1.output, activation1.output)}, "
          f"loss {loss:.4f} (ln 3 = {np.log(3):.4f})")
