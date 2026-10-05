"""Post 25, sections 1 to 8: Optimizer_Adagrad, three small examples, and the documented spiral run.

Run from the series root:
    python posts/25-adagrad/snippets/adagrad.py

Contents: the classes of posts 16 and 19 and the Optimizer_SGD of post 24, unchanged; the new
Optimizer_Adagrad; the training loop as one function; a two-parameter loss under gradient descent
and under AdaGrad; a constant gradient followed for 10,000 steps; a frequent and a rare
parameter; and the documented run, with the cache and the effective rate printed along the way.

Needs NumPy and the nnfs package. Every run is seeded.
"""
from types import SimpleNamespace

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()                                     # seed 0, float32 arrays, a patched np.dot


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

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.momentum              = momentum
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:
            # The velocity buffers are created on the first call, full of zeros.
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)

            # New velocity = beta * old velocity - current learning rate * gradient.
            weight_updates = self.momentum * layer.weight_momentums \
                           - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums \
                         - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # No momentum: the update of post 23.
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbiases

        # The step is the velocity, added.
        layer.weights += weight_updates
        layer.biases  += bias_updates

    def post_update_params(self):
        self.iterations += 1


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


def train(optimizer, seed=0, epochs=10001, watch=None):
    """Train the shared network of Part VI; return the last loss, the accuracy and the dead neurons."""
    np.random.seed(seed)                        # the data and the weights both follow from the seed
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    for epoch in range(epochs):
        # Forward pass.
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)
        dead = int(np.sum(np.max(activation1.output, axis=0) == 0))   # hidden neurons at 0 on every sample

        # Backward pass.
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update.
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

        if watch is not None:
            watch(epoch, loss, accuracy, dead, (dense1, dense2), optimizer)

    return float(loss), float(accuracy), dead


SETUP = ("nnfs.init() (float32, patched np.dot), np.random.seed(s), spiral_data(samples=100, classes=3), "
         "Layer_Dense(2, 64), ReLU, Layer_Dense(64, 3), softmax and cross-entropy, 10,001 full-batch epochs")


def seed_table(label, make_optimizer, seeds):
    """Run one optimiser setting once per seed and print a row per run and the range."""
    print("setup:", SETUP)
    print(label)
    print("seed   loss    accuracy   dead neurons")
    rows = []
    for seed in seeds:
        rows.append(train(make_optimizer(), seed=seed))
        print(f"{seed:4d}   {rows[-1][0]:.4f}  {rows[-1][1]:.4f}     {rows[-1][2]:2d} of 64")
    losses, accuracies, dead = zip(*rows)
    print(f"range  loss {min(losses):.4f} to {max(losses):.4f}   accuracy {min(accuracies):.4f} to "
          f"{max(accuracies):.4f}   dead neurons {min(dead)} to {max(dead)}")
    return rows


def toy_layer(weights):
    """A stand-in for a layer: the four attributes an optimiser reads and writes, in float64."""
    return SimpleNamespace(weights=np.array([weights], dtype=np.float64), biases=np.array([[0.0]]),
                           dweights=None, dbiases=np.array([[0.0]]))


def run_toy(optimizer, steps, gradient_of, start):
    """Apply an optimiser for some steps to a toy layer whose gradient is gradient_of(weights)."""
    layer = toy_layer(start)
    for _ in range(steps):
        layer.dweights = gradient_of(layer.weights)
        optimizer.pre_update_params()
        optimizer.update_params(layer)
        optimizer.post_update_params()
    return layer


if __name__ == "__main__":
    print("== Section 1: one learning rate, two gradient scales")
    scales = np.array([[2 / 100, 2.0]])         # L = w1^2 / 100 + w2^2, so dL/dw = scales * w

    def bowl(weights):
        return scales * weights

    print("gradient at (1, 1):", bowl(np.array([[1.0, 1.0]]))[0])
    for alpha, steps in ((0.1, 100), (0.5, 100), (10.0, 5)):
        first = run_toy(Optimizer_SGD(learning_rate=alpha), 1, bowl, [1.0, 1.0]).weights[0]
        last = run_toy(Optimizer_SGD(learning_rate=alpha), steps, bowl, [1.0, 1.0]).weights[0]
        print(f"gradient descent, alpha = {alpha:4.1f}: first steps {1 - first[0]:.4g} and {1 - first[1]:.4g}; "
              f"after {steps:3d} steps w1 = {last[0]:.4f}, w2 = {last[1]:.4g}")
    for steps in (1, 10, 100):
        layer = run_toy(Optimizer_Adagrad(learning_rate=0.1), steps, bowl, [1.0, 1.0])
        print(f"AdaGrad, alpha = 0.1: after {steps:3d} steps w1 = {layer.weights[0, 0]:.6f}, "
              f"w2 = {layer.weights[0, 1]:.6f}, caches {layer.weight_cache[0, 0]:.4g} and "
              f"{layer.weight_cache[0, 1]:.4g}")

    print()
    print("== Section 3: a constant gradient of 0.5, alpha = 1")
    print("      t        G     sqrt(G)   alpha/sqrt(G)   distance moved")
    for steps in (1, 2, 10, 100, 1000, 10000):
        layer = run_toy(Optimizer_Adagrad(learning_rate=1.0), steps, lambda w: np.array([[0.5]]), [0.0])
        cache = layer.weight_cache[0, 0]
        print(f"{steps:7d}  {cache:7.2f}  {np.sqrt(cache):8.3f}  {1 / np.sqrt(cache):11.4f}  "
              f"{abs(layer.weights[0, 0]):13.2f}")
    for size in (2.0, 0.2):
        cache = 1000 * size ** 2
        print(f"|g| = {size}: after 1,000 steps G = {cache:g}, sqrt(G) = {np.sqrt(cache):.3g}, "
              f"alpha/sqrt(G) = {1 / np.sqrt(cache):.3g}")

    print()
    print("== Section 8: a parameter that sees a gradient every step and one that sees it every 100th")
    count = [0]

    def sparse(weights):
        count[0] += 1
        return np.array([[1.0, 1.0 if count[0] % 100 == 0 else 0.0]])

    layer = run_toy(Optimizer_Adagrad(learning_rate=1.0), 1000, sparse, [0.0, 0.0])
    rates = 1 / (np.sqrt(layer.weight_cache[0]) + 1e-7)
    print(f"after 1,000 steps: caches {layer.weight_cache[0, 0]:g} and {layer.weight_cache[0, 1]:g}, "
          f"effective rates {rates[0]:.4f} and {rates[1]:.4f}, ratio {rates[1] / rates[0]:.1f}")

    print()
    print("== Section 7: the documented run, Optimizer_Adagrad(learning_rate=1.0, decay=1e-4), seed 0")
    print("setup:", SETUP)
    print("G and rate are taken over all 387 parameters; rate = current_learning_rate / (sqrt(G) + epsilon)")
    previous = {}
    smallest_change = [np.inf]
    split = []

    def watch(epoch, loss, accuracy, dead, layers, optimizer):
        cache = np.concatenate([c.ravel() for layer in layers for c in (layer.weight_cache, layer.bias_cache)])
        gradient = np.concatenate([g.ravel() for layer in layers for g in (layer.dweights, layer.dbiases)])
        if "cache" in previous:
            smallest_change[0] = min(smallest_change[0], float(np.min(cache - previous["cache"])))
        previous["cache"] = cache
        rate = optimizer.current_learning_rate / (np.sqrt(cache) + optimizer.epsilon)
        step = rate * np.abs(gradient)
        if epoch == 0:
            size = np.abs(gradient)
            print(f"first update: |gradient| from {size.min():.1e} to {size.max():.1e} (median {np.median(size):.1e}); "
                  f"|step| from {step.min():.3f} to {step.max():.3f} (median {np.median(step):.3f})")
            print(" epoch    loss   acc    dead   lr      G median   G max     rate median  rate min  below 1   mean |step|")
        if epoch in (0, 1, 2, 10, 100, 1000, 2000, 5000, 9000, 10000):
            print(f"{epoch:6d}  {loss:.4f}  {accuracy:.3f}  {dead:3d}   {optimizer.current_learning_rate:.4f}  "
                  f"{np.median(cache):.2e}  {cache.max():.2e}  {np.median(rate):10.2f}  {rate.min():8.3f}  "
                  f"{int(np.sum(rate < 1)):6d}   {step.mean():.2e}")
        if epoch in (100, 10000):
            gone = np.max(layers[0].output, axis=0) <= 0        # the dead neurons of this epoch's forward pass
            theirs = np.concatenate([np.tile(gone, 2), gone, np.repeat(gone, 3), np.zeros(3, dtype=bool)])
            split.append(f"epoch {epoch:5d}: {int(theirs.sum())} parameters of dead neurons, rate median "
                         f"{np.median(rate[theirs]):.2f}, {int(np.sum(rate[theirs] > 1))} above 1; "
                         f"the other {int(np.sum(~theirs))}, rate median {np.median(rate[~theirs]):.2f}, "
                         f"{int(np.sum(rate[~theirs] > 1))} above 1")

    loss, accuracy, dead = train(Optimizer_Adagrad(learning_rate=1.0, decay=1e-4), seed=0, watch=watch)
    print(f"final: loss {loss:.4f}, accuracy {accuracy:.4f}, dead neurons {dead} of 64")
    print(f"smallest change of any cache entry between two consecutive epochs: {smallest_change[0]:.1e}")
    for line in split:
        print(line)
