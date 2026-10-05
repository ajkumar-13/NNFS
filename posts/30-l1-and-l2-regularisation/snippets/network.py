"""Post 30: the classes with the two penalties, the training loop, and the shared setup.

Run from the series root:
    python posts/30-l1-and-l2-regularisation/snippets/network.py

Contents: Layer_Dense of post 16 with four regulariser arguments and the extra terms in backward;
Loss of post 19 with a regularization_loss method; Activation_ReLU, Activation_Softmax,
Loss_CategoricalCrossentropy, the combined class and Optimizer_Adam (post 27), all unchanged;
build(), train(), evaluate(), weight_figures() and spread(), which the other scripts import.

Run on its own, the script prints the gradient table of section 2 and the worked layer of
sections 4 and 5, in float64 and without training.
Needs NumPy and the nnfs package (spiral_data and nnfs.init). Takes about a second.
"""
import sys

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0.0, weight_regularizer_l2=0.0,
                 bias_regularizer_l1=0.0, bias_regularizer_l2=0.0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Added in post 30: one strength per penalty and per parameter array.
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs                                    # cached for backward
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)          # shape of weights
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)   # shape of biases

        # Added in post 30: the gradient of each penalty, added to the data gradient.
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

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

    # Added in post 30: the penalty of one layer, zero when no strength is set.
    def regularization_loss(self, layer):
        regularization_loss = 0.0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights ** 2)
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases ** 2)

        return regularization_loss


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
         "X, y = spiral_data(samples=100, classes=3), X_test, y_test = spiral_data(samples=100, classes=3), "
         "Layer_Dense(2, 64, penalties on weights and biases), ReLU, Layer_Dense(64, 3, no penalty), "
         "softmax and cross-entropy, 0.01 * randn weights, Optimizer_Adam(learning_rate=0.05, decay=1e-5), "
         "10,001 full-batch epochs")
SEEDS = (0, 1, 2, 3, 4)
NEAR_ZERO = 1e-3


def build(seed, l1=0.0, l2=0.0, samples=100):
    """Training data, test data, the four objects and the optimiser. nnfs.init() must have been called.

    The draws from the global stream, in order: the training points, the test points, the weights
    of dense1, the weights of dense2. Seed 0 is the seed nnfs.init() itself sets.
    """
    np.random.seed(seed)
    X, y = spiral_data(samples=samples, classes=3)
    X_test, y_test = spiral_data(samples=samples, classes=3)

    dense1 = Layer_Dense(2, 64,
                         weight_regularizer_l1=l1, weight_regularizer_l2=l2,
                         bias_regularizer_l1=l1, bias_regularizer_l2=l2)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.05, decay=1e-5)
    return X, y, X_test, y_test, dense1, activation1, dense2, loss_activation, optimizer


def train(X, y, dense1, activation1, dense2, loss_activation, optimizer, epochs=10001):
    """The full-batch loop of Part VI with the penalty added to the reported loss.

    Returns the data loss, the penalty and the accuracy of the last epoch, measured before that
    epoch's update. Draws no random numbers.
    """
    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        data_loss = loss_activation.forward(dense2.output, y)

        regularization_loss = (loss_activation.loss.regularization_loss(dense1) +
                               loss_activation.loss.regularization_loss(dense2))
        loss = data_loss + regularization_loss

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    return float(data_loss), float(regularization_loss), float(accuracy)


def evaluate(X, y, dense1, activation1, dense2, loss_activation):
    """Data loss (no penalty) and accuracy on (X, y): four forward calls and nothing else."""
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    data_loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    return float(data_loss), float(accuracy)


def weight_figures(layer):
    """Of a layer's weights: sum of |w|, sum of w squared, count below NEAR_ZERO, count equal to 0."""
    w = layer.weights
    return (float(np.sum(np.abs(w))), float(np.sum(w ** 2)),
            int(np.sum(np.abs(w) < NEAR_ZERO)), int(np.sum(w == 0)))


def run(seed, l1=0.0, l2=0.0, samples=100):
    """Train one network and measure it forward-only after the last update."""
    X, y, X_test, y_test, dense1, activation1, dense2, loss_activation, optimizer = build(seed, l1, l2, samples)
    network = (dense1, activation1, dense2, loss_activation)
    train(X, y, *network, optimizer)
    train_loss, train_accuracy = evaluate(X, y, *network)
    dead = np.all(activation1.output == 0, axis=0)      # neurons with no output on any training point
    near = np.abs(dense1.weights) < NEAR_ZERO
    test_loss, test_accuracy = evaluate(X_test, y_test, *network)
    penalty = float(loss_activation.loss.regularization_loss(dense1))
    return dict(train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss,
                test_accuracy=test_accuracy, penalty=penalty, figures=weight_figures(dense1),
                dead=int(dead.sum()), near_dead=int(near[:, dead].sum()), weights_dead=int(near[:, dead].size),
                near_live=int(near[:, ~dead].sum()), weights_live=int(near[:, ~dead].size))


def spread(l1=0.0, l2=0.0, samples=100, seeds=None):
    """One row per seed and the ranges. Losses are data losses; the penalty has its own column.

    The seeds are the argument, or else the integers on the command line, or else SEEDS.
    """
    if seeds is None:
        seeds = tuple(int(argument) for argument in sys.argv[1:]) or SEEDS
    print(f"dense1: weight and bias regularizer l1 = {l1:g}, l2 = {l2:g}; {samples} samples per class; seeds {seeds}")
    print("seed  train acc  test acc  gap in points  train data loss  test data loss  penalty  "
          "sum |w|  sum w^2  |w| < 0.001  w == 0")
    rows = []
    for seed in seeds:
        r = run(seed, l1, l2, samples)
        rows.append(r)
        sum_abs, sum_sq, near, exact = r["figures"]
        print(f"{seed:4d}  {r['train_accuracy']:9.4f}  {r['test_accuracy']:8.4f}  "
              f"{100 * (r['train_accuracy'] - r['test_accuracy']):13.2f}  {r['train_loss']:15.4f}  "
              f"{r['test_loss']:14.4f}  {r['penalty']:7.4f}  {sum_abs:7.2f}  {sum_sq:7.2f}  "
              f"{near:11d}  {exact:6d}", flush=True)
    train_acc = [100 * r["train_accuracy"] for r in rows]
    test_acc = [100 * r["test_accuracy"] for r in rows]
    gaps = [a - b for a, b in zip(train_acc, test_acc)]
    test_losses = [r["test_loss"] for r in rows]
    print(f"train accuracy {min(train_acc):.2f} to {max(train_acc):.2f} percent, "
          f"test accuracy {min(test_acc):.2f} to {max(test_acc):.2f} (mean {np.mean(test_acc):.2f}) percent, "
          f"gap {min(gaps):.2f} to {max(gaps):.2f} (mean {np.mean(gaps):.2f}) points, "
          f"test data loss {min(test_losses):.3f} to {max(test_losses):.3f}")
    data_losses = [r["train_loss"] for r in rows]
    penalties = [r["penalty"] for r in rows]
    print(f"mean over the seeds: training data loss {np.mean(data_losses):.4f}, penalty {np.mean(penalties):.4f}, "
          f"their sum {np.mean(data_losses) + np.mean(penalties):.4f}")
    print(f"dead neurons of dense1 (no output on any training point), of 64, per seed: {[r['dead'] for r in rows]}")
    print(f"weights of dense1 with |w| < 0.001, per seed: in dead neurons {[r['near_dead'] for r in rows]}, "
          f"{sum(r['near_dead'] for r in rows)} of their {sum(r['weights_dead'] for r in rows)} weights; "
          f"in live neurons {[r['near_live'] for r in rows]}, "
          f"{sum(r['near_live'] for r in rows)} of their {sum(r['weights_live'] for r in rows)} weights")
    return rows


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print("== Section 2: the gradient of each penalty at lambda = 0.01")
    lam = 0.01
    print("    w   L1: lam * sign(w)   L2: 2 * lam * w")
    for w in (0.01, 0.1, 0.5, 1.0, 10.0, 100.0):
        print(f"{w:5g}   {lam * np.sign(w):17g}   {2 * lam * w:15g}")

    print()
    print("== Sections 4 and 5: one layer, two inputs, three neurons")
    np.random.seed(0)
    layer = Layer_Dense(2, 3, weight_regularizer_l1=0.01, weight_regularizer_l2=0.01)
    layer.weights = np.array([[0.5, -2.0, 0.0],
                              [1.0, 0.1, -0.2]])
    loss_function = Loss_CategoricalCrossentropy()
    print("sum |w| =", np.sum(np.abs(layer.weights)), "  sum w^2 =", np.sum(layer.weights ** 2))
    print("regularization_loss:", round(float(loss_function.regularization_loss(layer)), 6))

    layer.forward(np.zeros((1, 2)))
    layer.backward(np.zeros((1, 3)))            # a zero data gradient leaves the penalty terms alone
    print("dweights, penalty terms only:")
    print(layer.dweights)

    plain = Layer_Dense(2, 3)
    print("no strengths set: regularization_loss", loss_function.regularization_loss(plain))
