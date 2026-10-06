"""Post 34, section 7: the two-moons network with a sigmoid head and with a two-output softmax head.

Run from the series root:
    python posts/34-sigmoid-and-binary-cross-entropy/snippets/network.py

Contents: Layer_Dense of post 30 with the init argument of post 33; Activation_ReLU (post 16);
Activation_Softmax, Loss, Loss_CategoricalCrossentropy and the combined softmax class (posts 19
and 30); Optimizer_Adam (post 27), all unchanged; the sigmoid classes of this post, imported from
binary_classes.py; make_moons(), train_test_split(), build(), train(), evaluate() and spread(),
which seeds.py imports.

Run on its own, the script trains both heads on seed 0 and prints the log of each.
Float64 weights, no nnfs.init(). Needs only NumPy. Takes 3 to 10 seconds.
"""
import numpy as np

from binary_classes import Activation_Sigmoid_Loss_BinaryCrossentropy


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, init="he",
                 weight_regularizer_l1=0.0, weight_regularizer_l2=0.0,
                 bias_regularizer_l1=0.0, bias_regularizer_l2=0.0):
        # The init argument of post 33: the standard deviation of the initial weights.
        if init == "he":
            scale = np.sqrt(2.0 / n_inputs)
        elif init == "xavier" or init == "glorot":
            scale = np.sqrt(2.0 / (n_inputs + n_neurons))
        elif init == "small":
            scale = 0.01
        else:
            raise ValueError(f"unknown init: {init!r}")

        self.weights = scale * np.random.randn(n_inputs, n_neurons)
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


SETUP = ("setup: no nnfs.init(), float64 weights; per seed s: make_moons(1000, noise 0.1, seed s) from "
         "np.random.default_rng(s), split 800 to 200, np.random.seed(s), Layer_Dense(2, 16), ReLU, "
         "Layer_Dense(16, 16), ReLU, Layer_Dense(16, outputs), all init='he', L2 of 1e-4 on the weights "
         "of the two hidden layers, Optimizer_Adam(learning_rate=0.01), 2,000 full-batch epochs")
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def make_moons(n_samples=1000, noise=0.10, seed=0):
    """Two interleaved half-circles with Gaussian noise, shuffled; the generator of nn-p02.

    Returns X of shape (n_samples, 2), float32, and y of shape (n_samples,), int64, values 0 and 1.
    """
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = np.linspace(0, np.pi, n_outer)
    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    theta_inner = np.linspace(0, np.pi, n_inner)
    inner = np.column_stack([1.0 - np.cos(theta_inner), 0.5 - np.sin(theta_inner)])

    X = np.vstack([outer, inner]).astype(np.float32)
    X += rng.normal(0.0, noise, X.shape).astype(np.float32)
    y = np.hstack([np.zeros(n_outer, dtype=np.int64), np.ones(n_inner, dtype=np.int64)])

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def train_test_split(X, y, test_frac=0.2, seed=0):
    """A seeded random split; returns X_train, y_train, X_test, y_test."""
    n_test = round(len(X) * test_frac)
    idx = np.random.default_rng(seed).permutation(len(X))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def build(seed, head="sigmoid"):
    """The data, the layers, the head and the optimiser of one run.

    head="sigmoid": one output neuron and the combined class of this post.
    head="softmax": two output neurons and the combined class of post 19.
    The weights are drawn in the order dense1, dense2, dense3, so the two heads share the
    weights of dense1 and dense2 on a seed and differ in dense3.
    """
    X, y = make_moons(seed=seed)
    X, y, X_test, y_test = train_test_split(X, y, seed=seed)

    np.random.seed(seed)
    dense1 = Layer_Dense(2, 16, init="he", weight_regularizer_l2=1e-4)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(16, 16, init="he", weight_regularizer_l2=1e-4)
    activation2 = Activation_ReLU()
    if head == "sigmoid":
        dense3 = Layer_Dense(16, 1, init="he")
        loss_activation = Activation_Sigmoid_Loss_BinaryCrossentropy()
    else:
        dense3 = Layer_Dense(16, 2, init="he")
        loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.01)
    layers = (dense1, activation1, dense2, activation2, dense3)
    return X, y, X_test, y_test, layers, loss_activation, optimizer


def predict(loss_activation):
    """Class 0 or 1 for each row of the output the head cached, shape (N,)."""
    if loss_activation.output.shape[1] == 1:
        return (loss_activation.output >= 0.5).astype(np.int64).ravel()
    return np.argmax(loss_activation.output, axis=1)


def evaluate(X, y, layers, loss_activation):
    """Data loss (no penalty) and the number of correct predictions on (X, y), forward only."""
    dense1, activation1, dense2, activation2, dense3 = layers
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    data_loss = loss_activation.forward(dense3.output, y)
    return float(data_loss), int(np.sum(predict(loss_activation) == y))


def train(X, y, layers, loss_activation, optimizer, epochs=2000, log_every=0):
    """The full-batch loop of Part VI on three dense layers. Draws no random numbers."""
    dense1, activation1, dense2, activation2, dense3 = layers
    for epoch in range(1, epochs + 1):
        data_loss, correct = evaluate(X, y, layers, loss_activation)

        loss_activation.backward(loss_activation.output, y)
        dense3.backward(loss_activation.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.post_update_params()

        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(f"epoch {epoch:5d} | data loss {data_loss:.4f} | train correct {correct}/{len(X)}")


def run(seed, head, log_every=0):
    X, y, X_test, y_test, layers, loss_activation, optimizer = build(seed, head)
    train(X, y, layers, loss_activation, optimizer, log_every=log_every)
    train_loss, train_correct = evaluate(X, y, layers, loss_activation)
    test_loss, test_correct = evaluate(X_test, y_test, layers, loss_activation)
    parameters = sum(layer.weights.size + layer.biases.size for layer in layers[::2])
    return parameters, train_loss, train_correct, len(X), test_loss, test_correct, len(X_test)


def spread(head, seeds=SEEDS):
    """One row per seed, measured forward-only after the last update; then the range."""
    print(f"head: {head}")
    print("seed   parameters   train correct   test correct   train data loss   test data loss")
    rows = []
    for seed in seeds:
        row = run(seed, head)
        rows.append(row)
        print(f"{seed:>4}   {row[0]:>10}   {row[2]:>9}/{row[3]}   {row[5]:>8}/{row[6]}   {row[1]:>15.4f}   {row[4]:>14.4f}")
    test = [row[5] for row in rows]
    print(f"test points correct of {rows[0][6]}, over {len(rows)} seeds: lowest {min(test)}, highest {max(test)}, "
          f"mean {np.mean(test):.1f}, seeds with all correct {sum(t == rows[0][6] for t in test)}")
    print(f"test data loss: {min(row[4] for row in rows):.4f} to {max(row[4] for row in rows):.4f}")
    return rows


if __name__ == "__main__":
    print(SETUP)
    for head in ("sigmoid", "softmax"):
        print()
        print(f"== Seed 0, head: {head}")
        row = run(0, head, log_every=500)
        print(f"parameters {row[0]} | train {row[2]}/{row[3]}, data loss {row[1]:.4f} | "
              f"test {row[5]}/{row[6]}, data loss {row[4]:.4f}")
