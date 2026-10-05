"""Post 28: the network of post 27 and its training loop, shared by the other scripts of this post.

Run from the series root:
    python posts/28-generalization-and-testing/snippets/network.py

Contents: the network classes of posts 16 and 19 and Optimizer_Adam of post 27, all unchanged;
build(), which draws the training data and creates the four objects and the optimiser; train(),
the full-batch loop of Part VI; evaluate(), the forward-only pass this post adds; and
boundary_length(), a count taken on a grid of points.

Run on its own, the script builds the network and prints its starting loss, without training.
Needs NumPy and the nnfs package (spiral_data and nnfs.init). Takes about a second.
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


SETUP = ("setup: nnfs.init() (seed 0, float32, float32 np.dot), spiral_data(samples=100, classes=3), "
         "Layer_Dense(2, 64), ReLU, Layer_Dense(64, 3), softmax and cross-entropy, 0.01 * randn weights, "
         "Optimizer_Adam(learning_rate=0.02, decay=1e-5), 10,001 full-batch epochs")


def build(seed, n_neurons=64):
    """Training data, the four objects and the optimiser. nnfs.init() must have been called."""
    np.random.seed(seed)                          # seed 0 is the seed nnfs.init() itself sets
    X, y = spiral_data(samples=100, classes=3)    # (300, 2), (300,): 300 normal draws

    dense1 = Layer_Dense(2, n_neurons)            # 2 * n_neurons normal draws
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(n_neurons, 3)            # 3 * n_neurons normal draws
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)
    return X, y, dense1, activation1, dense2, loss_activation, optimizer


def train(X, y, dense1, activation1, dense2, loss_activation, optimizer,
          epochs=10001, on_check=None, check_every=100):
    """The full-batch loop of Part VI. Draws no random numbers.

    on_check(epoch), if given, is called before the forward pass of every check_every-th epoch,
    when the parameters have had exactly that many updates. Returns the loss and the accuracy
    of the last epoch, which are measured before that epoch's update.
    """
    for epoch in range(epochs):
        if on_check is not None and epoch % check_every == 0:
            on_check(epoch)

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

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    return float(loss), float(accuracy)


def evaluate(X, y, dense1, activation1, dense2, loss_activation):
    """Loss and accuracy on (X, y): four forward calls, no backward call, no optimiser."""
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    return float(loss), float(accuracy)


_AXIS = np.linspace(-1, 1, 201)
GRID = np.stack(np.meshgrid(_AXIS, _AXIS), axis=-1).reshape(-1, 2).astype(np.float32)   # (40401, 2)


def boundary_length(dense1, activation1, dense2):
    """Pairs of neighbouring points, on a 201 by 201 grid over the square, given different classes."""
    dense1.forward(GRID)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    classes = np.argmax(dense2.output, axis=1).reshape(201, 201)
    return int(np.sum(classes[:, 1:] != classes[:, :-1]) + np.sum(classes[1:, :] != classes[:-1, :]))


def weight_norm(dense1, dense2):
    """The square root of the sum of the squares of all the weights, biases left out."""
    return float(np.sqrt(np.sum(dense1.weights ** 2) + np.sum(dense2.weights ** 2)))


if __name__ == "__main__":
    nnfs.init()
    print(SETUP)
    X, y, dense1, activation1, dense2, loss_activation, optimizer = build(seed=0)
    parameters = sum(a.size for a in (dense1.weights, dense1.biases, dense2.weights, dense2.biases))
    loss, accuracy = evaluate(X, y, dense1, activation1, dense2, loss_activation)
    print(f"training data {X.shape} {X.dtype}, labels {y.shape}, parameters {parameters}")
    print(f"before any update: loss {loss:.4f} (ln 3 = {np.log(3):.4f}), accuracy {accuracy:.4f}")
