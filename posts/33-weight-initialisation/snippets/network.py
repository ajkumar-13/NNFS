"""Post 33: Layer_Dense with the init argument, and the classes the other scripts of this post share.

Run from the series root:
    python posts/33-weight-initialisation/snippets/network.py

Contents: Layer_Dense of post 30 with one new argument, init, and the lines that turn it into a
weight scale; Activation_ReLU (post 16), Activation_Tanh (post 17), Activation_Softmax, Loss,
Loss_CategoricalCrossentropy and the combined class (post 19, Loss as extended in post 30),
Optimizer_SGD (post 22) and Optimizer_Adam (post 27), all unchanged; build_stack(), forward_stack(),
backward_stack(), train_stack() and the helpers built on them, which the other scripts import.

Run on its own, the script prints what the init argument does (section 6), in float64 and
without training. Needs NumPy and the nnfs package (spiral_data; the training scripts also call
nnfs.init). Takes about a second.
"""
import sys

import numpy as np
from nnfs.datasets import spiral_data


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, init="he",
                 weight_regularizer_l1=0.0, weight_regularizer_l2=0.0,
                 bias_regularizer_l1=0.0, bias_regularizer_l2=0.0):
        # Added in post 33: the standard deviation of the initial weights.
        if init == "he":
            scale = np.sqrt(2.0 / n_inputs)
        elif init == "xavier" or init == "glorot":
            scale = np.sqrt(2.0 / (n_inputs + n_neurons))
        elif init == "small":
            scale = 0.01                        # the fixed scale of posts 04 to 32
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


class Activation_Tanh:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        # f'(z) = 1 - tanh(z)^2, read from the cached output
        self.dinputs = dvalues * (1 - self.output ** 2)


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


class Optimizer_SGD:

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


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


def build_stack(sizes, init, activation=Activation_ReLU):
    """Dense layers of the given sizes, each followed by an activation except the last.

    sizes = [2, 64, 64, 3] gives Layer_Dense(2, 64), activation, Layer_Dense(64, 64), activation,
    Layer_Dense(64, 3). The weights are drawn from the global stream in that order. A number as
    init is used as a fixed scale for every layer (init_scale.py of post 04 does the same).
    """
    dense, activations = [], []
    for n_inputs, n_neurons in zip(sizes[:-1], sizes[1:]):
        if isinstance(init, str):
            layer = Layer_Dense(n_inputs, n_neurons, init=init)
        else:
            layer = Layer_Dense(n_inputs, n_neurons, init="small")
            layer.weights = layer.weights * (init / 0.01)       # the class drew 0.01 * randn; rescale it
        dense.append(layer)
        activations.append(activation())
    return dense, activations[:-1]


def forward_stack(X, dense, activations):
    """Forward through the stack; returns the output of the last dense layer."""
    values = X
    for layer, activation in zip(dense, activations):
        layer.forward(values)
        activation.forward(layer.output)
        values = activation.output
    dense[-1].forward(values)
    return dense[-1].output


def backward_stack(dvalues, dense, activations):
    """Backward through the stack, from the gradient of the last dense layer's output."""
    dense[-1].backward(dvalues)
    gradient = dense[-1].dinputs
    for layer, activation in zip(reversed(dense[:-1]), reversed(activations)):
        activation.backward(gradient)
        layer.backward(activation.dinputs)
        gradient = layer.dinputs


SEEDS = (0, 1, 2, 3, 4)


def seeds_from_command_line():
    """The integers on the command line, or else SEEDS."""
    return tuple(int(argument) for argument in sys.argv[1:]) or SEEDS


def train_stack(seed, hidden, init, optimizer, epochs, width=64):
    """Train 2 -> hidden layers of width neurons with ReLU -> 3 on the spiral, full batch.

    nnfs.init() must have been called once by the script. The draws from the global stream, in
    order: np.random.seed(seed), the 300 training points, the weights layer by layer. The loop is
    the one of Part VI and draws nothing. Returns a dict: the loss of the first epoch, the loss and
    accuracy of the last epoch (both measured before that epoch's update), the first epoch whose
    loss is below 1.0 (None if there is none), the first epoch whose loss is NaN (the loop stops
    there), the largest |dweights| of the first backward pass, and the objects.
    """
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)
    dense, activations = build_stack([2] + [width] * hidden + [3], init)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    result = dict(first_below=None, nan_epoch=None)
    for epoch in range(epochs):
        loss = loss_activation.forward(forward_stack(X, dense, activations), y)
        accuracy = np.mean(np.argmax(loss_activation.output, axis=1) == y)
        if epoch == 0:
            result["first_loss"] = float(loss)
        if np.isnan(loss):
            result["nan_epoch"] = epoch
            break
        if result["first_below"] is None and loss < 1.0:
            result["first_below"] = epoch

        loss_activation.backward(loss_activation.output, y)
        backward_stack(loss_activation.dinputs, dense, activations)
        if epoch == 0:
            result["first_gradient"] = max(float(np.abs(layer.dweights).max()) for layer in dense)

        if hasattr(optimizer, "pre_update_params"):             # Optimizer_SGD of post 22 has one method
            optimizer.pre_update_params()
        for layer in dense:
            optimizer.update_params(layer)
        if hasattr(optimizer, "post_update_params"):
            optimizer.post_update_params()

    result.update(loss=float(loss), accuracy=float(accuracy), X=X, y=y,
                  dense=dense, activations=activations)
    return result


def accuracy_of(X, y, dense, activations):
    """Accuracy on (X, y), forward only."""
    return float(np.mean(np.argmax(forward_stack(X, dense, activations), axis=1) == y))


def depth_sweep(init, depths, make_optimizer, epochs, seeds):
    """One line per depth: per seed the last loss and the first epoch with a loss below 1.0; the first gradients."""
    print(f"init = {init!r}; one cell per seed {seeds}: loss at epoch {epochs - 1} (first epoch below 1.0)")
    for hidden in depths:
        cells, gradients, stuck = [], [], 0
        for seed in seeds:
            r = train_stack(seed, hidden, init, make_optimizer(), epochs)
            cells.append(f"{r['loss']:.4f} ({r['first_below']})")
            gradients.append(r["first_gradient"])
            stuck += r["first_below"] is None
        print(f"hidden layers {hidden}: " + "  ".join(cells) + f"   never below 1.0: {stuck} of {len(seeds)}"
              f"   largest |dweights|, first pass: {min(gradients):.1e} to {max(gradients):.1e}", flush=True)


if __name__ == "__main__":
    print("== The scale each value of init gives, Layer_Dense(64, 64)")
    print("init        target scale   measured std of the 4,096 weights   largest |bias|")
    targets = {"he": np.sqrt(2 / 64), "xavier": np.sqrt(2 / 128), "glorot": np.sqrt(2 / 128), "small": 0.01}
    for init, target in targets.items():
        np.random.seed(0)
        layer = Layer_Dense(64, 64, init=init)
        print(f"{init:<10}  {target:<13.4f}  {layer.weights.std():<34.4f}  {np.abs(layer.biases).max():.1f}")

    print()
    print("== The same draws, another scale: Layer_Dense(2, 3) after np.random.seed(0), first weight row")
    for init in ("small", "xavier", "he"):
        np.random.seed(0)
        layer = Layer_Dense(2, 3, init=init)
        print(f"init={init!r:<9} {layer.weights[0]}")
    np.random.seed(0)
    print("default            ", Layer_Dense(2, 3).weights[0], " (the default is 'he')")

    print()
    print("== An unknown name")
    try:
        Layer_Dense(2, 3, init="He")
    except ValueError as error:
        print("ValueError:", error)

    print()
    print("== The Glorot uniform form has the same variance as the normal form used here")
    a = np.sqrt(6 / (64 + 64))
    np.random.seed(0)
    uniform = np.random.uniform(-a, a, (64, 64))
    print(f"limit a = sqrt(6 / 128) = {a:.4f}; a^2 / 3 = {a * a / 3:.6f}; 2 / 128 = {2 / 128:.6f}; "
          f"measured variance of 4,096 uniform draws {uniform.var():.6f}")
