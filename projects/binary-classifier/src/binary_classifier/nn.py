"""The series' classes, as this project uses them.

Every class here is the one a post of the series builds, copied with its names
and its arithmetic unchanged so that the code can be read beside the post:

    Layer_Dense                                 nn-016 (forward, backward),
                                                nn-030 (L1 and L2 terms),
                                                nn-033 (the init argument)
    Activation_ReLU                             nn-016
    Activation_Sigmoid                          the stable sigmoid of nn-034
                                                and its derivative, as a class
    Activation_Sigmoid_Loss_BinaryCrossentropy  nn-034, after the softmax
                                                shortcut of nn-019
    Optimizer_Adam                              nn-027
    regularization_loss                         nn-030, as a function

The combined sigmoid and binary cross-entropy class exists for the reason the
combined softmax class of nn-019 does: the gradient with respect to the logits
reduces to (prediction - target) / N, with no division by the sigmoid output.

Layer_Dense draws its weights from NumPy's global legacy generator
(np.random.randn), as the posts do. Seed it with np.random.seed before
building layers; binary_classifier.train does.
"""

import numpy as np


class Layer_Dense:
    def __init__(
        self,
        n_inputs,
        n_neurons,
        init="he",
        weight_regularizer_l1=0.0,
        weight_regularizer_l2=0.0,
        bias_regularizer_l1=0.0,
        bias_regularizer_l2=0.0,
    ):
        # nn-033: the standard deviation of the initial weights. "he" keeps the
        # variance of ReLU activations constant from layer to layer; "small" is
        # the 0.01 the early posts use, which starves a layer with two inputs.
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

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

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

        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        # Stable in both directions: 1 / (1 + exp(-x)) where x >= 0 and
        # exp(x) / (1 + exp(x)) where x < 0, so exp never sees a large
        # positive argument.
        out = np.empty_like(inputs, dtype=np.float64)
        pos = inputs >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-inputs[pos]))
        neg = ~pos
        ex = np.exp(inputs[neg])
        out[neg] = ex / (1.0 + ex)
        self.output = out

    def backward(self, dvalues):
        # d sigmoid(x) / dx = sigmoid(x) * (1 - sigmoid(x))
        self.dinputs = dvalues * self.output * (1.0 - self.output)


class Activation_Sigmoid_Loss_BinaryCrossentropy:
    """Last-layer activation + loss for binary classification.

    Forward expects raw logits of shape (N, 1) and integer labels y of
    shape (N,) with values in {0, 1}. Returns the mean BCE loss.

    Backward populates self.dinputs of shape (N, 1) with the simplified
    gradient (y_pred - y_true) / N.
    """

    def __init__(self):
        self._sigmoid = Activation_Sigmoid()

    def forward(self, logits, y_true):
        self._sigmoid.forward(logits)
        self.output = self._sigmoid.output  # (N, 1) probabilities

        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        y_pred = np.clip(self.output, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.mean(sample_losses))

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        # The shortcut: d(loss)/d(logits) = (sigmoid(logits) - y) / N.
        self.dinputs = (dvalues - y_true) / samples


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        )

        # The step counter is incremented in post_update_params, after this call.
        t = self.iterations + 1
        w_m_hat = layer.weight_momentums / (1 - self.beta_1**t)
        b_m_hat = layer.bias_momentums / (1 - self.beta_1**t)

        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        )
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        w_v_hat = layer.weight_cache / (1 - self.beta_2**t)
        b_v_hat = layer.bias_cache / (1 - self.beta_2**t)

        layer.weights -= self.current_learning_rate * w_m_hat / (np.sqrt(w_v_hat) + self.epsilon)
        layer.biases -= self.current_learning_rate * b_m_hat / (np.sqrt(b_v_hat) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


def regularization_loss(layer):
    loss = 0.0
    if layer.weight_regularizer_l1 > 0:
        loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
    if layer.weight_regularizer_l2 > 0:
        loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)
    if layer.bias_regularizer_l1 > 0:
        loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
    if layer.bias_regularizer_l2 > 0:
        loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)
    return loss
