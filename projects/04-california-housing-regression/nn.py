"""
From-scratch neural-network classes for the regression project.

Differences vs projects/01-mnist-from-scratch/nn.py:
  * Replaces Activation_Softmax_Loss_CategoricalCrossentropy with
    Loss_MSE — the natural loss for predicting a continuous target.
  * No softmax (the output is a scalar, not a probability distribution).
  * No Layer_Dropout (overkill for a small tabular regression).
  * Adds Layer_Linear (identity activation) for symmetry, though the
    final Dense layer's output is fed straight into Loss_MSE in this
    project.

Same Layer_Dense, Activation_ReLU, Optimizer_Adam, regularization_loss
as the classification projects.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Dense layer (posts/04 + posts/30 regularisation hooks)
# -----------------------------------------------------------------------------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0.0, weight_regularizer_l2=0.0,
                 bias_regularizer_l1=0.0,   bias_regularizer_l2=0.0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1   = bias_regularizer_l1
        self.bias_regularizer_l2   = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights); dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases); dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


# -----------------------------------------------------------------------------
# ReLU activation
# -----------------------------------------------------------------------------
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# -----------------------------------------------------------------------------
# Mean-squared-error loss for regression.
# -----------------------------------------------------------------------------
class Loss_MSE:
    """
    Mean-squared error between predicted scalar (or vector) and target.

    Forward:  L = mean( (y_pred - y_true) ** 2 )
    Backward: dL/d(y_pred) = 2 * (y_pred - y_true) / N

    Inputs are expected as shape (N, 1) for y_pred and either (N,) or
    (N, 1) for y_true; the class normalises y_true to (N, 1) internally.
    """

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = np.asarray(y_true, dtype=np.float64).reshape(y_pred.shape)
        return float(np.mean((y_pred - self.y_true) ** 2))

    def backward(self, y_pred, y_true=None):
        # y_true is captured from forward(); arg is accepted for API symmetry
        # with the classification loss classes.
        y_true = self.y_true if y_true is None else \
            np.asarray(y_true, dtype=np.float64).reshape(y_pred.shape)
        samples = len(y_pred)
        outputs = y_pred.shape[1] if y_pred.ndim > 1 else 1
        self.dinputs = 2 * (y_pred - y_true) / (samples * outputs)


# -----------------------------------------------------------------------------
# Adam optimiser (posts/27)
# -----------------------------------------------------------------------------
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
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
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache     = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)
            layer.bias_cache       = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums   = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases

        t = self.iterations + 1
        w_m_hat = layer.weight_momentums / (1 - self.beta_1 ** t)
        b_m_hat = layer.bias_momentums   / (1 - self.beta_1 ** t)

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache   = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases ** 2

        w_v_hat = layer.weight_cache / (1 - self.beta_2 ** t)
        b_v_hat = layer.bias_cache   / (1 - self.beta_2 ** t)

        layer.weights -= self.current_learning_rate * w_m_hat / \
            (np.sqrt(w_v_hat) + self.epsilon)
        layer.biases  -= self.current_learning_rate * b_m_hat / \
            (np.sqrt(b_v_hat) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# -----------------------------------------------------------------------------
# Regularisation helper (posts/30)
# -----------------------------------------------------------------------------
def regularization_loss(layer):
    loss = 0.0
    if layer.weight_regularizer_l1 > 0:
        loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
    if layer.weight_regularizer_l2 > 0:
        loss += layer.weight_regularizer_l2 * np.sum(layer.weights ** 2)
    if layer.bias_regularizer_l1 > 0:
        loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
    if layer.bias_regularizer_l2 > 0:
        loss += layer.bias_regularizer_l2 * np.sum(layer.biases ** 2)
    return loss
