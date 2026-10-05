"""The from-scratch classes, in the form the series "Neural Networks from Scratch" derives them.

Each class computes what the final version in the post named in its docstring computes, and keeps
the series' names (``Layer_Dense``, ``dvalues``, ``dinputs``) so that a reader can hold the post
and the code side by side. Two classes are written more compactly than in the posts and give the
same numbers: the combined softmax and loss class contains the softmax and the cross-entropy that
the post's version delegates to, and the penalty is a function here where the post makes it a
method of the loss. The only additions are the argument checks in the constructors, which reject
values the mathematics does not allow.

Random numbers come from NumPy's global generator (``np.random.randn``, ``np.random.binomial``), as
in the posts. A run is therefore reproducible when ``np.random.seed`` is called once before the
first layer is built, and the order in which layers are built and called is part of the result.

No dependency beyond NumPy.
"""

import numpy as np


class Layer_Dense:
    """A fully connected layer, ``output = inputs @ weights + biases``.

    Forward and backward pass from ``nn-016``; the L1 and L2 penalty terms from ``nn-030``.
    ``weights`` has shape ``(n_inputs, n_neurons)`` and starts at ``0.01 * randn``; ``biases`` has
    shape ``(1, n_neurons)`` and starts at zero.
    """

    def __init__(
        self,
        n_inputs,
        n_neurons,
        weight_regularizer_l1=0.0,
        weight_regularizer_l2=0.0,
        bias_regularizer_l1=0.0,
        bias_regularizer_l2=0.0,
    ):
        if n_inputs < 1 or n_neurons < 1:
            raise ValueError(
                f"a dense layer needs at least one input and one neuron, got {n_inputs} and "
                f"{n_neurons}"
            )
        strengths = (
            weight_regularizer_l1,
            weight_regularizer_l2,
            bias_regularizer_l1,
            bias_regularizer_l2,
        )
        if min(strengths) < 0:
            raise ValueError(f"regularisation strengths must not be negative, got {strengths}")

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
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
    """ReLU, ``output = max(0, inputs)``. The backward pass (``nn-016``) zeroes the gradient
    wherever the input was not positive."""

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax_Loss_CategoricalCrossentropy:
    """Softmax and categorical cross-entropy as one step (``nn-019``).

    ``forward`` stores the class probabilities in ``output`` and returns the mean loss.
    ``backward`` is the combined gradient $(\\hat{y} - y) / N$ with respect to the logits.
    Labels may be class indices of shape ``(N,)`` or one-hot rows of shape ``(N, C)``.
    """

    def forward(self, inputs, y_true):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        y_pred_clipped = np.clip(self.output, 1e-7, 1 - 1e-7)
        if y_true.ndim == 1:
            correct_confidences = y_pred_clipped[range(len(y_true)), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return float(np.mean(-np.log(correct_confidences)))

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


class Layer_Dropout:
    """Inverted dropout (``nn-031``).

    ``rate`` is the probability of dropping a unit. In training the kept units are scaled by
    ``1 / (1 - rate)`` so the expected activation is unchanged; with ``training=False`` the layer
    passes its input through untouched and draws no random numbers.
    """

    def __init__(self, rate):
        if not 0 <= rate < 1:
            raise ValueError(f"the dropout rate must be in [0, 1), got {rate}")
        # `rate` is the drop probability; the keep probability is what the arithmetic needs.
        self.rate = 1 - rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Optimizer_Adam:
    """Adam with bias correction and inverse-time learning-rate decay (``nn-027``).

    The three-call contract of the series' optimisers: ``pre_update_params`` once per step,
    ``update_params`` once per layer, ``post_update_params`` once per step.
    """

    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        if learning_rate <= 0:
            raise ValueError(f"the learning rate must be positive, got {learning_rate}")
        if decay < 0:
            raise ValueError(f"the decay must not be negative, got {decay}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not (0 <= beta_1 < 1 and 0 <= beta_2 < 1):
            raise ValueError(f"beta_1 and beta_2 must be in [0, 1), got {beta_1} and {beta_2}")
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
        weight_m_hat = layer.weight_momentums / (1 - self.beta_1**t)
        bias_m_hat = layer.bias_momentums / (1 - self.beta_1**t)

        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        )
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_v_hat = layer.weight_cache / (1 - self.beta_2**t)
        bias_v_hat = layer.bias_cache / (1 - self.beta_2**t)

        layer.weights -= (
            self.current_learning_rate * weight_m_hat / (np.sqrt(weight_v_hat) + self.epsilon)
        )
        layer.biases -= (
            self.current_learning_rate * bias_m_hat / (np.sqrt(bias_v_hat) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


def regularization_loss(layer):
    """The L1 and L2 penalty of one dense layer (``nn-030``); zero when no strength is set."""
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
