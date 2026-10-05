"""Post 24, sections 5 to 7: Optimizer_SGD with momentum, the training loop, and the documented run.

Run from the series root:
    python posts/24-momentum/snippets/momentum_sgd.py

Contents: the network classes of posts 16 and 19, unchanged; Optimizer_SGD of post 23 with a
momentum argument added; train(), one full-batch run of the shared setup of Part VI that also
records what the path does; and the documented run, decay only against decay plus momentum 0.9.

Needs NumPy and the nnfs package (spiral_data and nnfs.init). Takes about 15 seconds.
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


SETUP = ("setup: nnfs.init() (float32, float32 np.dot), spiral_data(samples=100, classes=3), "
         "Layer_Dense(2, 64), ReLU, Layer_Dense(64, 3), softmax and cross-entropy, "
         "0.01 * randn weights, 10,001 full-batch epochs")


def flat_parameters(layers):
    """Every weight and bias of the network as one float64 vector."""
    return np.concatenate([a.ravel() for layer in layers
                           for a in (layer.weights, layer.biases)]).astype(np.float64)


def train(optimizer, seed=0, epochs=10001):
    """One run of the shared setup. nnfs.init() must have been called once before."""
    np.random.seed(seed)                          # seed 0 is the seed nnfs.init() itself sets
    X, y = spiral_data(samples=100, classes=3)    # (300, 2), (300,)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    losses, accuracies, opposed = [], [], []
    position = flat_parameters((dense1, dense2))
    previous_step = np.zeros_like(position)

    for epoch in range(epochs):
        # Forward, accuracy, backward: unchanged from post 23.
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

        # Update: the three calls of post 23.
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

        # Bookkeeping for the post: does this step point against the one before it?
        new_position = flat_parameters((dense1, dense2))
        step = new_position - position
        opposed.append(step @ previous_step < 0)
        position, previous_step = new_position, step
        losses.append(float(loss))
        accuracies.append(float(accuracy))

    losses = np.array(losses)
    return {
        "losses": losses,
        "accuracies": np.array(accuracies),
        "loss": losses[-1],
        "accuracy": accuracies[-1],
        "peak_loss": losses.max(),
        "peak_epoch": int(losses.argmax()),
        "rises": int(np.sum(np.diff(losses) > 0)),          # epochs whose loss is above the one before
        "opposed": int(np.sum(opposed)),                    # steps that point against the step before
    }


if __name__ == "__main__":
    nnfs.init()
    print(SETUP)
    print("seed: 0, the seed nnfs.init() sets")

    runs = [("decay only (post 23)", Optimizer_SGD(learning_rate=1.0, decay=1e-3)),
            ("decay and momentum 0.9", Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9))]
    results = []
    for name, optimizer in runs:
        result = train(optimizer)
        results.append(result)
        print()
        print(name)
        for epoch in (0, 100, 1000, 2000, 5000, 10000):
            print(f"  epoch {epoch:5d}  loss {result['losses'][epoch]:.4f}  "
                  f"acc {result['accuracies'][epoch]:.4f}")
        print(f"  learning rate at the last epoch {optimizer.current_learning_rate:.4f}")

    print()
    print("what the path does (10,001 epochs, so 10,000 pairs of consecutive epochs and steps)")
    print("                          final loss  accuracy  peak loss (epoch)  loss rose  step against the last")
    for (name, _), r in zip(runs, results):
        print(f"{name:24s}  {r['loss']:10.4f}  {r['accuracy']:8.4f}  {r['peak_loss']:9.3f} ({r['peak_epoch']:5d})"
              f"  {r['rises']:9,d}  {r['opposed']:21,d}")
