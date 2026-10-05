"""Post 19, sections 5 to 7: the softmax backward, the combined class, and a check of both.

Run from the series root:
    python posts/19-softmax-derivatives-and-the-combined-backward-pass/snippets/combined_class.py

Needs only NumPy.
"""
import numpy as np


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


def central_difference(loss_of, logits, h=1e-5):
    """The slope of loss_of(logits) with respect to every logit, one entry moved at a time."""
    slopes = np.zeros_like(logits)
    for index in np.ndindex(*logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += h
        minus[index] -= h
        slopes[index] = (loss_of(plus) - loss_of(minus)) / (2 * h)
    return slopes


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, floatmode="fixed")

    rng = np.random.default_rng(4)
    logits = rng.normal(size=(5, 4))            # 5 samples, 4 classes, float64
    y = rng.integers(0, 4, size=5)

    # Route 1: the combined class.
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
    loss = softmax_loss.forward(logits, y)
    softmax_loss.backward(softmax_loss.output, y)

    # Route 2: the loss backward of post 18, then the softmax backward.
    activation = Activation_Softmax()
    loss_fn = Loss_CategoricalCrossentropy()
    activation.forward(logits)
    loss_fn.backward(activation.output, y)
    activation.backward(loss_fn.dinputs)

    print("labels:", y, f" loss: {loss:.6f}")
    print("combined dinputs, shape", softmax_loss.dinputs.shape)
    print(softmax_loss.dinputs)
    print("row sums:", softmax_loss.dinputs.sum(axis=1).round(12) + 0.0)
    print("largest |combined - Jacobian route|     =", f"{np.abs(softmax_loss.dinputs - activation.dinputs).max():.2e}")

    measured = central_difference(lambda z: softmax_loss.forward(z, y), logits)
    print("largest |combined - central difference| =", f"{np.abs(softmax_loss.dinputs - measured).max():.2e}")

    softmax_loss.forward(logits, y)
    softmax_loss.backward(softmax_loss.output, np.eye(4)[y])
    print("largest |one-hot labels - central difference| =", f"{np.abs(softmax_loss.dinputs - measured).max():.2e}")

    # The worked batch of section 4, through both routes.
    worked = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
    worked_y = np.array([0, 1, 1])
    softmax_loss.backward(worked, worked_y)
    activation.forward(np.log(worked))          # logits whose softmax is the worked batch
    loss_fn.backward(activation.output, worked_y)
    activation.backward(loss_fn.dinputs)
    print("section 4 batch, largest |combined - Jacobian route| =", f"{np.abs(softmax_loss.dinputs - activation.dinputs).max():.2e}")

    samples, classes = logits.shape
    print("numbers built: Jacobian route", samples * classes ** 2, " combined", samples * classes)
