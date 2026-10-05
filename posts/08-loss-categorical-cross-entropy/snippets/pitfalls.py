"""Post 08, section 9 and the pitfalls: each failure, run once so the symptom is on the page.

Run from the series root:
    python posts/08-loss-categorical-cross-entropy/snippets/pitfalls.py

Needs NumPy and the nnfs helper package (pip install nnfs).
"""
import warnings

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        shifted       = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values    = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities


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


def attempt(description, function):
    """Run function and print what it returns or the exception it raises."""
    try:
        print(f"   {description}: {function()}")
    except Exception as error:
        print(f"   {description}: {type(error).__name__}: {error}")


loss_fn = Loss_CategoricalCrossentropy()

softmax_outputs = np.array([[0.7,  0.1, 0.2 ],
                            [0.1,  0.5, 0.4 ],
                            [0.02, 0.9, 0.08]])
labels        = np.array([0, 1, 1])
labels_onehot = np.eye(3, dtype=int)[labels]

# The same two formats at the size of the spiral batch: 300 samples, 3 classes.
big_outputs = np.full((300, 3), 1 / 3)
big_labels  = np.arange(300) % 3
big_onehot  = np.eye(3, dtype=int)[big_labels]

print("1. a hand-written path given the other label format")
attempt("integer labels in the multiply-and-sum path, 3 samples",
        lambda: np.sum(softmax_outputs * labels, axis=1))
attempt("integer labels in the multiply-and-sum path, 300 samples",
        lambda: np.sum(big_outputs * big_labels, axis=1))
attempt("one-hot labels in the indexing path, 3 samples: result of shape",
        lambda: softmax_outputs[range(3), labels_onehot].shape)
attempt("one-hot labels in the indexing path, 300 samples",
        lambda: big_outputs[range(300), big_onehot])

print("2. the class given integer labels 0, 1, 2 as a column of shape (3, 1)")
column = np.array([[0], [1], [2]])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    column_losses = loss_fn.forward(softmax_outputs, column) + 0.0      # + 0.0 turns a -0.0 into 0.0
print("   per-sample losses:", column_losses)
print("   with the same labels as shape (3,):", loss_fn.forward(softmax_outputs, column.ravel()))

print("3. a soft label through the one-hot path")
soft_label = np.array([[0.9, 0.05, 0.05]])
prediction = np.array([[0.7, 0.1, 0.2]])
print(f"   the class returns          {loss_fn.forward(prediction, soft_label)[0]:.4f}")
print(f"   -sum(y * log(y_hat)) gives {-np.sum(soft_label * np.log(prediction), axis=1)[0]:.4f}")

print("4. the full sum -sum(y * log(y_hat)) without a clip, zero on a wrong class")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    full_sum = -np.sum(np.array([[1, 0, 0]]) * np.log(np.array([[0.6, 0.0, 0.4]])), axis=1)
print("   loss:", full_sum, "| the class on the same row:", loss_fn.forward(np.array([[0.6, 0.0, 0.4]]), np.array([0])))

print("5. summing instead of averaging: the worked batch once, then repeated ten times")
for copies in (1, 10):
    losses = loss_fn.forward(np.tile(softmax_outputs, (copies, 1)), np.tile(labels, copies))
    print(f"   {len(losses):>2d} samples: sum {np.sum(losses):.3f}, mean {np.mean(losses):.3f}")

print("6. a loss above the uniform-guess baseline with most predictions right")
nine_right = np.tile([0.9, 0.05, 0.05], (9, 1))
one_wrong  = np.array([[1e-6, 0.5, 0.5 - 1e-6]])
batch      = np.vstack([nine_right, one_wrong])
truth      = np.zeros(10, dtype=int)
print(f"   accuracy {np.mean(np.argmax(batch, axis=1) == truth):.1f}, loss {loss_fn.calculate(batch, truth):.3f},",
      f"log(3) = {np.log(3):.3f}")

print("7. the loss handed logits instead of probabilities (section 8's network)")
nnfs.init()
X, y = spiral_data(samples=100, classes=3)
dense1      = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2      = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(f"   on activation2.output: {loss_fn.calculate(activation2.output, y):.4f}")
print(f"   on dense2.output:      {loss_fn.calculate(dense2.output, y):.4f}")
