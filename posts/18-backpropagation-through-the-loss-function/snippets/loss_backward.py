"""Post 18, sections 2 to 5: the cross-entropy gradient by hand, then as a backward method.

Run from the series root:
    python posts/18-backpropagation-through-the-loss-function/snippets/loss_backward.py

Needs only NumPy.
"""
import numpy as np

# Section 2.2: one sample.
y_true = np.array([[1, 0, 0]])
y_pred = np.array([[0.7, 0.2, 0.1]])
print(-y_true / y_pred)

# Section 3.1: a batch of three.
y_true = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

y_pred = np.array([[0.7, 0.2, 0.1],
                   [0.1, 0.6, 0.3],
                   [0.2, 0.3, 0.5]])

N = len(y_pred)
dinputs = (-y_true / y_pred) / N
print(dinputs)

# Section 4: integer labels become one-hot rows.
y_true_indices = np.array([0, 1, 2])

n_labels = y_pred.shape[1]                  # number of classes
y_true_onehot = np.eye(n_labels)[y_true_indices]
print(y_true_onehot)


# Section 5: the classes of post 08, with the backward method added.
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


loss_fn = Loss_CategoricalCrossentropy()
print("loss:", loss_fn.calculate(y_pred, y_true_indices))

loss_fn.backward(y_pred, y_true_indices)
from_integers = loss_fn.dinputs
loss_fn.backward(y_pred, y_true)
from_onehot = loss_fn.dinputs

print(from_integers)
print("same array from one-hot labels:", np.array_equal(from_integers, from_onehot))
print("shape:", from_integers.shape, " non-zero entries per row:", np.count_nonzero(from_integers, axis=1))
