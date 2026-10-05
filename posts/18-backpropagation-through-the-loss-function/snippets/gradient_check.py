"""Post 18, sections 6 and 7: the backward method against a central difference, and the 1/N.

Run from the series root:
    python posts/18-backpropagation-through-the-loss-function/snippets/gradient_check.py

Needs only NumPy. Everything is float64; the seed is fixed.
"""
import numpy as np


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


def numerical_gradient(loss_fn, y_pred, y_true, h=1e-5):
    """Central difference of the batch loss with respect to every prediction."""
    grad = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        for k in range(y_pred.shape[1]):
            plus, minus = y_pred.copy(), y_pred.copy()
            plus[i, k]  += h
            minus[i, k] -= h
            grad[i, k] = (loss_fn.calculate(plus, y_true) - loss_fn.calculate(minus, y_true)) / (2 * h)
    return grad


rng = np.random.default_rng(18)
N, K = 5, 4
raw    = rng.random((N, K)) + 0.1
y_pred = raw / raw.sum(axis=1, keepdims=True)      # rows that sum to 1, like a softmax output
y_true = rng.integers(0, K, size=N)

loss_fn = Loss_CategoricalCrossentropy()
print("labels:", y_true, " loss:", round(float(loss_fn.calculate(y_pred, y_true)), 6))

for name, labels in (("integer", y_true), ("one-hot", np.eye(K)[y_true])):
    loss_fn.backward(y_pred, labels)
    numeric = numerical_gradient(loss_fn, y_pred, labels)
    print(f"{name} labels: largest |backward - central difference| = "
          f"{np.max(np.abs(loss_fn.dinputs - numeric)):.2e}")

i, k = 0, y_true[0]
print(f"sample 0, true class {k}: prediction {y_pred[i, k]:.6f}, "
      f"backward {loss_fn.dinputs[i, k]:.6f}, central difference {numeric[i, k]:.6f}")

without = -np.eye(K)[y_true] / y_pred              # the 1/N left out
ratio = without[range(N), y_true] / numeric[range(N), y_true]
print("1/N left out: backward / central difference =", np.round(ratio, 4))

print()
print("the same three samples repeated: one entry, and the sum over the rows")
base_pred = np.array([[0.7, 0.2, 0.1],
                      [0.1, 0.6, 0.3],
                      [0.2, 0.3, 0.5]])
base_true = np.array([0, 1, 2])
for copies in (1, 10, 100):
    pred, true = np.tile(base_pred, (copies, 1)), np.tile(base_true, copies)
    loss_fn.backward(pred, true)
    with_n    = loss_fn.dinputs
    without_n = loss_fn.dinputs * len(pred)
    print(f"N = {len(pred):3d}  loss {loss_fn.calculate(pred, true):.4f}"
          f"  entry [0, 0]: {with_n[0, 0]:9.6f}"
          f"  summed over rows with 1/N: {np.round(with_n.sum(axis=0), 4) + 0.0}"
          f"  without: {np.round(without_n.sum(axis=0), 2) + 0.0}")
