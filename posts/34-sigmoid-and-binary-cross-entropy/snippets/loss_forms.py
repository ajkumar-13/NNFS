"""Post 34, section 3: binary cross-entropy without a clip, with the clip, and from the logit.

Run from the series root:
    python posts/34-sigmoid-and-binary-cross-entropy/snippets/loss_forms.py

Float64, no nnfs.init(). Needs only NumPy. Takes about a second.
"""
import warnings

import numpy as np

from binary_classes import Activation_Sigmoid


def bce_unclipped(y_hat, y):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def bce_clipped(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def bce_from_logits(z, y):
    # max(z, 0) - z * y + log(1 + exp(-|z|)): no probability, no clip
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def sigmoid(z):
    activation = Activation_Sigmoid()
    activation.forward(z)
    return activation.output


print("== One term is active: the loss of a prediction for each label")
print("y_hat      y = 1: -log(y_hat)    y = 0: -log(1 - y_hat)")
for y_hat in (0.99, 0.9, 0.5, 0.1, 0.01):
    print(f"{y_hat:<10} {bce_unclipped(y_hat, 1.0):<21.4f} {bce_unclipped(y_hat, 0.0):.4f}")

print()
print("== The three forms on logits of growing size")
z = np.array([-40.0, -17.0, -5.0, 0.0, 5.0, 17.0, 40.0])
y_hat = sigmoid(z)
print("z                   ", "".join(f"{v:>11g}" for v in z))
for y in (0.0, 1.0):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unclipped = bce_unclipped(y_hat, y) + 0.0
    messages = sorted({str(w.message) for w in caught})
    print(f"y = {y:.0f}  unclipped    ", "".join(f"{v:>11.4g}" for v in unclipped), "  warnings:", messages if messages else "none")
    print(f"y = {y:.0f}  clipped      ", "".join(f"{v:>11.4g}" for v in bce_clipped(y_hat, y)))
    print(f"y = {y:.0f}  from logits  ", "".join(f"{v:>11.4g}" for v in bce_from_logits(z, y)))

print()
print("== Where the clip acts")
bound = np.log((1 - 1e-7) / 1e-7)
print(f"the clip changes y_hat when |z| exceeds log((1 - 1e-7) / 1e-7) = {bound:.3f}")
print(f"largest loss the clipped form can report, -log(1e-7): {-np.log(1e-7):.3f}")
print(f"smallest, -log(1 - 1e-7): {-np.log(1 - 1e-7):.3e}")

print()
print("== The clipped form against the logits form")
rng = np.random.default_rng(0)
for limit in (16.0, 40.0):
    z = rng.uniform(-limit, limit, size=100_000)
    y = rng.integers(0, 2, size=100_000).astype(np.float64)
    gap = np.abs(bce_clipped(sigmoid(z), y) - bce_from_logits(z, y))
    print(f"100,000 logits in [-{limit:.0f}, {limit:.0f}]: largest gap {gap.max():.2e}, "
          f"mean loss clipped {bce_clipped(sigmoid(z), y).mean():.4f}, from logits {bce_from_logits(z, y).mean():.4f}")
