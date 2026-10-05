"""Post 08, section 6: what log(0) does to the loss, and what the clip changes.

Run from the series root:
    python posts/08-loss-categorical-cross-entropy/snippets/clipping.py

Needs only NumPy.
"""
import warnings

import numpy as np

y_pred = np.array([[0.0, 1.0, 0.0],      # probability exactly 0 on the true class
                   [1.0, 0.0, 0.0],      # probability exactly 1 on the true class
                   [0.7, 0.1, 0.2]])     # an ordinary prediction
y_true = np.array([0, 0, 0])

y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

rows = range(len(y_pred))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    unclipped = -np.log(y_pred[rows, y_true]) + 0.0      # + 0.0 turns the -0.0 of log(1) into 0.0
clipped = -np.log(y_pred_clipped[rows, y_true])

print("1. three predictions for true class 0, without and with the clip")
print("   without:", unclipped, " mean:", np.mean(unclipped))
print("   raised: ", caught[0].category.__name__ + ":", caught[0].message)
print("   with:   ", clipped, " mean:", np.mean(clipped))
print(f"   the largest loss the clip allows, -log(1e-7): {-np.log(1e-7):.3f}")
print(f"   the smallest, -log(1 - 1e-7): {-np.log(1 - 1e-7):.3e}")

print("2. softmax can output an exact 0: the smallest whole-number gap between two logits that does it")
for dtype in (np.float32, np.float64):
    gap = 1
    while np.exp(dtype(-gap)) > 0:
        gap += 1
    print(f"   {dtype.__name__}: exp(-{gap}) == 0, exp(-{gap - 1}) = {np.exp(dtype(-(gap - 1)))}")

print("3. an ordinary batch is untouched: the worked batch of section 4")
batch  = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
labels = [0, 1, 1]
plain  = np.mean(-np.log(batch[range(3), labels]))
safe   = np.mean(-np.log(np.clip(batch, 1e-7, 1 - 1e-7)[range(3), labels]))
print("   mean loss without the clip:", plain, " with:", safe, " equal:", plain == safe)

print("4. why 1e-7: the upper bound has to stay below 1 in float32, the type nnfs.init() sets")
for eps in (1e-7, 1e-8):
    upper = np.float32(1 - eps)
    print(f"   float32(1 - {eps:g}) = {upper!r}, below 1: {bool(upper < 1)}")
perfect = np.clip(np.float32(1.0), 1e-7, 1 - 1e-7)
print(f"   float32 loss of a prediction of exactly 1 after the clip: {-np.log(perfect):.3e}")
