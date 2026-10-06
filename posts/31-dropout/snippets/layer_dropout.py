"""Post 31, sections 3 to 5: the mask, the scale, the expectation and the two modes of Layer_Dropout.

Run from the series root:
    python posts/31-dropout/snippets/layer_dropout.py

The class is imported from network.py. This file does not call nnfs.init(), so every array is
float64 and the seed is the one set here.

Needs NumPy (and the nnfs package, which network.py imports). Takes about a second.
"""
import numpy as np

from network import Layer_Dropout

np.set_printoptions(suppress=True, linewidth=120)

print("== Section 4: five activations of 1, drop rate 0.2")
p = 0.2
a = np.ones((1, 5))
np.random.seed(3)
raw_mask = np.random.binomial(1, 1 - p, size=a.shape)
print("mask of 0 and 1:       ", raw_mask[0], " sum of a * mask:", (a * raw_mask).sum())
print("mask / (1 - p):        ", raw_mask[0] / (1 - p), " sum:", round(float((a * raw_mask / (1 - p)).sum()), 4))

layer = Layer_Dropout(p)
np.random.seed(3)
layer.forward(a)
print("Layer_Dropout(0.2):    ", layer.output[0], " stored self.rate:", layer.rate)

print()
print("== Section 4: the expectation, over 200,000 masks")
row = np.array([0.5, 1.0, 2.0, 0.0, 3.0])
many = np.tile(row, (200_000, 1))
np.random.seed(0)
layer.forward(many)
inverted = layer.output
plain = many * (layer.binary_mask > 0)                  # the same masks without the scale
print("activations a:              ", row)
print("mean, mask only:            ", plain.mean(axis=0).round(4), " closed form (1 - p) * a:", (1 - p) * row)
print("mean, mask / (1 - p):       ", inverted.mean(axis=0).round(4), " closed form a")
print("variance, mask / (1 - p):   ", inverted.var(axis=0).round(4), " closed form a^2 * p / (1 - p):",
      row ** 2 * p / (1 - p))
print(f"fraction of entries dropped: {np.mean(layer.binary_mask == 0):.4f}")

print()
print("== Section 3: how many of 64 neurons one row keeps, by drop rate")
print("   p   mean kept   standard deviation   P(all 64 kept)   scale 1 / (1 - p)")
for rate in (0.1, 0.2, 0.5, 0.8):
    print(f"{rate:4g}   {64 * (1 - rate):9.1f}   {np.sqrt(64 * rate * (1 - rate)):18.2f}   "
          f"{(1 - rate) ** 64:14.2e}   {1 / (1 - rate):17.4g}")
print(f"masks of 64 neurons: 2^64 = {2 ** 64:,}")

print()
print("== Section 5: evaluation mode")
np.random.seed(0)
state_before = np.random.get_state()[1].copy()
layer.forward(a * 7.0, training=False)
state_after = np.random.get_state()[1]
print("output:", layer.output[0], " equals the input:", np.array_equal(layer.output, a * 7.0),
      " random stream untouched:", np.array_equal(state_before, state_after))

print()
print("== Section 5: backward uses the mask of the forward pass")
np.random.seed(3)
layer.forward(a)
layer.backward(np.array([[0.1, -0.2, 0.3, 0.4, -0.5]]))
print("mask:   ", layer.binary_mask[0])
print("dinputs:", layer.dinputs[0])
