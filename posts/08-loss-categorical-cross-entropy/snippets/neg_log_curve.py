"""Post 08, sections 2.1, 3 and 8.1: the -log(p) curve as numbers.

Run from the series root:
    python posts/08-loss-categorical-cross-entropy/snippets/neg_log_curve.py

Needs only NumPy.
"""
import numpy as np

print("section 3: the loss -log(p) for a probability p on the true class")
for p in [1.00, 0.90, 0.70, 0.50, 0.10, 0.01]:
    loss = abs(-np.log(p))          # abs only turns the -0.0 at p = 1 into 0.0
    print(f"   p = {p:.2f}   -log(p) = {loss:.3f}")
print("   the points marked in the first figure:",
      ", ".join(f"{p} -> {abs(-np.log(p)):.3f}" for p in [0.01, 0.1, 0.5, 1.0]))

print("section 3: the curve charges a low probability far more than a high one")
print(f"   -log(0.01) / -log(0.5) = {np.log(0.01) / np.log(0.5):.2f}")
print(f"   raising p from 0.01 to 0.10 removes {np.log(0.10) - np.log(0.01):.3f} of loss")
print(f"   raising p from 0.90 to 0.99 removes {np.log(0.99) - np.log(0.90):.3f} of loss")
print(f"   a straight-line loss 1 - p would remove {(1 - 0.01) - (1 - 0.10):.2f} and {(1 - 0.90) - (1 - 0.99):.2f}")

print("section 2.1: 300 samples, each with probability 1/3 on its true class")
p64 = np.full(300, 1 / 3)
p32 = p64.astype(np.float32)
print(f"   product of the 300 probabilities in float64: {np.prod(p64):.1e}")
print(f"   product of the 300 probabilities in float32: {np.prod(p32)}")
print(f"   sum of the 300 logs: {np.sum(np.log(p64)):.2f}   mean of -log: {np.mean(-np.log(p64)):.4f}")

print("section 8.1: the loss of a uniform guess over K classes, -log(1/K) = log(K)")
for K in [2, 3, 10, 100, 1000]:
    print(f"   K = {K:<4d}   log(K) = {np.log(K):.3f}")
