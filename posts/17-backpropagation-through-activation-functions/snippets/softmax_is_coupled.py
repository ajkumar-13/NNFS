"""Softmax is not element-wise: its Jacobian is full, and the one-line backward gets it wrong.

Run from the series root:
    python posts/17-backpropagation-through-activation-functions/snippets/softmax_is_coupled.py
"""
import numpy as np


def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)


def weighted_sum_loss(z, dvalues):
    """A stand-in loss whose gradient with respect to a = softmax(z) is exactly dvalues."""
    return float(np.sum(dvalues * softmax(z)))


np.set_printoptions(precision=6, suppress=True, floatmode="fixed")

z = np.array([1.0, -2.0, 3.0])
dvalues = np.array([5.0, 6.0, 7.0])
a = softmax(z)
print("== Section 4: one sample through softmax")
print("z =", z)
print("a =", a, " sum =", f"{a.sum():.6f}")

print()
print("== Raising z_1 alone by 0.1 moves every output")
moved = softmax(z + np.array([0.1, 0.0, 0.0]))
print("a after  =", moved)
print("change   =", moved - a)

print()
print("== The Jacobian, entry (k, j) = d a_k / d z_j")
jacobian = np.diagflat(a) - np.outer(a, a)      # a_k (1 - a_k) on the diagonal, -a_k a_j off it
print(jacobian)
print("non-zero entries:", int(np.count_nonzero(jacobian)), "of", jacobian.size)
print("every column sums to zero:", bool(np.allclose(jacobian.sum(axis=0), 0.0)))

print()
print("== Two candidate backward passes for dvalues =", dvalues)
full = dvalues @ jacobian                       # sum over k of dL/da_k * d a_k / d z_j
diagonal_only = dvalues * a * (1 - a)           # the element-wise line, which keeps k = j only
h = 1e-5
numerical = np.zeros_like(z)
for j in range(z.size):
    step = np.zeros_like(z)
    step[j] = h
    numerical[j] = (weighted_sum_loss(z + step, dvalues) - weighted_sum_loss(z - step, dvalues)) / (2 * h)
print("full Jacobian product :", full)
print("central difference    :", numerical)
print("element-wise line     :", diagonal_only)
print("largest gap, full Jacobian product against the central difference:", f"{np.abs(full - numerical).max():.1e}")
print("largest gap, element-wise line against the central difference    :", f"{np.abs(diagonal_only - numerical).max():.3f}")
