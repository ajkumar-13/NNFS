"""Post 19, sections 2 and 3: the softmax Jacobian, and what cross-entropy does to it.

Run from the series root:
    python posts/19-softmax-derivatives-and-the-combined-backward-pass/snippets/softmax_jacobian.py

Needs only NumPy.
"""
import numpy as np


def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)


np.set_printoptions(precision=6, suppress=True, floatmode="fixed")
h = 1e-5

print("== Section 2.1: the slope of exp(x) is exp(x)")
for x in (0.0, 1.0, 2.5):
    slope = (np.exp(x + h) - np.exp(x - h)) / (2 * h)
    print(f"x = {x:.1f}  central difference {slope:.6f}  exp(x) {np.exp(x):.6f}")

print()
print("== Section 2.3: the Jacobian of one sample, entry (k, j) = d y_hat_k / d z_j")
y_hat = np.array([0.7, 0.2, 0.1])
z = np.log(y_hat)                               # logits whose softmax is exactly y_hat
print("softmax(z) =", softmax(z))
jacobian = np.diagflat(y_hat) - np.outer(y_hat, y_hat)
print(jacobian)

measured = np.zeros((3, 3))
for j in range(3):
    step = np.zeros(3)
    step[j] = h
    measured[:, j] = (softmax(z + step) - softmax(z - step)) / (2 * h)
print("largest gap to a central difference:", f"{np.abs(jacobian - measured).max():.1e}")
print("column sums:", jacobian.sum(axis=0) + 0.0, " symmetric:", bool(np.allclose(jacobian, jacobian.T)))

print()
print("== Section 3: the cross-entropy gradient times the Jacobian, true class 0")
y = np.array([1.0, 0.0, 0.0])
dvalues = -y / y_hat                            # post 18: the gradient with respect to y_hat
print("dvalues           =", dvalues + 0.0)
print("dvalues @ J       =", dvalues @ jacobian)
print("y_hat - y         =", y_hat - y)

print()
print("== Section 3.1: the loss written in the logits")
loss_from_probability = -np.log(softmax(z)[0])
loss_from_logits = -z[0] + np.log(np.sum(np.exp(z)))
print(f"-log(y_hat_t) = {loss_from_probability:.6f}   -z_t + log(sum(exp(z))) = {loss_from_logits:.6f}")

print()
print("== Section 3.2: numbers computed per sample by each route")
for classes in (3, 1000, 50000):
    print(f"K = {classes:>6,}  Jacobian entries {classes ** 2:>13,}  combined entries {classes:>6,}")
