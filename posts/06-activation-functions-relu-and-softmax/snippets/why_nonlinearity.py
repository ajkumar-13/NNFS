"""Post 06, section 1: one small network run twice, without and with a ReLU between its layers.

Run from the series root:
    python posts/06-activation-functions-relu-and-softmax/snippets/why_nonlinearity.py

Needs NumPy only. Every number is an integer or a half, so the arithmetic is exact.
"""
import numpy as np

# One input feature, three hidden neurons, one output neuron.
# Weights are stored as (n_inputs, n_neurons), the layout of post 04.
W1 = np.array([[1.0, 1.0, 1.0]])            # (1, 3)
b1 = np.array([[0.0, -1.0, -2.0]])          # (1, 3)
W2 = np.array([[1.0], [-2.0], [2.0]])       # (3, 1)
b2 = np.array([[0.0]])                      # (1, 1)

X = np.arange(-1.0, 3.5, 0.5).reshape(-1, 1)    # nine inputs from -1 to 3, shape (9, 1)

# Without an activation: two dense layers, one after the other.
Z1        = np.dot(X, W1) + b1
no_activ  = np.dot(Z1, W2) + b2

# The single layer the two collapse into.
W_star    = np.dot(W1, W2)                  # (1, 1)
b_star    = np.dot(b1, W2) + b2             # (1, 1)
one_layer = np.dot(X, W_star) + b_star

# With ReLU between the same two layers.
A1        = np.maximum(0, Z1)
with_relu = np.dot(A1, W2) + b2

print("W_star =", W_star.ravel(), " b_star =", b_star.ravel())
print("two layers without an activation equal the one layer:", np.array_equal(no_activ, one_layer))
print()
print("    x   no activation   one layer   with ReLU")
for x, a, b, c in zip(X.ravel(), no_activ.ravel(), one_layer.ravel(), with_relu.ravel()):
    print(f"{x:5.1f}   {a:13.1f}   {b:9.1f}   {c:9.1f}")

# Slope of each version between neighbouring inputs: a single value means a straight line.
step = 0.5
print()
print("slopes without an activation:", np.unique(np.diff(no_activ.ravel()) / step))
print("slopes with ReLU:            ", np.unique(np.diff(with_relu.ravel()) / step))

# A straight line sends the midpoint of two inputs to the midpoint of their outputs.
f = with_relu.ravel()                       # f[2], f[4], f[6] are the outputs at x = 0, 1, 2
print("with ReLU: f(0) =", f[2], " f(2) =", f[6], " their midpoint =", (f[2] + f[6]) / 2, " but f(1) =", f[4])
