"""Post 03: the same call repeated for five layers, with every shape printed.

Run from the series root:  python posts/03-stacking-layers-and-the-forward-pass/snippets/deeper_stack.py
"""
import numpy as np

np.random.seed(0)

# 4 input features, then layers of 6, 5, 3, 4 and 2 neurons.
sizes = [4, 6, 5, 3, 4, 2]
X = np.random.randn(3, sizes[0])                 # a batch of 3 samples

# One row of weights per neuron: layer l has shape (sizes[l], sizes[l - 1]).
w1, w2, w3, w4, w5 = [np.random.randn(m, n) for n, m in zip(sizes[:-1], sizes[1:])]
b1, b2, b3, b4, b5 = [np.random.randn(m) for m in sizes[1:]]

z1 = np.dot(X,  w1.T) + b1
z2 = np.dot(z1, w2.T) + b2
z3 = np.dot(z2, w3.T) + b3
z4 = np.dot(z3, w4.T) + b4
z5 = np.dot(z4, w5.T) + b5    # final output

print("X", X.shape)
for number, w, b, z in [(1, w1, b1, z1), (2, w2, b2, z2), (3, w3, b3, z3), (4, w4, b4, z4), (5, w5, b5, z5)]:
    print(f"layer {number}: W {w.shape}, b {b.shape} -> Z {z.shape}")

# The same chain as a loop: one line of arithmetic, whatever the depth.
weights = [w1, w2, w3, w4, w5]
biases  = [b1, b2, b3, b4, b5]
z = X
for w, b in zip(weights, biases):
    z = np.dot(z, w.T) + b
print("the loop reproduces z5:", np.array_equal(z, z5))

# No activation anywhere, so the five layers are still one layer from 4 inputs to 2 neurons.
w_star, b_star = w1.T, b1
for w, b in zip(weights[1:], biases[1:]):
    w_star = np.dot(w_star, w.T)
    b_star = np.dot(b_star, w.T) + b
collapsed = np.dot(X, w_star) + b_star
print("W_star", w_star.shape, "b_star", b_star.shape)
print("one layer reproduces z5:", np.allclose(collapsed, z5))
print("parameters in the five layers:", sum(w.size + b.size for w, b in zip(weights, biases)))
print("parameters in the equivalent layer:", w_star.size + b_star.size)
