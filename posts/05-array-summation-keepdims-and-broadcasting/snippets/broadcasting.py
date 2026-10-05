"""Post 05, sections 4 to 7: the broadcasting rules written out by hand and checked against NumPy.

Run from the series root:
    python posts/05-array-summation-keepdims-and-broadcasting/snippets/broadcasting.py

Needs only NumPy. The one random array is drawn from a generator seeded with 0, so the output is
the same on every run.
"""
from itertools import product

import numpy as np


def broadcast_shape(left, right):
    """Return the shape NumPy gives two operands of these shapes, or raise ValueError."""
    # Rule 1: left-pad the shorter shape with 1s until both have the same length.
    ndim = max(len(left), len(right))
    left = (1,) * (ndim - len(left)) + tuple(left)
    right = (1,) * (ndim - len(right)) + tuple(right)
    result = []
    for m, n in zip(left, right):
        # Rule 2: two aligned sizes must be equal, or one of them must be 1.
        if m != n and m != 1 and n != 1:
            raise ValueError(f"sizes {m} and {n} differ and neither is 1")
        # Rule 3: a size of 1 stretches to the other size.
        result.append(n if m == 1 else m)
    return tuple(result)


def by_hand(left, right):
    try:
        return broadcast_shape(left, right)
    except ValueError:
        return "error"


def by_numpy(left, right):
    try:
        return (np.zeros(left) + np.zeros(right)).shape     # rule 4: operate element-wise
    except ValueError:
        return "error"


print("== Sections 4 and 7: the worked compatibility checks, by hand and by NumPy")
pairs = [((3, 3), (3, 1)), ((3, 3), (1, 3)), ((3, 3), (3,)), ((300, 3), (3,)),
         ((3, 1), (3,)), ((3, 3), (2, 3)), ((3, 3), (3, 2)), ((5, 3), (5,))]
for left, right in pairs:
    print(f"{str(left):8s} with {str(right):6s} -> by hand {str(by_hand(left, right)):8s} NumPy {by_numpy(left, right)}")

print("== Section 4: NumPy's own message for the first failing pair")
try:
    np.zeros((3, 3)) + np.zeros((2, 3))
except ValueError as err:
    print("ValueError:", str(err).strip())

print("== Section 7: the hand-written rules against NumPy on every pair of small shapes")
# Every shape with 0 to 3 axes whose sizes are 1, 2 or 3: (), (1,), (2,), ..., (3, 3, 3).
shapes = [shape for ndim in range(4) for shape in product((1, 2, 3), repeat=ndim)]
agree = sum(by_hand(s, t) == by_numpy(s, t) for s in shapes for t in shapes)
errors = sum(by_numpy(s, t) == "error" for s in shapes for t in shapes)
print(f"{len(shapes)} shapes, {len(shapes) ** 2:,d} ordered pairs: the two agree on {agree:,d}; NumPy raises on {errors:,d}")

print("== Section 5: the bias row is broadcast across the batch")
rng = np.random.default_rng(0)
X = rng.standard_normal((300, 2))           # 300 samples, 2 features
W = 0.01 * rng.standard_normal((2, 3))      # 2 inputs, 3 neurons
b = np.array([[0.1, 0.2, 0.3]])             # one bias per neuron

out = np.dot(X, W) + b          # X: (300, 2), W: (2, 3), b: (1, 3)
# np.dot(X, W) has shape (300, 3)
# b has shape (1, 3)
# (300, 3) + (1, 3) broadcasts to (300, 3)
print(np.dot(X, W).shape, b.shape, out.shape)

tiled = np.tile(b, (300, 1))                        # the explicit copy that broadcasting avoids
print(tiled.shape, np.array_equal(out, np.dot(X, W) + tiled))
stretched = np.broadcast_to(b, (300, 3))            # the stretched operand, as a view of b
print(stretched.shape, np.shares_memory(stretched, b))

print("== Section 6: the shape diary on a (3, 4) array, and each result subtracted from the array")
c = np.arange(12).reshape(3, 4)                     # N = 3 rows, F = 4 columns
diary = [("np.sum(c)", np.sum(c)),
         ("np.sum(c, keepdims=True)", np.sum(c, keepdims=True)),
         ("np.sum(c, axis=0)", np.sum(c, axis=0)),
         ("np.sum(c, axis=1)", np.sum(c, axis=1)),
         ("np.sum(c, axis=0, keepdims=True)", np.sum(c, axis=0, keepdims=True)),
         ("np.sum(c, axis=1, keepdims=True)", np.sum(c, axis=1, keepdims=True))]
for call, result in diary:
    shape = np.shape(result)
    print(f"{call:33s} shape {str(shape):7s} c - result: {by_numpy(c.shape, shape)}")
