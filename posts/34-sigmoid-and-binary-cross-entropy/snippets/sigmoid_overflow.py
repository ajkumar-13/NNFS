"""Post 34, section 2.1: what the one-line sigmoid does on large logits, and the form that avoids it.

Run from the series root:
    python posts/34-sigmoid-and-binary-cross-entropy/snippets/sigmoid_overflow.py

Float64 unless a line says float32; no nnfs.init(). Needs only NumPy. Takes about a second.
"""
import warnings

import numpy as np

from binary_classes import Activation_Sigmoid


def sigmoid_naive(z):
    return 1.0 / (1.0 + np.exp(-z))          # the forward of post 17


def sigmoid_other(z):
    return np.exp(z) / (1.0 + np.exp(z))     # the same function, written the other way


def sigmoid_stable(z):
    activation = Activation_Sigmoid()
    activation.forward(z)
    return activation.output


def run(function, z):
    """The values, and the distinct warnings NumPy raised on the way."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = function(z)
    return values, sorted({str(w.message) for w in caught})


z = np.array([-1000.0, -720.0, -40.0, 0.0, 40.0, 720.0, 1000.0])
print("== Three ways to write the sigmoid, float64")
print("z      ", "".join(f"{v:>12g}" for v in z))
for name, function in (("naive", sigmoid_naive), ("other", sigmoid_other), ("stable", sigmoid_stable)):
    values, messages = run(function, z)
    print(f"{name:7}", "".join(f"{v:>12.4g}" for v in values), "  warnings:", messages if messages else "none")

print()
print("== Where the overflow starts")
for dtype in (np.float64, np.float32):
    limit = np.log(np.finfo(dtype).max)
    print(f"{dtype.__name__}: np.exp overflows above {limit:.2f}, so the naive form warns for z below {-limit:.2f}")

print()
print("== What the naive form loses: the range where it returns 0 and the stable form does not")
z_grid = -np.arange(700.0, 760.0, 1.0)
naive, _ = run(sigmoid_naive, z_grid)
stable, _ = run(sigmoid_stable, z_grid)
lost = (naive == 0) & (stable > 0)
print(f"whole-number z from -700 to -759 with naive == 0 and stable > 0: {z_grid[lost].max():.0f} to {z_grid[lost].min():.0f}")
print(f"largest stable value among them: {stable[lost].max():.1e}")
z_wide = np.arange(-700.0, 700.5, 0.5)
print(f"largest |naive - stable| over z from -700 to 700 in steps of 0.5: "
      f"{np.abs(run(sigmoid_naive, z_wide)[0] - sigmoid_stable(z_wide)).max():.1e}")

print()
print("== With overflow raised as an error, as np.seterr(over='raise') does")
with np.errstate(over="raise"):
    for name, function in (("naive", sigmoid_naive), ("stable", sigmoid_stable)):
        try:
            function(np.array([-720.0, 3.0]))
            print(f"{name:7} returns")
        except FloatingPointError as error:
            print(f"{name:7} FloatingPointError: {error}")

print()
print("== Saturation is not symmetric, and it depends on the dtype of the logits")
for dtype in (np.float64, np.float32):
    up = 1
    while sigmoid_stable(np.array([up], dtype=dtype))[0] < 1.0:
        up += 1
    down = 1
    while sigmoid_stable(np.array([-down], dtype=dtype))[0] > 0.0:
        down += 1
    print(f"{dtype.__name__} logits: smallest whole z with sigma(z) == 1: {up}    "
          f"smallest whole z with sigma(-z) == 0: {down}    "
          f"sigma(-{up}) = {sigmoid_stable(np.array([-up], dtype=dtype))[0]:.3e}, still far from 0")
