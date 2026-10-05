"""Post 04, section 2: generate the spiral dataset and look at what comes back.

Run from the series root:
    python posts/04-dense-layer-class-and-spiral-data/snippets/spiral_dataset.py

Needs NumPy and the nnfs helper package (pip install nnfs).
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()                                 # seed 0, float32 arrays, a patched np.dot

X, y = spiral_data(samples=100, classes=3)
# X.shape = (300, 2)   : 100 samples per class, 3 classes, two coordinates per sample
# y.shape = (300,)     : class label (0, 1, or 2) for each row

print("X:", X.shape, X.dtype)
print("y:", y.shape, y.dtype)
print("samples per class:", np.bincount(y))
print("first three rows of X:")
print(X[:3])
print("labels of rows 98 to 101:", y[98:102])

# Each class is one arm: it starts at the origin and ends on the unit circle.
radius = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
for k in range(3):
    arm = radius[y == k]
    print(f"class {k}: distance from the origin runs from {arm[0]:.2f} to {arm[-1]:.2f}")
print(f"smallest and largest coordinate: {X.min():.2f} and {X.max():.2f}")

# The shape follows the arguments: samples * classes rows, always two columns.
for samples, classes in [(10, 3), (1000, 3), (100, 5)]:
    X_other, y_other = spiral_data(samples=samples, classes=classes)
    print(f"spiral_data(samples={samples}, classes={classes}): X {X_other.shape}, y {y_other.shape}")
