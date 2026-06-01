"""
Synthetic 2-D binary classification data.

A NumPy-only implementation of the classic "two moons" dataset:
two interleaved half-rings, slight Gaussian noise per point, optional
train/test split. Identical output format to sklearn.datasets.make_moons
so you can swap in sklearn's version one-for-one if you prefer.

The dataset is the binary analogue of the spiral dataset used in the
posts: small, 2-D, non-linearly separable, easy to plot.
"""

import numpy as np


def make_moons(n_samples=1000, noise=0.20, seed=0):
    """
    Generate the two-moons dataset.

    Returns:
        X  (n_samples, 2)  float32
        y  (n_samples,)    int64 in {0, 1}
    """
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Outer (upper) moon, class 0: half-circle centred at the origin.
    theta_outer = np.linspace(0, np.pi, n_outer)
    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])

    # Inner (lower) moon, class 1: half-circle shifted to (1, -0.5).
    theta_inner = np.linspace(0, np.pi, n_inner)
    inner = np.column_stack([1.0 - np.cos(theta_inner),
                             0.5 - np.sin(theta_inner)])

    X = np.vstack([outer, inner]).astype(np.float32)
    X += rng.normal(0.0, noise, X.shape).astype(np.float32)

    y = np.hstack([np.zeros(n_outer, dtype=np.int64),
                   np.ones(n_inner, dtype=np.int64)])

    # Shuffle so the order isn't class-sorted.
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def train_test_split(X, y, test_frac=0.2, seed=0):
    """
    Reproducible random split. test_frac of the data becomes the test set.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


if __name__ == "__main__":
    X, y = make_moons(n_samples=1000, noise=0.2, seed=0)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_frac=0.2)
    print(f"full: X={X.shape}  y={y.shape}  class balance={np.bincount(y).tolist()}")
    print(f"train: X={X_tr.shape}  y={y_tr.shape}")
    print(f"test:  X={X_te.shape}  y={y_te.shape}")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
