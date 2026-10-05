"""The two-moons dataset, generated in NumPy, and a seeded train and test split.

Two interleaved half-circles of unit radius in the plane, one per class, with
Gaussian noise added to every coordinate. The noiseless construction is the
one scikit-learn's make_moons uses, so the picture is the familiar one; the
noise comes from NumPy's default_rng, so the values are this project's own.
Nothing is downloaded and nothing is read from disk.
"""

import hashlib
import math

import numpy as np


def _check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be between 0 and 4294967295, got {seed}")


def make_moons(n_samples=1000, noise=0.10, seed=0):
    """Generate the two-moons dataset.

    Class 0 is the upper half-circle centred at the origin; class 1 is the
    lower half-circle centred at (1, 0.5). Returns the points in shuffled
    order:

        X  (n_samples, 2)  float32
        y  (n_samples,)    int64, values in {0, 1}
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError(f"n_samples must be an integer, got {n_samples!r}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2 (one point per class), got {n_samples}")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError(f"noise must be a finite number that is not negative, got {noise}")
    _check_seed(seed)

    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = np.linspace(0, np.pi, n_outer)
    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])

    theta_inner = np.linspace(0, np.pi, n_inner)
    inner = np.column_stack([1.0 - np.cos(theta_inner), 0.5 - np.sin(theta_inner)])

    X = np.vstack([outer, inner]).astype(np.float32)
    X += rng.normal(0.0, noise, X.shape).astype(np.float32)

    y = np.hstack([np.zeros(n_outer, dtype=np.int64), np.ones(n_inner, dtype=np.int64)])

    # Shuffle, so that the order is not sorted by class.
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def train_test_split(X, y, test_frac=0.2, seed=0):
    """Split X and y at random, reproducibly; test_frac of the rows form the test set.

    Returns X_train, y_train, X_test, y_test.
    """
    n = len(X)
    if len(y) != n:
        raise ValueError(f"X has {n} rows and y has {len(y)}; they must match")
    if not math.isfinite(test_frac) or not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be strictly between 0 and 1, got {test_frac}")
    _check_seed(seed)
    n_test = round(n * test_frac)
    if n_test < 1 or n_test >= n:
        raise ValueError(
            f"a split of {n} rows with test_frac {test_frac} leaves {n_test} test rows and "
            f"{n - n_test} training rows; both sets need at least one row"
        )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def dataset_checksum(X, y):
    """SHA-256 of the points and labels, as a hex string.

    The digest covers X as little-endian float32 and y as little-endian int64,
    row by row. Two machines that print the same digest trained on the same
    numbers, bit for bit.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(X, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(y, dtype="<i8").tobytes())
    return digest.hexdigest()
