"""
MNIST loading. Two backends:

- `load_mnist(backend="sklearn")` uses sklearn.datasets.fetch_openml.
- `load_mnist(backend="manual")` downloads the four idx-ubyte.gz files
  directly from the ossci-datasets S3 mirror and parses them with stdlib.

Both return identical NumPy arrays:

    X_train  (60000, 784) float32 in [0, 1]
    y_train  (60000,)     int64
    X_test   (10000, 784) float32 in [0, 1]
    y_test   (10000,)     int64
"""

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Backend A: sklearn (easiest)
# -----------------------------------------------------------------------------
def _load_via_sklearn():
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------------
# Backend B: manual download (no sklearn dependency)
# -----------------------------------------------------------------------------
_MNIST_URLS = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def _load_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def _load_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def _load_via_manual(cache_dir="mnist_cache"):
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    paths = {}
    for name, fname in _MNIST_URLS.items():
        dest = cache / fname
        if not dest.exists():
            print(f"  downloading {fname}...")
            urllib.request.urlretrieve(_MIRROR + fname, dest)
        paths[name] = dest

    return (
        _load_images(paths["train_images"]),
        _load_labels(paths["train_labels"]),
        _load_images(paths["test_images"]),
        _load_labels(paths["test_labels"]),
    )


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
def load_mnist(backend="sklearn"):
    """Return (X_train, y_train, X_test, y_test) with sklearn or manual backend."""
    if backend == "sklearn":
        return _load_via_sklearn()
    if backend == "manual":
        return _load_via_manual()
    raise ValueError(f"unknown backend: {backend!r}")


if __name__ == "__main__":
    # Smoke test: load and report shapes.
    X_train, y_train, X_test, y_test = load_mnist(backend="sklearn")
    print(f"X_train: {X_train.shape}  dtype={X_train.dtype}  range=[{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"y_train: {y_train.shape}  dtype={y_train.dtype}  classes={sorted(np.unique(y_train).tolist())}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
