"""Helpers shared by the test modules: IDX bytes, finite differences, the fixture record."""

import gzip
import struct
from pathlib import Path

import numpy as np

from fashion_mnist.data import DatasetFile, sha256_of

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Sizes and checksums printed by scripts/make_test_fixture.py for the committed files.
FIXTURE_FILES = {
    "train_images": DatasetFile(
        "train-images-idx3-ubyte.gz",
        1316,
        "001002aad074432be084b207889ec55a7c2ec55fc2bd5303be3ffbc9926c4234",
    ),
    "train_labels": DatasetFile(
        "train-labels-idx1-ubyte.gz",
        40,
        "0c275d1a67ea68094f3b21ffdd43f3dde83d856580e92b9e40a8830c582fce3f",
    ),
    "test_images": DatasetFile(
        "t10k-images-idx3-ubyte.gz",
        943,
        "4828b917678dd19bfada08eb5509561e9776e7d26aefb35f65816755db5a781c",
    ),
    "test_labels": DatasetFile(
        "t10k-labels-idx1-ubyte.gz",
        34,
        "7f0da28255fef4962690bf0612320f233b2563dd057162bc84e6bc5316da58be",
    ),
}
FIXTURE_TRAIN = 12
FIXTURE_TEST = 6


def fixture_pixel(image: int, row: int, col: int, test: bool = False) -> int:
    """The byte the fixture holds at one pixel (see scripts/make_test_fixture.py)."""
    return (37 * (image + (100 if test else 0)) + 7 * row + 13 * col) % 256


def numerical_gradient(function, array, step=1e-6):
    """Central differences of a scalar ``function()`` with respect to every entry of ``array``."""
    gradient = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + step
        upper = function()
        array[index] = original - step
        lower = function()
        array[index] = original
        gradient[index] = (upper - lower) / (2 * step)
    return gradient


def generator_state():
    """The full state of NumPy's global generator, in a form that compares with ``==``."""
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return name, keys.tobytes(), position, has_gauss, cached_gaussian


def image_bytes(pixels, magic=2051, count=None, rows=None, cols=None):
    """An IDX image file for a ``(count, rows, cols)`` array; the header fields can be falsified."""
    pixels = np.asarray(pixels, dtype=np.uint8)
    header = struct.pack(
        ">IIII",
        magic,
        pixels.shape[0] if count is None else count,
        pixels.shape[1] if rows is None else rows,
        pixels.shape[2] if cols is None else cols,
    )
    return header + pixels.tobytes()


def label_bytes(labels, magic=2049, count=None):
    """An IDX label file for a vector of labels; the header fields can be falsified."""
    labels = np.asarray(labels, dtype=np.uint8)
    return struct.pack(">II", magic, len(labels) if count is None else count) + labels.tobytes()


def write_dataset(directory, train_images, train_labels, test_images, test_labels):
    """Write four gzip IDX files under the real names and return their file record."""
    contents = {
        "train_images": ("train-images-idx3-ubyte.gz", image_bytes(train_images)),
        "train_labels": ("train-labels-idx1-ubyte.gz", label_bytes(train_labels)),
        "test_images": ("t10k-images-idx3-ubyte.gz", image_bytes(test_images)),
        "test_labels": ("t10k-labels-idx1-ubyte.gz", label_bytes(test_labels)),
    }
    files = {}
    for key, (name, raw) in contents.items():
        path = Path(directory) / name
        path.write_bytes(gzip.compress(raw, mtime=0))
        files[key] = DatasetFile(name, path.stat().st_size, sha256_of(path))
    return files
