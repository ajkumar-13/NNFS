"""Write the tiny dataset that the tests read, in MNIST's own file format.

    uv run python scripts/make_test_fixture.py

Four gzip files go to tests/fixtures/, named as the real files are: 12 training images and 6 test
images of 28 x 28 pixels, with their labels. The content is synthetic and follows a formula, so the
tests can state the expected value of any pixel:

    pixel(image k, row r, column c) = (37 k + 7 r + 13 c) mod 256      test images use k + 100
    training label k = k mod 10                                        test label k = 3 k mod 10

The gzip header carries no file name and no timestamp, so the same bytes are written every time.
The script prints each file's size and SHA-256; tests/conftest.py records them, and the tests fail
if the files and the record ever disagree.
"""

import gzip
import hashlib
import io
import struct
import sys
from pathlib import Path

import numpy as np

ROWS = COLS = 28
N_TRAIN = 12
N_TEST = 6
TEST_OFFSET = 100


def images(count: int, offset: int) -> np.ndarray:
    k = np.arange(count).reshape(count, 1, 1) + offset
    r = np.arange(ROWS).reshape(1, ROWS, 1)
    c = np.arange(COLS).reshape(1, 1, COLS)
    return ((37 * k + 7 * r + 13 * c) % 256).astype(np.uint8)


def idx_images(pixels: np.ndarray) -> bytes:
    return struct.pack(">IIII", 2051, len(pixels), ROWS, COLS) + pixels.tobytes()


def idx_labels(labels: np.ndarray) -> bytes:
    return struct.pack(">II", 2049, len(labels)) + labels.astype(np.uint8).tobytes()


def gzip_bytes(raw: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


def main() -> int:
    target = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    target.mkdir(parents=True, exist_ok=True)
    files = {
        "train-images-idx3-ubyte.gz": idx_images(images(N_TRAIN, 0)),
        "train-labels-idx1-ubyte.gz": idx_labels(np.arange(N_TRAIN) % 10),
        "t10k-images-idx3-ubyte.gz": idx_images(images(N_TEST, TEST_OFFSET)),
        "t10k-labels-idx1-ubyte.gz": idx_labels((3 * np.arange(N_TEST)) % 10),
    }
    for name, raw in files.items():
        packed = gzip_bytes(raw)
        (target / name).write_bytes(packed)
        print(f"{name}  {len(packed)} bytes  sha256 {hashlib.sha256(packed).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
