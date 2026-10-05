"""MNIST in its original IDX format: find the four files, check them, parse them.

The files are the gzip archives the dataset is distributed as. Nothing here touches the network:
:mod:`mnist_from_scratch.download` fetches the files, and this module refuses to parse a file whose
SHA-256 is not the one recorded in :data:`MNIST_FILES`, so a run always starts from the same bytes.

``load_mnist`` returns

    X_train  (60000, 784) float32 in [0, 1]
    y_train  (60000,)     int64
    X_test   (10000, 784) float32 in [0, 1]
    y_test   (10000,)     int64

in the order of the files, which is the standard split.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = "mnist_cache"
DOWNLOAD_COMMAND = "uv run python scripts/download_mnist.py"

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049


class DataError(Exception):
    """The dataset is missing, damaged, or not the files this project expects."""


@dataclass(frozen=True)
class DatasetFile:
    """One file of the dataset: its name, its size in bytes and the SHA-256 of its bytes."""

    name: str
    size: int
    sha256: str


MNIST_FILES: dict[str, DatasetFile] = {
    "train_images": DatasetFile(
        "train-images-idx3-ubyte.gz",
        9_912_422,
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    ),
    "train_labels": DatasetFile(
        "train-labels-idx1-ubyte.gz",
        28_881,
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    ),
    "test_images": DatasetFile(
        "t10k-images-idx3-ubyte.gz",
        1_648_877,
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    ),
    "test_labels": DatasetFile(
        "t10k-labels-idx1-ubyte.gz",
        4_542,
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    ),
}


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file's bytes, as lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_idx_images(raw: bytes, source: str = "the image file") -> np.ndarray:
    """Parse an uncompressed IDX image file into ``(count, rows * cols)`` float32 in [0, 1]."""
    if len(raw) < 16:
        raise DataError(f"{source} is too short to be an IDX image file ({len(raw)} bytes)")
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != IMAGE_MAGIC:
        raise DataError(
            f"{source} is not an IDX image file: magic number {magic}, expected {IMAGE_MAGIC}"
        )
    expected = count * rows * cols
    if len(raw) - 16 != expected:
        raise DataError(
            f"{source} declares {count} images of {rows} x {cols} pixels ({expected} bytes) but "
            f"holds {len(raw) - 16} bytes of pixels"
        )
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
    images = pixels.reshape(count, rows * cols).astype(np.float32)
    images /= 255.0
    return images


def parse_idx_labels(raw: bytes, source: str = "the label file") -> np.ndarray:
    """Parse an uncompressed IDX label file into ``(count,)`` int64."""
    if len(raw) < 8:
        raise DataError(f"{source} is too short to be an IDX label file ({len(raw)} bytes)")
    magic, count = struct.unpack(">II", raw[:8])
    if magic != LABEL_MAGIC:
        raise DataError(
            f"{source} is not an IDX label file: magic number {magic}, expected {LABEL_MAGIC}"
        )
    if len(raw) - 8 != count:
        raise DataError(f"{source} declares {count} labels but holds {len(raw) - 8}")
    return np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)


def _decompress(path: Path) -> bytes:
    try:
        with gzip.open(path, "rb") as handle:
            return handle.read()
    except (OSError, EOFError, zlib.error) as error:
        raise DataError(f"{path} is not a readable gzip file: {error}") from error


def check_files(data_dir: Path, files: dict[str, DatasetFile]) -> dict[str, Path]:
    """Return the path of every dataset file after checking that it is present and unaltered."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DataError(
            f"the data directory {data_dir} does not exist. Fetch the dataset first:\n"
            f"    {DOWNLOAD_COMMAND}"
        )
    missing = [spec.name for spec in files.values() if not (data_dir / spec.name).is_file()]
    if missing:
        raise DataError(
            f"{data_dir} is missing {', '.join(missing)}. Fetch the dataset first:\n"
            f"    {DOWNLOAD_COMMAND}"
        )
    paths = {}
    for key, spec in files.items():
        path = data_dir / spec.name
        found = sha256_of(path)
        if found != spec.sha256:
            raise DataError(
                f"{path} is not the expected file: its SHA-256 is {found}, expected "
                f"{spec.sha256}. Delete it and fetch it again:\n    {DOWNLOAD_COMMAND}"
            )
        paths[key] = path
    return paths


def _load_split(paths: dict[str, Path], split: str) -> tuple[np.ndarray, np.ndarray]:
    images_path, labels_path = paths[f"{split}_images"], paths[f"{split}_labels"]
    X = parse_idx_images(_decompress(images_path), str(images_path))
    y = parse_idx_labels(_decompress(labels_path), str(labels_path))
    if len(X) != len(y):
        raise DataError(
            f"{images_path} holds {len(X)} images but {labels_path} holds {len(y)} labels"
        )
    return X, y


def load_test_set(
    data_dir: str | Path = DEFAULT_DATA_DIR, files: dict[str, DatasetFile] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_test, y_test)``. All four files are still checked, only two are parsed."""
    paths = check_files(Path(data_dir), MNIST_FILES if files is None else files)
    return _load_split(paths, "test")


def load_mnist(
    data_dir: str | Path = DEFAULT_DATA_DIR, files: dict[str, DatasetFile] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X_train, y_train, X_test, y_test)`` from the four checked files in ``data_dir``."""
    paths = check_files(Path(data_dir), MNIST_FILES if files is None else files)
    X_train, y_train = _load_split(paths, "train")
    X_test, y_test = _load_split(paths, "test")
    if X_train.shape[1] != X_test.shape[1]:
        raise DataError(
            f"training images have {X_train.shape[1]} pixels each but test images have "
            f"{X_test.shape[1]}"
        )
    return X_train, y_train, X_test, y_test
