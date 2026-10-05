"""The IDX parser and the loader, on hand-made bytes and on the tiny committed fixture."""

import gzip
import re
import shutil

import numpy as np
import pytest

from fashion_mnist import data
from fashion_mnist.data import (
    DataError,
    DatasetFile,
    check_files,
    load_fashion_mnist,
    load_test_set,
    parse_idx_images,
    parse_idx_labels,
    sha256_of,
)
from helpers import (
    FIXTURE_FILES,
    FIXTURE_TEST,
    FIXTURE_TRAIN,
    fixture_pixel,
    image_bytes,
    label_bytes,
)


def copy_fixture(fixture_dir, target):
    for spec in FIXTURE_FILES.values():
        shutil.copy(fixture_dir / spec.name, target / spec.name)
    return target


# ----------------------------------------------------------------------------------- the parser


def test_parse_images_flattens_rows_and_scales_bytes_to_the_unit_interval():
    raw = image_bytes([[[0, 51], [102, 255]], [[255, 0], [0, 153]]])
    images = parse_idx_images(raw)
    assert images.shape == (2, 4)
    assert images.dtype == np.float32
    expected = np.array([[0, 51, 102, 255], [255, 0, 0, 153]], dtype=np.float32) / 255.0
    assert np.array_equal(images, expected)
    assert images[0, 0] == 0.0
    assert images[0, 3] == 1.0
    assert images[0, 1] == pytest.approx(0.2)


def test_parse_labels_returns_int64_class_indices():
    labels = parse_idx_labels(label_bytes([7, 0, 9, 3]))
    assert labels.dtype == np.int64
    assert labels.tolist() == [7, 0, 9, 3]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\x00\x00\x08", "too short to be an IDX image file"),
        (image_bytes(np.zeros((1, 2, 2)), magic=2049), "magic number 2049, expected 2051"),
        (image_bytes(np.zeros((1, 2, 2)), count=3), "declares 3 images of 2 x 2 pixels"),
        (image_bytes(np.zeros((2, 2, 2)))[:-1], "holds 7 bytes of pixels"),
    ],
)
def test_parse_images_rejects_damaged_files(raw, message):
    with pytest.raises(DataError, match=message):
        parse_idx_images(raw, "broken.gz")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\x00\x00", "too short to be an IDX label file"),
        (label_bytes([1, 2], magic=2051), "magic number 2051, expected 2049"),
        (label_bytes([1, 2], count=5), "declares 5 labels but holds 2"),
    ],
)
def test_parse_labels_rejects_damaged_files(raw, message):
    with pytest.raises(DataError, match=message):
        parse_idx_labels(raw, "broken.gz")


# ------------------------------------------------------------------------- the committed fixture


def test_the_committed_fixture_is_the_recorded_one(fixture_dir):
    for spec in FIXTURE_FILES.values():
        path = fixture_dir / spec.name
        assert path.stat().st_size == spec.size
        assert sha256_of(path) == spec.sha256


def test_load_fashion_mnist_on_the_fixture(fixture_dir, fixture_files):
    X_train, y_train, X_test, y_test = load_fashion_mnist(fixture_dir, fixture_files)

    assert X_train.shape == (FIXTURE_TRAIN, 784)
    assert X_test.shape == (FIXTURE_TEST, 784)
    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32
    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0

    assert y_train.dtype == np.int64
    assert y_train.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
    assert y_test.tolist() == [0, 3, 6, 9, 2, 5]

    for image, row, col in [(0, 0, 0), (0, 27, 27), (5, 3, 9), (11, 14, 20)]:
        expected = np.float32(fixture_pixel(image, row, col)) / np.float32(255.0)
        assert X_train[image, row * 28 + col] == expected
    for image, row, col in [(0, 0, 0), (5, 27, 0)]:
        expected = np.float32(fixture_pixel(image, row, col, test=True)) / np.float32(255.0)
        assert X_test[image, row * 28 + col] == expected


def test_load_test_set_returns_the_test_half_only(fixture_dir, fixture_files):
    X_test, y_test = load_test_set(fixture_dir, fixture_files)
    _, _, X_all, y_all = load_fashion_mnist(fixture_dir, fixture_files)
    assert np.array_equal(X_test, X_all)
    assert np.array_equal(y_test, y_all)


def test_load_uses_the_real_file_list_by_default(as_fashion_mnist):
    X_train, y_train, X_test, y_test = load_fashion_mnist(as_fashion_mnist)
    assert (len(X_train), len(y_train), len(X_test), len(y_test)) == (12, 12, 6, 6)
    assert len(load_test_set(as_fashion_mnist)[0]) == 6


# ------------------------------------------------------------------------- the expected dataset


def test_the_real_dataset_is_pinned_by_four_sha256_checksums():
    assert [spec.name for spec in data.FASHION_MNIST_FILES.values()] == [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for spec in data.FASHION_MNIST_FILES.values():
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        assert spec.size > 0
    assert sum(spec.size for spec in data.FASHION_MNIST_FILES.values()) == 30_878_645


def test_the_ten_classes_are_named_in_label_order():
    assert data.CLASS_NAMES == (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    )


# --------------------------------------------------------------------------------- the failures


def test_a_missing_directory_names_the_download_command(tmp_path):
    with pytest.raises(DataError, match=r"does not exist(.|\n)*scripts/download_fashion_mnist\.py"):
        load_fashion_mnist(tmp_path / "nowhere")


def test_a_missing_file_is_named_with_the_download_command(tmp_path, fixture_dir, fixture_files):
    copy_fixture(fixture_dir, tmp_path)
    (tmp_path / "t10k-labels-idx1-ubyte.gz").unlink()
    with pytest.raises(
        DataError, match=r"is missing t10k-labels-idx1-ubyte\.gz(.|\n)*download_fashion_mnist\.py"
    ):
        load_fashion_mnist(tmp_path, fixture_files)


def test_an_altered_file_fails_its_checksum(tmp_path, fixture_dir, fixture_files):
    copy_fixture(fixture_dir, tmp_path)
    path = tmp_path / "train-labels-idx1-ubyte.gz"
    path.write_bytes(gzip.compress(label_bytes([9] * 12), mtime=0))
    with pytest.raises(DataError, match="is not the expected file: its SHA-256 is"):
        check_files(tmp_path, fixture_files)


def test_a_file_that_is_not_gzip_is_reported(tmp_path, fixture_dir, fixture_files):
    copy_fixture(fixture_dir, tmp_path)
    path = tmp_path / "train-images-idx3-ubyte.gz"
    path.write_bytes(b"plain text, not gzip")
    fixture_files["train_images"] = DatasetFile(path.name, path.stat().st_size, sha256_of(path))
    with pytest.raises(DataError, match="is not a readable gzip file"):
        load_fashion_mnist(tmp_path, fixture_files)


def test_a_truncated_gzip_file_is_reported(tmp_path, fixture_dir, fixture_files):
    copy_fixture(fixture_dir, tmp_path)
    path = tmp_path / "train-images-idx3-ubyte.gz"
    path.write_bytes(path.read_bytes()[:200])
    fixture_files["train_images"] = DatasetFile(path.name, path.stat().st_size, sha256_of(path))
    with pytest.raises(DataError, match="is not a readable gzip file"):
        load_fashion_mnist(tmp_path, fixture_files)


def test_images_and_labels_must_agree_in_number(tmp_path, fixture_dir, fixture_files):
    copy_fixture(fixture_dir, tmp_path)
    path = tmp_path / "train-labels-idx1-ubyte.gz"
    path.write_bytes(gzip.compress(label_bytes([1, 2, 3]), mtime=0))
    fixture_files["train_labels"] = DatasetFile(path.name, path.stat().st_size, sha256_of(path))
    with pytest.raises(DataError, match=r"holds 12 images but .* holds 3 labels"):
        load_fashion_mnist(tmp_path, fixture_files)


def test_training_and_test_images_must_have_one_size(tmp_path, fixture_dir, fixture_files):
    copy_fixture(fixture_dir, tmp_path)
    path = tmp_path / "t10k-images-idx3-ubyte.gz"
    path.write_bytes(gzip.compress(image_bytes(np.zeros((6, 4, 4))), mtime=0))
    fixture_files["test_images"] = DatasetFile(path.name, path.stat().st_size, sha256_of(path))
    with pytest.raises(DataError, match="784 pixels each but test images have 16"):
        load_fashion_mnist(tmp_path, fixture_files)
