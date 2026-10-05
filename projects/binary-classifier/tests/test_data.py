"""The two-moons generator, the split, and the dataset checksum."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from binary_classifier.data import dataset_checksum, make_moons, train_test_split

FIXTURE = Path(__file__).parent / "fixtures" / "moons_n8_noise01_seed0.json"


# --- make_moons ----------------------------------------------------------------------------


def test_make_moons_reproduces_the_committed_fixture():
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    X, y = make_moons(
        n_samples=expected["n_samples"], noise=expected["noise"], seed=expected["seed"]
    )
    np.testing.assert_allclose(X, expected["X"], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(y, expected["y"])


def test_make_moons_shapes_types_and_balance():
    X, y = make_moons(n_samples=1000, noise=0.1, seed=0)
    assert X.shape == (1000, 2)
    assert X.dtype == np.float32
    assert y.shape == (1000,)
    assert y.dtype == np.int64
    assert np.bincount(y).tolist() == [500, 500]


def test_make_moons_gives_the_extra_point_of_an_odd_count_to_class_1():
    _, y = make_moons(n_samples=7, noise=0.1, seed=0)
    assert np.bincount(y).tolist() == [3, 4]


def test_make_moons_without_noise_lies_on_the_two_half_circles():
    X, y = make_moons(n_samples=200, noise=0.0, seed=4)
    upper, lower = X[y == 0].astype(np.float64), X[y == 1].astype(np.float64)
    # Class 0: unit circle about the origin, upper half.
    np.testing.assert_allclose(np.hypot(upper[:, 0], upper[:, 1]), 1.0, atol=1e-6)
    assert (upper[:, 1] >= -1e-6).all()
    # Class 1: unit circle about (1, 0.5), lower half.
    np.testing.assert_allclose(np.hypot(lower[:, 0] - 1.0, lower[:, 1] - 0.5), 1.0, atol=1e-6)
    assert (lower[:, 1] <= 0.5 + 1e-6).all()


def test_make_moons_noise_has_the_requested_standard_deviation():
    clean, y_clean = make_moons(n_samples=2000, noise=0.0, seed=3)
    noisy, y_noisy = make_moons(n_samples=2000, noise=0.1, seed=3)
    # The same seed shuffles both the same way, so the difference is the noise alone:
    # 4,000 draws with standard deviation 0.1.
    np.testing.assert_array_equal(y_clean, y_noisy)
    difference = noisy.astype(np.float64) - clean.astype(np.float64)
    assert difference.std() == pytest.approx(0.1, abs=0.005)
    assert abs(difference.mean()) < 0.006


def test_make_moons_is_fixed_by_its_seed():
    first = make_moons(n_samples=50, noise=0.1, seed=11)
    again = make_moons(n_samples=50, noise=0.1, seed=11)
    other = make_moons(n_samples=50, noise=0.1, seed=12)
    np.testing.assert_array_equal(first[0], again[0])
    np.testing.assert_array_equal(first[1], again[1])
    assert not np.array_equal(first[0], other[0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_samples": 1}, "n_samples must be at least 2"),
        ({"n_samples": 2.5}, "n_samples must be an integer"),
        ({"n_samples": True}, "n_samples must be an integer"),
        ({"noise": -0.1}, "noise must be a finite number that is not negative"),
        ({"noise": float("nan")}, "noise must be a finite number"),
        ({"noise": float("inf")}, "noise must be a finite number"),
        ({"seed": -1}, "seed must be between 0 and 4294967295"),
        ({"seed": 2**32}, "seed must be between 0 and 4294967295"),
        ({"seed": 1.5}, "seed must be an integer"),
    ],
)
def test_make_moons_rejects_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_moons(**kwargs)


# --- train_test_split ----------------------------------------------------------------------


def numbered_rows(n):
    """Row i is (2i, 2i + 1) and its label is i mod 2, so every row names itself."""
    X = np.arange(2 * n, dtype=np.float32).reshape(n, 2)
    return X, (np.arange(n) % 2).astype(np.int64)


def test_split_sizes_of_the_documented_run():
    X, y = make_moons(n_samples=1000, noise=0.1, seed=0)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_frac=0.2, seed=0)
    assert X_train.shape == (800, 2)
    assert y_train.shape == (800,)
    assert X_test.shape == (200, 2)
    assert y_test.shape == (200,)


def test_split_is_a_partition_and_keeps_each_label_with_its_row():
    X, y = numbered_rows(50)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_frac=0.2, seed=5)
    assert len(X_test) == 10
    assert len(X_train) == 40
    rows = np.concatenate([X_train[:, 0], X_test[:, 0]]) / 2
    assert sorted(rows.tolist()) == list(range(50))
    np.testing.assert_array_equal(y_train, (X_train[:, 0] / 2) % 2)
    np.testing.assert_array_equal(y_test, (X_test[:, 0] / 2) % 2)


def test_split_is_fixed_by_its_seed():
    X, y = numbered_rows(50)
    first = train_test_split(X, y, test_frac=0.2, seed=5)
    again = train_test_split(X, y, test_frac=0.2, seed=5)
    other = train_test_split(X, y, test_frac=0.2, seed=6)
    np.testing.assert_array_equal(first[2], again[2])
    assert not np.array_equal(first[2], other[2])


@pytest.mark.parametrize(
    ("n", "kwargs", "message"),
    [
        (10, {"test_frac": 0.0}, "test_frac must be strictly between 0 and 1"),
        (10, {"test_frac": 1.0}, "test_frac must be strictly between 0 and 1"),
        (10, {"test_frac": float("nan")}, "test_frac must be strictly between 0 and 1"),
        (2, {"test_frac": 0.2}, "leaves 0 test rows and 2 training rows"),
        (1, {"test_frac": 0.6}, "leaves 1 test rows and 0 training rows"),
        (10, {"seed": -3}, "seed must be between"),
    ],
)
def test_split_rejects_bad_arguments(n, kwargs, message):
    X, y = numbered_rows(n)
    with pytest.raises(ValueError, match=message):
        train_test_split(X, y, **kwargs)


def test_split_rejects_mismatched_lengths():
    X, y = numbered_rows(10)
    with pytest.raises(ValueError, match=r"X has 10 rows and y has 9"):
        train_test_split(X, y[:9])


# --- dataset_checksum ----------------------------------------------------------------------


def test_checksum_is_the_sha256_of_the_float32_points_then_the_int64_labels():
    X = np.array([[0.0, 1.0], [-2.5, 3.25]], dtype=np.float32)
    y = np.array([1, 0], dtype=np.int64)
    expected = hashlib.sha256(X.astype("<f4").tobytes() + y.astype("<i8").tobytes()).hexdigest()
    assert dataset_checksum(X, y) == expected
    assert len(expected) == 64


def test_checksum_ignores_memory_layout_and_sees_every_value():
    X, y = make_moons(n_samples=20, noise=0.1, seed=0)
    reference = dataset_checksum(X, y)
    assert dataset_checksum(np.asfortranarray(X), y.astype(np.int32)) == reference

    moved = X.copy()
    moved[7, 1] = np.nextafter(moved[7, 1], np.float32(np.inf))
    assert dataset_checksum(moved, y) != reference
    flipped = y.copy()
    flipped[0] = 1 - flipped[0]
    assert dataset_checksum(X, flipped) != reference
