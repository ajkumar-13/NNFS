"""Weights files: the round trip, and every way a file can fail to be one."""

import os
import pickle

import numpy as np
import pytest

from binary_classifier.model import (
    Network,
    WeightsError,
    load_weights,
    npz_path,
    output_path,
    save_weights,
)

CONFIG = {"noise": 0.1, "n_samples": 10, "seed": 3, "epochs": 5}


def make_split():
    X_train = np.array([[0.0, 1.0], [1.0, -0.5], [0.5, 0.5], [-1.0, 0.0]], dtype=np.float32)
    y_train = np.array([0, 1, 0, 0], dtype=np.int64)
    X_test = np.array([[2.0, 0.5], [0.0, 0.9]], dtype=np.float32)
    y_test = np.array([1, 0], dtype=np.int64)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def saved(tmp_path):
    """A valid weights file under tmp_path: (path, model)."""
    np.random.seed(0)
    model = Network(init="xavier")
    path = save_weights(tmp_path / "model.npz", model, make_split(), CONFIG)
    return path, model


def rewrite(path, **changes):
    """Rewrite the archive at path with some arrays replaced, or removed when the value is None."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, value in changes.items():
        if value is None:
            del arrays[key]
        else:
            arrays[key] = value
    np.savez(path, **arrays)


# --- the round trip ------------------------------------------------------------------------


def test_round_trip_restores_weights_split_and_configuration(saved):
    path, model = saved
    loaded, split, config = load_weights(path)
    for original, restored in zip(model.dense_layers, loaded.dense_layers, strict=True):
        np.testing.assert_array_equal(restored.weights, original.weights)
        np.testing.assert_array_equal(restored.biases, original.biases)
    for original, restored in zip(make_split(), split, strict=True):
        np.testing.assert_array_equal(restored, original)
        assert restored.dtype == original.dtype
    assert config == {"init": "xavier", **CONFIG}
    assert loaded.init == "xavier"

    X = np.array([[0.2, 0.1], [1.5, -0.3]], dtype=np.float32)
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))


def test_weights_holds_no_pickled_objects(saved):
    path, _ = saved
    with np.load(path, allow_pickle=False) as archive:
        kinds = {key: archive[key].dtype.kind for key in archive.files}
    assert set(kinds.values()) <= {"f", "i", "U"}
    assert len(kinds) == 16


def test_loading_accepts_a_string_path(saved):
    path, _ = saved
    _, _, config = load_weights(str(path))
    assert config["seed"] == 3


# --- names and locations -------------------------------------------------------------------


def test_missing_file_says_how_to_train_one(tmp_path):
    with pytest.raises(
        WeightsError,
        match=r"no weights file at .*absent.npz.* Train one with: "
        "uv run python -m binary_classifier.train",
    ):
        load_weights(tmp_path / "absent.npz")


def test_pickle_names_are_refused_for_reading_and_writing(tmp_path):
    old = tmp_path / "moons_weights.pkl"
    old.write_bytes(pickle.dumps({"dense1": None}))
    with pytest.raises(WeightsError, match=r"pickle file.*no longer read or written.*.npz"):
        load_weights(old)
    with pytest.raises(WeightsError, match=r"pickle file"):
        save_weights(old, Network(), make_split(), CONFIG)


def test_other_suffixes_are_refused():
    with pytest.raises(WeightsError, match=r"the weights name must end in .npz, got w.bin"):
        npz_path("w.bin")
    with pytest.raises(WeightsError, match=r"the decision grid name must end in .npz"):
        npz_path("grid", what="decision grid")


def test_writing_into_a_missing_directory_is_refused(tmp_path):
    target = tmp_path / "no_such_dir" / "model.npz"
    with pytest.raises(WeightsError, match=r"does not exist; create it or choose another"):
        output_path(target)
    with pytest.raises(WeightsError, match=r"does not exist"):
        save_weights(target, Network(), make_split(), CONFIG)
    assert not target.parent.exists()


def test_a_write_failure_is_reported(tmp_path):
    blocked = tmp_path / "taken.npz"
    blocked.mkdir()
    with pytest.raises(WeightsError, match=r"could not write the weights file"):
        save_weights(blocked, Network(), make_split(), CONFIG)


# --- files that are not weights files --------------------------------------------------------


class WouldRunOnUnpickle:
    """Unpickling an instance would create a directory: proof that code ran."""

    def __init__(self, target):
        self.target = target

    def __reduce__(self):
        return (os.mkdir, (self.target,))


def test_a_pickle_renamed_to_npz_is_refused_and_never_executed(tmp_path):
    marker = tmp_path / "code_ran"
    disguised = tmp_path / "disguised.npz"
    disguised.write_bytes(pickle.dumps(WouldRunOnUnpickle(str(marker))))
    with pytest.raises(WeightsError, match=r"cannot be read as a .npz weights file"):
        load_weights(disguised)
    assert not marker.exists()


def test_an_object_array_inside_the_archive_is_refused_and_never_executed(saved, tmp_path):
    path, _ = saved
    marker = tmp_path / "code_ran"
    payload = np.empty((), dtype=object)
    payload[()] = WouldRunOnUnpickle(str(marker))
    rewrite(path, init=payload)
    with pytest.raises(WeightsError, match=r"cannot be read as a .npz weights file"):
        load_weights(path)
    assert not marker.exists()


def test_garbage_and_truncated_files_are_refused(saved, tmp_path):
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"this is not a zip archive")
    with pytest.raises(WeightsError, match=r"cannot be read as a .npz weights file"):
        load_weights(garbage)

    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    with pytest.raises(WeightsError, match=r"cannot be read as a .npz weights file"):
        load_weights(empty)

    path, _ = saved
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(path.read_bytes()[:400])
    with pytest.raises(WeightsError, match=r"cannot be read as a .npz weights file"):
        load_weights(truncated)


def test_a_single_array_file_is_refused(tmp_path):
    single = tmp_path / "single.npz"
    with single.open("wb") as handle:
        np.save(handle, np.zeros((2, 16)))
    with pytest.raises(WeightsError, match=r"holds a single array, not a weights archive"):
        load_weights(single)


def test_a_missing_array_is_named(saved):
    path, _ = saved
    rewrite(path, dense2_biases=None, seed=None)
    with pytest.raises(WeightsError, match=r"it lacks dense2_biases, seed"):
        load_weights(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"dense1_weights": np.zeros((2, 8))},
            "dense1_weights must be a finite float array of "
            r"shape \(2, 16\); found shape \(2, 8\)",
        ),
        ({"dense3_biases": np.array([[np.nan]])}, "dense3_biases must be a finite float array"),
        (
            {"dense2_weights": np.zeros((16, 16), dtype=np.int64)},
            "dense2_weights must be a finite float array",
        ),
        (
            {"X_test": np.zeros((2, 3), dtype=np.float32)},
            "X_test must be a finite float array of "
            r"shape \(N, 2\)",
        ),
        ({"X_train": np.full((4, 2), np.inf, dtype=np.float32)}, "X_train must be a finite float"),
        ({"y_test": np.array([1, 2])}, "y_test must hold one label, 0 or 1"),
        (
            {"y_train": np.array([0, 1, 0])},
            "y_train must hold one label, 0 or 1, for each of the 4",
        ),
        ({"y_test": np.array([0.0, 1.0])}, "y_test must hold one label"),
        (
            {"X_test": np.zeros((0, 2), dtype=np.float32), "y_test": np.zeros(0, dtype=np.int64)},
            "the test set is empty",
        ),
        ({"format_version": np.array(2)}, "has weights format 2; this version reads format 1"),
        ({"format_version": np.array([1, 1])}, "has weights format"),
        ({"init": np.array("lecun")}, "init must be one of he, xavier, small; found 'lecun'"),
        ({"init": np.array(3)}, "init must be one of"),
        ({"noise": np.array([0.1, 0.2])}, "noise must be a single number"),
        ({"epochs": np.array(2.5)}, "epochs must be a single number"),
    ],
)
def test_wrong_contents_are_refused_with_the_reason(saved, changes, message):
    path, _ = saved
    rewrite(path, **changes)
    with pytest.raises(WeightsError, match=message):
        load_weights(path)
