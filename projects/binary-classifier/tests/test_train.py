"""The training loop on problems small enough to finish in a fraction of a second."""

import math

import numpy as np
import pytest

from binary_classifier.model import WeightsError, load_weights
from binary_classifier.train import train


def quiet_train(capsys, **kwargs):
    kwargs.setdefault("weights_path", None)
    result = train(**kwargs)
    return result, capsys.readouterr().out


def test_overfits_a_tiny_problem(capsys):
    result, _ = quiet_train(capsys, epochs=300, noise=0.05, n_samples=60, seed=1, log_every=100)
    first_epoch, first_loss, _ = result.history[0]
    assert first_epoch == 1
    assert [entry[0] for entry in result.history] == [1, 100, 200, 300]
    # 48 training points and 337 parameters: the network must fit every point.
    assert result.train_accuracy == 1.0
    assert result.loss < 0.05 < first_loss
    assert result.model.predict(result.split[0]).tolist() == result.split[1].tolist()


def test_small_initialisation_starts_at_ln_2(capsys):
    result, _ = quiet_train(capsys, epochs=1, n_samples=100, init="small")
    # Weights of scale 0.01 give logits near 0, probabilities near 1/2, and a loss near ln 2.
    assert result.loss == pytest.approx(math.log(2.0), abs=1e-3)
    assert result.config["init"] == "small"


def test_same_seed_same_weights_and_another_seed_other_weights(capsys):
    kwargs = {"epochs": 20, "n_samples": 40, "log_every": 20}
    first, _ = quiet_train(capsys, seed=4, **kwargs)
    again, _ = quiet_train(capsys, seed=4, **kwargs)
    other, _ = quiet_train(capsys, seed=5, **kwargs)
    for a, b in zip(first.model.dense_layers, again.model.dense_layers, strict=True):
        np.testing.assert_array_equal(a.weights, b.weights)
        np.testing.assert_array_equal(a.biases, b.biases)
    assert not np.array_equal(first.model.dense1.weights, other.model.dense1.weights)
    assert first.history == again.history


def test_split_is_eighty_to_twenty(capsys):
    result, out = quiet_train(capsys, epochs=1, n_samples=50)
    X_train, y_train, X_test, y_test = result.split
    assert (len(X_train), len(y_train), len(X_test), len(y_test)) == (40, 40, 10, 10)
    assert "train (40, 2), test (10, 2)" in out


def test_output_names_the_run(capsys):
    _, out = quiet_train(capsys, epochs=2, n_samples=1000, log_every=1)
    lines = out.splitlines()
    assert lines[0] == "two moons: 1,000 points, noise 0.1, seed 0 | train (800, 2), test (200, 2)"
    assert lines[1].startswith("data sha256: ")
    assert len(lines[1]) == len("data sha256: ") + 64
    assert lines[2] == (
        "model: 2-16-16-1, 337 parameters, init he | Adam lr 0.01, L2 0.0001, full batch"
    )
    assert lines[3].startswith("epoch     1 | loss ")
    assert lines[5].startswith("trained 2 epochs in ")
    assert len(lines) == 6


def test_one_epoch_is_reported_in_the_singular(capsys):
    _, out = quiet_train(capsys, epochs=1, n_samples=20)
    assert "trained 1 epoch in " in out


def test_weights_is_written_and_loads_back(tiny_run):
    assert tiny_run.weights_path.is_file()
    loaded, split, config = load_weights(tiny_run.weights_path)
    assert config == {"init": "he", "noise": 0.05, "n_samples": 40, "seed": 0, "epochs": 200}
    for trained, restored in zip(tiny_run.model.dense_layers, loaded.dense_layers, strict=True):
        np.testing.assert_array_equal(restored.weights, trained.weights)
        np.testing.assert_array_equal(restored.biases, trained.biases)
    np.testing.assert_array_equal(split[2], tiny_run.split[2])


def test_no_weights_is_written_when_none_is_asked_for(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, out = quiet_train(capsys, epochs=1, n_samples=20)
    assert result.weights_path is None
    assert "wrote weights" not in out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"epochs": 0}, ValueError, "epochs must be a whole number of at least 1, got 0"),
        ({"epochs": 2.0}, ValueError, "epochs must be a whole number"),
        ({"epochs": True}, ValueError, "epochs must be a whole number"),
        ({"log_every": 0}, ValueError, "log_every must be a whole number of at least 1"),
        ({"init": "lecun"}, ValueError, "init must be one of he, xavier, small"),
        ({"noise": -1.0}, ValueError, "noise must be a finite number that is not negative"),
        ({"n_samples": 2}, ValueError, "leaves 0 test rows"),
        ({"seed": -1}, ValueError, "seed must be between"),
        ({"weights_path": "weights.pkl"}, WeightsError, "pickle file"),
        ({"weights_path": "weights.txt"}, WeightsError, "must end in .npz"),
    ],
)
def test_bad_arguments_are_refused(capsys, kwargs, error, message):
    with pytest.raises(error, match=message):
        train(**{"epochs": 1, "n_samples": 20, "weights_path": None, **kwargs})


def test_a_bad_weights_path_is_refused_before_any_training(tmp_path, capsys):
    with pytest.raises(WeightsError, match=r"does not exist"):
        train(epochs=5, n_samples=20, weights_path=tmp_path / "missing" / "w.npz")
    assert capsys.readouterr().out == ""
