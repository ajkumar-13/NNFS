"""The training loop: it overfits a tiny problem, repeats a recorded run, rejects bad input."""

import numpy as np
import pytest

from fashion_mnist import data
from fashion_mnist import train as train_module
from fashion_mnist.data import DataError
from fashion_mnist.model import WEIGHT_KEYS, Network
from fashion_mnist.nn import Optimizer_Adam
from fashion_mnist.train import fit, main, train, weights_path_problem
from helpers import write_dataset

# What the project's original train.py printed and saved for the tiny fixture with
# --epochs 3 --batch-size 5 and seed 0, before the code was restructured into this package.
# The restructured loop must draw the same random numbers in the same order to match it.
RECORDED_LOSSES = [2.3107646804054944, 2.301770726616411, 2.302696283178792]
RECORDED_ACCURACIES = [2 / 12, 2 / 12, 3 / 12]
RECORDED_OUTPUT_BIASES = [
    0.004023235960483558,
    0.0046678527391337335,
    -0.00125301539667185,
    -0.0029174050292401768,
    -0.0032604063104434188,
    -0.0016343133098654517,
    -0.00203708436874281,
    -0.0027206447759346598,
    -0.0015223871583720306,
    -0.0009761439195558932,
]


def blobs(per_class=10, n_features=12, n_classes=3):
    """Three well-separated clusters: a problem a small network can fit exactly."""
    generator = np.random.RandomState(3)
    centres = generator.randn(n_classes, n_features) * 2
    X = np.concatenate([c + 0.3 * generator.randn(per_class, n_features) for c in centres])
    y = np.repeat(np.arange(n_classes), per_class)
    return X, y


def small_network():
    return Network(n_inputs=12, n_hidden=16, n_classes=3)


# ------------------------------------------------------------------------------------------- fit


def test_fit_overfits_a_tiny_synthetic_problem():
    X, y = blobs()
    network = small_network()
    optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-4)
    history = fit(network, optimizer, X, y, epochs=150, batch_size=10, log=lambda _: None)

    assert len(history) == 150
    assert history[0].loss == pytest.approx(np.log(3), abs=0.02)
    assert history[-1].loss < 0.05
    assert history[-1].accuracy == 1.0

    loss = network.forward(X, y, training=False)
    assert loss < 0.01
    assert np.array_equal(network.predictions(), y)


def test_fit_takes_one_step_per_batch_including_the_short_last_batch():
    X, y = blobs(per_class=4)
    optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-4)
    history = fit(small_network(), optimizer, X, y, epochs=2, batch_size=5, log=lambda _: None)
    # 12 samples in batches of 5, 5 and 2: three steps an epoch.
    assert optimizer.iterations == 6
    assert history[0].learning_rate == pytest.approx(0.01 / (1 + 1e-4 * 2))
    assert history[1].learning_rate == pytest.approx(0.01 / (1 + 1e-4 * 5))
    assert [stats.epoch for stats in history] == [1, 2]


def test_fit_is_repeatable_from_a_seed_and_differs_between_seeds():
    X, y = blobs()

    def run(seed):
        np.random.seed(seed)
        network = small_network()
        fit(network, Optimizer_Adam(), X, y, epochs=3, batch_size=8, log=lambda _: None)
        return network.weights()

    first, again, other = run(5), run(5), run(6)
    for key in WEIGHT_KEYS:
        assert np.array_equal(first[key], again[key])
    assert not np.array_equal(first["dense1_weights"], other["dense1_weights"])


def test_fit_logs_one_line_per_epoch_in_the_documented_format():
    X, y = blobs()
    lines = []
    history = fit(
        small_network(), Optimizer_Adam(), X, y, epochs=2, batch_size=10, log=lines.append
    )
    assert len(lines) == 2
    first = history[0]
    assert lines[0] == (
        f"epoch   1 | loss {first.loss:.4f} | train_acc {first.accuracy:.4f} | "
        f"lr {first.learning_rate:.6f}"
    )
    assert lines[1].startswith("epoch   2 | loss ")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"epochs": 0}, "epochs must be at least 1"),
        ({"batch_size": 0}, "batch size must be at least 1"),
        ({"X": np.zeros((30, 11))}, "takes rows of 12 features"),
        ({"X": np.zeros(12)}, "takes rows of 12 features"),
        ({"y": np.zeros(29, dtype=int)}, "one label per row"),
        ({"y": np.full(30, 3)}, "class indices from 0 to 2"),
        ({"y": np.full(30, -1)}, "class indices from 0 to 2"),
        ({"X": np.zeros((0, 12)), "y": np.zeros(0, dtype=int)}, "training set is empty"),
    ],
)
def test_fit_rejects_input_it_cannot_train_on(change, message):
    X, y = blobs()
    arguments = {"X": X, "y": y, "epochs": 1, "batch_size": 10} | change
    with pytest.raises(ValueError, match=message):
        fit(small_network(), Optimizer_Adam(), log=lambda _: None, **arguments)


# ----------------------------------------------------------------------------------------- train


def test_train_repeats_the_run_recorded_from_the_original_script(as_fashion_mnist, tmp_path):
    path = tmp_path / "weights.npz"
    lines = []
    history = train(as_fashion_mnist, path, epochs=3, batch_size=5, seed=0, log=lines.append)

    assert [stats.loss for stats in history] == pytest.approx(RECORDED_LOSSES, rel=1e-9)
    assert [stats.accuracy for stats in history] == pytest.approx(RECORDED_ACCURACIES)
    assert "epoch   1 | loss 2.3108 | train_acc 0.1667 | lr 0.001000" in lines
    assert "epoch   3 | loss 2.3027 | train_acc 0.2500 | lr 0.000999" in lines
    assert "  parameters 118,282" in lines
    assert lines[-1].startswith("weights sha256 ")

    restored = Network()
    restored.load(path)
    assert restored.dense3.biases[0].tolist() == pytest.approx(RECORDED_OUTPUT_BIASES, rel=1e-9)
    assert lines[-1] == f"weights sha256 {restored.fingerprint()}"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["weights.npz"]


def test_train_sets_the_seed_itself(as_fashion_mnist, tmp_path):
    np.random.seed(999)
    first = train(as_fashion_mnist, tmp_path / "a.npz", epochs=1, batch_size=5, log=lambda _: None)
    np.random.seed(111)
    second = train(as_fashion_mnist, tmp_path / "b.npz", epochs=1, batch_size=5, log=lambda _: None)
    assert first == second
    other = train(
        as_fashion_mnist, tmp_path / "c.npz", epochs=1, batch_size=5, seed=1, log=lambda _: None
    )
    assert other != first


def test_train_checks_the_weights_path_before_doing_any_work(tmp_path, as_fashion_mnist):
    with pytest.raises(ValueError, match=r"must end in \.npz"):
        train(as_fashion_mnist, tmp_path / "weights.pkl", epochs=1)
    with pytest.raises(ValueError, match="does not exist, so the weights cannot be written"):
        train(as_fashion_mnist, tmp_path / "absent" / "weights.npz", epochs=1)


def test_train_without_data_names_the_download_command(tmp_path):
    with pytest.raises(DataError, match=r"scripts/download_fashion_mnist\.py"):
        train(tmp_path / "no-data", tmp_path / "weights.npz", epochs=1, log=lambda _: None)
    assert not (tmp_path / "weights.npz").exists()


def test_train_rejects_images_of_another_size(tmp_path, monkeypatch):
    files = write_dataset(tmp_path, np.zeros((3, 4, 4)), [0, 1, 2], np.zeros((2, 4, 4)), [0, 1])
    monkeypatch.setattr(data, "FASHION_MNIST_FILES", files)
    with pytest.raises(DataError, match="takes 784 pixels per image, but the training images"):
        train(tmp_path, tmp_path / "weights.npz", epochs=1, log=lambda _: None)
    assert not (tmp_path / "weights.npz").exists()


def test_weights_path_problem(tmp_path):
    (tmp_path / "folder.npz").mkdir()
    assert weights_path_problem(tmp_path / "weights.npz") is None
    assert "must end in .npz" in weights_path_problem(tmp_path / "weights")
    assert "does not exist" in weights_path_problem(tmp_path / "absent" / "weights.npz")
    assert "is a directory" in weights_path_problem(tmp_path / "folder.npz")


# ------------------------------------------------------------------------------ the command line


def test_main_trains_and_writes_the_weights(as_fashion_mnist, tmp_path, capsys):
    path = tmp_path / "weights.npz"
    status = main(
        [
            "--epochs",
            "1",
            "--batch-size",
            "5",
            "--data-dir",
            str(as_fashion_mnist),
            "--weights",
            str(path),
        ]
    )
    assert status == 0
    output = capsys.readouterr().out
    assert "training for 1 epochs, batch_size=5 (3 steps/epoch), seed=0" in output
    assert "epoch   1 | loss 2.3108" in output
    assert f"wrote weights to {path}" in output
    assert path.is_file()


def test_main_passes_the_seed_through(as_fashion_mnist, tmp_path, capsys):
    path = tmp_path / "weights.npz"
    status = main(
        [
            "--epochs",
            "1",
            "--seed",
            "7",
            "--data-dir",
            str(as_fashion_mnist),
            "--weights",
            str(path),
        ]
    )
    assert status == 0
    output = capsys.readouterr().out
    assert "training for 1 epochs, batch_size=128 (1 steps/epoch), seed=7" in output
    assert "epoch   1 | loss 2.3108" not in output


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--epochs", "0"], "must be at least 1, got 0"),
        (["--epochs", "many"], "'many' is not a whole number"),
        (["--batch-size", "-4"], "must be at least 1, got -4"),
        (["--seed", "-1"], "must be between 0 and 4294967295"),
        (["--seed", "4294967296"], "must be between 0 and 4294967295"),
        (["--seed", "1.5"], "'1.5' is not a whole number"),
        (["--weights", "weights.pkl"], "must end in .npz"),
        (["--learning-rate", "0.1"], "unrecognized arguments"),
    ],
)
def test_main_rejects_bad_arguments_before_loading_anything(arguments, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_main_reports_missing_data_on_stderr_with_status_one(tmp_path, capsys):
    status = main(
        ["--data-dir", str(tmp_path / "no-data"), "--weights", str(tmp_path / "weights.npz")]
    )
    assert status == 1
    error = capsys.readouterr().err
    assert error.startswith("error: the data directory")
    assert "uv run python scripts/download_fashion_mnist.py" in error


def test_main_reports_exhausted_memory_with_status_one(tmp_path, monkeypatch, capsys):
    def exhausted(**_):
        raise MemoryError

    monkeypatch.setattr(train_module, "train", exhausted)
    assert main(["--weights", str(tmp_path / "weights.npz")]) == 1
    assert "not enough free memory" in capsys.readouterr().err
