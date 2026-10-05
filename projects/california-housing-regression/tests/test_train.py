"""The training loop: it overfits a tiny problem, repeats a recorded run, rejects bad input."""

import numpy as np
import pytest

from california_housing_regression.data import DataError
from california_housing_regression.model import WEIGHT_KEYS, Network
from california_housing_regression.nn import Optimizer_Adam
from california_housing_regression.train import fit, main, train, weights_path_problem
from helpers import write_archive

# What the project's original train.py printed and saved for the tiny fixture with
# --epochs 3 --batch-size 5 and seed 0, before the code was restructured into this package.
# The restructured loop must draw the same random numbers in the same order to match it.
RECORDED_LOSSES = [0.8467, 0.1250, 0.0709]
RECORDED_OUTPUT_BIAS = 0.008566993181689972
RECORDED_OUTPUT_WEIGHTS = [
    -0.04047802333535922,
    0.05735854492998782,
    0.0709751071259794,
    0.06950012249095996,
    -0.047825722598922996,
]
RECORDED_FIRST_BIASES = [
    0.0514130182384231,
    0.033772001494157586,
    0.050821721644554826,
    0.05481495550406248,
    -0.036341154845718274,
]


def curve(rows=30, n_features=6):
    """A small piecewise-linear regression problem, which a ReLU network can fit closely."""
    generator = np.random.RandomState(3)
    X = generator.randn(rows, n_features)
    y = np.abs(X[:, 0]) + 0.5 * np.maximum(X[:, 1], 0) - 0.3 * X[:, 3]
    return X, y


def small_network():
    return Network(n_inputs=6, n_hidden=16, n_outputs=1, l2=0.0)


# ------------------------------------------------------------------------------------------- fit


def test_fit_overfits_a_tiny_synthetic_problem():
    X, y = curve()
    network = small_network()
    optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-4)
    history = fit(network, optimizer, X, y, epochs=400, batch_size=10, log=lambda _: None)

    assert len(history) == 400
    # With weights near zero the first predictions are near zero, so the loss starts near the
    # mean square of the targets.
    assert history[0].mse == pytest.approx(float(np.mean(y**2)), rel=0.05)
    assert history[-1].mse < 0.002
    assert history[-1].mse < 0.002 * history[0].mse

    loss = network.forward(X, y)
    assert loss < 0.002
    assert np.max(np.abs(network.dense3.output[:, 0] - y)) < 0.15


def test_fit_reports_the_penalty_in_the_loss_and_not_in_the_mse():
    X, y = curve()
    plain = Network(n_inputs=6, n_hidden=16, n_outputs=1, l2=0.0)
    np.random.seed(1234)
    penalised = Network(n_inputs=6, n_hidden=16, n_outputs=1, l2=0.5)
    for layer in penalised.dense_layers:
        layer.weights = np.random.randn(*layer.weights.shape)

    history = fit(plain, Optimizer_Adam(), X, y, epochs=1, batch_size=30, log=lambda _: None)
    assert history[0].loss == history[0].mse

    before = float(penalised.regularization_loss())
    data_loss = penalised.forward(X, y)
    history = fit(penalised, Optimizer_Adam(), X, y, epochs=1, batch_size=30, log=lambda _: None)
    # One batch holds every row, so the epoch's figures are those of the single forward pass.
    assert history[0].mse == pytest.approx(data_loss)
    assert history[0].loss == pytest.approx(data_loss + before)


def test_fit_takes_one_step_per_batch_including_the_short_last_batch():
    X, y = curve(rows=12)
    optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-4)
    history = fit(small_network(), optimizer, X, y, epochs=2, batch_size=5, log=lambda _: None)
    # 12 samples in batches of 5, 5 and 2: three steps an epoch.
    assert optimizer.iterations == 6
    assert history[0].learning_rate == pytest.approx(0.01 / (1 + 1e-4 * 2))
    assert history[1].learning_rate == pytest.approx(0.01 / (1 + 1e-4 * 5))
    assert [stats.epoch for stats in history] == [1, 2]


def test_fit_is_repeatable_from_a_seed_and_differs_between_seeds():
    X, y = curve()

    def run(seed):
        np.random.seed(seed)
        network = small_network()
        fit(network, Optimizer_Adam(), X, y, epochs=3, batch_size=8, log=lambda _: None)
        return network.weights()

    first, again, other = run(5), run(5), run(6)
    for key in WEIGHT_KEYS:
        assert np.array_equal(first[key], again[key])
    assert not np.array_equal(first["dense1_weights"], other["dense1_weights"])


def test_fit_draws_one_permutation_per_epoch_and_nothing_else():
    X, y = curve()
    np.random.seed(21)
    network = small_network()
    reference = np.random.RandomState(21)
    for shape in ((6, 16), (16, 16), (16, 1)):
        reference.randn(*shape)
    fit(network, Optimizer_Adam(), X, y, epochs=4, batch_size=7, log=lambda _: None)
    for _ in range(4):
        reference.permutation(len(X))
    assert np.random.rand() == reference.rand()


def test_fit_logs_the_first_epoch_every_nth_and_the_last_in_the_documented_format():
    X, y = curve()
    lines = []
    history = fit(
        small_network(),
        Optimizer_Adam(),
        X,
        y,
        epochs=7,
        batch_size=10,
        log_every=3,
        log=lines.append,
    )
    assert len(history) == 7
    assert [line.split("|")[0].strip() for line in lines] == [
        "epoch   1",
        "epoch   3",
        "epoch   6",
        "epoch   7",
    ]
    first = history[0]
    assert lines[0] == (
        f"epoch   1 | loss {first.loss:.4f} | train_mse {first.mse:.4f} | "
        f"lr {first.learning_rate:.6f}"
    )


def test_fit_logs_every_epoch_by_default():
    X, y = curve()
    lines = []
    fit(small_network(), Optimizer_Adam(), X, y, epochs=3, batch_size=10, log=lines.append)
    assert len(lines) == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"epochs": 0}, "epochs must be at least 1"),
        ({"batch_size": 0}, "batch size must be at least 1"),
        ({"log_every": 0}, "log_every must be at least 1"),
        ({"X": np.zeros((30, 5))}, "takes rows of 6 features"),
        ({"X": np.zeros(6)}, "takes rows of 6 features"),
        ({"y": np.zeros(29)}, "one target per row"),
        ({"y": np.zeros((30, 1))}, "one target per row"),
        ({"y": np.full(30, np.nan)}, "must be finite numbers"),
        ({"X": np.full((30, 6), np.inf)}, "must be finite numbers"),
        ({"X": np.zeros((0, 6)), "y": np.zeros(0)}, "training set is empty"),
    ],
)
def test_fit_rejects_input_it_cannot_train_on(change, message):
    X, y = curve()
    arguments = {"X": X, "y": y, "epochs": 1, "batch_size": 10} | change
    with pytest.raises(ValueError, match=message):
        fit(small_network(), Optimizer_Adam(), log=lambda _: None, **arguments)


# ----------------------------------------------------------------------------------------- train


def test_train_repeats_the_run_recorded_from_the_original_script(as_housing, tmp_path):
    path = tmp_path / "weights.npz"
    lines = []
    history = train(as_housing, path, epochs=3, batch_size=5, seed=0, log_every=1, log=lines.append)

    assert [round(stats.loss, 4) for stats in history] == RECORDED_LOSSES
    assert "  train: (32, 8)    test: (8, 8)" in lines
    assert "  target: mean $311,094, standard deviation $136,465 (training fold)" in lines
    assert "  parameters 4,801" in lines
    assert "training for 3 epochs, batch_size=5 (7 steps/epoch), seed=0" in lines
    assert any(line.startswith("epoch   1 | loss 0.8467 | train_mse ") for line in lines)
    assert any(line.endswith("| lr 0.009980") for line in lines)
    assert lines[-1].startswith("weights sha256 ")

    restored = Network()
    scaler = restored.load(path)
    assert restored.dense3.biases[0, 0] == pytest.approx(RECORDED_OUTPUT_BIAS, rel=1e-9)
    assert restored.dense3.weights[:5, 0].tolist() == pytest.approx(
        RECORDED_OUTPUT_WEIGHTS, rel=1e-9
    )
    assert restored.dense1.biases[0, :5].tolist() == pytest.approx(RECORDED_FIRST_BIASES, rel=1e-9)
    assert scaler.y_mean == pytest.approx(3.110938787460327, rel=1e-6)
    assert scaler.y_std == pytest.approx(1.3646493958068848, rel=1e-6)
    assert lines[-1] == f"weights sha256 {restored.fingerprint()}"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["weights.npz"]


def test_train_sets_the_seed_itself(as_housing, tmp_path):
    np.random.seed(999)
    first = train(as_housing, tmp_path / "a.npz", epochs=1, batch_size=5, log=lambda _: None)
    np.random.seed(111)
    second = train(as_housing, tmp_path / "b.npz", epochs=1, batch_size=5, log=lambda _: None)
    assert first == second
    other = train(
        as_housing, tmp_path / "c.npz", epochs=1, batch_size=5, seed=1, log=lambda _: None
    )
    assert other != first


def test_train_stores_the_scaler_of_its_own_split(as_housing, tmp_path):
    train(as_housing, tmp_path / "zero.npz", epochs=1, seed=0, log=lambda _: None)
    train(as_housing, tmp_path / "one.npz", epochs=1, seed=1, log=lambda _: None)
    zero = Network().load(tmp_path / "zero.npz")
    one = Network().load(tmp_path / "one.npz")
    assert not zero.matches(one)


def test_train_checks_the_weights_path_and_the_seed_before_doing_any_work(tmp_path, as_housing):
    with pytest.raises(ValueError, match=r"must end in \.npz"):
        train(as_housing, tmp_path / "weights.pkl", epochs=1)
    with pytest.raises(ValueError, match="does not exist, so the weights cannot be written"):
        train(as_housing, tmp_path / "absent" / "weights.npz", epochs=1)
    with pytest.raises(ValueError, match="seed must be between 0 and 4294967295"):
        train(as_housing, tmp_path / "weights.npz", epochs=1, seed=-1)


def test_train_without_data_names_the_download_command(tmp_path):
    with pytest.raises(DataError, match=r"scripts/download_california_housing\.py"):
        train(tmp_path / "no-data", tmp_path / "weights.npz", epochs=1, log=lambda _: None)
    assert not (tmp_path / "weights.npz").exists()


def test_train_refuses_an_archive_that_is_not_the_published_one(tmp_path):
    write_archive(tmp_path, np.ones((10, 9)))
    with pytest.raises(DataError, match="is not the expected file"):
        train(tmp_path, tmp_path / "weights.npz", epochs=1, log=lambda _: None)
    assert not (tmp_path / "weights.npz").exists()


def test_weights_path_problem(tmp_path):
    (tmp_path / "folder.npz").mkdir()
    assert weights_path_problem(tmp_path / "weights.npz") is None
    assert "must end in .npz" in weights_path_problem(tmp_path / "weights")
    assert "does not exist" in weights_path_problem(tmp_path / "absent" / "weights.npz")
    assert "is a directory" in weights_path_problem(tmp_path / "folder.npz")


# ------------------------------------------------------------------------------ the command line


def test_main_trains_and_writes_the_weights(as_housing, tmp_path, capsys):
    path = tmp_path / "weights.npz"
    status = main(
        [
            *["--epochs", "1", "--batch-size", "5", "--log-every", "1"],
            *["--data-dir", str(as_housing), "--weights", str(path)],
        ]
    )
    assert status == 0
    output = capsys.readouterr().out
    assert f"loading California housing from {as_housing}" in output
    assert "training for 1 epochs, batch_size=5 (7 steps/epoch), seed=0" in output
    assert "epoch   1 | loss 0.8467" in output
    assert f"wrote weights to {path}" in output
    assert path.is_file()


def test_main_passes_the_seed_through(as_housing, tmp_path, capsys):
    path = tmp_path / "weights.npz"
    status = main(
        ["--epochs", "1", "--seed", "7", "--data-dir", str(as_housing), "--weights", str(path)]
    )
    assert status == 0
    output = capsys.readouterr().out
    assert "training for 1 epochs, batch_size=256 (1 steps/epoch), seed=7" in output
    assert "target: mean $311,094" not in output


def test_main_logs_every_twentieth_epoch_by_default(as_housing, tmp_path, capsys):
    path = tmp_path / "weights.npz"
    assert main(["--epochs", "41", "--data-dir", str(as_housing), "--weights", str(path)]) == 0
    epochs = [line[:9] for line in capsys.readouterr().out.splitlines() if line.startswith("epoch")]
    assert epochs == ["epoch   1", "epoch  20", "epoch  40", "epoch  41"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--epochs", "0"], "must be at least 1, got 0"),
        (["--epochs", "many"], "'many' is not a whole number"),
        (["--batch-size", "-4"], "must be at least 1, got -4"),
        (["--log-every", "0"], "must be at least 1, got 0"),
        (["--seed", "-1"], "must be between 0 and 4294967295"),
        (["--seed", "4294967296"], "must be between 0 and 4294967295"),
        (["--seed", "1.5"], "'1.5' is not a whole number"),
        (["--weights", "weights.pkl"], "must end in .npz"),
        (["--learning-rate", "0.1"], "unrecognized arguments"),
        (["--backend", "sklearn"], "unrecognized arguments"),
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
    assert "uv run python scripts/download_california_housing.py" in error
