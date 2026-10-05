"""The evaluation: metrics on hand-made cases, the report, the split check, the command line."""

import math
from dataclasses import replace

import numpy as np
import pytest

from california_housing_regression import evaluate as evaluate_module
from california_housing_regression.data import Housing, Scaler, load_california_housing
from california_housing_regression.evaluate import (
    Metrics,
    _dollars,
    evaluate,
    format_report,
    linear_baseline,
    main,
    predictions_path_problem,
    regression_metrics,
    score,
    to_dollars,
)
from california_housing_regression.model import Network, WeightsError
from california_housing_regression.train import train


def hand_made_housing():
    """Two features, five training rows and four test rows, with round statistics."""
    scaler = Scaler(
        x_mean=np.zeros(2, np.float32), x_std=np.ones(2, np.float32), y_mean=2.0, y_std=0.5
    )
    X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float32)
    X_test = np.array([[0, 0], [2, 0], [0, 2], [4, 3]], dtype=np.float32)
    # Published values in dollars; the last test row sits at the census cap.
    value_train = np.array([100_000.0, 150_000.0, 200_000.0, 250_000.0, 300_000.0])
    value_test = np.array([100_000.0, 200_000.0, 300_000.0, 500_001.0])
    return Housing(
        X_train=X_train,
        y_train=((value_train / 100_000 - 2.0) / 0.5).astype(np.float32),
        X_test=X_test,
        y_test=((value_test / 100_000 - 2.0) / 0.5).astype(np.float32),
        value_train=value_train,
        value_test=value_test,
        scaler=scaler,
        seed=0,
    )


def constant_network(output):
    """A 2-2-2-1 network that predicts ``output`` for every row."""
    network = Network(n_inputs=2, n_hidden=2, n_outputs=1)
    for layer in network.dense_layers:
        layer.weights = np.zeros_like(layer.weights)
    network.dense3.biases = np.array([[float(output)]])
    return network


# ----------------------------------------------------------------------------- regression_metrics


def test_metrics_on_a_hand_computed_case():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 330.0, 400.0])
    metrics = regression_metrics(y_true, y_pred)
    # Residuals 10, -10, 30, 0: squares sum to 1,100, absolute values to 50, signed to 30.
    assert metrics.n == 4
    assert metrics.rmse == pytest.approx(math.sqrt(1100 / 4))
    assert metrics.mae == pytest.approx(12.5)
    assert metrics.mean_error == pytest.approx(7.5)
    assert metrics.mean_prediction == pytest.approx(257.5)
    # The true values have mean 250 and total sum of squares 50,000.
    assert metrics.r2 == pytest.approx(1 - 1100 / 50_000)


def test_r2_is_one_for_a_perfect_prediction_and_zero_for_the_mean():
    y_true = np.array([1.0, 2.0, 6.0])
    perfect = regression_metrics(y_true, y_true)
    assert (perfect.rmse, perfect.mae, perfect.mean_error, perfect.r2) == (0.0, 0.0, 0.0, 1.0)
    mean = regression_metrics(y_true, np.full(3, 3.0))
    assert mean.r2 == pytest.approx(0.0)
    assert mean.mean_error == pytest.approx(0.0)


def test_r2_is_negative_for_a_model_worse_than_the_mean():
    y_true = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y_true, np.array([3.0, 2.0, 1.0]))
    # Residual sum of squares 8 against a total of 2.
    assert metrics.r2 == pytest.approx(-3.0)


def test_r2_is_undefined_when_every_true_value_is_the_same():
    metrics = regression_metrics(np.array([5.0, 5.0, 5.0]), np.array([4.0, 5.0, 9.0]))
    assert metrics.r2 is None
    assert metrics.mae == pytest.approx(5 / 3)
    assert metrics.mean_error == pytest.approx(1.0)


def test_metrics_do_not_depend_on_the_units_except_by_scale():
    y_true = np.array([1.0, 2.5, 4.0, 0.5])
    y_pred = np.array([1.5, 2.0, 3.0, 1.0])
    units = regression_metrics(y_true, y_pred)
    dollars = regression_metrics(y_true * 100_000, y_pred * 100_000)
    assert dollars.rmse == pytest.approx(units.rmse * 100_000)
    assert dollars.mae == pytest.approx(units.mae * 100_000)
    assert dollars.r2 == pytest.approx(units.r2)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        (np.zeros(3), np.zeros(4), "two vectors of one length"),
        (np.zeros((3, 1)), np.zeros((3, 1)), "two vectors of one length"),
        (np.zeros(0), np.zeros(0), "undefined for an empty set"),
    ],
)
def test_metrics_reject_what_they_cannot_score(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        regression_metrics(y_true, y_pred)


def test_to_dollars_undoes_the_scaling():
    scaler = Scaler(np.zeros(1, np.float32), np.ones(1, np.float32), y_mean=2.0, y_std=0.5)
    # (z * 0.5 + 2.0) * 100,000 for z = 0, 2 and -1.
    dollars = to_dollars(np.array([[0.0], [2.0], [-1.0]]), scaler)
    assert dollars.shape == (3,)
    assert dollars.tolist() == [200_000.0, 300_000.0, 150_000.0]


def test_dollars_are_printed_with_separators_and_the_sign_first():
    assert _dollars(49221.6) == "$49,222"
    assert _dollars(-31246.2) == "-$31,246"
    assert _dollars(-0.4) == "$0"
    assert _dollars(1_234_567) == "$1,234,567"


# ------------------------------------------------------------------------------------------ score


def test_score_of_a_constant_prediction_on_the_hand_made_folds():
    housing = hand_made_housing()
    # An output of 0 in standardised units is the training mean, 200,000 dollars.
    report = score(constant_network(0.0), housing)

    assert report.parameters == (2 * 2 + 2) + (2 * 2 + 2) + (2 * 1 + 1)
    assert report.seed == 0
    assert report.train_pred.tolist() == [200_000.0] * 5
    assert report.test_pred.tolist() == [200_000.0] * 4

    # Training residuals 100k, 50k, 0, -50k, -100k.
    assert report.train.n == 5
    assert report.train.rmse == pytest.approx(math.sqrt(25_000e6 / 5))
    assert report.train.mae == pytest.approx(60_000)
    assert report.train.r2 == pytest.approx(0.0)
    # In standardised units the targets are -2, -1, 0, 1, 2, so the mean squared error is 2.
    assert report.train_mse == pytest.approx(2.0)

    # Test residuals 100k, 0, -100k, -300,001.
    assert report.test.n == 4
    assert report.test.mae == pytest.approx(500_001 / 4)
    assert report.test.mean_error == pytest.approx(-300_001 / 4)
    assert report.test.rmse == pytest.approx(math.sqrt((2 * 1e10 + 300_001**2) / 4))
    assert report.test.r2 < 0

    assert report.at_cap.n == 1
    assert report.at_cap.mean_error == pytest.approx(-300_001)
    assert report.at_cap.mean_prediction == pytest.approx(200_000)
    assert report.at_cap.r2 is None
    assert report.below_cap.n == 3
    assert report.below_cap.mean_error == pytest.approx(0.0)
    assert report.below_cap.rmse == pytest.approx(math.sqrt(2e10 / 3))
    assert report.below_cap.r2 == pytest.approx(0.0)
    assert report.train_at_cap == 0

    # The training-mean baseline is this same constant prediction.
    assert report.mean_baseline == report.test


def test_score_measures_the_error_against_the_published_values():
    housing = hand_made_housing()
    # An output of 2 is (2 * 0.5 + 2) * 100,000 = 300,000 dollars.
    report = score(constant_network(2.0), housing)
    assert report.test_pred.tolist() == [300_000.0] * 4
    assert report.test.mean_error == pytest.approx((200_000 + 100_000 + 0 - 200_001) / 4)
    # The baseline still predicts the training mean, whatever the network does.
    assert report.mean_baseline.mean_prediction == pytest.approx(200_000)


def test_score_handles_a_test_fold_with_no_row_or_every_row_at_the_cap():
    housing = hand_made_housing()
    none = replace(housing, value_test=np.array([1e5, 2e5, 3e5, 4e5]))
    report = score(constant_network(0.0), none)
    assert report.at_cap is None
    assert report.below_cap.n == 4
    assert "  at the cap     n=0" in format_report(report)

    every = replace(housing, value_test=np.full(4, 500_001.0))
    report = score(constant_network(0.0), every)
    assert report.below_cap is None
    assert report.at_cap.n == 4
    assert report.test.r2 is None
    assert "R^2                       undefined" in format_report(report)


def test_linear_baseline_recovers_an_exactly_linear_target():
    generator = np.random.RandomState(8)
    scaler = Scaler(np.zeros(3, np.float32), np.ones(3, np.float32), y_mean=2.0, y_std=0.5)
    X_train = generator.randn(40, 3).astype(np.float32)
    X_test = generator.randn(10, 3).astype(np.float32)
    weights = np.array([0.5, -1.0, 0.25])

    def target(X):
        return X.astype(np.float64) @ weights + 0.3

    housing = Housing(
        X_train=X_train,
        y_train=target(X_train).astype(np.float32),
        X_test=X_test,
        y_test=target(X_test).astype(np.float32),
        value_train=(target(X_train) * 0.5 + 2.0) * 100_000,
        value_test=(target(X_test) * 0.5 + 2.0) * 100_000,
        scaler=scaler,
        seed=0,
    )
    predictions = linear_baseline(housing)
    assert predictions.shape == (10,)
    assert predictions == pytest.approx(housing.value_test, rel=1e-6)


def test_linear_baseline_is_fitted_on_the_training_fold_only():
    housing = hand_made_housing()
    before = linear_baseline(housing)
    moved = replace(housing, y_test=housing.y_test + 100)
    assert np.array_equal(linear_baseline(moved), before)


# ---------------------------------------------------------------------------------- format_report


def test_format_report_prints_every_figure():
    report = score(constant_network(0.0), hand_made_housing())
    text = format_report(report)
    lines = text.splitlines()
    assert lines[0] == "  parameters 15"
    assert lines[1] == f"  weights    sha256 {report.fingerprint}"
    assert lines[2] == "  split      seed 0: 5 training and 4 test block groups"
    assert "[train]  n=5" in lines
    assert "  MSE (standardised units)  2.0000" in lines
    assert "  RMSE (dollars)            $70,711" in lines
    assert "  MAE  (dollars)            $60,000" in lines
    assert "  mean error (dollars)      $0" in lines
    assert "  R^2                       0.0000" in lines
    assert "[test]  n=4" in lines
    assert "  mean error (dollars)      -$75,000" in lines
    assert (
        "Test fold by top-coding (the census records every value above the cap as $500,001):"
        in lines
    )
    assert (
        "  at the cap     n=1  RMSE $300,001  MAE $300,001  mean prediction $200,000  "
        "mean error -$300,001"
    ) in lines
    assert (
        "  below the cap  n=3  RMSE $81,650  MAE $66,667  mean prediction $200,000  mean error $0"
    ) in lines
    assert "  the training fold holds 0 block groups at the cap" in lines
    assert "Baselines on the test fold, fitted on the training fold:" in lines
    assert any(line.startswith("  training mean         RMSE $") for line in lines)
    assert any(line.startswith("  linear least squares  RMSE $") for line in lines)
    assert any(line.startswith("  this network          RMSE $") for line in lines)


# --------------------------------------------------------------------------------------- evaluate


@pytest.fixture
def trained(as_housing, tmp_path):
    """Weights trained for three epochs on the fixture, seed 0."""
    path = tmp_path / "weights.npz"
    train(as_housing, path, epochs=3, batch_size=5, seed=0, log=lambda _: None)
    return path


def test_evaluate_scores_the_folds_of_the_seed(as_housing, trained):
    lines = []
    report = evaluate(as_housing, trained, seed=0, log=lines.append)
    assert lines[0] == f"restoring weights from {trained}"
    assert lines[1] == f"loading California housing from {as_housing}"
    assert report.parameters == 4_801
    assert report.train.n == 32
    # Rows 36 to 39 of the fixture are at the cap, and seed 0 puts all four in the training fold.
    assert report.train_at_cap == 4
    assert report.at_cap is None
    assert report.test.n == 8
    assert "  split      seed 0: 32 training and 8 test block groups" in lines[2]

    network = Network()
    network.load(trained)
    assert report.fingerprint == network.fingerprint()
    housing = load_california_housing(as_housing, seed=0)
    expected = to_dollars(network.predict(housing.X_test), housing.scaler)
    assert np.array_equal(report.test_pred, expected)
    assert report.test == regression_metrics(housing.value_test, expected)
    assert report.test_mse == pytest.approx(network.forward(housing.X_test, housing.y_test))
    # Three epochs on 32 rows already beat the training mean on the test fold.
    assert report.test.r2 > 0.5
    assert report.mean_baseline.r2 < 0.05


def test_evaluate_refuses_weights_trained_on_another_split(as_housing, trained, tmp_path):
    with pytest.raises(WeightsError) as error:
        evaluate(as_housing, trained, seed=1, log=lambda _: None)
    message = str(error.value)
    assert "was not trained on the split that seed 1 gives" in message
    assert "would test it on rows it was trained on" in message
    assert "--seed SEED" in message

    other = tmp_path / "seed1.npz"
    train(as_housing, other, epochs=1, seed=1, log=lambda _: None)
    assert evaluate(as_housing, other, seed=1, log=lambda _: None).seed == 1
    with pytest.raises(WeightsError, match="was not trained on the split that seed 0 gives"):
        evaluate(as_housing, other, seed=0, log=lambda _: None)


def test_evaluate_writes_the_predictions_when_asked(as_housing, trained, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    path = out / "predictions.npz"
    lines = []
    report = evaluate(as_housing, trained, predictions_path=path, log=lines.append)
    assert lines[-1] == f"wrote predicted and published values to {path}"
    assert [p.name for p in out.iterdir()] == ["predictions.npz"]
    housing = load_california_housing(as_housing, seed=0)
    with np.load(path, allow_pickle=False) as archive:
        assert sorted(archive.files) == ["test_pred", "test_true", "train_pred", "train_true"]
        assert np.array_equal(archive["test_pred"], report.test_pred)
        assert np.array_equal(archive["test_true"], housing.value_test)
        assert np.array_equal(archive["train_pred"], report.train_pred)
        assert np.array_equal(archive["train_true"], housing.value_train)


def test_evaluate_writes_no_predictions_by_default(as_housing, trained, tmp_path):
    evaluate(as_housing, trained, log=lambda _: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["weights.npz"]


def test_evaluate_checks_the_predictions_path_before_doing_any_work(as_housing, trained, tmp_path):
    lines = []
    with pytest.raises(ValueError, match=r"predictions file must end in \.npz"):
        evaluate(as_housing, trained, predictions_path=tmp_path / "p.csv", log=lines.append)
    with pytest.raises(ValueError, match="must be different files"):
        evaluate(as_housing, trained, predictions_path=trained, log=lines.append)
    assert lines == []
    assert Network().load(trained) is not None


def test_predictions_path_problem(tmp_path):
    (tmp_path / "folder.npz").mkdir()
    assert predictions_path_problem(tmp_path / "predictions.npz") is None
    assert "must end in .npz" in predictions_path_problem(tmp_path / "predictions.csv")
    assert "does not exist" in predictions_path_problem(tmp_path / "absent" / "p.npz")
    assert "is a directory" in predictions_path_problem(tmp_path / "folder.npz")


def test_evaluate_without_weights_names_the_training_command(as_housing, tmp_path):
    with pytest.raises(WeightsError, match=r"Train first:\n.*california_housing_regression\.train"):
        evaluate(as_housing, tmp_path / "absent.npz", log=lambda _: None)


# ------------------------------------------------------------------------------ the command line


def arguments(data_dir, weights, *extra):
    return ["--data-dir", str(data_dir), "--weights", str(weights), *extra]


def test_main_prints_the_report_and_succeeds(as_housing, trained, capsys):
    assert main(arguments(as_housing, trained)) == 0
    output = capsys.readouterr().out
    assert "  parameters 4,801" in output
    assert "[test]  n=8" in output
    assert "Baselines on the test fold, fitted on the training fold:" in output


def test_main_turns_the_claim_into_a_check(as_housing, trained, capsys):
    assert main(arguments(as_housing, trained, "--min-r2", "0.5")) == 0
    capsys.readouterr()
    assert main(arguments(as_housing, trained, "--min-r2", "0.9999")) == 1
    captured = capsys.readouterr()
    assert "[test]  n=8" in captured.out
    assert captured.err.startswith("error: test R squared 0.")
    assert "is below the required 0.9999" in captured.err
    # A negative bound is a valid one: R squared can be negative.
    assert main(arguments(as_housing, trained, "--min-r2", "-1")) == 0


def test_main_fails_the_check_when_r2_is_undefined(as_housing, trained, monkeypatch, capsys):
    report = score(constant_network(0.0), hand_made_housing())
    undefined = Metrics(n=4, rmse=1.0, mae=1.0, mean_error=0.0, mean_prediction=1.0, r2=None)
    monkeypatch.setattr(evaluate_module, "evaluate", lambda **_: replace(report, test=undefined))
    assert main(arguments(as_housing, trained, "--min-r2", "0")) == 1
    assert "test R squared undefined is below the required 0.0000" in capsys.readouterr().err


def test_main_reports_the_wrong_seed_with_status_one(as_housing, trained, capsys):
    assert main(arguments(as_housing, trained, "--seed", "3")) == 1
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert "was not trained on the split that seed 3 gives" in error


def test_main_writes_the_predictions(as_housing, trained, tmp_path, capsys):
    path = tmp_path / "predictions.npz"
    assert main(arguments(as_housing, trained, "--predictions", str(path))) == 0
    assert path.is_file()
    assert f"wrote predicted and published values to {path}" in capsys.readouterr().out


def test_main_reports_missing_weights_and_missing_data(as_housing, trained, tmp_path, capsys):
    assert main(arguments(as_housing, tmp_path / "absent.npz")) == 1
    error = capsys.readouterr().err
    assert error.startswith("error: the weights file")
    assert "uv run python -m california_housing_regression.train" in error

    assert main(arguments(tmp_path / "no-data", trained)) == 1
    error = capsys.readouterr().err
    assert "uv run python scripts/download_california_housing.py" in error


def test_main_refuses_a_pickle_checkpoint(as_housing, tmp_path, capsys):
    path = tmp_path / "cal_housing_weights.pkl"
    path.write_bytes(b"not even looked at")
    assert main(arguments(as_housing, path)) == 1
    assert (
        "Pickle checkpoints written before version 1.0.0 are not loaded" in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--min-r2", "1.5"], "must be a number no greater than 1"),
        (["--min-r2", "high"], "'high' is not a number"),
        (["--min-r2", "nan"], "must be a number no greater than 1"),
        (["--seed", "-1"], "must be between 0 and 4294967295"),
        (["--seed", "x"], "'x' is not a whole number"),
        (["--predictions", "predictions.csv"], "must end in .npz"),
        (["--checkpoint", "old.pkl"], "unrecognized arguments"),
    ],
)
def test_main_rejects_bad_arguments_before_loading_anything(extra, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(extra)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_main_refuses_to_write_the_predictions_over_the_weights(as_housing, trained, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(arguments(as_housing, trained, "--predictions", str(trained)))
    assert exit_info.value.code == 2
    assert "--predictions and --weights must name different files" in capsys.readouterr().err
    assert Network().load(trained) is not None
