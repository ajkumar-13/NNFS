"""The evaluation metrics on hand-made cases, and the evaluation of a tiny trained run."""

import math

import numpy as np
import pytest

from binary_classifier.evaluate import (
    boundary_band,
    confusion_matrix,
    decision_grid,
    evaluate,
    logit_linearity,
    score,
    unit_activity,
)
from binary_classifier.model import WeightsError

SIGMOID_10 = 1.0 / (1.0 + math.exp(-10.0))
SIGMOID_5 = 1.0 / (1.0 + math.exp(-5.0))


# --- confusion_matrix ----------------------------------------------------------------------


def test_confusion_matrix_hand_made():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    # One true 0 kept, one true 0 called 1, one true 1 called 0, two true 1 kept.
    confusion = confusion_matrix(y_true, y_pred)
    np.testing.assert_array_equal(confusion, [[1, 1], [1, 2]])
    assert confusion.sum() == 5


def test_confusion_matrix_of_a_single_class():
    np.testing.assert_array_equal(confusion_matrix([1, 1, 1], [1, 0, 1]), [[0, 0], [1, 2]])


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([0, 1], [0, 1, 1], "must be one-dimensional and equally long"),
        ([[0, 1]], [[0, 1]], "must be one-dimensional and equally long"),
        ([0, 2], [0, 1], "y_true must hold only the labels 0 and 1"),
        ([0, 1], [0, -1], "y_pred must hold only the labels 0 and 1"),
    ],
)
def test_confusion_matrix_rejects_bad_input(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        confusion_matrix(y_true, y_pred)


# --- score ---------------------------------------------------------------------------------


def test_score_hand_made(ramp_model):
    # The logit is 10 * x0, so the points at x0 = 1 are called 1 and those at x0 = -1 are called 0.
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    y = np.array([1, 0, 0, 1])
    metrics = score(ramp_model, X, y)
    assert (metrics.correct, metrics.total) == (2, 4)
    assert metrics.accuracy == 0.5
    np.testing.assert_array_equal(metrics.confusion, [[1, 1], [1, 1]])
    # Two points are right with probability sigmoid(10), two are wrong with the same confidence:
    # the loss is the mean of 2 * ln(1 + e^-10) and 2 * ln(1 + e^10).
    expected = (2 * math.log1p(math.exp(-10.0)) + 2 * math.log1p(math.exp(10.0))) / 4
    assert metrics.loss == pytest.approx(expected, rel=1e-12)


def test_score_of_a_perfect_classifier(ramp_model):
    X = np.array([[0.5, 3.0], [-0.5, 3.0], [2.0, -1.0]])
    metrics = score(ramp_model, X, np.array([1, 0, 1]))
    assert (metrics.correct, metrics.total, metrics.accuracy) == (3, 3, 1.0)
    np.testing.assert_array_equal(metrics.confusion, [[1, 0], [0, 2]])


# --- decision_grid and boundary_band -------------------------------------------------------


def test_decision_grid_hand_made(ramp_model):
    XX, YY, probs = decision_grid(ramp_model, x_range=(-1.0, 1.0), y_range=(0.0, 3.0), n=5)
    assert XX.shape == YY.shape == probs.shape == (5, 5)
    # Columns run over x, rows over y.
    np.testing.assert_allclose(XX[0], [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(YY[:, 0], [0.0, 0.75, 1.5, 2.25, 3.0])
    # The probability is sigmoid(10 * x), the same in every row.
    row = [1 - SIGMOID_10, 1 - SIGMOID_5, 0.5, SIGMOID_5, SIGMOID_10]
    np.testing.assert_allclose(probs, np.tile(row, (5, 1)), rtol=1e-9)


def test_decision_grid_defaults(ramp_model):
    XX, YY, probs = decision_grid(ramp_model)
    assert probs.shape == (200, 200)
    assert (XX.min(), XX.max(), YY.min(), YY.max()) == (-1.5, 2.5, -1.0, 1.5)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_boundary_band_hand_made():
    probs = np.array([[0.10, 0.50], [0.46, 0.55]])
    # 0.50 and 0.46 lie strictly inside (0.45, 0.55); 0.10 and the end point 0.55 do not.
    assert boundary_band(probs) == 0.5
    assert boundary_band(probs, low=0.0, high=1.0) == 1.0


# --- unit_activity and logit_linearity -----------------------------------------------------


def test_unit_activity_hand_made(ramp_model):
    # Hidden unit 0 carries relu(x0) and hidden unit 1 carries relu(-x0), in both layers; the
    # other 14 units of each layer carry zeros. With one point on each side, both switch.
    X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert unit_activity(ramp_model, X) == [(14, 0, 2), (14, 0, 2)]
    # With every point on the positive side, unit 0 is always active and unit 1 never is.
    X = np.array([[1.0, 0.0], [0.5, 2.0]])
    assert unit_activity(ramp_model, X) == [(15, 1, 0), (15, 1, 0)]


def test_logit_linearity_hand_made(ramp_model):
    # The logit is 10 * x0 everywhere, so a plane fits it exactly.
    X = np.array([[-1.0, 0.3], [0.0, -2.0], [1.0, 0.7], [0.4, 0.1]])
    assert logit_linearity(ramp_model, X) == pytest.approx(1.0, abs=1e-12)

    # Without the unit that carries relu(-x0) the logit is 10 * relu(x0), which has a kink.
    ramp_model.dense3.weights[1, 0] = 0.0
    X = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    # Logits 0, 0, 10. The best line is 5 * x0 + 10 / 3, with residuals 5/3, -10/3, 5/3:
    # R^2 = 1 - (50 / 3) / (200 / 3) = 0.75.
    assert logit_linearity(ramp_model, X) == pytest.approx(0.75, rel=1e-12)

    # Constant logits count as linear.
    ramp_model.dense3.weights[:] = 0.0
    assert logit_linearity(ramp_model, X) == 1.0


# --- evaluate ------------------------------------------------------------------------------


def test_evaluate_a_tiny_run(tiny_run, capsys):
    results = evaluate(tiny_run.weights_path)
    out = capsys.readouterr().out

    # 32 training points fitted exactly; 8 test points.
    assert (results["train"].correct, results["train"].total) == (32, 32)
    assert results["test"].total == 8
    assert results["train"].loss < 0.05
    assert results["grid"] == tiny_run.weights_path.with_name("decision_grid.npz")

    lines = out.splitlines()
    assert lines[0].endswith("tiny.npz (init he, noise 0.05, 40 points, seed 0, 200 epochs)")
    assert lines[1] == "parameters: 337"
    assert "[train] loss " in out
    assert "| accuracy 1.0000 (32/32)" in out
    assert "[test] loss " in out
    assert "hidden units over the training set: never active | always active | switching" in out
    assert [sum(counts) for counts in results["activity"]] == [16, 16]
    assert f"logits fitted by a plane over the training set: R^2 {results['linearity']:.4f}" in out
    assert 0.0 <= results["linearity"] < 1.0
    assert "decision grid: 200 x 200 points, x from -1.5 to 2.5, y from -1 to 1.5" in out
    assert lines[-1].startswith("wrote decision grid to ")

    with np.load(results["grid"], allow_pickle=False) as grid:
        assert sorted(grid.files) == ["XX", "X_test", "X_train", "YY", "probs", "y_test", "y_train"]
        assert grid["probs"].shape == (200, 200)
        np.testing.assert_array_equal(grid["X_test"], tiny_run.split[2])
        assert results["band"] == pytest.approx(
            float(np.mean((grid["probs"] > 0.45) & (grid["probs"] < 0.55)))
        )


def test_evaluate_reports_the_confusion_matrix_it_computed(tiny_run, capsys):
    results = evaluate(tiny_run.weights_path)
    out = capsys.readouterr().out
    confusion = results["test"].confusion
    assert confusion.sum() == 8
    expected = (
        f"class 0: {confusion[0, 0]} of {confusion[0].sum()} correct | "
        f"class 1: {confusion[1, 1]} of {confusion[1].sum()} correct"
    )
    assert expected in out.split("[test]")[1]


def test_evaluate_writes_the_grid_where_it_is_told(tiny_run, tmp_path, capsys):
    target = tmp_path / "elsewhere.npz"
    results = evaluate(tiny_run.weights_path, grid=target)
    assert results["grid"] == target
    assert target.is_file()
    assert not tiny_run.weights_path.with_name("decision_grid.npz").exists()


def test_evaluate_refuses_a_bad_grid_path_before_printing(tiny_run, tmp_path, capsys):
    with pytest.raises(WeightsError, match=r"the decision grid name must end in .npz"):
        evaluate(tiny_run.weights_path, grid=tmp_path / "grid.csv")
    with pytest.raises(WeightsError, match=r"does not exist"):
        evaluate(tiny_run.weights_path, grid=tmp_path / "missing" / "grid.npz")
    assert capsys.readouterr().out == ""


def test_evaluate_reports_a_grid_write_failure(tiny_run, tmp_path, capsys):
    blocked = tmp_path / "blocked.npz"
    blocked.mkdir()
    with pytest.raises(WeightsError, match=r"could not write the decision grid"):
        evaluate(tiny_run.weights_path, grid=blocked)


def test_evaluate_without_a_weights_file_says_how_to_make_one(tmp_path):
    with pytest.raises(WeightsError, match=r"no weights file at .* Train one with"):
        evaluate(tmp_path / "moons_weights.npz")
