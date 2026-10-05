"""Score saved weights on both folds of the California housing data, in dollars.

    uv run python -m california_housing_regression.evaluate
    uv run python -m california_housing_regression.evaluate --min-r2 0.82

Prints the parameter count, the fingerprint of the weights, and for each fold the mean squared
error in standardised units and the RMSE, the MAE and R squared in dollars. The test fold is then
split into the block groups at the census cap and those below it, and two baselines fitted on the
training fold are scored on the test fold: the training mean and a linear least-squares fit. With
``--min-r2`` the command exits with status 1 when the test R squared is below the given value,
which turns the documented claim into a check that can fail.

``--seed`` must be the seed the weights were trained with, because the seed fixes the split. The
weights file carries the scaling statistics of its training fold; if they are not the ones this
seed gives, the command stops instead of scoring rows the network was trained on.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from california_housing_regression.data import (
    DEFAULT_DATA_DIR,
    DOLLARS_PER_UNIT,
    TOP_CODE_DOLLARS,
    DataError,
    Housing,
    Scaler,
    destandardise_y,
    load_california_housing,
)
from california_housing_regression.model import Network, WeightsError
from california_housing_regression.train import DEFAULT_SEED, DEFAULT_WEIGHTS, MAX_SEED


@dataclass(frozen=True)
class Metrics:
    """Errors of a set of predictions, in the units of the values (here dollars).

    ``mean_error`` is the mean of prediction minus truth, so a negative figure is
    under-prediction. ``r2`` is ``None`` when every true value is the same, because R squared
    divides by their variance.
    """

    n: int
    rmse: float
    mae: float
    mean_error: float
    mean_prediction: float
    r2: float | None


@dataclass(frozen=True)
class Report:
    """The result of scoring a network on the two folds."""

    parameters: int
    fingerprint: str
    seed: int
    train_mse: float
    train: Metrics
    test_mse: float
    test: Metrics
    at_cap: Metrics | None
    below_cap: Metrics | None
    train_at_cap: int
    mean_baseline: Metrics
    linear_baseline: Metrics
    train_pred: np.ndarray
    test_pred: np.ndarray


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """RMSE, MAE, mean error and R squared of ``y_pred`` against ``y_true``.

    $R^2 = 1 - \\sum (\\hat{y} - y)^2 / \\sum (y - \\bar{y})^2$, which is 0 for a model that always
    predicts the mean of ``y_true`` and negative for a worse one.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(
            f"expected two vectors of one length, got shapes {y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("the metrics are undefined for an empty set")
    residuals = y_pred - y_true
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return Metrics(
        n=y_true.size,
        rmse=math.sqrt(ss_res / y_true.size),
        mae=float(np.mean(np.abs(residuals))),
        mean_error=float(np.mean(residuals)),
        mean_prediction=float(np.mean(y_pred)),
        r2=None if ss_tot == 0 else 1 - ss_res / ss_tot,
    )


def to_dollars(y_scaled: np.ndarray, scaler: Scaler) -> np.ndarray:
    """Standardised predictions as dollars, shape ``(N,)``."""
    return destandardise_y(np.asarray(y_scaled, dtype=np.float64).ravel(), scaler) * (
        DOLLARS_PER_UNIT
    )


def linear_baseline(housing: Housing) -> np.ndarray:
    """Test-fold predictions, in dollars, of a least-squares line fitted on the training fold.

    The model is $\\hat{y} = \\mathbf{x} \\cdot \\mathbf{w} + b$ on the eight standardised
    features, the best that a network with no hidden layer could do under the same loss.
    """
    design = np.column_stack([housing.X_train.astype(np.float64), np.ones(len(housing.X_train))])
    coefficients, *_ = np.linalg.lstsq(design, housing.y_train.astype(np.float64), rcond=None)
    test = np.column_stack([housing.X_test.astype(np.float64), np.ones(len(housing.X_test))])
    return to_dollars(test @ coefficients, housing.scaler)


def score(network: Network, housing: Housing) -> Report:
    """One forward pass over each fold, and every figure the report prints."""
    train_mse = network.forward(housing.X_train, housing.y_train)
    train_pred = to_dollars(network.dense3.output, housing.scaler)
    test_mse = network.forward(housing.X_test, housing.y_test)
    test_pred = to_dollars(network.dense3.output, housing.scaler)

    capped = housing.value_test >= TOP_CODE_DOLLARS
    at_cap = below_cap = None
    if capped.any():
        at_cap = regression_metrics(housing.value_test[capped], test_pred[capped])
    if not capped.all():
        below_cap = regression_metrics(housing.value_test[~capped], test_pred[~capped])

    training_mean = np.full(len(housing.value_test), housing.scaler.y_mean * DOLLARS_PER_UNIT)
    return Report(
        parameters=network.parameter_count(),
        fingerprint=network.fingerprint(),
        seed=housing.seed,
        train_mse=train_mse,
        train=regression_metrics(housing.value_train, train_pred),
        test_mse=test_mse,
        test=regression_metrics(housing.value_test, test_pred),
        at_cap=at_cap,
        below_cap=below_cap,
        train_at_cap=int(np.sum(housing.value_train >= TOP_CODE_DOLLARS)),
        mean_baseline=regression_metrics(housing.value_test, training_mean),
        linear_baseline=regression_metrics(housing.value_test, linear_baseline(housing)),
        train_pred=train_pred,
        test_pred=test_pred,
    )


def _r2(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _dollars(value: float) -> str:
    """A whole number of dollars with thousands separators, the sign before the dollar sign."""
    rounded = round(value)
    return f"-${-rounded:,}" if rounded < 0 else f"${rounded:,}"


def _fold(name: str, mse: float, metrics: Metrics) -> list[str]:
    return [
        "",
        f"[{name}]  n={metrics.n:,}",
        f"  MSE (standardised units)  {mse:.4f}",
        f"  RMSE (dollars)            {_dollars(metrics.rmse)}",
        f"  MAE  (dollars)            {_dollars(metrics.mae)}",
        f"  mean error (dollars)      {_dollars(metrics.mean_error)}",
        f"  R^2                       {_r2(metrics.r2)}",
    ]


def _group(name: str, metrics: Metrics | None) -> str:
    if metrics is None:
        return f"  {name:<14} n=0"
    return (
        f"  {name:<14} n={metrics.n:,}  RMSE {_dollars(metrics.rmse)}  "
        f"MAE {_dollars(metrics.mae)}  mean prediction {_dollars(metrics.mean_prediction)}  "
        f"mean error {_dollars(metrics.mean_error)}"
    )


def _baseline(name: str, metrics: Metrics) -> str:
    return (
        f"  {name:<21} RMSE {_dollars(metrics.rmse)}  MAE {_dollars(metrics.mae)}  "
        f"R^2 {_r2(metrics.r2)}"
    )


def format_report(report: Report) -> str:
    """The report as the text the command prints."""
    lines = [
        f"  parameters {report.parameters:,}",
        f"  weights    sha256 {report.fingerprint}",
        f"  split      seed {report.seed}: {report.train.n:,} training and {report.test.n:,} test "
        "block groups",
    ]
    lines += _fold("train", report.train_mse, report.train)
    lines += _fold("test", report.test_mse, report.test)
    lines += [
        "",
        f"Test fold by top-coding (the census records every value above the cap as "
        f"${TOP_CODE_DOLLARS:,}):",
        _group("at the cap", report.at_cap),
        _group("below the cap", report.below_cap),
        f"  the training fold holds {report.train_at_cap:,} block groups at the cap",
        "",
        "Baselines on the test fold, fitted on the training fold:",
        _baseline("training mean", report.mean_baseline),
        _baseline("linear least squares", report.linear_baseline),
        _baseline("this network", report.test),
    ]
    return "\n".join(lines)


def save_predictions(path: str | Path, report: Report, housing: Housing) -> None:
    """Write predicted and published values of both folds, in dollars, to an ``.npz`` file."""
    path = Path(path)
    partial = path.with_name(path.name + ".part")
    with open(partial, "wb") as handle:
        np.savez(
            handle,
            train_pred=report.train_pred,
            train_true=housing.value_train,
            test_pred=report.test_pred,
            test_true=housing.value_test,
        )
    os.replace(partial, path)


def predictions_path_problem(path: Path) -> str | None:
    """Why ``path`` cannot receive the predictions, or ``None`` when it can."""
    if path.suffix != ".npz":
        return f"the predictions file must end in .npz, got {path}"
    if not path.parent.is_dir():
        return (
            f"the directory {path.parent} does not exist, so the predictions cannot be written "
            "there"
        )
    if path.is_dir():
        return f"{path} is a directory, not a file"
    return None


def evaluate(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    seed: int = DEFAULT_SEED,
    predictions_path: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> Report:
    """Load the weights and the data, check that they belong together, score, print, return."""
    if predictions_path is not None:
        problem = predictions_path_problem(Path(predictions_path))
        if problem:
            raise ValueError(problem)
        if Path(predictions_path).resolve() == Path(weights_path).resolve():
            raise ValueError("the predictions file and the weights file must be different files")

    network = Network()
    log(f"restoring weights from {weights_path}")
    stored = network.load(weights_path)

    log(f"loading California housing from {data_dir}")
    housing = load_california_housing(data_dir, seed=seed)
    if not stored.matches(housing.scaler):
        raise WeightsError(
            f"{weights_path} was not trained on the split that seed {seed} gives: the scaling "
            "statistics stored in it are not those of this training fold. Scoring it here would "
            "test it on rows it was trained on. Pass the seed the weights were trained with:\n"
            "    uv run python -m california_housing_regression.evaluate --seed SEED --weights "
            f"{weights_path}"
        )

    report = score(network, housing)
    log(format_report(report))
    if predictions_path is not None:
        save_predictions(predictions_path, report, housing)
        log("")
        log(f"wrote predicted and published values to {predictions_path}")
    return report


def _r2_bound(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not (math.isfinite(value) and value <= 1):
        raise argparse.ArgumentTypeError(f"must be a number no greater than 1, got {text}")
    return value


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if not 0 <= value <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"must be between 0 and {MAX_SEED}, got {value}")
    return value


def _predictions_path(text: str) -> Path:
    path = Path(text)
    problem = predictions_path_problem(path)
    if problem:
        raise argparse.ArgumentTypeError(problem)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m california_housing_regression.evaluate",
        description="Score saved weights on both folds of the California housing data.",
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=DEFAULT_SEED,
        help="the seed the weights were trained with; it fixes the split (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory holding cal_housing.tgz (default: %(default)s)",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="the .npz file written by the trainer (default: %(default)s)",
    )
    parser.add_argument(
        "--min-r2",
        type=_r2_bound,
        default=None,
        help="exit with status 1 if the test R squared is below this value",
    )
    parser.add_argument(
        "--predictions",
        type=_predictions_path,
        default=None,
        help="also write predicted and published values in dollars to this .npz file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.predictions is not None and args.predictions.resolve() == Path(args.weights).resolve():
        parser.error("--predictions and --weights must name different files")
    try:
        report = evaluate(
            data_dir=args.data_dir,
            weights_path=args.weights,
            seed=args.seed,
            predictions_path=args.predictions,
        )
    except (DataError, WeightsError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.min_r2 is not None and (report.test.r2 is None or report.test.r2 < args.min_r2):
        print(
            f"error: test R squared {_r2(report.test.r2)} is below the required {args.min_r2:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
