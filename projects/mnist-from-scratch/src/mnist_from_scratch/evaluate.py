"""Score saved weights on the 10,000 MNIST test images.

    uv run python -m mnist_from_scratch.evaluate
    uv run python -m mnist_from_scratch.evaluate --min-accuracy 0.975

Prints the parameter count, the fingerprint of the weights, the test loss and accuracy, the
accuracy of each digit, the confusion matrix, and the first misclassified test indices. With
``--min-accuracy`` the command exits with status 1 when the test accuracy is below the given
fraction, which turns the documented claim into a check that can fail.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mnist_from_scratch.data import DEFAULT_DATA_DIR, DataError, load_test_set
from mnist_from_scratch.model import Network, WeightsError
from mnist_from_scratch.train import DEFAULT_WEIGHTS

FIRST_ERRORS_SHOWN = 20


@dataclass(frozen=True)
class Report:
    """The result of scoring a network on a labelled set."""

    parameters: int
    fingerprint: str
    loss: float
    accuracy: float
    n_samples: int
    n_wrong: int
    confusion: np.ndarray
    wrong_indices: np.ndarray


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts with the true class in rows and the predicted class in columns."""
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(
            f"expected two label vectors of one length, got shapes {y_true.shape} and "
            f"{y_pred.shape}"
        )
    for name, labels in (("true", y_true), ("predicted", y_pred)):
        if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
            raise ValueError(f"{name} labels must lie between 0 and {n_classes - 1}")
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    return confusion


def accuracy(confusion: np.ndarray) -> float:
    """The fraction of samples on the diagonal of a confusion matrix."""
    total = int(confusion.sum())
    if total == 0:
        raise ValueError("accuracy is undefined for an empty set")
    return int(np.trace(confusion)) / total


def per_class_accuracy(confusion: np.ndarray) -> list[tuple[int, float, int]]:
    """``(class, accuracy, samples)`` for every class that has at least one sample."""
    rows = []
    for label in range(len(confusion)):
        samples = int(confusion[label].sum())
        if samples:
            rows.append((label, int(confusion[label, label]) / samples, samples))
    return rows


def score(network: Network, X: np.ndarray, y: np.ndarray) -> Report:
    """One forward pass with dropout off, and every metric the report prints."""
    loss = network.forward(X, y, training=False)
    predictions = network.predictions()
    confusion = confusion_matrix(y, predictions, network.n_classes)
    wrong = np.flatnonzero(predictions != y)
    return Report(
        parameters=network.parameter_count(),
        fingerprint=network.fingerprint(),
        loss=loss,
        accuracy=accuracy(confusion),
        n_samples=len(y),
        n_wrong=len(wrong),
        confusion=confusion,
        wrong_indices=wrong,
    )


def format_report(report: Report) -> str:
    """The report as the text the command prints."""
    classes = range(len(report.confusion))
    lines = [
        f"  parameters {report.parameters:,}",
        f"  weights    sha256 {report.fingerprint}",
        "",
        f"  loss     {report.loss:.4f}",
        f"  accuracy {report.accuracy:.4f}  "
        f"({report.n_samples - report.n_wrong}/{report.n_samples})",
        "",
        "Per-class accuracy:",
    ]
    for label, class_accuracy, samples in per_class_accuracy(report.confusion):
        lines.append(f"  digit {label}: {class_accuracy:.4f}  ({samples} samples)")
    lines += [
        "",
        "Confusion matrix (rows = true class, cols = predicted class):",
        "         " + "  ".join(f"{label:>5d}" for label in classes),
    ]
    for label in classes:
        row = "  ".join(f"{int(count):>5d}" for count in report.confusion[label])
        lines.append(f"  true {label}: {row}")
    lines += [
        "",
        f"{report.n_wrong} misclassified samples. First {FIRST_ERRORS_SHOWN} indices: "
        f"{report.wrong_indices[:FIRST_ERRORS_SHOWN].tolist()}",
    ]
    return "\n".join(lines)


def evaluate(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    log: Callable[[str], None] = print,
) -> Report:
    """Load the weights and the test set, score, print the report and return it."""
    network = Network()
    log(f"restoring weights from {weights_path}")
    network.load(weights_path)

    log(f"loading MNIST from {data_dir}")
    X_test, y_test = load_test_set(data_dir)
    if X_test.shape[1] != network.n_inputs:
        raise DataError(
            f"the network takes {network.n_inputs} pixels per image, but the test images have "
            f"{X_test.shape[1]}"
        )

    report = score(network, X_test, y_test)
    log(format_report(report))
    return report


def _fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be a fraction between 0 and 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mnist_from_scratch.evaluate",
        description="Score saved weights on the 10,000 MNIST test images.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory holding the four MNIST files (default: %(default)s)",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="the .npz file written by the trainer (default: %(default)s)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=_fraction,
        default=None,
        help="exit with status 1 if the test accuracy is below this fraction",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate(data_dir=args.data_dir, weights_path=args.weights)
    except (DataError, WeightsError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except MemoryError:
        print(
            "error: not enough free memory. The 10,000 test images take 31 MB as float32; "
            "close other programs and run again.",
            file=sys.stderr,
        )
        return 1
    if args.min_accuracy is not None and report.accuracy < args.min_accuracy:
        print(
            f"error: test accuracy {report.accuracy:.4f} is below the required "
            f"{args.min_accuracy:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
