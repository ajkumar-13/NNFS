"""Score trained weights and sample their decision boundary.

    uv run python -m binary_classifier.evaluate
    uv run python -m binary_classifier.evaluate --weights small.npz --grid small_grid.npz
    uv run python -m binary_classifier.evaluate --min-accuracy 1.0

Prints, for the training set and the test set stored in the weights file, the
mean binary cross-entropy (without the L2 term), the accuracy, the confusion
matrix, and the number of points of each class classified correctly. Two
diagnostics follow, both over the training set: how many hidden units switch
between active and inactive, and how nearly the logits are a linear function of
the input. Then the predicted probability is sampled on a 200 by 200 grid and
written, with the two sets of points, to a .npz file that a plotting program
can read.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binary_classifier.model import (
    DEFAULT_WEIGHTS,
    WeightsError,
    load_weights,
    output_path,
)

GRID_NAME = "decision_grid.npz"
X_RANGE = (-1.5, 2.5)
Y_RANGE = (-1.0, 1.5)
GRID_POINTS = 200


@dataclass
class Metrics:
    """Loss, counts, and the confusion matrix (rows: true class, columns: predicted class)."""

    loss: float
    correct: int
    total: int
    confusion: np.ndarray

    @property
    def accuracy(self):
        return self.correct / self.total


def confusion_matrix(y_true, y_pred):
    """The 2 by 2 matrix of counts: entry [t, p] is the number of class-t points predicted p."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be one-dimensional and equally long; "
            f"got shapes {y_true.shape} and {y_pred.shape}"
        )
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must hold only the labels 0 and 1")
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return confusion


def score(model, X, y):
    """Loss and classification counts of model on the points X with labels y."""
    loss = model.data_loss(X, y)
    predictions = (model.loss_activation.output >= 0.5).astype(np.int64).ravel()
    confusion = confusion_matrix(y, predictions)
    return Metrics(loss=loss, correct=int(np.trace(confusion)), total=len(y), confusion=confusion)


def decision_grid(model, x_range=X_RANGE, y_range=Y_RANGE, n=GRID_POINTS):
    """Sample the predicted probability of class 1 on an n by n grid.

    Returns XX, YY, probs, each of shape (n, n): row i is the i-th value of y,
    column j is the j-th value of x.
    """
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)
    probs = model.predict_proba(grid).reshape(n, n)
    return XX, YY, probs


def boundary_band(probs, low=0.45, high=0.55):
    """Fraction of grid cells whose probability lies strictly between low and high."""
    return float(np.mean((probs > low) & (probs < high)))


def unit_activity(model, X):
    """Count, for each hidden layer, how its 16 units behave over the points X.

    A unit is active on a point when its ReLU output there is positive. Returns
    one (never, always, switching) triple per hidden layer: the units active on
    no point, on every point, and on some points but not others. Only a
    switching unit can bend the decision boundary.
    """
    model.forward(X)
    counts = []
    for activation in (model.activation1, model.activation2):
        active = activation.output > 0
        never = int((~active.any(axis=0)).sum())
        always = int(active.all(axis=0).sum())
        counts.append((never, always, active.shape[1] - never - always))
    return counts


def logit_linearity(model, X):
    """R squared of the least-squares plane fitted to the logits over the points X.

    A value of 1 means that on these points the network is a linear classifier,
    with a straight line for a decision boundary; a network that bends its
    boundary scores lower. Constant logits count as linear.
    """
    logits = model.forward(X).ravel()
    design = np.column_stack([X, np.ones(len(X))]).astype(np.float64)
    coefficients = np.linalg.lstsq(design, logits, rcond=None)[0]
    residual = logits - design @ coefficients
    centred = logits - logits.mean()
    total = float(centred @ centred)
    if total == 0.0:
        return 1.0
    return 1.0 - float(residual @ residual) / total


def _report(name, metrics):
    confusion = metrics.confusion
    print(
        f"\n[{name}] loss {metrics.loss:.4f} | accuracy {metrics.accuracy:.4f} "
        f"({metrics.correct:,}/{metrics.total:,})"
    )
    print("            predicted 0   predicted 1")
    for label in (0, 1):
        print(f"  true {label}   {confusion[label, 0]:>11,}   {confusion[label, 1]:>11,}")
    per_class = " | ".join(
        f"class {label}: {confusion[label, label]:,} of {confusion[label].sum():,} correct"
        for label in (0, 1)
    )
    print(f"  {per_class}")


def evaluate(weights_path=DEFAULT_WEIGHTS, grid=None):
    """Print the metrics of a weights file and write its decision grid.

    grid is the .npz file to write; None means decision_grid.npz beside the
    weights file. Returns a dict with the keys train and test (Metrics), activity
    and linearity (the two diagnostics, over the training set), band, and grid
    (the path written). Raises WeightsError when the weights file cannot be
    read or the grid cannot be written.
    """
    weights_path = Path(weights_path)
    model, split, config = load_weights(weights_path)
    grid = output_path(
        weights_path.with_name(GRID_NAME) if grid is None else grid, what="decision grid"
    )
    X_train, y_train, X_test, y_test = split

    print(
        f"weights: {weights_path} (init {config['init']}, noise {config['noise']:g}, "
        f"{config['n_samples']:,} points, seed {config['seed']}, "
        f"{config['epochs']:,} epochs)"
    )
    print(f"parameters: {model.parameter_count():,}")

    train_metrics = score(model, X_train, y_train)
    _report("train", train_metrics)
    test_metrics = score(model, X_test, y_test)
    _report("test", test_metrics)

    activity = unit_activity(model, X_train)
    linearity = logit_linearity(model, X_train)
    print("\nhidden units over the training set: never active | always active | switching")
    for number, (never, always, switching) in enumerate(activity, start=1):
        print(f"  layer {number}: {never:>2} | {always:>2} | {switching:>2}")
    print(f"logits fitted by a plane over the training set: R^2 {linearity:.4f}")

    XX, YY, probs = decision_grid(model)
    band = boundary_band(probs)
    print(
        f"\ndecision grid: {GRID_POINTS} x {GRID_POINTS} points, "
        f"x from {X_RANGE[0]:g} to {X_RANGE[1]:g}, y from {Y_RANGE[0]:g} to {Y_RANGE[1]:g}"
    )
    print(f"  probability range: {probs.min():.3f} to {probs.max():.3f}")
    print(f"  cells with 0.45 < p < 0.55: {band:.3f} of the grid")

    try:
        np.savez(
            grid,
            XX=XX,
            YY=YY,
            probs=probs,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    except OSError as err:
        raise WeightsError(f"could not write the decision grid {grid}: {err}") from err
    print(f"wrote decision grid to {grid}")
    return {
        "train": train_metrics,
        "test": test_metrics,
        "activity": activity,
        "linearity": linearity,
        "band": band,
        "grid": grid,
    }


def _fraction(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be a fraction between 0 and 1, got {value}")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m binary_classifier.evaluate",
        description="Score trained weights and sample their decision boundary.",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"the .npz file written by training (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--grid",
        default=None,
        help=f"the .npz file to write the decision grid to "
        f"(default: {GRID_NAME} beside the weights file)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=_fraction,
        default=None,
        help="exit with status 1 if the test accuracy is below this fraction",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        results = evaluate(weights_path=args.weights, grid=args.grid)
    except (WeightsError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    accuracy = results["test"].accuracy
    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        print(
            f"error: test accuracy {accuracy:.4f} is below the required {args.min_accuracy:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
