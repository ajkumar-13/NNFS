"""Train the 8-64-64-1 network on California housing from a fixed seed and write its weights.

    uv run python -m california_housing_regression.train                 the documented run
    uv run python -m california_housing_regression.train --epochs 5      a short smoke run

The documented run is seed 0, 200 epochs, mini-batches of 256 reshuffled every epoch, Adam with
learning rate 0.01 and decay 1e-4. The seed fixes the split into folds as well as the initial
weights and the shuffles. The optimiser settings are constants of this module, not options, because
they are part of the baseline that other work cites.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from california_housing_regression.data import (
    DEFAULT_DATA_DIR,
    DOLLARS_PER_UNIT,
    DataError,
    load_california_housing,
)
from california_housing_regression.model import Network
from california_housing_regression.nn import Optimizer_Adam

DEFAULT_WEIGHTS = "cal_housing_weights.npz"
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 0
DEFAULT_LOG_EVERY = 20
LEARNING_RATE = 0.01
DECAY = 1e-4

MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class EpochStats:
    """What one epoch of training reports.

    ``loss`` is the mean over the epoch's batches of the data loss plus the L2 penalty, and ``mse``
    is the same mean of the data loss alone. Both are in the units of the targets.
    """

    epoch: int
    loss: float
    mse: float
    learning_rate: float


def fit(
    network: Network,
    optimizer: Optimizer_Adam,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    log_every: int = 1,
    log: Callable[[str], None] = print,
) -> list[EpochStats]:
    """Mini-batch training: reshuffle every epoch, one forward, backward and update per batch.

    Every epoch is in the returned history. A line is logged for the first epoch, for every
    ``log_every``-th epoch and for the last.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"the batch size must be at least 1, got {batch_size}")
    if log_every < 1:
        raise ValueError(f"log_every must be at least 1, got {log_every}")
    if X.ndim != 2 or X.shape[1] != network.n_inputs:
        raise ValueError(
            f"the network takes rows of {network.n_inputs} features, got an array of shape "
            f"{X.shape}"
        )
    if y.shape != (len(X),):
        raise ValueError(f"expected one target per row, got {y.shape} targets for {len(X)} rows")
    if len(X) == 0:
        raise ValueError("there is nothing to train on: the training set is empty")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("the features and the targets must be finite numbers")

    n_samples = len(X)
    history = []
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n_samples)

        epoch_loss = 0.0
        epoch_mse = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            # Indexing through the permutation gives the batches a shuffled copy would give,
            # without holding a second copy of the dataset.
            batch = order[start : start + batch_size]
            X_batch = X[batch]
            y_batch = y[batch]

            data_loss = network.forward(X_batch, y_batch)
            loss = data_loss + network.regularization_loss()

            network.backward()

            optimizer.pre_update_params()
            for layer in network.dense_layers:
                optimizer.update_params(layer)
            optimizer.post_update_params()

            epoch_loss += float(loss)
            epoch_mse += data_loss
            n_batches += 1

        stats = EpochStats(
            epoch=epoch,
            loss=epoch_loss / n_batches,
            mse=epoch_mse / n_batches,
            learning_rate=optimizer.current_learning_rate,
        )
        history.append(stats)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            log(
                f"epoch {stats.epoch:3d} | loss {stats.loss:.4f} | "
                f"train_mse {stats.mse:.4f} | lr {stats.learning_rate:.6f}"
            )
    return history


def train(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    log_every: int = DEFAULT_LOG_EVERY,
    log: Callable[[str], None] = print,
) -> list[EpochStats]:
    """Seed, load and split the data, build the network, train it, and write the weights."""
    weights_path = Path(weights_path)
    problem = weights_path_problem(weights_path)
    if problem:
        raise ValueError(problem)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"the seed must be between 0 and {MAX_SEED}, got {seed}")

    # The seed is set once, before the first layer draws its weights. Layer initialisation and
    # the shuffles read this generator, in that order. The split into folds draws from its own
    # generator, started from the same seed inside the loader.
    np.random.seed(seed)

    log(f"loading California housing from {data_dir}")
    housing = load_california_housing(data_dir, seed=seed)
    scaler = housing.scaler
    log(f"  train: {housing.X_train.shape}    test: {housing.X_test.shape}")
    log(
        f"  target: mean ${scaler.y_mean * DOLLARS_PER_UNIT:,.0f}, standard deviation "
        f"${scaler.y_std * DOLLARS_PER_UNIT:,.0f} (training fold)"
    )

    network = Network()
    optimizer = Optimizer_Adam(learning_rate=LEARNING_RATE, decay=DECAY)
    steps = (len(housing.X_train) + batch_size - 1) // batch_size
    log(f"  parameters {network.parameter_count():,}")
    log(f"training for {epochs} epochs, batch_size={batch_size} ({steps} steps/epoch), seed={seed}")
    log("")

    started = time.perf_counter()
    history = fit(
        network, optimizer, housing.X_train, housing.y_train, epochs, batch_size, log_every, log
    )
    elapsed = time.perf_counter() - started

    network.save(weights_path, scaler)
    log("")
    log(f"trained in {elapsed:.1f} s")
    log(f"wrote weights to {weights_path}")
    log(f"weights sha256 {network.fingerprint()}")
    return history


def weights_path_problem(path: Path) -> str | None:
    """Why ``path`` cannot receive the weights, or ``None`` when it can.

    Checked before training starts, so a mistyped path costs nothing.
    """
    if path.suffix != ".npz":
        return f"the weights file must end in .npz, got {path}"
    if not path.parent.is_dir():
        return f"the directory {path.parent} does not exist, so the weights cannot be written there"
    if path.is_dir():
        return f"{path} is a directory, not a file"
    return None


def _weights_path(text: str) -> Path:
    path = Path(text)
    problem = weights_path_problem(path)
    if problem:
        raise argparse.ArgumentTypeError(problem)
    return path


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if not 0 <= value <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"must be between 0 and {MAX_SEED}, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m california_housing_regression.train",
        description="Train the 8-64-64-1 network on California housing and write its weights.",
    )
    parser.add_argument(
        "--epochs", type=_positive_int, default=DEFAULT_EPOCHS, help="default: %(default)s"
    )
    parser.add_argument(
        "--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE, help="default: %(default)s"
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=DEFAULT_SEED,
        help="fixes the split, the initial weights and the shuffles (default: %(default)s)",
    )
    parser.add_argument(
        "--log-every",
        type=_positive_int,
        default=DEFAULT_LOG_EVERY,
        help="print a line every this many epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory holding cal_housing.tgz (default: %(default)s)",
    )
    parser.add_argument(
        "--weights",
        type=_weights_path,
        default=DEFAULT_WEIGHTS,
        help="where to write the trained weights, an .npz file (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train(
            data_dir=args.data_dir,
            weights_path=args.weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            log_every=args.log_every,
        )
    except (DataError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
