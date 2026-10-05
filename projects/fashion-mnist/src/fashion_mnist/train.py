"""Train the 784-128-128-10 network on Fashion-MNIST from a fixed seed and write its weights.

    uv run python -m fashion_mnist.train                  the documented run
    uv run python -m fashion_mnist.train --epochs 1       a short smoke run

The documented run is seed 0, 20 epochs, mini-batches of 128 reshuffled every epoch, Adam with
learning rate 0.001 and decay 1e-4. The optimiser settings are constants of this module, not
options, because they are part of the baseline that other work cites.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fashion_mnist.data import DEFAULT_DATA_DIR, DataError, load_fashion_mnist
from fashion_mnist.model import Network
from fashion_mnist.nn import Optimizer_Adam

DEFAULT_WEIGHTS = "fashion_mnist_weights.npz"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 128
DEFAULT_SEED = 0
LEARNING_RATE = 0.001
DECAY = 1e-4

MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class EpochStats:
    """What one epoch of training reports."""

    epoch: int
    loss: float
    accuracy: float
    learning_rate: float


def fit(
    network: Network,
    optimizer: Optimizer_Adam,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    log: Callable[[str], None] = print,
) -> list[EpochStats]:
    """Mini-batch training: reshuffle every epoch, one forward, backward and update per batch.

    The reported loss is the mean over batches of data loss plus L2 penalty, and the reported
    accuracy counts the predictions made during training, with dropout active.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"the batch size must be at least 1, got {batch_size}")
    if X.ndim != 2 or X.shape[1] != network.n_inputs:
        raise ValueError(
            f"the network takes rows of {network.n_inputs} features, got an array of shape "
            f"{X.shape}"
        )
    if y.shape != (len(X),):
        raise ValueError(f"expected one label per row, got {y.shape} labels for {len(X)} rows")
    if len(X) == 0:
        raise ValueError("there is nothing to train on: the training set is empty")
    if y.min() < 0 or y.max() >= network.n_classes:
        raise ValueError(
            f"labels must be class indices from 0 to {network.n_classes - 1}, got values from "
            f"{y.min()} to {y.max()}"
        )

    n_samples = len(X)
    history = []
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n_samples)

        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            # Indexing through the permutation gives the batches a shuffled copy would give,
            # without holding a second copy of the dataset.
            batch = order[start : start + batch_size]
            X_batch = X[batch]
            y_batch = y[batch]

            data_loss = network.forward(X_batch, y_batch, training=True)
            loss = data_loss + network.regularization_loss()
            epoch_correct += int(np.sum(network.predictions() == y_batch))

            network.backward(y_batch)

            optimizer.pre_update_params()
            for layer in network.dense_layers:
                optimizer.update_params(layer)
            optimizer.post_update_params()

            epoch_loss += float(loss)
            n_batches += 1

        stats = EpochStats(
            epoch=epoch,
            loss=epoch_loss / n_batches,
            accuracy=epoch_correct / n_samples,
            learning_rate=optimizer.current_learning_rate,
        )
        history.append(stats)
        log(
            f"epoch {stats.epoch:3d} | loss {stats.loss:.4f} | "
            f"train_acc {stats.accuracy:.4f} | lr {stats.learning_rate:.6f}"
        )
    return history


def train(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    log: Callable[[str], None] = print,
) -> list[EpochStats]:
    """Seed, load the data, build the network, train it, and write the weights."""
    weights_path = Path(weights_path)
    problem = weights_path_problem(weights_path)
    if problem:
        raise ValueError(problem)

    # The seed is set once, before the first layer draws its weights. Layer initialisation, the
    # shuffles and the dropout masks all read the same generator, in that order.
    np.random.seed(seed)

    log(f"loading Fashion-MNIST from {data_dir}")
    X_train, y_train, X_test, _ = load_fashion_mnist(data_dir)
    log(f"  train: {X_train.shape}    test: {X_test.shape}")

    network = Network()
    if X_train.shape[1] != network.n_inputs:
        raise DataError(
            f"the network takes {network.n_inputs} pixels per image, but the training images "
            f"have {X_train.shape[1]}"
        )
    optimizer = Optimizer_Adam(learning_rate=LEARNING_RATE, decay=DECAY)
    steps = (len(X_train) + batch_size - 1) // batch_size
    log(f"  parameters {network.parameter_count():,}")
    log(f"training for {epochs} epochs, batch_size={batch_size} ({steps} steps/epoch), seed={seed}")
    log("")

    started = time.perf_counter()
    history = fit(network, optimizer, X_train, y_train, epochs, batch_size, log)
    elapsed = time.perf_counter() - started

    network.save(weights_path)
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
        prog="python -m fashion_mnist.train",
        description="Train the 784-128-128-10 network on Fashion-MNIST and write its weights.",
    )
    parser.add_argument(
        "--epochs", type=_positive_int, default=DEFAULT_EPOCHS, help="default: %(default)s"
    )
    parser.add_argument(
        "--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE, help="default: %(default)s"
    )
    parser.add_argument("--seed", type=_seed, default=DEFAULT_SEED, help="default: %(default)s")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory holding the four Fashion-MNIST files (default: %(default)s)",
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
        )
    except (DataError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except MemoryError:
        print(
            "error: not enough free memory. The 70,000 images alone take 220 MB as float32; "
            "close other programs and run again.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
