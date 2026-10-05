"""Train the 2-16-16-1 classifier on two moons and write a weights file.

    uv run python -m binary_classifier.train                  the documented run
    uv run python -m binary_classifier.train --init small     the initialisation that underfits
    uv run python -m binary_classifier.train --noise 0.2      a harder dataset

The documented run: 1,000 points at noise 0.1, split 800 to 200, He
initialisation, Adam at learning rate 0.01 with no decay, an L2 penalty of
1e-4 on the two hidden layers, 2,000 full-batch epochs, seed 0.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binary_classifier.data import dataset_checksum, make_moons, train_test_split
from binary_classifier.model import (
    DEFAULT_WEIGHTS,
    INIT_CHOICES,
    L2_LAMBDA,
    Network,
    WeightsError,
    output_path,
    save_weights,
)
from binary_classifier.nn import Optimizer_Adam

LEARNING_RATE = 0.01
TEST_FRACTION = 0.2


@dataclass
class TrainResult:
    """What a training run leaves behind.

    history holds one (epoch, loss, train_accuracy) entry for every epoch that
    was printed, the last epoch always among them. Each entry is measured
    before that epoch's update, and the loss includes the L2 term;
    binary_classifier.evaluate scores the final weights.
    """

    model: Network
    split: tuple
    config: dict
    history: list
    seconds: float
    weights_path: Path | None

    @property
    def loss(self):
        return self.history[-1][1]

    @property
    def train_accuracy(self):
        return self.history[-1][2]


def _check_count(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a whole number of at least 1, got {value!r}")


def train(
    epochs=2000,
    noise=0.10,
    n_samples=1000,
    weights_path=DEFAULT_WEIGHTS,
    seed=0,
    log_every=200,
    init="he",
):
    """Train the model and return a TrainResult.

    weights_path is the .npz file to write; None trains without writing one.
    Raises ValueError for an argument out of range and WeightsError for a
    weights path that cannot be written, both before any training is done.
    """
    _check_count("epochs", epochs)
    _check_count("log_every", log_every)
    if weights_path is not None:
        weights_path = output_path(weights_path)

    X, y = make_moons(n_samples=n_samples, noise=noise, seed=seed)
    split = train_test_split(X, y, test_frac=TEST_FRACTION, seed=seed)
    X_train, y_train, X_test, _ = split

    # The dense layers draw their initial weights from NumPy's global generator.
    np.random.seed(seed)
    model = Network(init=init)
    optimizer = Optimizer_Adam(learning_rate=LEARNING_RATE)

    print(
        f"two moons: {n_samples:,} points, noise {noise:g}, seed {seed} | "
        f"train {X_train.shape}, test {X_test.shape}"
    )
    print(f"data sha256: {dataset_checksum(X, y)}")
    print(
        f"model: 2-16-16-1, {model.parameter_count():,} parameters, init {init} | "
        f"Adam lr {LEARNING_RATE:g}, L2 {L2_LAMBDA:g}, full batch"
    )

    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        data_loss = model.data_loss(X_train, y_train)
        loss = data_loss + model.regularization_loss()

        predictions = (model.loss_activation.output >= 0.5).astype(np.int64).ravel()
        train_accuracy = float(np.mean(predictions == y_train))

        model.backward(y_train)

        optimizer.pre_update_params()
        for layer in model.dense_layers:
            optimizer.update_params(layer)
        optimizer.post_update_params()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            history.append((epoch, float(loss), train_accuracy))
            print(f"epoch {epoch:5d} | loss {loss:.4f} | train_acc {train_accuracy:.4f}")
    seconds = time.perf_counter() - started
    print(f"trained {epochs:,} {'epoch' if epochs == 1 else 'epochs'} in {seconds:.2f} s")

    config = {
        "init": init,
        "noise": float(noise),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "epochs": int(epochs),
    }
    if weights_path is not None:
        save_weights(weights_path, model, split, config)
        print(f"wrote weights to {weights_path}")
    return TrainResult(
        model=model,
        split=split,
        config=config,
        history=history,
        seconds=seconds,
        weights_path=weights_path,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m binary_classifier.train",
        description="Train the 2-16-16-1 classifier on two moons and write a weights file.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2000, help="full-batch epochs, at least 1 (default: 2000)"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.10,
        help="standard deviation of the noise on each coordinate, not negative (default: 0.1)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="points to generate before the 80 to 20 split (default: 1000)",
    )
    parser.add_argument(
        "--init", choices=INIT_CHOICES, default="he", help="weight initialisation (default: he)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the data, the split, and the weights (default: 0)",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"the .npz file to write (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="print the loss every this many epochs (default: 200)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train(
            epochs=args.epochs,
            noise=args.noise,
            n_samples=args.n_samples,
            weights_path=args.weights,
            seed=args.seed,
            log_every=args.log_every,
            init=args.init,
        )
    except (ValueError, WeightsError) as err:
        parser.error(str(err))
    return 0


if __name__ == "__main__":
    sys.exit(main())
