"""Train and score the model for several seeds and every initialisation.

    uv run python scripts/seed_sweep.py
    uv run python scripts/seed_sweep.py --seeds 3 --noise 0.2

A seed fixes the data, the split, and the initial weights together, so each
row is a different dataset as well as a different starting point. Every run is
the documented one (2,000 full-batch epochs) apart from its seed, its
initialisation, and the noise if one is given. The last column is the R squared
of a plane fitted to the logits over the training set: 1 means the network is a
linear classifier there. Nothing is written to disk.
"""

import argparse
import contextlib
import io
import sys

from binary_classifier.evaluate import logit_linearity, score
from binary_classifier.model import INIT_CHOICES
from binary_classifier.train import train


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python scripts/seed_sweep.py",
        description="Train and score the model for several seeds and every initialisation.",
    )
    parser.add_argument(
        "--seeds", type=int, default=10, help="number of seeds, counted from 0 (default: 10)"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.10,
        help="standard deviation of the noise on each coordinate (default: 0.1)",
    )
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error(f"--seeds must be at least 1, got {args.seeds}")

    print(f"noise {args.noise:g}, 2,000 epochs, seeds 0 to {args.seeds - 1}")
    print("init     seed   train correct   test correct   plane R^2")
    test_counts = {}
    for init in INIT_CHOICES:
        for seed in range(args.seeds):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = train(noise=args.noise, seed=seed, init=init, weights_path=None)
            except ValueError as err:
                parser.error(str(err))
            X_train, y_train, X_test, y_test = result.split
            on_train = score(result.model, X_train, y_train)
            on_test = score(result.model, X_test, y_test)
            test_counts.setdefault(init, []).append(on_test.correct)
            print(
                f"{init:<8} {seed:>4}   {on_train.correct:>7}/{on_train.total}"
                f"   {on_test.correct:>8}/{on_test.total}"
                f"   {logit_linearity(result.model, X_train):>9.4f}"
            )

    print(f"\ntest points correct, of {on_test.total}, over the {args.seeds} seeds")
    for init, counts in test_counts.items():
        print(
            f"{init:<8} lowest {min(counts)}   highest {max(counts)}   "
            f"mean {sum(counts) / len(counts):.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
