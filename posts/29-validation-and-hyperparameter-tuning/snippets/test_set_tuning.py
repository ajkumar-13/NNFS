"""Post 29, section 2: what choosing a setting by its test accuracy does to the reported number.

Run from the series root:
    python posts/29-validation-and-hyperparameter-tuning/snippets/test_set_tuning.py

For each of five seeds: eight learning rates are trained on the same 300 points, 200 test sets of
300 points are drawn, and 30,000 further points stand in for unseen data. On every test set the
setting with the highest test accuracy is chosen, and its test accuracy is compared with its
accuracy on the further points. 40 trainings of 1,000 epochs: about 35 seconds.
Needs NumPy and the nnfs package.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from kfold import accuracy, train

CANDIDATES = [0.01, 0.014, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]    # eight learning rates, width 64
N_TEST_SETS = 200


def experiment(seed):
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)                     # the training data of the seed
    test_sets = [spiral_data(samples=100, classes=3) for _ in range(N_TEST_SETS)]
    fresh = [spiral_data(samples=100, classes=3) for _ in range(100)]
    X_fresh = np.concatenate([f[0] for f in fresh])                # 30,000 points nobody tunes on
    y_fresh = np.concatenate([f[1] for f in fresh])
    rng = np.random.default_rng(seed)
    small_sets = [(X_t[keep], y_t[keep]) for X_t, y_t in test_sets
                  for keep in [rng.permutation(300)[:60]]]         # 60 random points of each test set

    models = [train(X, y, lr, seed=100 + seed) for lr in CANDIDATES]
    fresh_acc = np.array([accuracy(m, X_fresh, y_fresh) for m in models])
    test_acc = np.array([[accuracy(m, X_t, y_t) for X_t, y_t in test_sets] for m in models])
    small_acc = np.array([[accuracy(m, X_t, y_t) for X_t, y_t in small_sets] for m in models])

    sets = np.arange(N_TEST_SETS)
    chosen = test_acc.argmax(axis=0)                 # tuned on the test set: one choice per test set
    tuned = test_acc[chosen, sets] - fresh_acc[chosen]
    # The three-way protocol: choose on one set (the validation set), report on the next (the test set).
    clean = test_acc[chosen, (sets + 1) % N_TEST_SETS] - fresh_acc[chosen]
    chosen_small = small_acc.argmax(axis=0)
    tuned_small = small_acc[chosen_small, sets] - fresh_acc[chosen_small]
    return {
        "fresh_acc": fresh_acc,
        "first": (CANDIDATES[chosen[0]], test_acc[chosen[0], 0], fresh_acc[chosen[0]]),
        "tuned": tuned, "clean": clean, "tuned_small": tuned_small,
    }


if __name__ == "__main__":
    nnfs.init()
    print("setup: nnfs.init(); per seed s: np.random.seed(s), then spiral_data(samples=100, classes=3) for the")
    print("training data, 200 times for the test sets and 100 times for the 30,000 further points;")
    print("2 -> 64 -> 3 network, weights after np.random.seed(100 + s), Adam, no decay, 1,000 full-batch epochs")
    print("candidates:", CANDIDATES)
    print()

    means = {"tuned": [], "clean": [], "tuned_small": []}
    for seed in range(5):
        r = experiment(seed)
        lr, reported, on_fresh = r["first"]
        print(f"seed {seed}  accuracy on the 30,000 further points, per candidate: {np.round(r['fresh_acc'], 3)}")
        print(f"        first test set: chosen lr {lr}, test accuracy {reported:.3f}, on further points {on_fresh:.3f}")
        print(f"        200 test sets of 300: chosen by test accuracy, mean (test - further) {r['tuned'].mean():+.4f}, "
              f"positive on {int(np.sum(r['tuned'] > 0))}, negative on {int(np.sum(r['tuned'] < 0))}")
        print(f"        chosen on one set, reported on the next: mean (test - further) {r['clean'].mean():+.4f}")
        print(f"        test sets of 60: chosen by test accuracy, mean (test - further) {r['tuned_small'].mean():+.4f}")
        for key in means:
            means[key].append(r[key].mean())

    print()
    for key, label in (("tuned", "chosen by test accuracy, 300 points"),
                       ("tuned_small", "chosen by test accuracy, 60 points"),
                       ("clean", "chosen on one set, reported on the next")):
        values = np.array(means[key])
        print(f"{label:40s} over the five seeds: {values.min():+.4f} to {values.max():+.4f}, mean {values.mean():+.4f}")
