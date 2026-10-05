"""Post 29, section 5: five learning rates compared by 5-fold cross-validation, seed 0.

Run from the series root:
    python posts/29-validation-and-hyperparameter-tuning/snippets/search.py

25 trainings of 1,000 epochs: about 25 seconds. Needs NumPy and the nnfs package.
seeds_1_2.py and seeds_3_4.py repeat the comparison for four more seeds.
"""
import numpy as np
import nnfs

from kfold import SETUP, draw_data, k_fold_accuracies

CANDIDATES = [0.01, 0.05, 0.1, 0.5, 1.0]            # learning rates to compare


def sweep(seed, verbose=True):
    """5-fold validation accuracies of every candidate learning rate, for one seed."""
    X, y, X_test, y_test = draw_data(seed)           # the test set stays unused here
    results = {}
    for lr in CANDIDATES:
        fold_accs = k_fold_accuracies(X, y, lr, seed=seed)
        results[lr] = fold_accs
        if verbose:
            print(f'lr={lr:<6} mean_acc={np.mean(fold_accs):.3f} '
                  f'std={np.std(fold_accs):.3f}  folds {np.round(fold_accs, 3)}')
    return results


def summary(seed, results):
    """One line per seed: the mean of every candidate, and the lead of the best over the second."""
    means = {lr: float(np.mean(accs)) for lr, accs in results.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    cells = "  ".join(f"{lr}: {means[lr]:.3f}" for lr in CANDIDATES)
    return (f"seed {seed}  {cells}  | best {ranked[0]}, ahead of {ranked[1]} "
            f"by {means[ranked[0]] - means[ranked[1]]:.3f}")


if __name__ == "__main__":
    nnfs.init()
    print(SETUP)
    print("s = 0; candidates:", CANDIDATES)
    print()
    results = sweep(0)
    print()
    print(summary(0, results))

    best_per_fold = [[lr for lr in CANDIDATES if results[lr][i] == max(results[c][i] for c in CANDIDATES)]
                     for i in range(5)]
    print("best learning rate on each single fold:", best_per_fold)
