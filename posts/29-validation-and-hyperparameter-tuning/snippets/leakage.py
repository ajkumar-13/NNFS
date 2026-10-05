"""Post 29, section 6: a preprocessing step fitted before the split, and the same step fitted inside each fold.

Run from the series root:
    python posts/29-validation-and-hyperparameter-tuning/snippets/leakage.py

The data are pure noise: 100 rows, 2,000 standard-normal features, and labels that are a random
half 0 and half 1, so no classifier can do better than 0.5 on new rows. The preprocessing step keeps
the 20 features whose class means differ most. Ten seeds, no training loop: about 3 seconds.
Needs NumPy and the nnfs package (kfold.py imports it, and the last block uses spiral_data).
"""
import numpy as np

from kfold import draw_data, k_fold_split


def select_features(X, y, keep=20):
    """The preprocessing step: the columns whose two class means lie furthest apart."""
    gap = np.abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
    return np.argsort(gap)[-keep:]


def centroid_accuracy(tr_X, tr_y, va_X, va_y):
    """Nearest class mean: fitted on the training rows, scored on the validation rows."""
    mean0 = tr_X[tr_y == 0].mean(axis=0)
    mean1 = tr_X[tr_y == 1].mean(axis=0)
    nearer_1 = np.sum((va_X - mean1) ** 2, axis=1) < np.sum((va_X - mean0) ** 2, axis=1)
    return np.mean(nearer_1.astype(int) == va_y)


def leaky(X, y, seed):
    columns = select_features(X, y)                   # fitted on all rows, validation rows included
    return np.mean([centroid_accuracy(tr_X[:, columns], tr_y, va_X[:, columns], va_y)
                    for tr_X, tr_y, va_X, va_y in k_fold_split(X, y, k=5, seed=seed)])


def clean(X, y, seed):
    fold_accs = []
    for tr_X, tr_y, va_X, va_y in k_fold_split(X, y, k=5, seed=seed):
        columns = select_features(tr_X, tr_y)         # fitted on the training rows of this fold
        fold_accs.append(centroid_accuracy(tr_X[:, columns], tr_y, va_X[:, columns], va_y))
    return np.mean(fold_accs)


if __name__ == "__main__":
    print("data: 100 rows, 2,000 noise features, random labels; 5-fold accuracy of a nearest-class-mean classifier")
    leaky_accs, clean_accs, new_accs = [], [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((100, 2000))
        y = rng.permutation(np.arange(100) % 2)       # 50 zeros and 50 ones in random order
        X_new = rng.standard_normal((2000, 2000))     # new rows for the leaky pipeline's final model
        y_new = rng.permutation(np.arange(2000) % 2)
        columns = select_features(X, y)
        leaky_accs.append(leaky(X, y, seed))
        clean_accs.append(clean(X, y, seed))
        new_accs.append(centroid_accuracy(X[:, columns], y, X_new[:, columns], y_new))
    for label, accs in (("selected before the split ", leaky_accs),
                        ("selected inside each fold ", clean_accs),
                        ("leaky pipeline on new rows", new_accs)):
        print(f"{label}  {np.round(accs, 2)}  mean {np.mean(accs):.3f}")

    print()
    print("the spiral: per-feature mean and standard deviation, all 300 rows against each training fold")
    X, y, X_test, y_test = draw_data(0)
    mean_shift = max(np.abs(X.mean(axis=0) - tr_X.mean(axis=0)).max() for tr_X, _, _, _ in k_fold_split(X, y))
    std_shift = max(np.abs(X.std(axis=0) - tr_X.std(axis=0)).max() for tr_X, _, _, _ in k_fold_split(X, y))
    print(f"standard deviation of the features {np.round(X.std(axis=0), 3)}")
    print(f"largest shift of a mean {mean_shift:.3f}, of a standard deviation {std_shift:.3f}")
