"""Post 29, section 3: the spread of one validation split against the spread of a 5-fold mean.

Run from the series root:
    python posts/29-validation-and-hyperparameter-tuning/snippets/steadiness.py

The 300 points of seed 0, a learning rate of 0.1 and the same initial weights throughout; only the
shuffle that makes the folds changes, eight times. 40 trainings of 1,000 epochs: about 25 seconds.
Needs NumPy and the nnfs package.
"""
import numpy as np
import nnfs

from kfold import accuracy, draw_data, k_fold_split, train

if __name__ == "__main__":
    nnfs.init()
    X, y, X_test, y_test = draw_data(0)               # the test set is not used in this script
    print("setup: nnfs.init(), the 300 points of seed 0, 2 -> 64 -> 3 network, weights after np.random.seed(100),")
    print("Optimizer_Adam(learning_rate=0.1), 1,000 full-batch epochs; folds from np.random.default_rng(r), r = 0..7")
    print()

    single, means = [], []
    for r in range(8):
        fold_accs = [accuracy(train(tr_X, tr_y, 0.1, seed=100), va_X, va_y)
                     for tr_X, tr_y, va_X, va_y in k_fold_split(X, y, k=5, seed=r)]
        single.extend(fold_accs)
        means.append(np.mean(fold_accs))
        print(f"shuffle {r}  folds {np.round(fold_accs, 3)}  mean {means[-1]:.3f}")

    print()
    print(f"one split of 60 (40 values):  {min(single):.3f} to {max(single):.3f}, standard deviation {np.std(single):.3f}")
    print(f"5-fold mean (8 values):       {min(means):.3f} to {max(means):.3f}, standard deviation {np.std(means):.3f}")
    print(f"ratio of the standard deviations {np.std(single) / np.std(means):.2f}   sqrt(5) = {np.sqrt(5):.2f}")
