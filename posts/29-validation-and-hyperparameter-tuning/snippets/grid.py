"""Post 29, section 8: a two-by-two grid scored by 5-fold cross-validation, and the test set opened once.

Run from the series root:
    python posts/29-validation-and-hyperparameter-tuning/snippets/grid.py

21 trainings of 1,000 epochs: about 20 seconds. Needs NumPy and the nnfs package.
seeds_1_2.py and seeds_3_4.py repeat the grid for four more seeds.
"""
import numpy as np
import nnfs

from kfold import SETUP, accuracy, draw_data, k_fold_accuracies, train


def grid(seed, known=None, verbose=True):
    """Score the grid by 5-fold validation, retrain the winner on all 300 points, open the test set."""
    X, y, X_test, y_test = draw_data(seed)
    scores = dict(known or {})                       # (learning rate, width) -> fold accuracies

    for lr in [0.05, 0.1]:
        for n_neurons in [16, 64]:
            if (lr, n_neurons) not in scores:
                scores[(lr, n_neurons)] = k_fold_accuracies(X, y, lr, n_neurons, seed=seed)
            fold_accs = scores[(lr, n_neurons)]
            if verbose:
                print(f'lr={lr:<5} n_neurons={n_neurons:<3} mean_acc={np.mean(fold_accs):.3f} '
                      f'std={np.std(fold_accs):.3f}')

    grid_scores = {c: scores[c] for c in scores if c[0] in (0.05, 0.1) and c[1] in (16, 64)}
    best_lr, best_n = max(grid_scores, key=lambda c: np.mean(grid_scores[c]))

    # Every decision is made. Retrain the winner on all 300 points, then open the test set, once.
    final_model = train(X, y, best_lr, best_n, seed=100 + seed)
    return {
        "scores": grid_scores,
        "best": (best_lr, best_n),
        "k_fold_mean": float(np.mean(grid_scores[(best_lr, best_n)])),
        "train_accuracy": accuracy(final_model, X, y),
        "test_accuracy": accuracy(final_model, X_test, y_test),
    }


def summary(seed, result):
    """One line per seed: the four means, the winner, its k-fold mean and its one test accuracy."""
    cells = "  ".join(f"{lr}/{n}: {np.mean(accs):.3f}" for (lr, n), accs in sorted(result["scores"].items()))
    return (f"seed {seed}  {cells}  | winner {result['best']}  k-fold {result['k_fold_mean']:.3f}  "
            f"train {result['train_accuracy']:.3f}  test {result['test_accuracy']:.3f}")


if __name__ == "__main__":
    nnfs.init()
    print(SETUP)
    print("s = 0; grid: learning rate 0.05 or 0.1, hidden width 16 or 64")
    print()
    result = grid(0)
    print()
    print("winner (learning rate, width):", result["best"])
    print(f"k-fold mean of the winner {result['k_fold_mean']:.3f}")
    print(f"retrained on all 300 points: training accuracy {result['train_accuracy']:.3f}, "
          f"test accuracy {result['test_accuracy']:.3f} "
          f"({int(round(result['test_accuracy'] * 300))} of 300)")
