"""Post 29, sections 5 and 8: the learning-rate comparison and the grid for seeds 1 and 2.

Run from the series root:
    python posts/29-validation-and-hyperparameter-tuning/snippets/seeds_1_2.py

Each seed draws its own data, folds and initial weights. 72 trainings of 1,000 epochs:
about 60 seconds. Needs NumPy and the nnfs package.
"""
import nnfs

import grid
import search
from kfold import SETUP


def run(seeds):
    nnfs.init()
    print(SETUP)
    print()
    for seed in seeds:
        results = search.sweep(seed, verbose=False)
        print("learning rates  ", search.summary(seed, results))
        known = {(lr, 64): accs for lr, accs in results.items()}     # width 64 is already scored
        print("grid (rate/width)", grid.summary(seed, grid.grid(seed, known, verbose=False)))


if __name__ == "__main__":
    run([1, 2])
