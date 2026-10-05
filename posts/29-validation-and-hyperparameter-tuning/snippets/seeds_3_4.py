"""Post 29, sections 5 and 8: the learning-rate comparison and the grid for seeds 3 and 4.

Run from the series root:
    python posts/29-validation-and-hyperparameter-tuning/snippets/seeds_3_4.py

The same runs as seeds_1_2.py for two more seeds: about 60 seconds. Needs NumPy and the nnfs package.
"""
from seeds_1_2 import run

if __name__ == "__main__":
    run([3, 4])
