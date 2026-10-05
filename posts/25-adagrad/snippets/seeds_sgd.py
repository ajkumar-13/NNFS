"""Post 25, section 7: the plain gradient-descent run of post 22 on the same five seeds.

Run from the series root:
    python posts/25-adagrad/snippets/seeds_sgd.py

Needs NumPy and the nnfs package. The classes and the training loop are imported from adagrad.py
in the same directory. Five full runs of 10,001 epochs: 35 to 50 seconds.
"""
import sys

sys.dont_write_bytecode = True                  # keep the snippets directory free of __pycache__

from adagrad import Optimizer_SGD, seed_table

seed_table("Optimizer_SGD(learning_rate=1.0)",
           lambda: Optimizer_SGD(learning_rate=1.0), seeds=(0, 1, 2, 3, 4))
