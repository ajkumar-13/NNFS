"""Post 25, section 7: the momentum run of post 24 on the same five seeds.

Run from the series root:
    python posts/25-adagrad/snippets/seeds_momentum.py

Needs NumPy and the nnfs package. The classes and the training loop are imported from adagrad.py
in the same directory. Five full runs of 10,001 epochs: 35 to 50 seconds.
"""
import sys

sys.dont_write_bytecode = True                  # keep the snippets directory free of __pycache__

from adagrad import Optimizer_SGD, seed_table

seed_table("Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)",
           lambda: Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9), seeds=(0, 1, 2, 3, 4))
