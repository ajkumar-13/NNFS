"""Post 25, section 7: the documented AdaGrad setting on five seeds.

Run from the series root:
    python posts/25-adagrad/snippets/seeds_adagrad.py

Needs NumPy and the nnfs package. The classes and the training loop are imported from adagrad.py
in the same directory. Five full runs of 10,001 epochs: 35 to 50 seconds.
"""
import sys

sys.dont_write_bytecode = True                  # keep the snippets directory free of __pycache__

from adagrad import Optimizer_Adagrad, seed_table

seed_table("Optimizer_Adagrad(learning_rate=1.0, decay=1e-4)",
           lambda: Optimizer_Adagrad(learning_rate=1.0, decay=1e-4), seeds=(0, 1, 2, 3, 4))
