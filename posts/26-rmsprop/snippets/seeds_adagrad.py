"""Post 26, section 7: post 25's AdaGrad setting on seeds 0 to 4, for the comparison with RMSProp.

Run from the series root:
    python posts/26-rmsprop/snippets/seeds_adagrad.py

Needs NumPy and the nnfs package. The classes, train() and seed_table() are imported from
rmsprop.py in the same directory. Five full runs of 10,001 epochs: about 35 seconds.
"""
import sys

sys.dont_write_bytecode = True                  # keep the snippets directory free of __pycache__

from rmsprop import Optimizer_Adagrad, seed_table

seed_table("Optimizer_Adagrad(learning_rate=1.0, decay=1e-4)",
           lambda: Optimizer_Adagrad(learning_rate=1.0, decay=1e-4), seeds=(0, 1, 2, 3, 4))
