"""Post 26, section 7: the documented RMSProp setting on seeds 0 to 4.

Run from the series root:
    python posts/26-rmsprop/snippets/seeds_rmsprop.py

Needs NumPy and the nnfs package. The classes, train() and seed_table() are imported from
rmsprop.py in the same directory. Five full runs of 10,001 epochs: about 35 seconds.
"""
import sys

sys.dont_write_bytecode = True                  # keep the snippets directory free of __pycache__

from rmsprop import Optimizer_RMSprop, seed_table

seed_table("Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)",
           lambda: Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999), seeds=(0, 1, 2, 3, 4))
