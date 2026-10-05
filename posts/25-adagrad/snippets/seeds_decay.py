"""Post 25, section 6: AdaGrad with ten times the documented decay, on the same five seeds.

Run from the series root:
    python posts/25-adagrad/snippets/seeds_decay.py

Needs NumPy and the nnfs package. The classes and the training loop are imported from adagrad.py
in the same directory. Five full runs of 10,001 epochs: 35 to 50 seconds.
"""
import sys

sys.dont_write_bytecode = True                  # keep the snippets directory free of __pycache__

from adagrad import Optimizer_Adagrad, seed_table

for decay in (1e-4, 1e-3):
    print(f"decay = {decay:g}: the schedule alone leaves 1 / (1 + {decay:g} * 10000) = "
          f"{1 / (1 + decay * 10000):.4f} of the learning rate at the last epoch")
seed_table("Optimizer_Adagrad(learning_rate=1.0, decay=1e-3)",
           lambda: Optimizer_Adagrad(learning_rate=1.0, decay=1e-3), seeds=(0, 1, 2, 3, 4))
