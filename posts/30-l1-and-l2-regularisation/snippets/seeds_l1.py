"""Post 30, section 7: five seeds, L1 at 5e-4 on the weights and biases of dense1.

Run from the series root:
    python posts/30-l1-and-l2-regularisation/snippets/seeds_l1.py

One row per seed, measured forward-only after the last update. Both loss columns are data losses
(cross-entropy alone); the penalty of dense1 is printed in its own column. The last four columns
describe the 128 weights of dense1.

Needs NumPy and the nnfs package. Takes 30 to 50 seconds.
"""
import nnfs

from network import SETUP, spread

nnfs.init()
print(SETUP)
spread(l1=5e-4)
