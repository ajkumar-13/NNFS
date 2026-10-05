"""Post 30, section 7: five seeds, no penalty: the baseline every other script of this post is compared with.

Run from the series root:
    python posts/30-l1-and-l2-regularisation/snippets/seeds_none.py

One row per seed, measured forward-only after the last update. Both loss columns are data losses
(cross-entropy alone); the penalty of dense1 is printed in its own column. The last four columns
describe the 128 weights of dense1.

Needs NumPy and the nnfs package. Takes 30 to 50 seconds.
"""
import nnfs

from network import SETUP, spread

nnfs.init()
print(SETUP)
spread()
