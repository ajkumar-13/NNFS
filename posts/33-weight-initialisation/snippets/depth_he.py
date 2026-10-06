"""Post 33, section 7: the depth sweep of depth_small.py with He initialisation.

Run from the series root:
    python posts/33-weight-initialisation/snippets/depth_he.py

The same setup as depth_small.py with init="he": 2, 4, 5 or 6 hidden layers of 64 ReLU neurons on
the spiral, Optimizer_Adam(learning_rate=0.02, decay=1e-5), 501 full-batch epochs, seeds 0 to 4 or
the seeds given as arguments.

Needs NumPy and the nnfs package. Takes 30 to 40 seconds.
"""
import nnfs

from network import Optimizer_Adam, depth_sweep, seeds_from_command_line

nnfs.init()
depth_sweep("he", (2, 4, 5, 6), lambda: Optimizer_Adam(learning_rate=0.02, decay=1e-5), 501,
            seeds_from_command_line())
