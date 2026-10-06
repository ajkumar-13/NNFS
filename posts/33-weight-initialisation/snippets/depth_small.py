"""Post 33, section 7: how deep a ReLU network trains from the 0.01 scale under Adam.

Run from the series root:
    python posts/33-weight-initialisation/snippets/depth_small.py

Setup, as train_stack() in network.py builds it: nnfs.init() once, then per run np.random.seed(s),
spiral_data(samples=100, classes=3), 2, 4, 5 or 6 hidden Layer_Dense layers of 64 neurons with
ReLU, Layer_Dense(64, 3), softmax and cross-entropy, Optimizer_Adam(learning_rate=0.02, decay=1e-5),
501 full-batch epochs (not the 10,001 of Part VI: the question is whether the loss leaves ln 3).
Other seeds can be given as arguments.

Needs NumPy and the nnfs package. Takes 30 to 40 seconds.
"""
import nnfs

from network import Optimizer_Adam, depth_sweep, seeds_from_command_line

nnfs.init()
depth_sweep("small", (2, 4, 5, 6), lambda: Optimizer_Adam(learning_rate=0.02, decay=1e-5), 501,
            seeds_from_command_line())
