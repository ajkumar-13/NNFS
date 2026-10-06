"""Post 33, section 7: the 0.01 scale against He under plain gradient descent, one to three hidden layers.

Run from the series root:
    python posts/33-weight-initialisation/snippets/depth_sgd.py

Setup, as train_stack() in network.py builds it: nnfs.init() once, then per run np.random.seed(s),
spiral_data(samples=100, classes=3), 1, 2 or 3 hidden Layer_Dense layers of 64 neurons with ReLU,
Layer_Dense(64, 3), softmax and cross-entropy, Optimizer_SGD(learning_rate=1.0) of post 22,
501 full-batch epochs, seeds 0 to 4 or the seeds given as arguments.

Needs NumPy and the nnfs package. Takes 20 to 30 seconds.
"""
import nnfs

from network import Optimizer_SGD, depth_sweep, seeds_from_command_line

nnfs.init()
for init in ("small", "he"):
    depth_sweep(init, (1, 2, 3), lambda: Optimizer_SGD(learning_rate=1.0), 501, seeds_from_command_line())
