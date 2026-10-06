"""Post 31, section 8: the 64-neuron runs repeated with a learning rate of 0.002 instead of 0.02.

Run from the series root:
    python posts/31-dropout/snippets/jobs/smaller_step.py

Ten runs of 10,001 full-batch epochs: no dropout layer, then Layer_Dropout(0.1), for seeds 0 to 4.
Everything else is the setup of network.py. The question is whether the damage dropout does to
this network is an effect of the step size.

Needs NumPy and the nnfs package. Takes 2 to 4 minutes, which is why it is under jobs/.
"""
import sys
from pathlib import Path

import nnfs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from network import SETUP, paired, spread

nnfs.init()
print(SETUP)
print("changed: Optimizer_Adam(learning_rate=0.002, decay=1e-5)")
print()
base = spread(rate=None, learning_rate=0.002)
print()
rows = spread(rate=0.1, learning_rate=0.002)
print()
paired(base, rows, "Layer_Dropout(0.1)")
