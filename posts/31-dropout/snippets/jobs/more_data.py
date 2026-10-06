"""Post 31, section 8: the 64-neuron network with 1,000 samples per class instead of 100.

Run from the series root:
    python posts/31-dropout/snippets/jobs/more_data.py

Ten runs of 10,001 full-batch epochs: no dropout layer, then Layer_Dropout(0.1), for seeds 0 to 4.
Everything else is the setup of network.py; the training set and the test set are 3,000 points each.

Needs NumPy and the nnfs package. Takes about 20 to 40 minutes, which is why it is under jobs/.
"""
import sys
from pathlib import Path

import nnfs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from network import SETUP, paired, spread

nnfs.init()
print(SETUP)
print("changed: spiral_data(samples=1000, classes=3) for the training set and for the test set")
print()
base = spread(rate=None, samples=1000)
print()
rows = spread(rate=0.1, samples=1000)
print()
paired(base, rows, "Layer_Dropout(0.1)")
