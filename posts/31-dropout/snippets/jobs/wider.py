"""Post 31, section 8: a hidden layer of 512 neurons instead of 64, 100 samples per class.

Run from the series root:
    python posts/31-dropout/snippets/jobs/wider.py

Ten runs of 10,001 full-batch epochs: no dropout layer, then Layer_Dropout(0.1), for seeds 0 to 4.
Everything else is the setup of network.py. A width changes how many draws the weights take, so
the test points of a seed are not those of the 64-neuron runs.

Needs NumPy and the nnfs package. Takes about 25 to 45 minutes, which is why it is under jobs/.
"""
import sys
from pathlib import Path

import nnfs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from network import SETUP, paired, spread

nnfs.init()
print(SETUP)
print("changed: Layer_Dense(2, 512) and Layer_Dense(512, 3)")
print()
base = spread(rate=None, n_neurons=512)
print()
rows = spread(rate=0.1, n_neurons=512)
print()
paired(base, rows, "Layer_Dropout(0.1)")
