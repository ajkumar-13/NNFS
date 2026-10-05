"""Post 30, section 7: the same two networks with 1,000 samples per class instead of 100.

Run from the series root:
    python posts/30-l1-and-l2-regularisation/snippets/jobs/more_data.py

Ten runs of 10,001 full-batch epochs on 3,000 points: no penalty, then L2 at 5e-4 on dense1,
for seeds 0 to 4. Everything else is the setup of network.py; the test set is a second draw of
1,000 samples per class. Losses printed are data losses; the penalty has its own column.

Needs NumPy and the nnfs package. Takes about 15 to 20 minutes, which is why it is under jobs/.
"""
import sys
from pathlib import Path

import nnfs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from network import SETUP, spread

nnfs.init()
print(SETUP)
print("changed: spiral_data(samples=1000, classes=3) for the training set and for the test set")
print()
spread(samples=1000)
print()
spread(l2=5e-4, samples=1000)
