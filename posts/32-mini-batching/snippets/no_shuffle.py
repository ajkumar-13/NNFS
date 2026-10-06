"""Post 32, section 10: the batch loop without the shuffle, on data stored class by class.

Run from the series root:
    python posts/32-mini-batching/snippets/no_shuffle.py
    python posts/32-mini-batching/snippets/no_shuffle.py 5 6 7 8 9      (other seeds)

spiral_data returns the 100 rows of class 0, then class 1, then class 2. With order=np.arange
the batches are cut from that stored order in every epoch. 1,000 epochs each; the shuffled
runs to compare with are batch_100.py and batch_32.py.

Needs NumPy and the nnfs package. Takes about 25 seconds.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from network import SETUP, spread

nnfs.init()
print(SETUP)

np.random.seed(0)
X, y = spiral_data(samples=100, classes=3)
print()
print("== The labels of each batch when the rows keep their stored order, classes counted as (0, 1, 2)")
for batch_size in (100, 32):
    counts = [tuple(int(v) for v in np.bincount(y[start:start + batch_size], minlength=3))
              for start in range(0, len(X), batch_size)]
    print(f"batch size {batch_size}: {counts}")

print()
print("== Batches of 100, no shuffle")
spread(epochs=1000, batch_size=100, order=np.arange)
print()
print("== Batches of 32, no shuffle")
spread(epochs=1000, batch_size=32, order=np.arange)
