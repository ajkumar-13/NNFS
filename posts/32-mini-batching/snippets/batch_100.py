"""Post 32, section 6: batch size 100 over five seeds, at 1,000 and at 3,334 epochs.

Run from the series root:
    python posts/32-mini-batching/snippets/batch_100.py
    python posts/32-mini-batching/snippets/batch_100.py 5 6 7 8 9      (other seeds)

Three updates per epoch, so 1,000 epochs are 3,000 updates and 3,334 epochs are 10,002 updates.
Each seed is one training run, measured at both epoch counts.
Needs NumPy and the nnfs package. Takes about 20 seconds.
"""
import nnfs

from network import SETUP, spread_checkpoints

nnfs.init()
print(SETUP)
spread_checkpoints(batch_size=100, checkpoints=(1000, 3334))
