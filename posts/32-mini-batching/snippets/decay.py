"""Post 32, section 5: a decay chosen for full-batch epochs, kept and then chosen again.

Run from the series root:
    python posts/32-mini-batching/snippets/decay.py
    python posts/32-mini-batching/snippets/decay.py 0 1 2 3 4 5 6 7 8 9      (the ten seeds of the post)

1,000 epochs each. decay = 1e-2 brings the rate to a tenth after 1,000 full-batch updates.
With batches of 32 the same 1,000 epochs are 10,000 updates: the unchanged decay brings the
rate to a hundredth, and decay = 1e-3 restores the tenth.

Needs NumPy and the nnfs package. Takes about 35 seconds.
"""
import nnfs

from network import SETUP, spread

nnfs.init()
print(SETUP)
print("in this script the decay is the argument printed with each table, not 1e-5")
print()
print("== Full batch, the decay as chosen")
full = spread(epochs=1000, batch_size=300, decay=1e-2)
print()
print("== Batches of 32, the decay unchanged")
kept = spread(epochs=1000, batch_size=32, decay=1e-2)
print()
print("== Batches of 32, the decay divided by the 10 updates of an epoch")
divided = spread(epochs=1000, batch_size=32, decay=1e-3)

n = len(full)
print()
print(f"== Counts over the {n} seeds, training loss, seed by seed")
print(f"decay unchanged: batches of 32 end higher than the full batch on "
      f"{sum(k['train_loss'] > f['train_loss'] for k, f in zip(kept, full))} of {n}")
print(f"decay divided by 10: batches of 32 end lower than with the decay unchanged on "
      f"{sum(d['train_loss'] < k['train_loss'] for d, k in zip(divided, kept))} of {n} "
      f"and lower than the full batch on {sum(d['train_loss'] < f['train_loss'] for d, f in zip(divided, full))} of {n}")
