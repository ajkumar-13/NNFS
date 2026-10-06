"""Post 32, sections 6 and 7: batch size 8 over five seeds, at 264 epochs, at two learning rates.

Run from the series root:
    python posts/32-mini-batching/snippets/batch_8.py
    python posts/32-mini-batching/snippets/batch_8.py 0 1 2 3 4 5 6 7 8 9      (the ten seeds of the post)

Thirty-eight updates per epoch, so 264 epochs are 10,032 updates.
Needs NumPy and the nnfs package. Takes about 30 seconds.
"""
import nnfs

from network import SETUP, spread

nnfs.init()
print(SETUP)
print()
print("== 264 epochs, the learning rate of the other runs")
full_rate = spread(epochs=264, batch_size=8)
print()
print("== 264 epochs, half the learning rate")
half_rate = spread(epochs=264, batch_size=8, learning_rate=0.01)

n = len(full_rate)
pairs = list(zip(half_rate, full_rate))
print()
print(f"== Half the learning rate against the full one, seed by seed, {n} seeds")
print(f"fewer dead neurons on {sum(h['dead'] < f['dead'] for h, f in pairs)} of {n}, "
      f"higher training accuracy on {sum(h['train_accuracy'] > f['train_accuracy'] for h, f in pairs)} of {n}, "
      f"lower training loss on {sum(h['train_loss'] < f['train_loss'] for h, f in pairs)} of {n}")
