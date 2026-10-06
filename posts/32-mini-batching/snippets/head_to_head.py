"""Post 32, section 6: batches of 32 against the full batch, each seed's two runs paired.

Run from the series root:
    python posts/32-mini-batching/snippets/head_to_head.py
    python posts/32-mini-batching/snippets/head_to_head.py 0 1 2 3 4 5 6 7 8 9      (the ten seeds of the post)

Each seed trains two networks from the same data and weights. The full batch (300 rows, one
update per epoch) is measured after 1,000 and after 10,000 epochs of one run; the batches of 32
(ten updates per epoch) after 100 and after 1,000 epochs of one run. The four tables are
followed by three comparisons, seed by seed: equal epochs (1,000 each), equal updates late
(10,000 each) and equal updates early (1,000 each). Every figure is forward-only on all 300
training rows or on the test set.

Needs NumPy and the nnfs package. Takes about 50 seconds for five seeds and twice that for ten.
"""
import nnfs

from network import SETUP, command_line_seeds, spread_checkpoints

nnfs.init()
print(SETUP)
seeds = command_line_seeds()
n = len(seeds)

full_early, full_late = spread_checkpoints(batch_size=300, checkpoints=(1000, 10000), seeds=seeds)
mini_early, mini_late = spread_checkpoints(batch_size=32, checkpoints=(100, 1000), seeds=seeds)


def count(rows_a, rows_b, key, sign):
    """On how many seeds the figure of rows_a is lower (sign -1) or higher (sign +1) than that of rows_b."""
    return sum(sign * (a[key] - b[key]) > 0 for a, b in zip(rows_a, rows_b))


print()
print(f"== Counts over the {n} seeds {seeds}")
print(f"equal epochs (1,000): batches of 32 have the lower training loss on "
      f"{count(mini_late, full_early, 'train_loss', -1)} of {n} and the higher test accuracy on "
      f"{count(mini_late, full_early, 'test_accuracy', +1)} of {n}")
print(f"equal updates (10,000): batches of 32 have the lower training loss on "
      f"{count(mini_late, full_late, 'train_loss', -1)} of {n} and the higher test accuracy on "
      f"{count(mini_late, full_late, 'test_accuracy', +1)} of {n}")
print(f"equal updates (1,000): the full batch has the lower training loss on "
      f"{count(full_early, mini_early, 'train_loss', -1)} of {n}")
