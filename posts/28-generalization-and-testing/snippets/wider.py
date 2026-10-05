"""Post 28, sections 4 and 6: the same run with 128 hidden neurons in place of 64.

Run from the series root:
    python posts/28-generalization-and-testing/snippets/wider.py

Everything except the width is the setup of seed_spread.py: the same five seeds, 10,001 epochs,
and 300 test points drawn after the weights. Compare the rows with those of seed_spread.py.

Needs NumPy and the nnfs package. Takes about 70 seconds, the longest script of the post.
"""
import nnfs

from network import SETUP
from seed_spread import SEEDS, final_table

nnfs.init()
print(SETUP)
print(f"changed: Layer_Dense(2, 128) and Layer_Dense(128, 3), 771 parameters; seeds {SEEDS}")
print()
print("== 128 neurons, after the last update")
final_table(128, follow=False)
