"""Post 31, section 8: five seeds without a dropout layer, the baseline (post 28's documented run).

Run from the series root:
    python posts/31-dropout/snippets/seeds_none.py

No_Dropout stands where the dropout layer will stand, so this is the network of posts 27 and 28
and the columns are the figures of post 28, section 3. One row per seed, measured
forward-only after the last update. The last line scores the same networks through a
Layer_Dropout(0.1) they were not trained with. Other seeds can be given as arguments.

Needs NumPy and the nnfs package. Takes 40 to 90 seconds.
"""
import nnfs

from network import SETUP, spread

nnfs.init()
print(SETUP)
spread(rate=None)
