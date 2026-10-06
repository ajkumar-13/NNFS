"""Post 31, section 8: five seeds with Layer_Dropout(0.2) after the ReLU.

Run from the series root:
    python posts/31-dropout/snippets/seeds_rate_20.py

One row per seed, measured forward-only after the last update: the training accuracy through
100 fresh masks (mean, lowest and highest), the training and test accuracy with the mask off,
the two differences in points, the test data loss and the number of dead hidden neurons.
Other seeds can be given as arguments, as in seeds_rate_20.py 5 6 7 8 9.

Needs NumPy and the nnfs package. Takes 40 to 90 seconds.
"""
import nnfs

from network import SETUP, spread

nnfs.init()
print(SETUP)
spread(rate=0.2)
