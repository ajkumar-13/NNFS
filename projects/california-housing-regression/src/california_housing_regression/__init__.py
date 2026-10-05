"""California housing regression: an 8-64-64-1 dense network in NumPy alone.

The classes in :mod:`california_housing_regression.nn` are the ones the series "Neural Networks
from Scratch" derives post by post, with a mean squared error in place of the softmax and
cross-entropy. This package assembles them into one model, trains it on four fifths of the 20,640
California block groups and scores it, in dollars, on the fifth it never saw.

    uv run python scripts/download_california_housing.py           fetch the archive, check it
    uv run python -m california_housing_regression.train           train from the fixed seed
    uv run python -m california_housing_regression.evaluate        score the saved weights
"""

__version__ = "1.0.0"
