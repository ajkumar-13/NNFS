"""MNIST from scratch: a 784-128-128-10 dense network in NumPy alone.

The classes in :mod:`mnist_from_scratch.nn` are the ones the series "Neural Networks from Scratch"
derives post by post. This package assembles them into one model, trains it on the 60,000 MNIST
training digits and scores it on the 10,000 test digits.

    uv run python scripts/download_mnist.py         fetch the four MNIST files, check their SHA-256
    uv run python -m mnist_from_scratch.train       train from the fixed seed, write the weights
    uv run python -m mnist_from_scratch.evaluate    score the saved weights on the test set
"""

__version__ = "1.0.0"
