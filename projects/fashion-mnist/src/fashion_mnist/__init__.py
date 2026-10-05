"""Fashion-MNIST: the series' 784-128-128-10 dense network, unchanged, on clothing images.

The classes in :mod:`fashion_mnist.nn` are the ones the series "Neural Networks from Scratch"
derives post by post, and the network in :mod:`fashion_mnist.model` is the one the series' MNIST
project trains. This package trains it on the 60,000 Fashion-MNIST training images, scores it on
the 10,000 test images, and reports the errors class by class.

    uv run python scripts/download_fashion_mnist.py    fetch the four files, check their SHA-256
    uv run python -m fashion_mnist.train               train from the fixed seed, write the weights
    uv run python -m fashion_mnist.evaluate            score the saved weights on the test set
"""

__version__ = "1.0.0"
