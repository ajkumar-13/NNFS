"""Fetch the four Fashion-MNIST files into fashion_mnist_cache/ and check each against its SHA-256.

    uv run python scripts/download_fashion_mnist.py
    uv run python scripts/download_fashion_mnist.py --data-dir D:\\datasets\\fashion-mnist

The file names, sizes and checksums are in fashion_mnist.data.FASHION_MNIST_FILES, and the work is
done by fashion_mnist.download, so the loader and the downloader cannot disagree about what the
dataset is.
"""

import sys

from fashion_mnist.download import main

if __name__ == "__main__":
    sys.exit(main())
