"""Fetch the four MNIST files into mnist_cache/ and check each against its SHA-256.

    uv run python scripts/download_mnist.py
    uv run python scripts/download_mnist.py --data-dir D:\\datasets\\mnist

The file names, sizes and checksums are in mnist_from_scratch.data.MNIST_FILES, and the work is
done by mnist_from_scratch.download, so the loader and the downloader cannot disagree about what
the dataset is.
"""

import sys

from mnist_from_scratch.download import main

if __name__ == "__main__":
    sys.exit(main())
