"""Fetch cal_housing.tgz into cal_housing_cache/ and check it against its SHA-256.

    uv run python scripts/download_california_housing.py
    uv run python scripts/download_california_housing.py --data-dir D:\\datasets\\cal_housing

The file name, size and checksum are in california_housing_regression.data.ARCHIVE, and the work is
done by california_housing_regression.download, so the loader and the downloader cannot disagree
about what the dataset is.
"""

import sys

from california_housing_regression.download import main

if __name__ == "__main__":
    sys.exit(main())
