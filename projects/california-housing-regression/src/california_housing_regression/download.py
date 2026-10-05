"""Fetch the California housing archive and keep it only if its SHA-256 is the expected one.

    uv run python scripts/download_california_housing.py
    uv run python scripts/download_california_housing.py --data-dir D:\\datasets\\cal_housing

This is the only code in the project that opens a network connection. An archive that is already
present with the right checksum is left alone, so the command is safe to repeat. A download is
written beside its destination under a ``.part`` name and moved into place only after its size and
checksum match, so an interrupted or tampered download never looks like the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from california_housing_regression import data
from california_housing_regression.data import DEFAULT_DATA_DIR, DatasetFile, sha256_of

# The copy of the StatLib archive that scikit-learn's own loader downloads.
ARCHIVE_URL = "https://ndownloader.figshare.com/files/5976036"
TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1 << 16


class DownloadError(Exception):
    """The archive could not be fetched, or what arrived is not the expected file."""


def _fetch(url: str, destination: Path, spec: DatasetFile, opener: Callable) -> None:
    """Stream ``url`` into ``destination``, refusing anything but the expected bytes."""
    digest = hashlib.sha256()
    received = 0
    try:
        with opener(url, timeout=TIMEOUT_SECONDS) as response, open(destination, "wb") as out:
            while chunk := response.read(CHUNK_BYTES):
                received += len(chunk)
                if received > spec.size:
                    raise DownloadError(
                        f"{url} sent more than the expected {spec.size:,} bytes; stopped"
                    )
                digest.update(chunk)
                out.write(chunk)
    except (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
    ) as error:
        raise DownloadError(f"could not fetch {url}: {error}") from error
    if received != spec.size:
        raise DownloadError(f"{url} sent {received:,} bytes, expected {spec.size:,}")
    if digest.hexdigest() != spec.sha256:
        raise DownloadError(
            f"{url} has SHA-256 {digest.hexdigest()}, expected {spec.sha256}; the file was not kept"
        )


def download(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    url: str = ARCHIVE_URL,
    archive: DatasetFile | None = None,
    opener: Callable | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Make ``data_dir`` hold the archive, fetching it from ``url`` if it is absent.

    ``opener`` is called as ``opener(url, timeout=seconds)`` and defaults to
    ``urllib.request.urlopen``; the tests pass a stand-in so that they never touch the network.
    """
    if not url.startswith("https://"):
        raise DownloadError(f"the URL must start with https://, got {url}")
    spec = data.ARCHIVE if archive is None else archive
    opener = urllib.request.urlopen if opener is None else opener
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    path = data_dir / spec.name
    if path.exists():
        found = sha256_of(path)
        if found != spec.sha256:
            raise DownloadError(
                f"{path} exists but is not the expected file: its SHA-256 is {found}, "
                f"expected {spec.sha256}. Move or delete it and run the download again."
            )
        log(f"  {spec.name}: already present, checksum ok")
        return path

    log(f"  {spec.name}: downloading {spec.size:,} bytes")
    partial = path.with_name(path.name + ".part")
    try:
        _fetch(url, partial, spec, opener)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    log(f"  {spec.name}: checksum ok")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/download_california_housing.py",
        description="Fetch the California housing archive and check it against its SHA-256.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory to hold the archive, created if absent (default: %(default)s)",
    )
    parser.add_argument(
        "--url",
        default=ARCHIVE_URL,
        help="https address that serves the archive (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"California housing into {args.data_dir}")
    try:
        download(data_dir=args.data_dir, url=args.url)
    except (DownloadError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("the archive is present and checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
