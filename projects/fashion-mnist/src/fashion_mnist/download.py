"""Fetch the four Fashion-MNIST files and keep only those whose SHA-256 is the expected one.

    uv run python scripts/download_fashion_mnist.py
    uv run python scripts/download_fashion_mnist.py --data-dir D:\\datasets\\fashion-mnist

This is the only code in the project that opens a network connection. A file that is already
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

from fashion_mnist import data
from fashion_mnist.data import DEFAULT_DATA_DIR, DatasetFile, sha256_of

# The bucket of the dataset's authors, Zalando Research, addressed over https.
MIRROR = "https://fashion-mnist.s3.eu-central-1.amazonaws.com/"
TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1 << 16


class DownloadError(Exception):
    """A file could not be fetched, or what arrived is not the expected file."""


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
    base_url: str = MIRROR,
    files: dict[str, DatasetFile] | None = None,
    opener: Callable | None = None,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Make ``data_dir`` hold every dataset file, fetching those that are absent.

    ``opener`` is called as ``opener(url, timeout=seconds)`` and defaults to
    ``urllib.request.urlopen``; the tests pass a stand-in so that they never touch the network.
    """
    if not base_url.startswith("https://"):
        raise DownloadError(f"the base URL must start with https://, got {base_url}")
    if not base_url.endswith("/"):
        base_url += "/"
    files = data.FASHION_MNIST_FILES if files is None else files
    opener = urllib.request.urlopen if opener is None else opener
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for spec in files.values():
        path = data_dir / spec.name
        if path.exists():
            found = sha256_of(path)
            if found != spec.sha256:
                raise DownloadError(
                    f"{path} exists but is not the expected file: its SHA-256 is {found}, "
                    f"expected {spec.sha256}. Move or delete it and run the download again."
                )
            log(f"  {spec.name}: already present, checksum ok")
        else:
            log(f"  {spec.name}: downloading {spec.size:,} bytes")
            partial = path.with_name(path.name + ".part")
            try:
                _fetch(base_url + spec.name, partial, spec, opener)
                os.replace(partial, path)
            finally:
                partial.unlink(missing_ok=True)
            log(f"  {spec.name}: checksum ok")
        paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/download_fashion_mnist.py",
        description="Fetch the four Fashion-MNIST files and check each against its SHA-256.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory to hold the files, created if absent (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        default=MIRROR,
        help="https mirror that serves the four files (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Fashion-MNIST into {args.data_dir}")
    try:
        download(data_dir=args.data_dir, base_url=args.base_url)
    except (DownloadError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("all four files present and checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
