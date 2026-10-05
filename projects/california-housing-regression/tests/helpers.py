"""Helpers shared by the test modules: archive bytes, finite differences, the fixture record."""

import gzip
import io
import tarfile
from pathlib import Path

import numpy as np

from california_housing_regression.data import MEMBER, DatasetFile, sha256_of

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Size and checksum printed by scripts/make_test_fixture.py for the committed file.
FIXTURE_ARCHIVE = DatasetFile(
    "cal_housing.tgz",
    1052,
    "681c464a2234205b237cc666b06395cafb682b7daea9b8f929e0e2abf04e1fd0",
)
FIXTURE_ROWS = 40
FIXTURE_TRAIN = 32
FIXTURE_TEST = 8
# The first 8 entries of the seed 0 permutation of 40 rows are the test fold; the rest train.
FIXTURE_TEST_ROWS = [11, 27, 4, 24, 23, 2, 3, 34]
FIXTURE_TRAIN_ROWS = [
    *[18, 1, 10, 22, 20, 26, 28, 30, 37, 38, 17, 0, 21, 35, 9, 6],
    *[32, 25, 19, 36, 8, 16, 13, 12, 7, 39, 5, 14, 29, 33, 15, 31],
]


def fixture_row(k: int) -> list[float]:
    """The nine values the fixture holds in row ``k`` (see scripts/make_test_fixture.py)."""
    households = 90 + 5 * k
    return [
        -124.30 + 0.25 * k,
        32.50 + 0.20 * k,
        1 + (7 * k) % 52,
        households * (4 + k % 5),
        households + 2 * k,
        250 + 31 * k,
        households,
        1.5 + 0.21 * k,
        min(60_000 + 12_500 * k, 500_001),
    ]


def fixture_table() -> np.ndarray:
    """The fixture's 40 rows as the file prints them, with six decimal places."""
    return np.array([[float(f"{value:.6f}") for value in fixture_row(k)] for k in range(40)])


def numerical_gradient(function, array, step=1e-6):
    """Central differences of a scalar ``function()`` with respect to every entry of ``array``."""
    gradient = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + step
        upper = function()
        array[index] = original - step
        lower = function()
        array[index] = original
        gradient[index] = (upper - lower) / (2 * step)
    return gradient


def generator_state():
    """The full state of NumPy's global generator, in a form that compares with ``==``."""
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return name, keys.tobytes(), position, has_gauss, cached_gaussian


def table_bytes(table) -> bytes:
    """A table as the comma-separated text the dataset uses."""
    lines = [",".join(f"{value:.6f}" for value in row) for row in np.asarray(table)]
    return ("\n".join(lines) + "\n").encode("ascii")


def archive_bytes(members: dict[str, bytes], directories: tuple[str, ...] = ()) -> bytes:
    """A gzip tar archive holding ``members``, and empty ``directories``."""
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as packed,
        tarfile.open(fileobj=packed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for name in directories:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def write_archive(directory, table, name: str = "cal_housing.tgz") -> DatasetFile:
    """Write ``table`` as a dataset archive under ``directory`` and return its file record."""
    path = Path(directory) / name
    path.write_bytes(archive_bytes({MEMBER: table_bytes(table)}))
    return DatasetFile(name, path.stat().st_size, sha256_of(path))
