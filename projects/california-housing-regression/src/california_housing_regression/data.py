"""The California housing data: find the archive, check it, parse it, split it, scale it.

The data is the StatLib file of Pace and Barry (1997), 20,640 block groups of the 1990 census,
distributed as ``cal_housing.tgz``. Nothing here touches the network:
:mod:`california_housing_regression.download` fetches the archive, and this module refuses to parse
an archive whose SHA-256 is not the one recorded in :data:`ARCHIVE`, so a run always starts from
the same bytes.

``load_california_housing`` returns a :class:`Housing` with

    X_train      (16512, 8) float32, standardised
    y_train      (16512,)   float32, standardised
    X_test       (4128, 8)  float32, standardised with the training fold's statistics
    y_test       (4128,)    float32, standardised with the training fold's statistics
    value_train  (16512,)   float64, the published median house value in dollars
    value_test   (4128,)    float64
    scaler       the four statistics, computed from the training fold alone

The rule of ``nn-029`` is the point of this module: the split comes first, the scaling statistics
are computed from the training rows only, and the test rows are scaled with those same numbers.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = "cal_housing_cache"
DOWNLOAD_COMMAND = "uv run python scripts/download_california_housing.py"

# The file inside the archive, and its nine comma-separated columns in file order.
MEMBER = "CaliforniaHousing/cal_housing.data"
COLUMNS = (
    "longitude",
    "latitude",
    "housingMedianAge",
    "totalRooms",
    "totalBedrooms",
    "population",
    "households",
    "medianIncome",
    "medianHouseValue",
)
# No real table is near this size; the published one is 2,071,484 bytes.
MAX_TABLE_BYTES = 16 * 1024 * 1024

# The eight features the network sees, in column order.
FEATURE_NAMES = (
    "MedInc",  # median income of the block group, in tens of thousands of dollars
    "HouseAge",  # median house age in years
    "AveRooms",  # rooms per household
    "AveBedrms",  # bedrooms per household
    "Population",  # people in the block group
    "AveOccup",  # people per household
    "Latitude",
    "Longitude",
)

TEST_FRACTION = 0.2
# The network's target is the median house value in units of 100,000 dollars.
DOLLARS_PER_UNIT = 100_000
# The census top-coded the value: every block group above the cap is recorded as this figure.
TOP_CODE_DOLLARS = 500_001
# Added to each standard deviation so that a constant column cannot divide by zero.
EPSILON = 1e-7
# Relative difference below which two sets of scaling statistics count as the same.
SCALER_TOLERANCE = 1e-6


class DataError(Exception):
    """The dataset is missing, damaged, or not the file this project expects."""


@dataclass(frozen=True)
class DatasetFile:
    """One file of the dataset: its name, its size in bytes and the SHA-256 of its bytes."""

    name: str
    size: int
    sha256: str


ARCHIVE = DatasetFile(
    "cal_housing.tgz",
    441_963,
    "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
)


@dataclass(frozen=True)
class Scaler:
    """The statistics that standardise the data, all computed from the training fold.

    ``x_mean`` and ``x_std`` are float32 vectors with one entry per feature; ``y_mean`` and
    ``y_std`` are the target's, in units of 100,000 dollars.
    """

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def matches(self, other: Scaler) -> bool:
        """True when the two scalers hold the same statistics.

        The comparison allows a relative difference of one part in a million, a few times the
        rounding of a float32 sum, so that statistics recomputed on another processor still match.
        Two different splits of the data differ by far more than that.
        """
        ours = np.concatenate([self.x_mean, self.x_std, [self.y_mean, self.y_std]])
        theirs = np.concatenate([other.x_mean, other.x_std, [other.y_mean, other.y_std]])
        return bool(np.allclose(ours, theirs, rtol=SCALER_TOLERANCE, atol=0.0))


@dataclass(frozen=True)
class Housing:
    """The two folds, standardised, with the published dollar values and the scaler."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    value_train: np.ndarray
    value_test: np.ndarray
    scaler: Scaler
    seed: int


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file's bytes, as lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(data_dir: str | Path, archive: DatasetFile) -> Path:
    """Return the path of the archive after checking that it is present and unaltered."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DataError(
            f"the data directory {data_dir} does not exist. Fetch the dataset first:\n"
            f"    {DOWNLOAD_COMMAND}"
        )
    path = data_dir / archive.name
    if not path.is_file():
        raise DataError(
            f"{data_dir} is missing {archive.name}. Fetch the dataset first:\n"
            f"    {DOWNLOAD_COMMAND}"
        )
    found = sha256_of(path)
    if found != archive.sha256:
        raise DataError(
            f"{path} is not the expected file: its SHA-256 is {found}, expected "
            f"{archive.sha256}. Delete it and fetch it again:\n    {DOWNLOAD_COMMAND}"
        )
    return path


def read_member(path: Path, member: str = MEMBER) -> bytes:
    """The bytes of one file inside a gzip tar archive. Nothing is extracted to disk."""
    try:
        with tarfile.open(path, "r:gz") as archive:
            try:
                info = archive.getmember(member)
            except KeyError:
                raise DataError(f"{path} does not contain {member}") from None
            if not info.isfile():
                raise DataError(f"{member} in {path} is not a regular file")
            if info.size > MAX_TABLE_BYTES:
                raise DataError(
                    f"{member} in {path} declares {info.size:,} bytes, more than the "
                    f"{MAX_TABLE_BYTES:,} this project reads"
                )
            with archive.extractfile(info) as handle:
                return handle.read()
    except (tarfile.TarError, OSError, EOFError, zlib.error) as error:
        raise DataError(f"{path} is not a readable gzip tar archive: {error}") from error


def parse_table(raw: bytes, source: str = "the data file") -> np.ndarray:
    """Parse the comma-separated table into ``(rows, 9)`` float64, columns as in :data:`COLUMNS`."""
    if not raw.strip():
        raise DataError(f"{source} is empty")
    try:
        table = np.loadtxt(io.BytesIO(raw), delimiter=",", ndmin=2)
    except ValueError as error:
        raise DataError(f"{source} is not a table of comma-separated numbers: {error}") from error
    if table.shape[1] != len(COLUMNS):
        raise DataError(
            f"{source} has {table.shape[1]} columns, expected the {len(COLUMNS)} columns "
            f"{', '.join(COLUMNS)}"
        )
    if not np.all(np.isfinite(table)):
        raise DataError(f"{source} contains values that are not finite")
    if np.any(table[:, COLUMNS.index("households")] <= 0):
        raise DataError(
            f"{source} has a block group with no households, so its per-household features are "
            "undefined"
        )
    return table


def features_and_target(table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive the eight features and the target from the nine columns of the file.

    Returns ``X`` of shape ``(rows, 8)`` and ``y`` of shape ``(rows,)``, both float32, with ``y``
    in units of 100,000 dollars, and ``value``, the published median house value in dollars as
    float64. The arithmetic is done in float64 and rounded to float32 once, which gives the arrays
    that ``sklearn.datasets.fetch_california_housing`` returns, bit for bit.
    """
    longitude, latitude, age, rooms, bedrooms, population, households, income, value = table.T
    X = np.column_stack(
        [
            income,
            age,
            rooms / households,
            bedrooms / households,
            population,
            population / households,
            latitude,
            longitude,
        ]
    ).astype(np.float32)
    y = (value / float(DOLLARS_PER_UNIT)).astype(np.float32)
    return X, y, value.copy()


def split_indices(
    n: int, test_fraction: float = TEST_FRACTION, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle ``range(n)`` from ``seed`` and return ``(train_indices, test_indices)``.

    The test fold is the first ``round(n * test_fraction)`` entries of the permutation and the
    training fold is the rest, so the two never share a row.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"the test fraction must be between 0 and 1, got {test_fraction}")
    n_test = round(n * test_fraction)
    if n_test < 1 or n - n_test < 2:
        raise DataError(
            f"{n} rows are too few to split into a training fold of at least 2 rows and a test "
            f"fold of at least 1 with a test fraction of {test_fraction}"
        )
    order = np.random.default_rng(seed).permutation(n)
    return order[n_test:], order[:n_test]


def standardise_fit(X: np.ndarray, y: np.ndarray) -> Scaler:
    """Compute the scaling statistics. Call this on the training fold, and on nothing else."""
    return Scaler(
        x_mean=X.mean(axis=0),
        x_std=X.std(axis=0) + EPSILON,
        y_mean=float(y.mean()),
        y_std=float(y.std()) + EPSILON,
    )


def standardise_apply(
    X: np.ndarray, y: np.ndarray, scaler: Scaler
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a fold with statistics that are already fixed. Nothing is fitted here."""
    X_scaled = (X - scaler.x_mean) / scaler.x_std
    y_scaled = (y - scaler.y_mean) / scaler.y_std
    return X_scaled.astype(np.float32), y_scaled.astype(np.float32)


def destandardise_y(y_scaled: np.ndarray, scaler: Scaler) -> np.ndarray:
    """Map standardised targets or predictions back to units of 100,000 dollars."""
    return y_scaled * scaler.y_std + scaler.y_mean


def load_table(data_dir: str | Path = DEFAULT_DATA_DIR, archive: DatasetFile | None = None):
    """Return the checked archive's table, ``(rows, 9)`` float64."""
    path = check_file(data_dir, ARCHIVE if archive is None else archive)
    return parse_table(read_member(path), f"{MEMBER} in {path}")


def load_california_housing(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    seed: int = 0,
    archive: DatasetFile | None = None,
) -> Housing:
    """Load the checked archive, split it from ``seed``, and scale both folds.

    The order is the rule of ``nn-029``: split, then fit the scaler on the training fold, then
    apply it to both folds.
    """
    X, y, value = features_and_target(load_table(data_dir, archive))
    train, test = split_indices(len(X), TEST_FRACTION, seed)

    scaler = standardise_fit(X[train], y[train])
    X_train, y_train = standardise_apply(X[train], y[train], scaler)
    X_test, y_test = standardise_apply(X[test], y[test], scaler)

    return Housing(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        value_train=value[train],
        value_test=value[test],
        scaler=scaler,
        seed=seed,
    )
