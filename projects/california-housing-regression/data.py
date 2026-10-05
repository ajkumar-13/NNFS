"""
California housing dataset loading + preprocessing for regression.

Source: sklearn.datasets.fetch_california_housing (20 640 samples,
8 numerical features, target is median house value in $100 000).

Two backends:
  - "sklearn" calls fetch_california_housing
  - "manual" downloads cal_housing.tgz from the StatLib mirror and
    parses it with stdlib + numpy (no sklearn required)

Preprocessing:
  - Features standardised to zero mean and unit variance (per-feature
    statistics computed on the training set ONLY, applied to the test
    set — the "no leakage" rule from posts/29).
  - Target standardised the same way (so the loss is on the same scale
    as the loss in the lectures; predictions are de-standardised at
    eval time so the reported error is back in dollars).
"""

import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "MedInc",     # median income in block group
    "HouseAge",   # median house age in block group
    "AveRooms",   # average rooms per household
    "AveBedrms",  # average bedrooms per household
    "Population", # block-group population
    "AveOccup",   # average household occupancy
    "Latitude",
    "Longitude",
]


# -----------------------------------------------------------------------------
# Backend A: sklearn
# -----------------------------------------------------------------------------
def _load_via_sklearn():
    from sklearn.datasets import fetch_california_housing
    raw = fetch_california_housing()
    return raw.data.astype(np.float32), raw.target.astype(np.float32)


# -----------------------------------------------------------------------------
# Backend B: manual download from StatLib mirror
# -----------------------------------------------------------------------------
_STATLIB_URL = "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz"


def _load_via_manual(cache_dir="cal_housing_cache"):
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    archive = cache / "cal_housing.tgz"
    csv_path = cache / "cal_housing.data"

    if not csv_path.exists():
        if not archive.exists():
            print(f"  downloading cal_housing.tgz...")
            urllib.request.urlretrieve(_STATLIB_URL, archive)
        with tarfile.open(archive) as tar:
            member = next(m for m in tar.getmembers()
                          if m.name.endswith("cal_housing.data"))
            with tar.extractfile(member) as src:
                csv_path.write_bytes(src.read())

    # Raw cal_housing.data columns (in order):
    #   longitude, latitude, housingMedianAge, totalRooms, totalBedrooms,
    #   population, households, medianIncome, medianHouseValue
    raw = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32)

    # Derive the same 8 features sklearn exposes.
    longitude     = raw[:, 0]
    latitude      = raw[:, 1]
    house_age     = raw[:, 2]
    total_rooms   = raw[:, 3]
    total_beds    = raw[:, 4]
    population    = raw[:, 5]
    households    = raw[:, 6]
    median_income = raw[:, 7]
    median_value  = raw[:, 8]

    X = np.column_stack([
        median_income,
        house_age,
        total_rooms / households,                          # AveRooms
        total_beds / households,                           # AveBedrms
        population,
        population / households,                           # AveOccup
        latitude,
        longitude,
    ]).astype(np.float32)
    # Target is in dollars; sklearn divides by 100 000 so it's in [0, 5].
    y = (median_value / 100_000).astype(np.float32)
    return X, y


# -----------------------------------------------------------------------------
# Train/test split and standardisation
# -----------------------------------------------------------------------------
def train_test_split(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def standardise_fit(X, y):
    """Compute per-column mean/std on training data only. Returns the stats."""
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-7
    y_mean = float(y.mean())
    y_std  = float(y.std()) + 1e-7
    return {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}


def standardise_apply(X, y, stats):
    """Apply pre-computed stats to a fold (no re-fitting, no leakage)."""
    X_std = (X - stats["X_mean"]) / stats["X_std"]
    y_std = (y - stats["y_mean"]) / stats["y_std"]
    return X_std.astype(np.float32), y_std.astype(np.float32)


def destandardise_y(y_std, stats):
    """Map model predictions in standardised units back to dollars (in $100k)."""
    return y_std * stats["y_std"] + stats["y_mean"]


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
def load_california_housing(backend="sklearn", test_frac=0.2, seed=0):
    """
    Returns:
        X_train, y_train, X_test, y_test  (standardised, float32)
        stats                              (dict for de-standardising predictions)
    """
    if backend == "sklearn":
        X, y = _load_via_sklearn()
    elif backend == "manual":
        X, y = _load_via_manual()
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    X_train_raw, y_train_raw, X_test_raw, y_test_raw = train_test_split(
        X, y, test_frac=test_frac, seed=seed)

    # Fit ONLY on the training fold.
    stats = standardise_fit(X_train_raw, y_train_raw)
    X_train, y_train = standardise_apply(X_train_raw, y_train_raw, stats)
    X_test,  y_test  = standardise_apply(X_test_raw,  y_test_raw,  stats)

    return X_train, y_train, X_test, y_test, stats


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, stats = load_california_housing(backend="sklearn")
    print(f"X_train: {X_tr.shape}  X_test: {X_te.shape}")
    print(f"y_train: {y_tr.shape}  y_test: {y_te.shape}")
    print(f"y_train range (standardised): [{y_tr.min():.2f}, {y_tr.max():.2f}]")
    print(f"y_mean (raw): ${stats['y_mean'] * 100_000:>10,.0f}")
    print(f"y_std  (raw): ${stats['y_std']  * 100_000:>10,.0f}")
    print(f"\nFeatures: {FEATURE_NAMES}")
