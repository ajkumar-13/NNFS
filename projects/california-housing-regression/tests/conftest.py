"""Fixtures for every test: no network, a seeded generator, and the tiny committed dataset."""

import urllib.request
from pathlib import Path

import numpy as np
import pytest

from california_housing_regression import data
from helpers import FIXTURE_ARCHIVE, FIXTURE_DIR


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any attempt to open a URL in a test is an error."""

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


@pytest.fixture(autouse=True)
def _seeded():
    """Every test starts from the same state of NumPy's global generator."""
    np.random.seed(1234)


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def as_housing(monkeypatch) -> Path:
    """Make the tiny fixture stand in for the real archive in code that takes no file record."""
    monkeypatch.setattr(data, "ARCHIVE", FIXTURE_ARCHIVE)
    return FIXTURE_DIR
