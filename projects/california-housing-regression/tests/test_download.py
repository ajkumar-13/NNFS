"""The downloader, against a stand-in for the network: nothing here opens a connection."""

import http.client
import io
import urllib.error
import urllib.request

import pytest

from california_housing_regression import data
from california_housing_regression import download as download_module
from california_housing_regression.data import load_california_housing, sha256_of
from california_housing_regression.download import ARCHIVE_URL, DownloadError, download
from helpers import FIXTURE_ARCHIVE, FIXTURE_DIR

URL = "https://example.invalid/files/cal_housing"
FIXTURE_BYTES = (FIXTURE_DIR / FIXTURE_ARCHIVE.name).read_bytes()


class FakeServer:
    """Serves the fixture archive, or other bytes, and records what was asked for."""

    def __init__(self, body=FIXTURE_BYTES, failure=None):
        self.body = body
        self.failure = failure
        self.requests = []

    def __call__(self, url, timeout=None):
        self.requests.append((url, timeout))
        if self.failure is not None:
            raise self.failure
        return io.BytesIO(self.body)


def names(directory):
    return sorted(path.name for path in directory.iterdir())


def test_download_fetches_the_archive_and_it_loads(tmp_path):
    server = FakeServer()
    messages = []
    target = tmp_path / "cache"
    path = download(target, URL, FIXTURE_ARCHIVE, opener=server, log=messages.append)

    assert path == target / "cal_housing.tgz"
    assert names(target) == ["cal_housing.tgz"]
    assert sha256_of(path) == FIXTURE_ARCHIVE.sha256
    assert server.requests == [(URL, download_module.TIMEOUT_SECONDS)]
    assert messages == [
        "  cal_housing.tgz: downloading 1,052 bytes",
        "  cal_housing.tgz: checksum ok",
    ]
    assert len(load_california_housing(target, archive=FIXTURE_ARCHIVE).X_train) == 32


def test_download_leaves_a_present_archive_alone(tmp_path):
    download(tmp_path, URL, FIXTURE_ARCHIVE, opener=FakeServer(), log=lambda _: None)
    server = FakeServer()
    messages = []
    download(tmp_path, URL, FIXTURE_ARCHIVE, opener=server, log=messages.append)
    assert server.requests == []
    assert messages == ["  cal_housing.tgz: already present, checksum ok"]


def test_download_discards_a_file_with_the_wrong_checksum(tmp_path):
    server = FakeServer(body=bytes(FIXTURE_ARCHIVE.size))
    with pytest.raises(DownloadError, match=r"has SHA-256 .* the file was not kept"):
        download(tmp_path, URL, FIXTURE_ARCHIVE, opener=server, log=lambda _: None)
    assert names(tmp_path) == []


def test_download_stops_when_more_bytes_arrive_than_expected(tmp_path):
    server = FakeServer(body=bytes(200_000))
    with pytest.raises(DownloadError, match="sent more than the expected 1,052 bytes"):
        download(tmp_path, URL, FIXTURE_ARCHIVE, opener=server, log=lambda _: None)
    assert names(tmp_path) == []


def test_download_rejects_a_short_file(tmp_path):
    server = FakeServer(body=b"short")
    with pytest.raises(DownloadError, match="sent 5 bytes, expected 1,052"):
        download(tmp_path, URL, FIXTURE_ARCHIVE, opener=server, log=lambda _: None)
    assert names(tmp_path) == []


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("no route to host"),
        TimeoutError("timed out"),
        ConnectionResetError("reset by peer"),
        http.client.IncompleteRead(b"partial"),
    ],
)
def test_download_reports_a_network_failure(tmp_path, failure):
    server = FakeServer(failure=failure)
    with pytest.raises(DownloadError, match=r"could not fetch https://example.invalid/files/"):
        download(tmp_path, URL, FIXTURE_ARCHIVE, opener=server, log=lambda _: None)
    assert names(tmp_path) == []


def test_download_refuses_to_replace_an_unexpected_file(tmp_path):
    path = tmp_path / "cal_housing.tgz"
    path.write_bytes(b"something the user put here")
    server = FakeServer()
    with pytest.raises(DownloadError, match="exists but is not the expected file"):
        download(tmp_path, URL, FIXTURE_ARCHIVE, opener=server, log=lambda _: None)
    assert path.read_bytes() == b"something the user put here"
    assert server.requests == []


@pytest.mark.parametrize("url", ["http://example.invalid/a.tgz", "file:///tmp/a.tgz", "a.tgz"])
def test_download_accepts_https_only(tmp_path, url):
    with pytest.raises(DownloadError, match="must start with https://"):
        download(tmp_path / "cache", url, FIXTURE_ARCHIVE, opener=FakeServer())
    assert not (tmp_path / "cache").exists()


def test_download_defaults_to_the_real_record_and_urlopen(tmp_path, monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(data, "ARCHIVE", FIXTURE_ARCHIVE)
    monkeypatch.setattr(urllib.request, "urlopen", server)
    download(tmp_path, log=lambda _: None)
    assert server.requests[0][0] == ARCHIVE_URL
    assert names(tmp_path) == ["cal_housing.tgz"]


def test_the_default_address_is_https():
    assert ARCHIVE_URL.startswith("https://")


def test_main_downloads_and_reports_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(data, "ARCHIVE", FIXTURE_ARCHIVE)
    monkeypatch.setattr(urllib.request, "urlopen", FakeServer())
    assert download_module.main(["--data-dir", str(tmp_path / "cache")]) == 0
    output = capsys.readouterr().out
    assert f"California housing into {tmp_path / 'cache'}" in output
    assert "the archive is present and checked" in output
    assert names(tmp_path / "cache") == ["cal_housing.tgz"]


def test_main_passes_the_address_through(tmp_path, monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(data, "ARCHIVE", FIXTURE_ARCHIVE)
    monkeypatch.setattr(urllib.request, "urlopen", server)
    assert download_module.main(["--data-dir", str(tmp_path), "--url", URL]) == 0
    assert server.requests[0][0] == URL


def test_main_reports_a_failure_on_stderr_with_status_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(data, "ARCHIVE", FIXTURE_ARCHIVE)
    monkeypatch.setattr(urllib.request, "urlopen", FakeServer(failure=TimeoutError("timed out")))
    assert download_module.main(["--data-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: could not fetch https://")
    assert "the archive is present and checked" not in captured.out


def test_main_rejects_an_address_that_is_not_https(tmp_path, capsys):
    assert download_module.main(["--data-dir", str(tmp_path), "--url", "http://x.invalid/a"]) == 1
    assert "must start with https://" in capsys.readouterr().err


def test_the_real_network_is_closed_in_tests(tmp_path):
    with pytest.raises(AssertionError, match="tried to open a network connection"):
        download(tmp_path, URL, FIXTURE_ARCHIVE, log=lambda _: None)
    assert names(tmp_path) == []
