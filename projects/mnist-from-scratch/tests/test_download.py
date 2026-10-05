"""The downloader, against a stand-in for the network: nothing here opens a connection."""

import http.client
import io
import urllib.error
import urllib.request

import pytest

from helpers import FIXTURE_DIR, FIXTURE_FILES
from mnist_from_scratch import data
from mnist_from_scratch import download as download_module
from mnist_from_scratch.data import load_mnist, sha256_of
from mnist_from_scratch.download import MIRROR, DownloadError, download

BASE = "https://example.invalid/mnist/"


class FakeServer:
    """Serves the fixture files by URL and records what was asked for."""

    def __init__(self, overrides=None, failure=None):
        self.overrides = overrides or {}
        self.failure = failure
        self.requests = []

    def __call__(self, url, timeout=None):
        self.requests.append((url, timeout))
        if self.failure is not None:
            raise self.failure
        name = url.rsplit("/", 1)[1]
        body = self.overrides.get(name, (FIXTURE_DIR / name).read_bytes())
        return io.BytesIO(body)


def names(directory):
    return sorted(path.name for path in directory.iterdir())


def test_download_fetches_all_four_files_and_they_load(tmp_path):
    server = FakeServer()
    messages = []
    target = tmp_path / "cache"
    paths = download(target, BASE, FIXTURE_FILES, opener=server, log=messages.append)

    assert [path.name for path in paths] == [spec.name for spec in FIXTURE_FILES.values()]
    assert names(target) == sorted(spec.name for spec in FIXTURE_FILES.values())
    for spec in FIXTURE_FILES.values():
        assert sha256_of(target / spec.name) == spec.sha256
    assert [url for url, _ in server.requests] == [
        BASE + spec.name for spec in FIXTURE_FILES.values()
    ]
    assert all(timeout == download_module.TIMEOUT_SECONDS for _, timeout in server.requests)
    assert sum("checksum ok" in message for message in messages) == 4
    assert len(load_mnist(target, FIXTURE_FILES)[0]) == 12


def test_download_leaves_present_files_alone(tmp_path):
    download(tmp_path, BASE, FIXTURE_FILES, opener=FakeServer(), log=lambda _: None)
    server = FakeServer()
    messages = []
    download(tmp_path, BASE, FIXTURE_FILES, opener=server, log=messages.append)
    assert server.requests == []
    assert all("already present, checksum ok" in message for message in messages)


def test_download_adds_the_missing_slash_to_the_base_url(tmp_path):
    server = FakeServer()
    download(tmp_path, BASE.rstrip("/"), FIXTURE_FILES, opener=server, log=lambda _: None)
    assert server.requests[0][0] == BASE + "train-images-idx3-ubyte.gz"


def test_download_discards_a_file_with_the_wrong_checksum(tmp_path):
    name = "train-labels-idx1-ubyte.gz"
    tampered = bytes(FIXTURE_FILES["train_labels"].size)
    server = FakeServer(overrides={name: tampered})
    with pytest.raises(DownloadError, match=r"has SHA-256 .* the file was not kept"):
        download(tmp_path, BASE, FIXTURE_FILES, opener=server, log=lambda _: None)
    assert names(tmp_path) == ["train-images-idx3-ubyte.gz"]


def test_download_stops_when_more_bytes_arrive_than_expected(tmp_path):
    name = "train-images-idx3-ubyte.gz"
    server = FakeServer(overrides={name: bytes(200_000)})
    with pytest.raises(DownloadError, match="sent more than the expected 1,316 bytes"):
        download(tmp_path, BASE, FIXTURE_FILES, opener=server, log=lambda _: None)
    assert names(tmp_path) == []


def test_download_rejects_a_short_file(tmp_path):
    name = "train-images-idx3-ubyte.gz"
    server = FakeServer(overrides={name: b"short"})
    with pytest.raises(DownloadError, match="sent 5 bytes, expected 1,316"):
        download(tmp_path, BASE, FIXTURE_FILES, opener=server, log=lambda _: None)
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
    with pytest.raises(DownloadError, match=r"could not fetch https://example.invalid/mnist/"):
        download(tmp_path, BASE, FIXTURE_FILES, opener=server, log=lambda _: None)
    assert names(tmp_path) == []


def test_download_refuses_to_replace_an_unexpected_file(tmp_path):
    path = tmp_path / "train-images-idx3-ubyte.gz"
    path.write_bytes(b"something the user put here")
    server = FakeServer()
    with pytest.raises(DownloadError, match="exists but is not the expected file"):
        download(tmp_path, BASE, FIXTURE_FILES, opener=server, log=lambda _: None)
    assert path.read_bytes() == b"something the user put here"
    assert server.requests == []


@pytest.mark.parametrize("url", ["http://example.invalid/mnist/", "file:///tmp/mnist/", "mnist/"])
def test_download_accepts_https_only(tmp_path, url):
    with pytest.raises(DownloadError, match="must start with https://"):
        download(tmp_path / "cache", url, FIXTURE_FILES, opener=FakeServer())
    assert not (tmp_path / "cache").exists()


def test_download_defaults_to_the_real_file_list_and_urlopen(tmp_path, monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(data, "MNIST_FILES", dict(FIXTURE_FILES))
    monkeypatch.setattr(urllib.request, "urlopen", server)
    download(tmp_path, log=lambda _: None)
    assert server.requests[0][0] == MIRROR + "train-images-idx3-ubyte.gz"
    assert len(names(tmp_path)) == 4


def test_the_mirror_is_https():
    assert MIRROR.startswith("https://")
    assert MIRROR.endswith("/")


def test_main_downloads_and_reports_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(data, "MNIST_FILES", dict(FIXTURE_FILES))
    monkeypatch.setattr(urllib.request, "urlopen", FakeServer())
    assert download_module.main(["--data-dir", str(tmp_path / "cache")]) == 0
    output = capsys.readouterr().out
    assert "all four files present and checked" in output
    assert len(names(tmp_path / "cache")) == 4


def test_main_reports_a_failure_on_stderr_with_status_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(data, "MNIST_FILES", dict(FIXTURE_FILES))
    monkeypatch.setattr(
        urllib.request, "urlopen", FakeServer(failure=urllib.error.URLError("offline"))
    )
    assert download_module.main(["--data-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "error: could not fetch" in captured.err
    assert "all four files" not in captured.out


def test_main_rejects_a_base_url_that_is_not_https(tmp_path, capsys):
    status = download_module.main(
        ["--data-dir", str(tmp_path), "--base-url", "http://example.invalid/"]
    )
    assert status == 1
    assert "must start with https://" in capsys.readouterr().err
