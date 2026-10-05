"""Write the tiny dataset that the tests read, in the California housing archive's own format.

    uv run python scripts/make_test_fixture.py

One file goes to tests/fixtures/, named as the real one is: cal_housing.tgz, a gzip tar archive
holding CaliforniaHousing/cal_housing.data with 40 rows of nine comma-separated columns. The
content is synthetic and follows a formula, so the tests can state the expected value of any cell.
For row k = 0 .. 39:

    longitude         = -124.30 + 0.25 k
    latitude          = 32.50 + 0.20 k
    housingMedianAge  = 1 + (7 k mod 52)
    households        = 90 + 5 k
    totalRooms        = households * (4 + (k mod 5))
    totalBedrooms     = households + 2 k
    population        = 250 + 31 k
    medianIncome      = 1.5 + 0.21 k
    medianHouseValue  = min(60000 + 12500 k, 500001)        rows 36 to 39 sit at the census cap

The tar header carries no owner and no timestamp and the gzip header no file name and no
timestamp, so the same bytes are written every time. The script prints the file's size and SHA-256;
tests/helpers.py records them, and the tests fail if the file and the record ever disagree.
"""

import gzip
import hashlib
import io
import sys
import tarfile
from pathlib import Path

N_ROWS = 40
MEMBER = "CaliforniaHousing/cal_housing.data"
TOP_CODE = 500_001


def row(k: int) -> tuple[float, ...]:
    households = 90 + 5 * k
    return (
        -124.30 + 0.25 * k,
        32.50 + 0.20 * k,
        1 + (7 * k) % 52,
        households * (4 + k % 5),
        households + 2 * k,
        250 + 31 * k,
        households,
        1.5 + 0.21 * k,
        min(60_000 + 12_500 * k, TOP_CODE),
    )


def table_bytes() -> bytes:
    lines = [",".join(f"{value:.6f}" for value in row(k)) for k in range(N_ROWS)]
    return ("\n".join(lines) + "\n").encode("ascii")


def archive_bytes(members: dict[str, bytes]) -> bytes:
    """A gzip tar archive of ``members`` whose bytes depend on nothing but its content."""
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as packed,
        tarfile.open(fileobj=packed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def main() -> int:
    target = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    target.mkdir(parents=True, exist_ok=True)
    packed = archive_bytes({MEMBER: table_bytes()})
    (target / "cal_housing.tgz").write_bytes(packed)
    print(f"cal_housing.tgz  {len(packed)} bytes  sha256 {hashlib.sha256(packed).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
