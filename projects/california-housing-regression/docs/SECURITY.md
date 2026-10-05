# Security

Audited: 2026-10-05.

## What the project touches

It reads one dataset archive and one weights file from the local disk, writes the weights file, the dataset archive and, when asked, a predictions file, and takes its options from the command line. One module, `download.py`, opens network connections, and only when the user runs `scripts/download_california_housing.py`. It runs no subprocess, evaluates no text as code, reads no environment variables, and starts no server.

## Threat notes

- **A weights file is data, not a program.** Weights are stored as ten arrays in an `.npz` archive, six of parameters and four of scaling statistics, and read with `allow_pickle=False`, so loading a file someone else produced cannot execute code from it. An archive that contains an object array is refused, not unpickled. Before any array data is read, the loader checks that the archive holds exactly the ten expected names and that every array's header declares the shape this network needs and a floating-point type, so a crafted file cannot make it allocate more memory than the network itself occupies. After reading, it rejects values that are not finite and standard deviations that are not positive. A path that does not end in `.npz`, including a pickle checkpoint from before version 1.0.0, is refused without being opened. The reasons are in [record 0002](adr/0002-weights-as-npz.md).
- **Residual risk of a weights file.** A file that passes every check is still someone's numbers. The evaluation will report whatever those numbers score; it cannot tell you where they came from. The fingerprint printed by both commands lets you confirm that a file holds the weights of a run you made. The scaling statistics in the file are compared with the ones recomputed from the data, and the recomputed ones are used for scoring, so a file cannot move the folds or the dollar scale.
- **The dataset is pinned by content.** The archive has its size and SHA-256 recorded in `data.py`. The downloader keeps a file only if both match, and the loader checks the SHA-256 again on every load before it opens the archive, so a mirror, a proxy or a later edit on disk cannot substitute other data unnoticed.
- **The archive is never extracted.** The loader reads one member, `CaliforniaHousing/cal_housing.data`, by its exact name into memory and writes nothing to disk, so a path inside the archive cannot point a write anywhere. A member that is not a regular file, or that declares more than 16,777,216 bytes, is refused. The parser then checks that the table is nine columns of finite numbers.
- **The download.** Transfers use `https` only, and Python's default certificate verification; an address with another scheme is rejected before any connection. The default address redirects to the host's storage service; the size and checksum apply to whatever arrives at the end of the redirects. A transfer is cut off as soon as it exceeds the recorded size, a connection that stalls times out after 60 seconds, and an incomplete or altered file is deleted, never left under the dataset's name. `--url` lets the user choose another source; the checksum applies to it equally.
- **Existing files are not overwritten blindly.** If a file with the archive's name is present and has the wrong checksum, the downloader stops and says so; it does not delete or replace it. The trainer replaces the weights file, and the evaluation the predictions file, only by renaming a completely written temporary file, and the evaluation refuses to write its predictions over the weights.
- **Resource use.** The whole dataset is $20{,}640 \times 9 \times 8 = 1{,}486{,}080$ bytes as 64-bit floats. No input can make a run longer than the epochs the user asked for.
- **Personal data.** None. The data is a public table of census aggregates, one row per block group and none per person or household, and the project stores nothing about its user.

## Secrets

None. The project needs no credentials and reads no configuration from the environment, so there is no `.env` file and no `.env.example`.

## Dependencies

- Runtime: NumPy 2.3.5, pinned exactly because the documented result depends on its build and on its random generators ([record 0003](adr/0003-pin-numpy-for-bit-reproducibility.md)).
- Development: pytest 9.1.1, pytest-cov 7.1.0 and ruff 0.16.10, with the packages they bring in (coverage 7.16.2, colorama, iniconfig, packaging, pluggy, pygments).
- All ten are locked with their hashes in `uv.lock`, and `uv sync --frozen` installs exactly those files.
- Audit on 2026-10-05: `uv audit --frozen` reported no known vulnerabilities and no adverse project statuses in the 10 locked packages.
- scikit-learn, which the project used to read the data before version 1.0.0, is no longer a dependency ([record 0004](adr/0004-statlib-archive-with-checksum.md)).
- The exact pin means a security fix in a later NumPy does not arrive by itself. The audit is repeated at each yearly maintenance check; if it reports NumPy, the pin moves and the evaluation is redone as record 0003 describes.

## Reporting a problem

The series takes no pull requests or issues. Report a security problem, like any other correction, through the series website.
