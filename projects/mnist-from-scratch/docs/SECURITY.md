# Security

Audited: 2026-10-05.

## What the project touches

It reads four dataset files and one weights file from the local disk, writes the weights file and the dataset files, and takes its options from the command line. One module, `download.py`, opens network connections, and only when the user runs `scripts/download_mnist.py`. It runs no subprocess, evaluates no text as code, reads no environment variables, and starts no server.

## Threat notes

- **A weights file is data, not a program.** Weights are stored as six arrays in an `.npz` archive and read with `allow_pickle=False`, so loading a file someone else produced cannot execute code from it. An archive that contains an object array is refused, not unpickled. Before any array data is read, the loader checks that the archive holds exactly the six expected names and that every array's header declares the shape of its layer and a floating-point type, so a crafted file cannot make it allocate more memory than the network itself occupies. After reading, it rejects values that are not finite. A path that does not end in `.npz`, including a pickle checkpoint from before version 1.0.0, is refused without being opened. The reasons are in [record 0002](adr/0002-weights-as-npz.md).
- **Residual risk of a weights file.** A file that passes every check is still someone's numbers. The evaluation will report whatever those numbers score; it cannot tell you where they came from. The fingerprint printed by both commands lets you confirm that a file holds the weights of a run you made.
- **The dataset is pinned by content.** Each of the four files has its size and SHA-256 recorded in `data.py`. The downloader keeps a file only if both match, and the loader checks the SHA-256 again on every load before it parses anything, so a mirror, a proxy or a later edit on disk cannot substitute other data unnoticed. The parser therefore sees only the published bytes; it still validates each header against the payload length and reports a mismatch.
- **The download.** Transfers use `https` only, and Python's default certificate verification; an address with another scheme is rejected before any connection. A transfer is cut off as soon as it exceeds the recorded size, a connection that stalls times out after 60 seconds, and an incomplete or altered file is deleted, never left under the dataset's name. `--base-url` lets the user choose another mirror; the checksums apply to it equally.
- **Existing files are not overwritten blindly.** If a file with a dataset name is present and has the wrong checksum, the downloader stops and says so; it does not delete or replace it. The trainer replaces the weights file only by renaming a completely written temporary file.
- **Resource use.** Training holds the dataset in memory: $70{,}000 \times 784 \times 4 = 219{,}520{,}000$ bytes of images. Running out of memory is reported as an error, not as a crash trace. No input can make a run longer than the epochs the user asked for.
- **Personal data.** None. MNIST is a public dataset of handwritten digits, and the project stores nothing about its user.

## Secrets

None. The project needs no credentials and reads no configuration from the environment, so there is no `.env` file and no `.env.example`.

## Dependencies

- Runtime: NumPy 2.3.5, pinned exactly because the documented result depends on its build ([record 0003](adr/0003-pin-numpy-for-bit-reproducibility.md)).
- Development: pytest 9.1.1, pytest-cov 7.1.0 and ruff 0.16.10, with the packages they bring in (coverage 7.16.2, colorama, iniconfig, packaging, pluggy, pygments).
- All ten are locked with their hashes in `uv.lock`, and `uv sync --frozen` installs exactly those files.
- Audit on 2026-10-05: `uv audit --frozen` reported no known vulnerabilities and no adverse project statuses in the 10 locked packages.
- The exact pin means a security fix in a later NumPy does not arrive by itself. The audit is repeated at each yearly maintenance check; if it reports NumPy, the pin moves and the evaluation is redone as record 0003 describes.

## Reporting a problem

The series takes no pull requests or issues. Report a security problem, like any other correction, through the series website.
