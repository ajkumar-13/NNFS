# Security

Audited: 2026-10-05.

## What the project touches

Two commands that a person runs on their own machine. They generate data from a seed, train a small network, read one weights file from the local disk, and write that file and a decision grid. The package opens no network connection, runs no subprocess, evaluates no text as code, reads no environment variables, and starts no server. Its inputs are the command-line options and the weights file.

## Threat notes

- **A weights file is data, not a program.** Before version 1.0.0 weights were written with `pickle`, and unpickling a file executes whatever the file tells Python to construct. From 1.0.0 the file is a NumPy `.npz` archive of numeric and string arrays, read with `numpy.load(..., allow_pickle=False)`. With that argument NumPy refuses any array of Python objects, which is the only part of the format that is unpickled. Before a network is built, `load_weights` also checks that the sixteen expected arrays are present and that each has the expected shape, a numeric type, and finite values; that labels are 0 or 1; and that the format version is the one this version writes. A file named `.pkl` is refused by name, with a message that says to train again ([record 0002](adr/0002-weights-as-npz.md)).
- **The tests hold this in place.** `tests/test_weights.py` builds a pickle whose loading would create a directory, presents it once as a file renamed to `.npz` and once as an object array inside an otherwise valid archive, and asserts both that loading raises `WeightsError` and that the directory was never created.
- **Residual risk of a weights file.** A crafted archive can still declare arrays large enough to exhaust memory before the shape check sees them, and the safety of parsing rests on NumPy's and Python's readers for the `.npy` and zip formats. A file that passes every check is still someone's numbers: the evaluation reports whatever they score on the points stored beside them. Load files you produced yourself or received from someone you trust.
- **The decision grid.** The grid written by evaluation is also a `.npz` archive of numeric arrays. The plotting snippet in the README reads it with `numpy.load`, whose default refuses pickled objects.
- **Command-line options.** Options are checked before any work starts: counts must be whole numbers of at least 1, the noise finite and not negative, the seed between 0 and 4,294,967,295, the initialisation one of the three offered, and output names must end in `.npz` inside a directory that exists. A bad option ends the command with a sentence saying what was wrong.
- **Existing files are overwritten.** `train` writes the weights and `evaluate` the grid, at the paths given or at `moons_weights.npz` and `decision_grid.npz` in the working directory, and an existing file at either path is replaced without a question.
- **Resource use.** The number of points and of epochs has no upper limit, so a very large value runs for a long time or runs out of memory; that affects only the person who typed it.
- **Personal data.** None. The data are generated, and the project stores nothing about its user.

## Secrets

None. The project needs no credentials and reads no configuration from the environment, so there is no `.env` file and no `.env.example`.

## Dependencies

- Runtime: NumPy 2.3.5, pinned exactly, as in every project of the series ([record 0003](adr/0003-pin-numpy-for-bit-reproducibility.md)).
- Development: pytest 9.1.1, pytest-cov 7.1.0 and ruff 0.16.10, with the packages they bring in (coverage 7.16.2, colorama, iniconfig, packaging, pluggy, pygments).
- All ten are locked with their hashes in `uv.lock`, and `uv sync --frozen` installs exactly those files. The build backend, hatchling, is fetched at the first build and is not in the lock.
- Audit on 2026-10-05: `uv audit --frozen` reported no known vulnerabilities and no adverse project statuses in the 10 locked packages.
- The exact pin means a security fix in a later NumPy does not arrive by itself. The audit is repeated at each yearly maintenance check; if it reports NumPy, the pin moves and the evaluation is redone as record 0003 describes.

## Reporting a problem

The series takes no pull requests or issues. Report a security problem, like any other correction, through the series website.
