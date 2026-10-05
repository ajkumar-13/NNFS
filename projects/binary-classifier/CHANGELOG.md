# Changelog

All notable changes to this project are recorded in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-05

The first released version. The model and its training configuration are those of the project as it stood before: the restructured code produces the same six weight arrays, bit for bit, as the scripts it replaces, from the same command-line defaults.

### Added

- The package `binary_classifier` under `src/`, with the commands `python -m binary_classifier.train` and `python -m binary_classifier.evaluate`.
- `pyproject.toml`, `.python-version` and `uv.lock`: NumPy pinned to 2.3.5 on Python 3.13, the environment of every project of the series, installed with `uv sync --frozen`.
- `--init he|xavier|small` on the training command, and the `init` argument of `Layer_Dense` as `nn-033` defines it. The default, `he`, is what the code did before.
- `scripts/seed_sweep.py`, which repeats the run for ten seeds and the three initialisations.
- In the training output: the number of parameters, the SHA-256 of the generated data, and the time taken.
- In the evaluation output: the configuration stored in the weights file, the count of each class classified correctly, how many hidden units are never active, always active, or switching over the training set, and the $R^2$ of a plane fitted to the logits.
- `--grid` on the evaluation command, to choose where the decision grid is written, and `--min-accuracy`, which makes the command exit with status 1 when the test accuracy is below the given fraction.
- Checks on every command-line option, with exit status 2 and a message that says what was wrong; a missing or unreadable weights file ends the evaluation with `error:` and exit status 1.
- A test suite of 155 tests, with the coverage floor of 95 percent enforced by the test command, and `ruff check` with `ruff format --check` as the lint command.
- `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, `docs/SECURITY.md`, six decision records under `docs/adr/`, this changelog, and `LICENSE`.

### Changed

- Weights are stored in a NumPy `.npz` archive read with pickling switched off. The option is `--weights` (it was `--checkpoint`) and the default name `moons_weights.npz`. **Files written by earlier versions (`moons_weights.pkl`) no longer load; train again to replace them.**
- Weights and grid names must end in `.npz`. Any other name is refused before training starts, because `numpy.savez` would append the ending and write to a name that was not asked for.
- The four scripts `nn.py`, `data.py`, `train.py`, and `evaluate.py` moved from the top of the project into the package. `python train.py` and `python evaluate.py` no longer exist.
- The figure moved from `diagrams/` to `docs/diagrams/`, and its colour tokens were renamed from `--ce-` to `--nn-`.
- The default noise of `make_moons` is 0.1, the value the training command passes; it was 0.2.
- The training log no longer prints the learning rate on every line; it is constant and is printed once.
- The README states measured results: 800 of 800 and 200 of 200. It previously gave "~99.9%" and "~98.5%" in one place and "~99-100%" in another.

### Removed

- `requirements.txt`, replaced by `pyproject.toml` and `uv.lock`.
- Reading and writing of pickle files.

### Fixed

- The evaluation printed the number of correct predictions as `int(accuracy * n)`, which can fall one short of the true count through rounding. The count now comes from the confusion matrix.
- The docstring of `make_moons` placed the lower moon at $(1, -0.5)$. Its centre is $(1, 0.5)$; the code was right.

### Security

- Loading a weights file can no longer execute code from the file. See `docs/SECURITY.md` and `docs/adr/0002-weights-as-npz.md`.

## Before 1.0.0

Unversioned states of the project, recorded for the history they explain.

- 2026-06-10: `Layer_Dense` changed from 0.01 times a standard normal to He initialisation, and the default noise from 0.2 to 0.1, after the first configuration scored 85 percent on the test set.
- 2026-06-01: the project was added to the series as four scripts and a README.
