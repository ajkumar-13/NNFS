# Changelog

All notable changes to this project are recorded here, in the form of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow semantic versioning: a change that alters the weights of the documented seed 0 run is a major version.

## [1.0.0] - 2026-10-05

The first release under the series' project contract. The four loose scripts became a tested package with a pinned environment, a checked dataset and a documented, repeatable result. The model and its training configuration did not change: from seed 0 the package produces, bit for bit, the weights the earlier scripts produced.

### Added

- The package `mnist_from_scratch` under `src/`, with `pyproject.toml` and `uv.lock`. The quickstart is `uv sync --frozen`, then `uv run python scripts/download_mnist.py`, `uv run python -m mnist_from_scratch.train` and `uv run python -m mnist_from_scratch.evaluate`.
- `scripts/download_mnist.py`, which fetches the four MNIST files over `https` and keeps each only if its size and SHA-256 match the record in `data.py`.
- A SHA-256 fingerprint of the weights, printed by the trainer and by the evaluation, so that two runs can be compared bit for bit.
- `--min-accuracy` on the evaluation, which exits with status 1 when the test accuracy is below the given fraction.
- `--data-dir` on all three commands.
- The parameter count and the training time in the printed output.
- Validation of every option, of the dataset files and of the weights file, with messages that name the command to run next.
- 157 tests with a coverage floor of 95 percent, a tiny synthetic dataset in MNIST's format under `tests/fixtures/`, and `scripts/make_test_fixture.py`, which generates it.
- `docs/ARCHITECTURE.md` with the architecture figure, `docs/EVALUATION.md`, `docs/SECURITY.md`, and four decision records under `docs/adr/`.

### Changed

- Weights are written to `mnist_weights.npz`, an archive of six plain arrays, and the option that names the file is `--weights` on both commands. It was `--checkpoint`, and the file was a pickle.
- The dataset is read only from the four original IDX files, and their SHA-256 is checked on every load. The trainer and the evaluation no longer download anything.
- NumPy is pinned to 2.3.5 and Python to 3.13, the versions the documented result was measured on.
- The trainer indexes each mini-batch through the epoch's permutation and no longer builds a shuffled copy of the dataset. The batches, and therefore the weights, are the same.
- The evaluation parses only the two test files.
- The README states measured figures. It said "roughly 97 percent" test accuracy; the documented run gives 98.00 percent, with a test loss of 0.0628 where the sample output showed 0.0892. It said 8 and 5 are the hardest digits; in the documented run 9 is the hardest, at 95.14 percent, and with another seed it was 2.

### Removed

- The scikit-learn backend and the `--backend` option. scikit-learn is no longer a dependency.
- `requirements.txt`, replaced by `pyproject.toml` and `uv.lock`.
- Support for pickle checkpoints. A `mnist_weights.pkl` written by the earlier scripts no longer loads; run the trainer again, which in the locked environment gives the same weights as an `.npz` file.

### Fixed

- The trainer's description said the L2 penalty was applied to every dense layer. It is applied to the two hidden layers' weights only, as it always was in the code, and the documentation now says so.

### Security

- Loading weights no longer unpickles a file, so a weights file cannot run code. Arrays are read with `allow_pickle=False` and checked for name, shape, type and finite values before they reach the network.
- Downloaded files are verified by size and SHA-256 and are never left in place when the check fails.
