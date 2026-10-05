# Changelog

All notable changes to this project are recorded here, in the form of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow semantic versioning: a change that alters the weights of the documented seed 0 run is a major version.

## [1.0.0] - 2026-10-05

The first release under the series' project contract. The four loose scripts became a tested package with a pinned environment, a checked dataset and a documented, repeatable result, laid out as the series' MNIST project (`nn-p01`) is. The model and its training configuration did not change: from seed 0 the package produces, bit for bit, the weights the earlier scripts produced.

### Added

- The package `fashion_mnist` under `src/`, with `pyproject.toml` and `uv.lock`. The quickstart is `uv sync --frozen`, then `uv run python scripts/download_fashion_mnist.py`, `uv run python -m fashion_mnist.train` and `uv run python -m fashion_mnist.evaluate`.
- `scripts/download_fashion_mnist.py`, which fetches the four Fashion-MNIST files over `https` and keeps each only if its size and SHA-256 match the record in `data.py`.
- A SHA-256 fingerprint of the weights, printed by the trainer and by the evaluation, so that two runs can be compared bit for bit.
- `--min-accuracy` on the evaluation, which exits with status 1 when the test accuracy is below the given fraction.
- `--data-dir` on all three commands.
- The parameter count, the training time and the first misclassified test indices in the printed output.
- Validation of every option, of the dataset files and of the weights file, with messages that name the command to run next.
- 163 tests with a coverage floor of 95 percent, a tiny synthetic dataset in the dataset's file format under `tests/fixtures/`, and `scripts/make_test_fixture.py`, which generates it.
- `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, `docs/SECURITY.md`, and four decision records under `docs/adr/`.

### Changed

- Weights are written to `fashion_mnist_weights.npz`, an archive of six plain arrays, and the option that names the file is `--weights` on both commands. It was `--checkpoint`, and the file was a pickle.
- The dataset is read only from the four original IDX files, and their SHA-256 is checked on every load. The trainer and the evaluation no longer download anything. The files come from the same bucket of the dataset's authors as before, now over `https`; the earlier address was plain `http`.
- NumPy is pinned to 2.3.5 and Python to 3.13, the versions the documented result was measured on.
- The trainer indexes each mini-batch through the epoch's permutation and no longer builds a shuffled copy of the dataset. The batches, and therefore the weights, are the same.
- The evaluation parses only the two test files.
- The most-confused pairs with equal counts are listed by true class and then by predicted class. They were listed in the reverse order. No two of the five pairs of the documented run have equal counts, so its list is unchanged.
- The figure moved from `diagrams/` to `docs/diagrams/`, and its colour tokens carry the series prefix `--nn-`. Nothing else in it changed; its numbers are illustrative and predate the measured run, as its caption in `docs/ARCHITECTURE.md` says.
- The README states measured figures. It said the test accuracy was about 87 percent, and the trainer's own description said 88 to 89; the documented run gives 87.12 percent with a test loss of 0.3419. It said MNIST reached about 97 percent with this network; the MNIST project's documented run gives 98.00. Its per-class table and its list of confusions did not come from a run: it gave Shirt 71.0 percent as the hardest class, and Pullover read as Coat and Shirt read as T-shirt/top, 100 times each, as the largest confusions, where the documented run has Pullover hardest at 72.60 percent, Shirt at 75.20, and T-shirt/top read as Shirt, 185 times, as the largest confusion.

### Removed

- The scikit-learn backend and the `--backend` option. scikit-learn is no longer a dependency.
- `requirements.txt`, replaced by `pyproject.toml` and `uv.lock`.
- Support for pickle checkpoints. A `fashion_mnist_weights.pkl` written by the earlier scripts no longer loads; run the trainer again, which in the locked environment gives the same weights as an `.npz` file.
- Claims the project never measured: accuracies for logistic regression, for wider or deeper dense networks and for convolutional networks on this dataset, and the table of stretch goals built on them.

### Fixed

- The evaluation stopped with a `UnicodeEncodeError` just before its list of most-confused pairs whenever its output was redirected to a file or a pipe on Windows, because the heading contained an arrow character that the default encoding there cannot write. The report is now plain ASCII.
- The trainer's description said the L2 penalty was applied to every dense layer. It is applied to the two hidden layers' weights only, as it always was in the code, and the documentation now says so.

### Security

- Loading weights no longer unpickles a file, so a weights file cannot run code. Arrays are read with `allow_pickle=False` and checked for name, shape, type and finite values before they reach the network.
- Downloaded files are verified by size and SHA-256 and are never left in place when the check fails, and the download uses `https` only.
