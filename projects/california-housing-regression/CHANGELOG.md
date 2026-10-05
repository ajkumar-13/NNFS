# Changelog

All notable changes to this project are recorded here, in the form of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow semantic versioning: a change that alters the weights of the documented seed 0 run is a major version.

## [1.0.0] - 2026-10-05

The first release under the series' project contract. The four loose scripts became a tested package with a pinned environment, a checked dataset and a documented, repeatable result. The model and its training configuration did not change: from seed 0 the package produces, bit for bit, the folds and the weights the earlier scripts produced.

### Added

- The package `california_housing_regression` under `src/`, with `pyproject.toml` and `uv.lock`. The quickstart is `uv sync --frozen`, then `uv run python scripts/download_california_housing.py`, `uv run python -m california_housing_regression.train` and `uv run python -m california_housing_regression.evaluate`.
- `scripts/download_california_housing.py`, which fetches `cal_housing.tgz` over `https` and keeps it only if its size and SHA-256 match the record in `data.py`.
- A SHA-256 fingerprint of the weights, printed by the trainer and by the evaluation, so that two runs can be compared bit for bit.
- `--min-r2` on the evaluation, which exits with status 1 when the test $R^2$ is below the given value.
- `--seed` on the evaluation. The seed fixes the split, and the evaluation stops if the weights were trained on another one.
- `--data-dir` on all three commands.
- In the evaluation report: the mean error of each fold, the test fold scored separately at and below the census cap of 500,001 dollars, and two baselines fitted on the training fold, the training mean and a linear least-squares fit.
- The parameter count and the training time in the printed output.
- Validation of every option, of the dataset archive and of the weights file, with messages that name the command to run next.
- 239 tests with a coverage floor of 95 percent, among them tests that the scaling statistics come from the training fold alone; a tiny synthetic dataset in the archive's own format under `tests/fixtures/`; and `scripts/make_test_fixture.py`, which generates it.
- `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, `docs/SECURITY.md`, and five decision records under `docs/adr/`.

### Changed

- Weights are written to `cal_housing_weights.npz`, an archive of ten plain arrays (six of parameters, four of scaling statistics), and the option that names the file is `--weights` on both commands. It was `--checkpoint`, and the file was a pickle that also held both folds.
- The dataset is read only from the original archive, and its SHA-256 is checked on every load. The trainer and the evaluation no longer download anything.
- The evaluation rebuilds the folds from the dataset instead of reading them from the checkpoint, and measures its dollar errors against the values in the file as 64-bit floats. The printed RMSE, MAE and $R^2$ of the documented run are unchanged.
- The evaluation writes the predicted and published values only when asked, with `--predictions FILE.npz`. It used to write `predictions.npz` beside the checkpoint on every run.
- The trainer prints `loss`, the data loss plus the penalty, and `train_mse`, the data loss alone. The earlier trainer printed the first under the name `train_mse`, and a `train_rmse` in dollars derived from it that therefore included the penalty.
- NumPy is pinned to 2.3.5 and Python to 3.13, the versions the documented result was measured on.
- The trainer indexes each mini-batch through the epoch's permutation and no longer builds a shuffled copy of the training fold. The batches, and therefore the weights, are the same.
- The README states measured figures. Its results table already agreed with the documented run; these statements did not:
  - It said that for block groups at the cap the network predicts about 200 to 280 thousand dollars, and elsewhere about 200 to 320 thousand. In the documented run the mean prediction for the 207 capped test block groups is 468,755 dollars.
  - It said the target is capped at 500,000 dollars. The file records 500,001.
  - It compared the result with gradient boosting and a tuned boosted-tree model at $R^2$ of about 0.81 and 0.85. Neither was measured, and the comparison is removed. The linear baseline it gave as about 0.64 is now measured, at 0.6374.
  - It said the L2 penalty "contributes a tiny fraction of the loss". In epoch 200 of the documented run it is 8.2 percent of it.
  - It listed reaching $R^2$ above 0.80 as a stretch goal needing a wider network. The documented network reaches 0.8231.
  - It said training takes about 30 seconds. The documented run took 17.9 s.
- The diagram moved from `diagrams/` to `docs/diagrams/`, with the colour tokens renamed from `--ce-` to `--nn-` and nothing else changed. The figures printed inside it are not the measured ones; `docs/ARCHITECTURE.md` says so in its caption.

### Removed

- The scikit-learn backend, the manual backend and the `--backend` option. scikit-learn is no longer a dependency.
- `requirements.txt`, replaced by `pyproject.toml` and `uv.lock`.
- Support for pickle checkpoints. A `cal_housing_weights.pkl` written by the earlier scripts no longer loads; run the trainer again, which in the locked environment gives the same weights as an `.npz` file.

### Fixed

- The trainer's description said the L2 penalty was applied to every dense layer. It is applied to the two hidden layers' weights only, as it always was in the code, and the documentation now says so.
- The description of `nn.py` announced a `Layer_Linear` class that the file never contained. The sentence is gone.

### Security

- Loading weights no longer unpickles a file, so a weights file cannot run code. Arrays are read with `allow_pickle=False` and checked for name, shape, type and finite values before they reach the network.
- The downloaded archive is verified by size and SHA-256 and is never left in place when the check fails. It is read in memory and never extracted to disk.
