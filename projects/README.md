# Projects

Four projects put the classes of Neural Networks from Scratch to work on real data. Each is a released reference model with one documented run, and each can be built once the post in the "After" column has been read.

| Id | Project | Kind | Status | After | Measured result (seed 0) |
|---|---|---|---|---|---|
| `nn-p01` | [MNIST from scratch](mnist-from-scratch/README.md) | model | released | `nn-032` | 98.00 percent test accuracy, 9,800 of 10,000 images |
| `nn-p02` | [Binary classifier on two moons](binary-classifier/README.md) | model | released | `nn-034` | 200 of 200 held-out points correct at noise 0.1 |
| `nn-p03` | [Fashion-MNIST with the same network](fashion-mnist/README.md) | model | released | `nn-033` | 87.12 percent test accuracy, 8,712 of 10,000 images |
| `nn-p04` | [California housing regression](california-housing-regression/README.md) | model | released | `nn-032` | test $R^2$ of 0.8231, RMSE 49,222 dollars |

Each figure is the headline of the project's `docs/EVALUATION.md`, which gives the commands that print it, the per-class or per-fold detail, the runs from other seeds, and the limits of the claim.

## What the projects share

Every project has the same layout: a package under `src/<package>/` whose `nn.py` holds the project's own copy of the series' classes, with `model.py`, `data.py`, `train.py`, and `evaluate.py` beside it; tests under `tests/`; `docs/` with the architecture, the evaluation, the security notes, and the decision records; and a `project.yaml` that names the commands. The environment is managed by [uv](https://docs.astral.sh/uv/) from a committed `uv.lock`, and NumPy, pinned at 2.3.5, is the only third-party package the code imports, so that a run from the fixed seed repeats on the same machine. Weights are saved as plain arrays in an `.npz` file, so loading weights never runs code from the file. The three public datasets are fetched by a script under `scripts/`, which keeps a file only if its SHA-256 is the expected one; the two-moons data are generated and need no download. Datasets and weights stay on the reader's machine and are never committed.

## Commands

From a project's directory, with its package name in place of `<package>`:

```bash
uv sync --frozen
uv run python -m <package>.train
uv run python -m <package>.evaluate
uv run pytest --cov=src --cov-fail-under=95
uv run ruff check . && uv run ruff format --check .
```

The packages are `mnist_from_scratch`, `binary_classifier`, `fashion_mnist`, and `california_housing_regression`. Before training, the three projects with a public dataset run their download script once: `uv run python scripts/download_mnist.py`, `uv run python scripts/download_fashion_mnist.py`, or `uv run python scripts/download_california_housing.py`. Each project's README has its quickstart in full.
