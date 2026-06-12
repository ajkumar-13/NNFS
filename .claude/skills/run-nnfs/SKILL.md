---
name: run-nnfs
description: Run, train, evaluate, smoke-test, or verify the NNFS projects (MNIST, two-moons binary classifier, Fashion-MNIST, California housing) and the cumulative notebook. Use when asked to run the nnfs code, retrain a model, check accuracy numbers, or confirm the projects still work.
---

# Run the NNFS projects

This repo is a 35-post "Neural Networks from Scratch" blog series whose runnable
surface is four NumPy-only Python projects under `projects/` (each with
`train.py` / `evaluate.py` CLIs) plus `cumulative_notebook.ipynb`. There is no
server or GUI; the driver is a smoke script that trains-if-needed, evaluates,
and asserts metric thresholds.

**All paths below are relative to the repo root, and everything must be run
from the repo root** — dataset caches (`mnist_cache/`, `fashion_mnist_cache/`)
and weight checkpoints (`*_weights.pkl`) are cwd-relative.

## Prerequisites

- Python 3.13 with `numpy`, `scikit-learn`, `nnfs` (`pip install numpy scikit-learn nnfs`).
  Verified with numpy 2.3.5, scikit-learn 1.9.0.
- All `*.pkl` / `*.gz` / `*.npz` artifacts are gitignored: a clean clone has no
  checkpoints and no dataset caches. The driver handles both (trains short runs,
  and the data loaders download from mirrors on first use — verified: MNIST pulls
  ~11 MB from `ossci-datasets.s3.amazonaws.com`).

## Run (agent path) — the smoke driver

```
python .claude/skills/run-nnfs/smoke.py
```

Runs five steps — `moons`, `mnist`, `fashion`, `housing`, `notebook` — and
exits non-zero if any fails. Each project step: if its canonical checkpoint
(`moons_weights.pkl`, `mnist_weights.pkl`, `fashion_mnist_weights.pkl`,
`cal_housing_weights.pkl` at repo root) is missing, it smoke-trains one
(1 epoch for MNIST/Fashion, 500 epochs moons, 50 epochs housing), then runs
`evaluate.py` and asserts a metric threshold. The notebook step executes all
11 code cells in-process and checks the final test accuracy.

Timing observed: ~12 s when checkpoints exist, ~55 s with `--fresh` training,
plus first-time dataset downloads on a clean machine.

Variants (all run this session):

```
python .claude/skills/run-nnfs/smoke.py mnist            # one step only
python .claude/skills/run-nnfs/smoke.py --fresh moons mnist fashion housing
```

`--fresh` retrains into temp-dir checkpoints (never clobbers the root ones).

Expected passing output ends with `ALL STEPS PASSED (5/5)`. Reference metrics
with fully-trained checkpoints: MNIST 0.98, Fashion 0.8712, moons 1.0,
housing R² 0.8458, notebook final test accuracy ~0.68. With `--fresh` smoke
training: MNIST ~0.92, Fashion ~0.80, housing R² ~0.81.

## Run (human path) — individual projects

Each project is a plain CLI, run from the repo root:

```
$env:PYTHONUTF8='1'   # see Gotchas — required on Windows
python projects/01-mnist-from-scratch/train.py --epochs 1 --backend manual --checkpoint $env:TEMP\scratch_mnist.pkl
python projects/01-mnist-from-scratch/evaluate.py --backend manual --checkpoint $env:TEMP\scratch_mnist.pkl
```

Pass an explicit `--checkpoint` for experiments — `train.py`'s default
silently **overwrites the canonical root checkpoint** (verified the hard way;
a 1-epoch run replaced the 0.98-accuracy `mnist_weights.pkl` with a 0.92 one).

Same shape for `02-binary-classifier` (no `--backend`),
`03-fashion-mnist` (`--backend manual`), and
`04-california-housing-regression` (no `--backend`; needs sklearn for the
dataset). Full training (`train.py` with default epochs) takes minutes, not
hours, but the smoke driver is the right default.

## Direct invocation (internals)

The network classes live in each project's `nn.py` and import standalone —
useful for PRs touching one class. The notebook cells also run bare; the
driver's `notebook` step is exactly that (`exec` of each cell in one
namespace), so `python .claude/skills/run-nnfs/smoke.py notebook` is the
fastest "does the library still work" check (~9 s).

## Gotchas

- **Windows cp1252 crash (will bite you):** `projects/03-fashion-mnist/evaluate.py`
  prints `→` (U+2192) and dies with `UnicodeEncodeError: 'charmap' codec can't
  encode character '→'` on a default Windows console. The driver sets
  `PYTHONUTF8=1` for its subprocesses; set it yourself for manual runs.
- **`nnfs.init()` patches `np.dot`** to enforce float32 and crashes on plain
  Python lists (`'list' object has no attribute 'astype'`). Notebook cells
  already wrap lists in `np.array()`; keep doing that in new demo code.
- **cwd matters.** `--checkpoint` defaults and dataset cache dirs resolve
  relative to the cwd. Run from the repo root or checkpoints/caches scatter.
- **`train.py` clobbers the canonical checkpoint by default.** Any manual
  training run without `--checkpoint` overwrites the root `*_weights.pkl`.
  Use a temp path for experiments (the driver's `--fresh` already does).
- **Prefer `--backend manual`** for MNIST/Fashion: it uses the local
  `*_cache/` dirs (or S3 mirrors on first run). The `sklearn` backend goes
  through `fetch_openml`, which is slower and flakier.
- **Moons evaluate output format** is `[test] loss=… accuracy=…` (key=value),
  unlike the other projects' `accuracy 0.98` style — relevant if you parse it.
- Evaluate runs write side artifacts at the root (`decision_grid.npz`,
  `predictions.npz`). They're gitignored; don't commit them.

## Troubleshooting

- `UnicodeEncodeError: 'charmap' codec can't encode character '→'` →
  `$env:PYTHONUTF8='1'` (PowerShell) before running any `evaluate.py`.
- `could not parse 'test accuracy' from evaluate output` in the driver →
  an `evaluate.py` output format changed; update the regex in `PROJECTS`
  at the top of `smoke.py`.
- `FileNotFoundError: …_weights.pkl` when running `evaluate.py` by hand →
  you're not in the repo root, or the checkpoint was never trained; run the
  driver (it trains missing checkpoints) or `train.py --epochs 1` first.
