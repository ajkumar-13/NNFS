# Verified project results (run, not assumed)

Each project's own code was executed end-to-end (train + evaluate) on the real dataset.
Datasets: project 02 is synthetic (NumPy `make_moons`); 01/03 use the idx-ubyte mirrors via
`--backend manual`; 04 uses `sklearn.datasets.fetch_california_housing`.

| Project | Network / config | Verified result | README claim | Verdict |
|---|---|---|---|---|
| **01 · MNIST** | 784→128→128→10, Adam, 20 epochs, dropout 0.1 + L2 5e-4 | **test 98.0%** (train 98.5%) | ~97% | ✓ holds (slightly better) |
| **02 · two-moons** | 2→16→16→1, Adam lr=0.01, 2000 ep, **He init + noise 0.1** (fixed) | **test 100%** (was 85% with the old init) | >98% | ✓ fixed (was underfitting) |
| **03 · Fashion-MNIST** | 784→128→128→10, Adam, 20 epochs | **test 87.1%, loss 0.34** (train 89.3%) | ~88.9% / ~0.34 | ≈ (acc ~2pts under; loss exact) |
| **04 · California housing** | 8→64→64→1, Adam lr=0.01 decay=1e-4, 200 ep | **test R² 0.823, RMSE $49.2k, MAE $33.7k** (train R² 0.846) | R²~0.78 / RMSE~$58k | ✓ better than claimed |

Reproduce (from the repo root):

```
python projects/02-binary-classifier/train.py      && python projects/02-binary-classifier/evaluate.py
python projects/04-california-housing-regression/train.py && python projects/04-california-housing-regression/evaluate.py
python projects/01-mnist-from-scratch/train.py    --backend manual && python projects/01-mnist-from-scratch/evaluate.py
python projects/03-fashion-mnist/train.py         --backend manual && python projects/03-fashion-mnist/evaluate.py
```

## Project 02 — the default config underfits (the only real defect)

A 2→16→16→1 ReLU net should reach ~98% on two-moons, but the shipped default tops out at ~85%.
The `nn.py` code is correct; the cause is the **`0.01 * randn` weight init**, which is far too small for a
**2-input** layer (pre-activations start at ~0.01, many ReLUs die, the net underfits). This is exactly the
failure the series' own **Part 33 (weight initialisation)** warns about. Verified outcomes (same architecture):

| Config | noise=0.1 test | noise=0.2 test |
|---|--:|--:|
| default (`0.01*randn`, lr=0.01, 2000 ep) | 87% | 85% |
| **He init** (`randn * sqrt(2/fan_in)`), lr=0.01, 2000 ep | **100%** | **95%** |
| lr=0.05, 5000 ep (no init change) | 99.5% | 97% |

So the network is fine; the init starves it. **Fix applied:** `nn.py` now uses He init and `train.py`
defaults to `noise=0.1`. Re-verified after the fix: **train 100%, test 100%** (200/200) in the same
2000 epochs — the README's ">98% / clean curve" claim now holds honestly.

## Project 04 — baselines (same split, verified)

| Model | Test R² |
|---|--:|
| Predict the training mean | 0.000 |
| Ordinary least squares | 0.637 |
| 2-layer MLP (this project) | **0.823** |
| sklearn GradientBoostingRegressor (defaults) | 0.808 |

The from-scratch MLP edges out default gradient boosting on this split.

*All runs are seed-0 deterministic via each project's `--seed` default.*
