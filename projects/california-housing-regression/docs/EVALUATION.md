# Evaluation

Measured: 2026-10-05. Every figure on this page was printed by the commands shown, run from this directory in a freshly created environment, or is a closed form written out beside it.

## The claim

Trained from seed 0 with the documented command, the 4,801-parameter network reaches $R^2 = 0.8231$ on the 4,128 block groups of the test fold, with an RMSE of 49,222 dollars and an MAE of 33,708 dollars, measured in dollars against the published values.

The claim is false if the commands below, run in the locked environment, print a lower $R^2$. The last command makes that a check: it exits with status 1 when the test $R^2$ is under 0.82.

## How to reproduce it

```bash
uv sync --frozen
uv run python scripts/download_california_housing.py
uv run python -m california_housing_regression.train
uv run python -m california_housing_regression.evaluate --min-r2 0.82
```

| Item | Value |
|---|---|
| Seed | 0 (the default of `--seed`); it fixes the split, the initial weights and the shuffles |
| Data | `cal_housing.tgz` with the SHA-256 recorded in `src/california_housing_regression/data.py`; 20,640 block groups, split into 16,512 for training and 4,128 for testing |
| Scaling | features and target standardised with the mean and standard deviation of the training fold only; the training fold's target has mean 206,663 dollars and standard deviation 114,978 dollars |
| Training | 200 epochs, mini-batches of 256, 65 steps an epoch, Adam with learning rate 0.01 and decay 0.0001, L2 0.0001 on the hidden weights |
| Software | Python 3.13.5, NumPy 2.3.5, installed from `uv.lock` by uv 0.11.7 |
| Machine | Intel Core i7-9750H (2.60 GHz), 7.8 GB of memory, Windows 10, CPU only |

## Results

| Figure | Training fold | Test fold |
|---|---|---|
| Block groups | 16,512 | 4,128 |
| $R^2$ | 0.8458 | 0.8231 |
| RMSE | 45,152 dollars | 49,222 dollars |
| MAE | 31,240 dollars | 33,708 dollars |
| Mean error, prediction minus published value | 4,408 dollars | 6,794 dollars |
| MSE in standardised units, without the penalty | 0.1542 | 0.1833 |

| Figure | Value |
|---|---|
| Parameters | 4,801 |
| Training loss in epoch 200, data loss plus L2 penalty | 0.1715 |
| Training time, 200 epochs | 17.9 s |
| Fingerprint of the weights | `ad8a93d75e5cffe293d02dc1ae55c453f496cf935819061fe852111a7f9a8069` |

The parameter count is also a closed form: $(8 \times 64 + 64) + (64 \times 64 + 64) + (64 \times 1 + 1) = 576 + 4{,}160 + 65 = 4{,}801$.

The test RMSE is $49{,}222 / 206{,}663 = 23.8$ percent of the mean house value of the training fold. The gap between the folds is $0.8458 - 0.8231 = 0.0227$ in $R^2$ and $49{,}222 - 45{,}152 = 4{,}070$ dollars in RMSE: the network fits the rows it trained on somewhat better than the rows it did not, as expected, and the gap is small beside the error itself.

### Against two baselines

Both baselines are fitted on the training fold and scored on the test fold, by the same evaluation command.

| Model | Test RMSE | Test MAE | Test $R^2$ |
|---|---|---|---|
| The training mean, 206,663 dollars for every block group | 117,040 dollars | 92,511 dollars | -0.0001 |
| A least-squares line on the eight standardised features | 70,477 dollars | 52,597 dollars | 0.6374 |
| This network | 49,222 dollars | 33,708 dollars | 0.8231 |

The mean's $R^2$ is not exactly zero because $R^2$ is measured against the mean of the test fold, and the training mean is a slightly different number. The line is what a network with no hidden layer could reach at best under the same loss. The two hidden layers take the RMSE from 70,477 to 49,222 dollars, a reduction of $(70{,}477 - 49{,}222) / 70{,}477 = 30.2$ percent.

### At the census cap

The census recorded every median value above the cap as 500,001 dollars. That is the value of 965 of the 20,640 block groups: 758 in the training fold and 207 in the test fold.

| Part of the test fold | Block groups | RMSE | MAE | Mean prediction | Mean error |
|---|---|---|---|---|---|
| At the cap | 207 | 71,679 dollars | 44,172 dollars | 468,755 dollars | -31,246 dollars |
| Below the cap | 3,921 | 47,744 dollars | 33,155 dollars | 200,994 dollars | 8,802 dollars |

For the 207 capped block groups the network predicts 468,755 dollars on average, 31,246 dollars under the recorded value. The recorded value is itself a floor, not the true median, so the error against the true value of those homes is larger than this table can show and cannot be measured from this data. Below the cap the network predicts 8,802 dollars too high on average. The two parts combine to the test fold's mean error: $(207 \times (-31{,}246) + 3{,}921 \times 8{,}802) / 4{,}128 = 6{,}794$ dollars.

### The training log

```text
loading California housing from cal_housing_cache
  train: (16512, 8)    test: (4128, 8)
  target: mean $206,663, standard deviation $114,978 (training fold)
  parameters 4,801
training for 200 epochs, batch_size=256 (65 steps/epoch), seed=0

epoch   1 | loss 0.8192 | train_mse 0.8171 | lr 0.009936
epoch  20 | loss 0.2274 | train_mse 0.2154 | lr 0.008850
epoch  40 | loss 0.2128 | train_mse 0.1999 | lr 0.007937
epoch  60 | loss 0.1990 | train_mse 0.1864 | lr 0.007195
epoch  80 | loss 0.1961 | train_mse 0.1833 | lr 0.006579
epoch 100 | loss 0.1955 | train_mse 0.1823 | lr 0.006061
epoch 120 | loss 0.1878 | train_mse 0.1745 | lr 0.005618
epoch 140 | loss 0.1834 | train_mse 0.1699 | lr 0.005236
epoch 160 | loss 0.1812 | train_mse 0.1675 | lr 0.004902
epoch 180 | loss 0.1775 | train_mse 0.1637 | lr 0.004609
epoch 200 | loss 0.1715 | train_mse 0.1575 | lr 0.004348

trained in 17.9 s
wrote weights to cal_housing_weights.npz
weights sha256 ad8a93d75e5cffe293d02dc1ae55c453f496cf935819061fe852111a7f9a8069
```

The `loss` column is the mean over the epoch's batches of the mean squared error plus the L2 penalty, and `train_mse` is the same mean of the mean squared error alone, both in standardised units. In epoch 200 the penalty is $0.1715 - 0.1575 = 0.0140$, or 8.2 percent of the loss. `train_mse` is averaged while the weights are still moving, which is why its last value, 0.1575, is not the 0.1542 that the evaluation measures on the training fold with the final weights.

### The evaluation report

```text
restoring weights from cal_housing_weights.npz
loading California housing from cal_housing_cache
  parameters 4,801
  weights    sha256 ad8a93d75e5cffe293d02dc1ae55c453f496cf935819061fe852111a7f9a8069
  split      seed 0: 16,512 training and 4,128 test block groups

[train]  n=16,512
  MSE (standardised units)  0.1542
  RMSE (dollars)            $45,152
  MAE  (dollars)            $31,240
  mean error (dollars)      $4,408
  R^2                       0.8458

[test]  n=4,128
  MSE (standardised units)  0.1833
  RMSE (dollars)            $49,222
  MAE  (dollars)            $33,708
  mean error (dollars)      $6,794
  R^2                       0.8231

Test fold by top-coding (the census records every value above the cap as $500,001):
  at the cap     n=207  RMSE $71,679  MAE $44,172  mean prediction $468,755  mean error -$31,246
  below the cap  n=3,921  RMSE $47,744  MAE $33,155  mean prediction $200,994  mean error $8,802
  the training fold holds 758 block groups at the cap

Baselines on the test fold, fitted on the training fold:
  training mean         RMSE $117,040  MAE $92,511  R^2 -0.0001
  linear least squares  RMSE $70,477  MAE $52,597  R^2 0.6374
  this network          RMSE $49,222  MAE $33,708  R^2 0.8231
```

## Repeatability

The fingerprint above was printed four times on 2026-10-05, on the one machine:

1. by the project's scripts as they stood before version 1.0.0, which read the data through scikit-learn 1.9.0;
2. by this package, reading the archive itself;
3. by this package again in a second, freshly created environment, which is the run reported on this page;
4. by this package under NumPy 2.5.3 in place of the locked 2.3.5.

So restructuring the code into a package and replacing the scikit-learn loader changed nothing in the model: the raw arrays are identical, the standardised folds are identical, and the weights are identical bit for bit. The first and second runs also printed the same test figures to every digit shown (record [0004](adr/0004-statlib-archive-with-checksum.md)). The fourth run shows that on this machine the result survived one change of NumPy version; it is one observation, and the pin stays (record [0003](adr/0003-pin-numpy-for-bit-reproducibility.md)).

## Other seeds

Two more runs, identical except for `--seed`, show how much of the headline belongs to the seed. The seed fixes the split as well as the initial weights, so each row below is a different network scored on a different test fold.

```bash
uv run python -m california_housing_regression.train --seed 1 --weights seed1.npz
uv run python -m california_housing_regression.evaluate --seed 1 --weights seed1.npz
```

| Seed | Test $R^2$ | Test RMSE | Test MAE | Test mean error | Training $R^2$ | Linear baseline, test $R^2$ |
|---|---|---|---|---|---|---|
| 0 | 0.8231 | 49,222 dollars | 33,708 dollars | 6,794 dollars | 0.8458 | 0.6374 |
| 1 | 0.8135 | 49,593 dollars | 33,692 dollars | 2,125 dollars | 0.8417 | 0.6232 |
| 2 | 0.8074 | 50,710 dollars | 34,647 dollars | 4,182 dollars | 0.8448 | 0.6155 |

| Seed | Test block groups at the cap | Mean prediction at the cap | Mean error at the cap |
|---|---|---|---|
| 0 | 207 | 468,755 dollars | -31,246 dollars |
| 1 | 192 | 457,814 dollars | -42,187 dollars |
| 2 | 196 | 451,827 dollars | -48,174 dollars |

The documented seed is the best of the three. The test $R^2$ runs from 0.8074 to 0.8231, a spread of 0.0157, and the RMSE from 49,222 to 50,710 dollars. In all three the network is far above the linear baseline, under-predicts at the cap by tens of thousands of dollars, and predicts too high on average over the whole fold.

## Limits of the claim

- **It is one run, and the best of three.** The headline is seed 0. Seeds 1 and 2 gave 0.8135 and 0.8074, both under the 0.82 that the check asks of seed 0. Read the result as "about 0.81 to 0.82", and quote a figure with its seed. Three runs are not a distribution, and no confidence interval over seeds is claimed.
- **The seed moves the split too.** A different seed tests on different block groups, so the spread above mixes the luck of the initial weights with the luck of the fold. The two were not measured separately.
- **The top-coded rows have no true value.** 207 of the 4,128 test rows record the cap, not the median value. Every error on this page is against the recorded value, so the figures say how well the network reproduces the census table, and understate the error on the most expensive block groups by an unknown amount.
- **The errors are not uniform.** An RMSE of 49,222 dollars is 23.8 percent of the mean value. For a block group worth 100,000 dollars the same absolute error is a much larger fraction; error by price band was not measured.
- **The network predicts too high on average,** by 6,794 dollars on the test fold and 4,408 dollars on the training fold in the documented run, and by smaller positive amounts with the other two seeds. A least-squares fit has zero mean error on its training data by construction; this network, stopped after 200 epochs of mini-batch steps, does not.
- **No validation set was held out.** The settings were fixed before this evaluation and were not tuned as part of it; the run trains on the whole training fold, and the test fold is used only to report. The figure is not evidence that these settings are the best ones. Anyone who changes a setting and compares test figures is tuning on the test fold.
- **A random split ignores geography.** Neighbouring block groups resemble each other, and latitude and longitude are features, so a test row usually has near neighbours in the training fold. The result says nothing about a region the network has not seen.
- **The data is the 1990 census.** The dollar figures are 1990 dollars, and the model is not a valuation tool for any later year.
- **Bit-identity is local.** It was shown on one processor. The split uses NumPy's `default_rng`, whose stream is not promised to stay the same across NumPy versions, and the matrix products go through the linear algebra library in the wheel, so the version is pinned.
- **The training time is a wall-clock reading** on a laptop that was running other work at the same time. The five 200-epoch runs made that day with this package printed 20.9, 17.7, 17.7, 19.0 and 17.9 s, so read it as "about 20 seconds".

## Use

- 2026-10-05: the series itself. `nn-008` sends its readers here as the place where the series meets mean squared error, which no post derives. No use outside the series is recorded yet.
