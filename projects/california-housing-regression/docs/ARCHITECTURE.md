# Architecture

The project is one Python package, `california_housing_regression`, with three commands: fetch the dataset, train the network, score the saved weights. The network is a stack of the classes that Neural Networks from Scratch derives, with a linear output and a mean squared error where the classifiers of the series have a softmax and a cross-entropy. Everything around it exists to make one training run repeatable, to keep the test fold out of every number used in training, and to report the error in dollars.

![A two-panel schematic: on the left a scatter of predicted against actual house value with a dashed diagonal and a column of points at the right edge, on the right the chain of layers from 8 inputs through two 64-unit ReLU layers to one linear output, the loss formula, and a card of metrics.](diagrams/01-predicted-vs-actual.svg)

*The model and the shape of its task. The figure is a schematic drawn before the measured run: its points are illustrative, the column at the cap is drawn lower than the network predicts, and the six figures in its metrics card and the R squared in its note are not the measured ones. The measured figures are in [EVALUATION.md](EVALUATION.md).*

## The network

| Stage | Operation | Output shape | Parameters |
|---|---|---|---|
| Input | eight features, each standardised with the training fold's mean and standard deviation | $(N, 8)$ | 0 |
| Hidden layer 1 | `Layer_Dense(8, 64)`, `Activation_ReLU` | $(N, 64)$ | $8 \times 64 + 64 = 576$ |
| Hidden layer 2 | `Layer_Dense(64, 64)`, `Activation_ReLU` | $(N, 64)$ | $64 \times 64 + 64 = 4{,}160$ |
| Output layer | `Layer_Dense(64, 1)`, no activation | $(N, 1)$ | $64 \times 1 + 1 = 65$ |
| Total | | | $4{,}801$ |

The output is linear: the one number the last dense layer produces is the prediction, in standardised units of the target. The loss is the mean squared error, plus an L2 penalty of $\lambda = 10^{-4}$ on the weights of each hidden layer; the output layer's weights and all biases are not penalised.

$$L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 + \lambda \sum \mathbf{W}_1^2 + \lambda \sum \mathbf{W}_2^2$$

The gradient of the first term with respect to the predictions is $2(\hat{y}_i - y_i)/N$, which is what `Loss_MSE.backward` computes. There is no combined activation and loss class as in `nn-019`, because there is no activation to combine. The evaluation reports the first term alone.

Training is Adam (`Optimizer_Adam`, learning rate 0.01, decay 0.0001, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$) for 200 epochs. Each epoch reshuffles the 16,512 training rows and takes $\lceil 16{,}512 / 256 \rceil = 65$ steps, so a run makes $200 \times 65 = 13{,}000$ updates and the learning rate falls from 0.01 to $0.01 / (1 + 0.0001 \times 12{,}999) = 0.004348$.

## The data

The archive `cal_housing.tgz` holds one table, `CaliforniaHousing/cal_housing.data`: 20,640 rows, one per census block group, of nine comma-separated columns. The loader derives the eight features and the target from them.

| Feature | From the columns of the file |
|---|---|
| `MedInc` | `medianIncome` |
| `HouseAge` | `housingMedianAge` |
| `AveRooms` | `totalRooms / households` |
| `AveBedrms` | `totalBedrooms / households` |
| `Population` | `population` |
| `AveOccup` | `population / households` |
| `Latitude` | `latitude` |
| `Longitude` | `longitude` |

The target is `medianHouseValue` divided by 100,000. The census top-coded it: every block group whose median value was above the cap is recorded as 500,001 dollars, which is true of 965 of the 20,640 rows. The arithmetic is done in 64-bit floats and rounded to 32-bit floats once, and that order gives, bit for bit, the arrays the project trained on before version 1.0.0, when it read the data through scikit-learn (record 0004).

## Components

| Module | Responsibility | Reads and writes |
|---|---|---|
| `nn.py` | The series' classes: `Layer_Dense`, `Activation_ReLU`, `Optimizer_Adam` and `regularization_loss`, and `Loss_MSE`, which no post derives. | Nothing; draws from NumPy's global generator. |
| `model.py` | `Network`: builds the layers in a fixed order, runs the forward and backward passes, counts parameters, fingerprints the weights, saves and loads them with the scaler. | The weights file. |
| `data.py` | The record of the archive (name, size, SHA-256), the checksum check, the parser, the feature arithmetic, the seeded split, the scaler, `load_california_housing`. | The dataset directory, read only. |
| `download.py` | Fetches the archive over `https`, checks size and SHA-256, moves it into place. The only module that opens a network connection. | The dataset directory. |
| `train.py` | `fit`, the mini-batch loop for any network and data; `train`, the documented run; the command line. | Dataset in, weights out. |
| `evaluate.py` | `regression_metrics`, the two baselines, `score`, the printed report, the command line with `--min-r2` and `--predictions`. | Dataset and weights in; the predictions file out, when asked. |

`scripts/download_california_housing.py` is the entry point for the download and contains no logic of its own. `scripts/make_test_fixture.py` writes the tiny archive under `tests/fixtures/` that the tests read. Nothing importable lives outside `src/california_housing_regression/`.

## Data flow

1. **Fetch.** `scripts/download_california_housing.py` streams the archive to `cal_housing_cache/cal_housing.tgz.part`, stops if more bytes arrive than the record allows, compares size and SHA-256 with the record in `data.ARCHIVE`, and renames the file into place. An archive that is already present and correct is left alone.
2. **Load.** `load_california_housing` checks the SHA-256 of the archive, reads the one table out of it without extracting anything to disk, checks that it is nine columns of finite numbers with no block group of zero households, and derives the features, the target, and the published value in dollars.
3. **Split.** `split_indices` shuffles the row numbers with a generator started from the seed. The first $\mathrm{round}(20{,}640 \times 0.2) = 4{,}128$ are the test fold and the remaining 16,512 are the training fold.
4. **Scale.** `standardise_fit` computes the mean and standard deviation of each feature and of the target from the training rows. `standardise_apply` then scales both folds with those numbers. This is the rule of `nn-029`, and the order of the three calls in `load_california_housing` is the whole of its implementation (record 0005).
5. **Train.** `train` seeds the generator, loads the data, builds `Network()`, and calls `fit`. For each batch `fit` runs the forward pass, adds the penalty to the data loss, backpropagates, and applies one Adam step to the three dense layers. Every twentieth epoch prints its mean loss, its mean data loss and the current learning rate. At the end the weights and the scaler go to `cal_housing_weights.npz` and the fingerprint of the weights is printed.
6. **Evaluate.** `evaluate` builds a network, loads and checks the weights file, loads the data and splits it with `--seed`, and compares the scaler it has just computed with the one stored in the file. If they differ, the weights were trained on another split and the command stops. Otherwise it runs one forward pass over each fold, turns the predictions into dollars with $\hat{v} = (\hat{y}\,\sigma_y + \mu_y) \times 100{,}000$, and scores them against the published values. The exit status is 0, or 1 when `--min-r2` is given and not met.

## What the evaluation reports

| Figure | Definition |
|---|---|
| MSE (standardised units) | the loss without the penalty, on the standardised target |
| RMSE | $\sqrt{\frac{1}{N} \sum (\hat{v}_i - v_i)^2}$ in dollars |
| MAE | $\frac{1}{N} \sum \lvert \hat{v}_i - v_i \rvert$ in dollars |
| Mean error | $\frac{1}{N} \sum (\hat{v}_i - v_i)$ in dollars; negative means the model predicts too low |
| $R^2$ | $1 - \sum (\hat{v}_i - v_i)^2 / \sum (v_i - \bar{v})^2$, with $\bar{v}$ the mean of the fold being scored |

The test fold is also scored in two parts, the block groups recorded at the cap and those below it, and two baselines fitted on the training fold are scored on the test fold: the training mean, and a least-squares line on the eight standardised features.

## Reproducibility

A run is determined by the seed, the dataset bytes, the code and the arithmetic of the NumPy build. The first is an option with a default of 0, the second is fixed by checksum, and the last is fixed by pinning NumPy (record 0003). For the code, the order in which random numbers are drawn is a contract:

1. `np.random.seed(seed)`, once, at the start of `train`.
2. The split, from its own generator `np.random.default_rng(seed)`, which does not touch the global one.
3. Three `randn` draws for the initial weights, in the order dense1, dense2, dense3.
4. In every epoch, one `permutation` of the training row numbers.

Nothing else draws during training, and evaluation draws only the split. `Network.fingerprint` is a SHA-256 over the six parameter arrays; two runs with the same fingerprint have identical weights. The seed fixes the split as well as the weights, so two seeds differ in which rows are tested, not only in how the network starts.

## Configuration

The commands take options and read nothing else: no environment variables, no configuration file, no secrets, so the project has no `.env.example`.

| Option | Command | Default | Checked for |
|---|---|---|---|
| `--epochs` | train | 200 | a whole number, at least 1 |
| `--batch-size` | train | 256 | a whole number, at least 1 |
| `--seed` | train, evaluate | 0 | a whole number from 0 to 4,294,967,295; for evaluate, that it gives the split the weights were trained on |
| `--log-every` | train | 20 | a whole number, at least 1 |
| `--data-dir` | all three | `cal_housing_cache` | the archive present with the recorded SHA-256 |
| `--weights` | train, evaluate | `cal_housing_weights.npz` | ends in `.npz`; for train, the directory exists; for evaluate, the content fits the network |
| `--min-r2` | evaluate | none | a number no greater than 1 |
| `--predictions` | evaluate | none | ends in `.npz`, the directory exists, and it is not the weights file |
| `--url` | download | the address in `download.py` | starts with `https://` |

The architecture and the optimiser settings are constants in `model.py` and `train.py`, not options (record 0001).

## Failures and what the user sees

| Situation | Result |
|---|---|
| An option has an impossible value | A usage message naming the option and the reason; exit status 2; nothing is loaded. |
| The dataset directory or the archive is missing | `error:` naming what is missing and the command `uv run python scripts/download_california_housing.py`; exit status 1. |
| The archive is not the published one | `error:` with the SHA-256 found and the one expected; exit status 1. |
| The download fails, is cut short, or arrives altered | `error:` with the address and the reason; the partial file is removed; exit status 1. |
| The weights file is missing | `error:` naming the file and the command `uv run python -m california_housing_regression.train`; exit status 1. |
| The weights file is a pickle, is damaged, or belongs to another architecture | `error:` saying which check failed; exit status 1. |
| The weights were trained with another seed than the one given to evaluate | `error:` saying that the split differs and that `--seed` must be the training seed; exit status 1; nothing is scored. |
| `--min-r2` is not met | The full report, then `error:` with the measured and required $R^2$; exit status 1. |

## Decisions

- [0001](adr/0001-california-housing-regression.md): the series' classes as they are, a linear output with mean squared error, and a frozen baseline.
- [0002](adr/0002-weights-as-npz.md): weights and scaler as plain arrays in an `.npz` file, never a pickle.
- [0003](adr/0003-pin-numpy-for-bit-reproducibility.md): NumPy pinned exactly, and the order of random draws as part of the result.
- [0004](adr/0004-statlib-archive-with-checksum.md): the original archive, fetched by a script and checked by SHA-256, parsed without scikit-learn.
- [0005](adr/0005-split-before-scaling.md): split first, fit the scaler on the training fold, and refuse to score weights on a split they were not trained on.
