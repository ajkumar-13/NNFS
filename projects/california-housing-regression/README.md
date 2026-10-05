# California housing regression

Every post in the series classifies, which leaves one question open: does the forward and backward machinery survive a change of loss? This project answers it. It assembles the classes the series builds, with their arithmetic unchanged, into a two-hidden-layer dense network ($8 \to 64 \to 64 \to 1$) with a linear output and a mean squared error, and predicts the median house value of a California census block group from eight features. It splits the 20,640 block groups into 16,512 for training and 4,128 for testing, computes every scaling statistic from the training fold alone, and reports its errors in dollars. From the fixed seed the documented run reaches $R^2 = 0.8231$ on the test fold, with an RMSE of 49,222 dollars, 4,801 parameters and nothing but NumPy.

It is a reference model with a reproducible result, not a library and not a valuation tool: the data is the 1990 census, the target is capped at 500,001 dollars, and the network is the smallest change to the series' classifier that makes it a regressor, on purpose.

## Who this is for

- Readers of Neural Networks from Scratch who have finished the post on mini-batching (`nn-032`) and want to see a regression built from the classes they wrote, with every number traceable to a command.
- Readers of the validation post (`nn-029`) who want its rule against data leakage as running code and as tests: the scaler is fitted on the training fold, applied to the test fold, and the evaluation refuses weights that were trained on another split.

## Quickstart

You need [uv](https://docs.astral.sh/uv/), which fetches Python 3.13 and the locked packages, and a network connection for the first two commands. From a clone of the series repository:

```bash
cd projects/california-housing-regression
uv sync --frozen
uv run python scripts/download_california_housing.py
uv run python -m california_housing_regression.train
uv run python -m california_housing_regression.evaluate
```

`download_california_housing.py` fetches one archive, `cal_housing.tgz` (441,963 bytes), into `cal_housing_cache/` and keeps it only if its SHA-256 is the expected one. `train` runs 200 epochs from seed 0 and writes `cal_housing_weights.npz`; `evaluate` loads that file and prints, for each fold, the RMSE, the MAE, the mean error and $R^2$ in dollars, then the test fold at and below the census cap, then two baselines. The dataset and the weights stay on your machine and are never committed.

To check the project itself:

```bash
uv run pytest --cov=src --cov-fail-under=95
uv run ruff check . && uv run ruff format --check .
```

These are the commands in `project.yaml`, and they were last run from a clean environment on the `verified` date there.

## What it does

- Trains the network from a fixed seed, so that two runs on the same machine give the same weights bit for bit. Both commands print a SHA-256 fingerprint of the weights, which is how you can tell.
- Keeps the test fold out of training. The split comes first, the mean and standard deviation of every feature and of the target come from the training rows only, and the same numbers scale the test rows.
- Scores the saved weights on both folds in dollars, against the values the census published, and separately on the block groups recorded at the cap, where the published value is a floor and not a measurement.
- Puts the result beside two baselines fitted on the same training fold: the training mean, and a least-squares line on the same eight features.
- Turns the headline into a check: `uv run python -m california_housing_regression.evaluate --min-r2 0.82` exits with status 1 if the test $R^2$ is lower.
- Refuses to start on a dataset that is not byte for byte the published one, and says which command fetches it.
- Stores weights as plain arrays in an `.npz` file, so loading weights never runs code from the file.

`--epochs`, `--batch-size`, `--seed`, `--log-every`, `--data-dir` and `--weights` change a training run; `--help` on any command lists them. The seed fixes the split as well as the weights, so `evaluate` takes the same `--seed`, and with `--predictions FILE.npz` it also writes the predicted and published values of both folds for plotting. The optimiser settings, the layer sizes, the L2 strength and the test fraction are fixed in the code, because they are the documented baseline.

## Architecture

Each block group is eight numbers: median income, median house age, rooms, bedrooms and people per household, population, latitude and longitude. They are standardised and passed through two hidden layers of 64 units, each a dense layer and a ReLU, and a dense layer of one unit with no activation after it. That one number is the prediction in standardised units, and $(\hat{y}\,\sigma_y + \mu_y) \times 100{,}000$ is the prediction in dollars. The loss is the mean squared error, and the two hidden layers carry an L2 penalty of 0.0001 on their weights. Training is Adam with learning rate 0.01 and decay 0.0001 for 200 epochs, each one 65 mini-batches of at most 256 block groups drawn in a fresh random order.

What changes from the series' MNIST classifier (`nn-p01`) is short:

| Piece | MNIST classifier | This project |
|---|---|---|
| Layer sizes | $784 \to 128 \to 128 \to 10$ | $8 \to 64 \to 64 \to 1$ |
| After the last dense layer | softmax | nothing; the output is the prediction |
| Loss | categorical cross-entropy, combined with the softmax in one class | `Loss_MSE`, with gradient $2(\hat{y} - y)/N$ |
| Target | a class index | a real number, standardised |
| Inputs | pixels scaled to $[0, 1]$ by a constant | features standardised with the training fold's statistics |
| Dropout | 0.1 after each hidden layer | none |
| What is reported | accuracy, confusion matrix | RMSE, MAE and mean error in dollars, $R^2$ |

The dense layer, the ReLU, the optimiser and the penalty are the same classes, and the backward pass is the same chain of calls with one loss swapped for another.

The code is one package, `src/california_housing_regression/`, of six small modules: `nn.py` holds the series' classes and the loss, `model.py` assembles them and reads and writes the weights, `data.py` checks, parses, splits and scales the dataset, `download.py` fetches it, and `train.py` and `evaluate.py` are the two commands. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the components and the data flow and carries the diagram, and the records under [docs/adr/](docs/adr/) give the reasons for the decisions a maintainer would otherwise have to rediscover.

## Results

Measured on 2026-10-05 with the quickstart commands, seed 0, on an Intel Core i7-9750H with 7.8 GB of memory, CPU only. [docs/EVALUATION.md](docs/EVALUATION.md) has the full output and the limits of the claim.

| Figure | Training fold | Test fold |
|---|---|---|
| Block groups | 16,512 | 4,128 |
| $R^2$ | 0.8458 | 0.8231 |
| RMSE | 45,152 dollars | 49,222 dollars |
| MAE | 31,240 dollars | 33,708 dollars |
| Mean error, prediction minus published value | 4,408 dollars | 6,794 dollars |

| Model, fitted on the training fold | Test RMSE | Test $R^2$ |
|---|---|---|
| The training mean | 117,040 dollars | -0.0001 |
| A least-squares line on the eight features | 70,477 dollars | 0.6374 |
| This network, 4,801 parameters | 49,222 dollars | 0.8231 |

Training takes 17.9 s for the 200 epochs; other runs that day took up to 20.9 s on the same busy laptop.

The census recorded every median value above its cap as 500,001 dollars, which is the value of 207 of the 4,128 test block groups. For those the network predicts 468,755 dollars on average, 31,246 dollars under the recorded figure, with an RMSE of 71,679 dollars; for the 3,921 below the cap the RMSE is 47,744 dollars. The network cannot learn a value the data never shows it, and the recorded figure is itself only a lower bound for those homes.

## Limits

- **One seed, one machine, and the best of three.** The headline is the seed 0 run. Seeds 1 and 2 gave a test $R^2$ of 0.8135 and 0.8074. The seed also decides which block groups are tested, so quote a figure with its seed.
- **The capped rows have no true value.** Every error here is against the recorded value. For the most expensive block groups that understates the real error by an amount this data cannot give.
- **1990 dollars, one state.** The model is not a way to value a house, then or now.
- **A random split is a kind one.** Latitude and longitude are features and neighbouring block groups resemble each other, so most test rows have close neighbours in the training fold. The result says nothing about a region held out whole.
- **Bit-for-bit reproduction needs the locked environment.** NumPy is pinned to 2.3.5. With another version or on another processor the last digits can differ; the fingerprint tells you whether your run is the documented one.
- **No validation split.** The settings are fixed and nothing is tuned. Anyone who changes a setting and compares test figures is tuning on the test fold, the mistake `nn-029` describes; hold out part of the training fold first, and fit the scaler on what remains.
- **A baseline, not the series' best practice.** The layers start from `0.01 * randn` weights, which `nn-033` shows is a poor choice for deeper networks. It is kept because this run is the documented baseline.
- **Weights and scaler only.** The `.npz` file holds the six parameter arrays and the four scaling statistics and no optimiser state, so a run cannot be resumed. Pickle checkpoints written by the scripts before version 1.0.0 are not loaded.

## Built from

- `nn-016`: `Layer_Dense` and `Activation_ReLU`, each with its backward pass.
- `nn-027`: `Optimizer_Adam`, with bias correction and learning-rate decay.
- `nn-029`: the rule that the scaler is fitted on the training fold and only applied to the test fold.
- `nn-030`: the L2 penalty in the dense layer's gradient and in the loss.
- `nn-032`: the epoch and mini-batch loops of the trainer.

The mean squared error has no post of its own. `Loss_MSE` is written here in the form of the series' loss classes, and `nn-008` points to this project as the place where the series meets it.

## Corrections

The series takes no pull requests or issues. If you find a mistake, send the correction through the series website.

## Licence

Code under MIT, prose and figures under CC BY 4.0, as stated in the series [LICENSE](../../LICENSE). The data is the StatLib California housing table of Pace and Barry (1997), built from the 1990 United States census.
