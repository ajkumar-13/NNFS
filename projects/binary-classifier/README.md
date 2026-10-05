# Binary classifier on two moons

A dense network with a single sigmoid output, written in NumPy alone, that learns to separate two interleaved half-circles of points and writes the grid of predictions needed to draw its decision boundary. It is the worked example for the sigmoid and binary cross-entropy post of Neural Networks from Scratch (`nn-034`): 337 parameters, two commands, and a result in about two seconds on a laptop CPU. It is not a general-purpose classifier and not a library. It trains one fixed network on one generated dataset, and its value is that every number it prints can be reproduced and traced to a line of code.

## Who this is for

Readers of the series who have reached `nn-034` and want to see the combined sigmoid and cross-entropy gradient train a model on a problem that can be drawn on one chart. It also serves anyone who wants a small, exactly reproducible case of an initialisation deciding whether a network learns: one option turns a model that classifies every test point correctly into one that draws a straight line through both moons and gets 26 of 200 test points wrong.

## Quickstart

You need [uv](https://docs.astral.sh/uv/), which fetches Python 3.13 and the locked packages. Nothing else is downloaded: the data are generated. From a clone of the series repository:

```bash
cd projects/binary-classifier
uv sync --frozen
uv run python -m binary_classifier.train
uv run python -m binary_classifier.evaluate
```

`train` runs 2,000 epochs from seed 0 and writes `moons_weights.npz`; `evaluate` loads that file, prints the results below, and writes `decision_grid.npz`. Both files land in the directory you run from and are never committed.

To check the project itself:

```bash
uv run pytest --cov=src --cov-fail-under=95
uv run ruff check . && uv run ruff format --check .
```

These are the commands in `project.yaml`, and they were last run from a clean environment on the `verified` date there.

## What it does

- Generates the two-moons dataset in NumPy from a seed: 1,000 points, 500 per class, Gaussian noise of standard deviation 0.1 on each coordinate, split 800 to 200.
- Trains a $2 \to 16 \to 16 \to 1$ network with ReLU hidden layers by full-batch Adam, using the combined sigmoid and binary cross-entropy class, whose gradient with respect to the logits is $(\hat{y} - y)/N$.
- Scores the trained network on both sets: loss, accuracy, confusion matrix, and the count of each class classified correctly.
- Samples the predicted probability on a $200 \times 200$ grid, so that the boundary, the curve where the probability is 0.5, can be drawn by any plotting program.
- Offers the initialisation as an option (`--init he`, `xavier`, or `small`), so that the failure `nn-033` describes can be produced and measured with the same command.
- Turns the headline into a check: `uv run python -m binary_classifier.evaluate --min-accuracy 1.0` exits with status 1 if a test point is misclassified.
- Stores weights as plain arrays in an `.npz` file, so loading weights never runs code from the file.

`--epochs`, `--noise`, `--n-samples`, `--init`, `--seed` and `--weights` change a run; `--help` on either command lists them. The layer sizes, the learning rate and the L2 strength are fixed in the code, because they are the documented baseline.

## Architecture

Five modules under `src/binary_classifier/`: `nn.py` holds the series' classes (`Layer_Dense`, `Activation_ReLU`, `Activation_Sigmoid`, `Activation_Sigmoid_Loss_BinaryCrossentropy`, `Optimizer_Adam`), `data.py` generates and splits the points, `model.py` assembles the network and reads and writes the weights, and `train.py` and `evaluate.py` are the two commands. The network has $2 \cdot 16 + 16 = 48$, $16 \cdot 16 + 16 = 272$, and $16 \cdot 1 + 1 = 17$ parameters in its three dense layers, 337 in all. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the components, the data flow, the file formats, and the diagram, and the records under [docs/adr/](docs/adr/) give the reasons for the decisions a maintainer would otherwise have to rediscover.

## Results

Measured on 2026-10-05 with the quickstart commands, seed 0, on an Intel Core i7-9750H with 7.8 GB of memory, CPU only. [docs/EVALUATION.md](docs/EVALUATION.md) has the full printed output, the commands for every row, and the limits of the claim.

| Run | Training set (800 points) | Test set (200 points) |
|---|---|---|
| Documented run: He initialisation, noise 0.1 | 800 correct, loss 0.0006 | 200 correct (100.0 percent), loss 0.0045 |
| The same with `--init small` (weights drawn as 0.01 times a standard normal) | 707 correct (88.4 percent), loss 0.2467 | 174 correct (87.0 percent), loss 0.2645 |
| The same with `--noise 0.2` | 788 correct (98.5 percent), loss 0.0377 | 192 correct (96.0 percent), loss 0.1846 |

Training takes between one and three seconds on that laptop. With the small initialisation the network ends as a linear classifier: after 2,000 epochs, and still after 20,000, a plane fits its logits over the training set with $R^2 = 1.0000$ to four decimals, against 0.7413 for the documented run. Its boundary is a straight line, and no straight line separates two moons.

The result is not an accident of seed 0. `uv run python scripts/seed_sweep.py` repeats the run for seeds 0 to 9, each of which draws a different dataset, split, and set of initial weights. With He initialisation the test score lies between 197 and 200 of 200 and is 200 on eight of the ten seeds; with the small initialisation it lies between 173 and 181.

## Drawing the boundary

`decision_grid.npz` holds the arrays `XX`, `YY`, and `probs`, each $200 \times 200$, covering $x$ from $-1.5$ to $2.5$ and $y$ from $-1$ to $1.5$, together with the training and test points (`X_train`, `y_train`, `X_test`, `y_test`). The boundary is the contour of `probs` at 0.5. matplotlib is not a dependency of this project; to draw the grid with it, save the lines below as `plot.py` beside the grid and run `uv run --with matplotlib python plot.py`.

```python
import matplotlib.pyplot as plt
import numpy as np

grid = np.load("decision_grid.npz")
plt.contourf(grid["XX"], grid["YY"], grid["probs"], levels=20, cmap="RdBu_r", alpha=0.5)
plt.contour(grid["XX"], grid["YY"], grid["probs"], levels=[0.5], colors="black", linewidths=2)
plt.scatter(*grid["X_test"].T, c=grid["y_test"], cmap="RdBu_r", edgecolors="black")
plt.savefig("decision_boundary.png", dpi=150)
```

## Limits

- **One architecture, one optimiser setting, one dataset.** The options change the seed, the noise, the number of points, the number of epochs, and the initialisation; the layer sizes, the learning rate, and the L2 penalty are fixed in the code.
- **Synthetic data.** The data are generated, and the test points come from the same generator as the training points. The test score measures interpolation between training points, not behaviour on data from anywhere else.
- **An easy setting.** A score of 200 of 200 belongs to noise 0.1: the two noiseless half-circles are 0.5 apart at their closest, five times the standard deviation of the noise. At noise 0.2 the same network gets 192.
- **No validation split.** The configuration was settled with the test score in view, so the seed sweep, which draws nine further datasets, is the evidence that it was not fitted to one split.
- **Bit-for-bit reproduction needs the locked environment.** NumPy is pinned to 2.3.5, as in every project of the series. On another processor the last digits of the losses can differ; `data sha256` in the training output tells you whether your data are the documented ones.
- **Weights, the split, and nothing else.** The `.npz` file holds the six parameter arrays, the training and test points, and the configuration, but no optimiser state, so a run cannot be resumed. Pickle files written by the scripts before version 1.0.0 are not loaded.

## Built from

- `nn-016`, coding backpropagation: `Layer_Dense` and `Activation_ReLU` with their backward passes.
- `nn-019`, softmax derivatives and the combined backward pass: the pattern of one class for the last activation and the loss, which `nn-034` carries over to the sigmoid.
- `nn-027`, Adam: `Optimizer_Adam`.
- `nn-030`, L1 and L2 regularisation: the penalty terms in `Layer_Dense` and `regularization_loss`.
- `nn-033`, weight initialisation: the `init` argument of `Layer_Dense`, with He as the default.
- `nn-034`, sigmoid and binary cross-entropy: the stable sigmoid and `Activation_Sigmoid_Loss_BinaryCrossentropy`.

## Corrections

The series takes no pull requests or issues. If you find a mistake, send the correction through the series website.

## Licence

Code under MIT, prose and figures under CC BY 4.0, as stated in the series [LICENSE](../../LICENSE). The data are generated by the code and carry no licence of their own.
