# Architecture

The project is one Python package, `binary_classifier`, with two commands. Training generates the data, fits the network, and writes one file; evaluation reads that file, prints the metrics, and writes a second file for plotting. NumPy is the only runtime dependency. The package opens no network connection, reads no file but the weights file, and writes no file but the weights file and the grid.

## The network

![A two-panel schematic: on the left two interleaved half-moons of points with a curve passing between them, on the right the chain of layers from 2 inputs through two 16-unit ReLU layers to one sigmoid output, with the loss formula below it](diagrams/01-decision-boundary.svg)

*The model and the shape of its task. The figure is a schematic drawn before the measured run: its points and its curve are illustrative, and the noise level and the two accuracies printed in it are not the measured ones. The measured figures are in [EVALUATION.md](EVALUATION.md).*

The input is a point in the plane. Two dense layers of 16 units with ReLU follow, then a dense layer with one output, the logit $z$. The sigmoid turns the logit into the probability of class 1, $\hat{y} = \sigma(z) = 1/(1 + e^{-z})$, and the prediction is class 1 where $\hat{y} \ge 0.5$.

| Layer | Weights | Biases | Parameters |
|---|---|---|---|
| `dense1`, $2 \to 16$, then ReLU | $2 \cdot 16 = 32$ | 16 | 48 |
| `dense2`, $16 \to 16$, then ReLU | $16 \cdot 16 = 256$ | 16 | 272 |
| `dense3`, $16 \to 1$, then sigmoid | $16 \cdot 1 = 16$ | 1 | 17 |
| Total | 304 | 33 | 337 |

The loss that training minimises is the mean binary cross-entropy plus an L2 penalty on the weights of the two hidden layers, with $\lambda = 10^{-4}$:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\bigl[y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\bigr] + \lambda \sum \mathbf{W}_1^2 + \lambda \sum \mathbf{W}_2^2$$

The gradient of the first term with respect to the logits is $(\hat{y}_i - y_i)/N$, which is what `Activation_Sigmoid_Loss_BinaryCrossentropy.backward` computes; the derivation is in `nn-034`. Evaluation reports the first term alone.

## Components

| Module | What it holds |
|---|---|
| `nn.py` | The series' classes, unchanged in name and arithmetic: `Layer_Dense`, `Activation_ReLU`, `Activation_Sigmoid`, `Activation_Sigmoid_Loss_BinaryCrossentropy`, `Optimizer_Adam`, and the function `regularization_loss`. |
| `data.py` | `make_moons`, the generator; `train_test_split`, the seeded split; `dataset_checksum`, the SHA-256 that training prints. |
| `model.py` | `Network`, which wires the classes into the $2 \to 16 \to 16 \to 1$ network with its forward and backward passes; `save_weights` and `load_weights`; the checks on file names and file contents. |
| `train.py` | `train`, the full-batch loop, and the command `python -m binary_classifier.train`. |
| `evaluate.py` | `score`, `confusion_matrix`, `unit_activity`, `logit_linearity`, `decision_grid`, and the command `python -m binary_classifier.evaluate`. |

`scripts/seed_sweep.py` is a driver, not part of the package: it calls `train` and `score` for ten seeds and three initialisations and prints one row each. The tests under `tests/` cover every module of the package and no script.

## Data flow

1. **Check.** `train` checks its arguments and the weights path before any work is done.
2. **Generate.** `make_moons(n_samples, noise, seed)` draws the points: `n_samples // 2` on the upper half of the unit circle about the origin (class 0), the rest on the lower half of the unit circle about $(1, 0.5)$ (class 1), each coordinate with Gaussian noise added, then shuffled. The points are `float32` and the labels `int64`. `train_test_split` takes a seeded random 20 percent as the test set.
3. **Train.** `train` seeds the generator and builds `Network(init)`. Each epoch runs the forward pass on the whole training set, adds the penalty to the data loss, backpropagates, and takes one Adam step (learning rate 0.01, no decay) on each dense layer. At the end the weights, the split, and the configuration of the run go to `moons_weights.npz`.
4. **Evaluate.** `evaluate` reads the archive with pickling switched off and checks every array's shape, type, and values before a network is built from it. It scores the stored training and test sets, describes the hidden layers over the training set with `unit_activity` and `logit_linearity`, samples the probability on a $200 \times 200$ grid, and writes the grid beside the weights file. The exit status is 0, or 1 when `--min-accuracy` is given and not met.

## Reproducibility

A run is determined by the seed, the code, and the arithmetic of the NumPy build. The seed is an option with a default of 0, and NumPy is pinned exactly (record 0003). For the code, the order in which random numbers are drawn is a contract:

1. `make_moons` draws the noise and the shuffle from its own `numpy.random.default_rng(seed)`, and `train_test_split` draws its permutation from another.
2. `np.random.seed(seed)`, once, in `train`.
3. Three `randn` draws for the initial weights, in the order dense1, dense2, dense3. Biases start at zero.

Nothing else draws during training. The data therefore do not depend on the initialisation: the He run and the `--init small` run of the same seed train on identical points, which the `data sha256` line of the training output shows.

## The weights file

A NumPy `.npz` archive, uncompressed, of sixteen arrays. It is about 23 kB for the documented run.

| Array | Shape | Type | Content |
|---|---|---|---|
| `dense1_weights`, `dense1_biases` | $(2, 16)$, $(1, 16)$ | `float64` | first hidden layer |
| `dense2_weights`, `dense2_biases` | $(16, 16)$, $(1, 16)$ | `float64` | second hidden layer |
| `dense3_weights`, `dense3_biases` | $(16, 1)$, $(1, 1)$ | `float64` | output layer |
| `X_train`, `y_train` | $(N_\text{train}, 2)$, $(N_\text{train},)$ | `float32`, `int64` | the training set |
| `X_test`, `y_test` | $(N_\text{test}, 2)$, $(N_\text{test},)$ | `float32`, `int64` | the test set |
| `format_version` | scalar | `int64` | 1 |
| `init` | scalar | string | `he`, `xavier`, or `small` |
| `noise`, `n_samples`, `seed`, `epochs` | scalars | `float64`, `int64` | the configuration of the run |

The split is stored, not regenerated at evaluation, so the test points scored are exactly the ones training did not see (record 0005).

## The decision grid file

A `.npz` archive of seven arrays: `XX`, `YY`, and `probs`, each $(200, 200)$ and `float64`, where row $i$ belongs to the $i$-th value of $y$ and column $j$ to the $j$-th value of $x$; and `X_train`, `y_train`, `X_test`, `y_test` as in the weights file. It is about 978 kB.

## Configuration

The commands take options and read nothing else: no environment variables, no configuration file, no secrets, so the project has no `.env.example`.

| Option | Command | Default | Checked for |
|---|---|---|---|
| `--epochs` | train | 2,000 | a whole number, at least 1 |
| `--noise` | train | 0.1 | a finite number, not negative |
| `--n-samples` | train | 1,000 | a whole number that leaves at least one training and one test point |
| `--init` | train | `he` | one of `he`, `xavier`, `small` |
| `--seed` | train | 0 | a whole number from 0 to 4,294,967,295 |
| `--log-every` | train | 200 | a whole number, at least 1 |
| `--weights` | train, evaluate | `moons_weights.npz` | ends in `.npz`; for train, the directory exists; for evaluate, the content fits the network |
| `--grid` | evaluate | `decision_grid.npz` beside the weights file | ends in `.npz`; the directory exists |
| `--min-accuracy` | evaluate | none | a fraction from 0 to 1 |

The layer sizes, ReLU, the sigmoid output and its loss, Adam with learning rate 0.01 and its default $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$, the L2 penalty, full-batch updates, the 20 percent test fraction, and the grid's range and resolution are constants in the code, not options (record 0001).

## Failures and what the user sees

| Situation | Result |
|---|---|
| An option of `train` has an impossible value, or its weights path cannot be written | A usage message with the reason; exit status 2; nothing is trained. |
| `--min-accuracy` is not a fraction | A usage message; exit status 2. |
| The weights file is missing | `error:` naming the file and the command `uv run python -m binary_classifier.train`; exit status 1. |
| The weights file is a pickle, is damaged, or does not hold what this project writes | `error:` saying which check failed; exit status 1. |
| The grid path does not end in `.npz`, or its directory is missing | `error:` saying so, before anything is printed; exit status 1. |
| `--min-accuracy` is not met | The full report, then `error:` with the measured and required accuracy; exit status 1. |

## Decisions

- [0001](adr/0001-binary-classifier.md): one fixed network on two moons, and what counts as the baseline.
- [0002](adr/0002-weights-as-npz.md): weights as plain arrays in an `.npz` file, never a pickle.
- [0003](adr/0003-pin-numpy-for-bit-reproducibility.md): NumPy pinned exactly, and the order of random draws as part of the result.
- [0004](adr/0004-he-initialisation.md): He initialisation by default, the small one kept as an option.
- [0005](adr/0005-generated-data-stored-split.md): data generated, checksummed, and stored with the weights.
- [0006](adr/0006-series-classes-copied.md): the series' classes copied under their own names.
