# Architecture

The project is one Python package, `fashion_mnist`, with three commands: fetch the dataset, train the network, score the saved weights. The network is the one the series' MNIST project (`nn-p01`) trains, a stack of the classes that Neural Networks from Scratch derives, and it is run here with that project's settings. Everything around it exists to make one training run repeatable, one test figure checkable, and the errors readable class by class.

Three modules are that project's with only names changed: `nn.py` is the same file byte for byte, and `model.py` and `train.py` differ in the package name, the dataset named in the messages and the default weights file. What belongs to this project is the dataset record and the class names in `data.py`, the mirror in `download.py`, and the named report and the most-confused pairs in `evaluate.py` (record 0001).

## The network

| Stage | Operation | Output shape | Parameters |
|---|---|---|---|
| Input | $28 \times 28$ pixels flattened, scaled to $[0, 1]$ | $(N, 784)$ | 0 |
| Hidden layer 1 | `Layer_Dense(784, 128)`, `Activation_ReLU`, `Layer_Dropout(0.1)` | $(N, 128)$ | $784 \times 128 + 128 = 100{,}480$ |
| Hidden layer 2 | `Layer_Dense(128, 128)`, `Activation_ReLU`, `Layer_Dropout(0.1)` | $(N, 128)$ | $128 \times 128 + 128 = 16{,}512$ |
| Output layer | `Layer_Dense(128, 10)`, softmax | $(N, 10)$ | $128 \times 10 + 10 = 1{,}290$ |
| Total | | | $118{,}282$ |

The loss is the mean categorical cross-entropy of the softmax output, plus an L2 penalty of $0.0005 \sum w^2$ over the weights of each hidden layer. The output layer's weights and all biases are not penalised. Softmax and cross-entropy are one class, `Activation_Softmax_Loss_CategoricalCrossentropy`, whose backward pass is the combined gradient $(\hat{\mathbf{y}} - \mathbf{y}) / N$ from `nn-019`. Dropout is inverted dropout: active in training, the identity in evaluation.

Training is Adam (`Optimizer_Adam`, learning rate 0.001, decay 0.0001, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$) for 20 epochs. Each epoch reshuffles the 60,000 training images and takes $\lceil 60{,}000 / 128 \rceil = 469$ steps, so a run makes $20 \times 469 = 9{,}380$ updates and the learning rate falls from 0.001 to $0.001 / (1 + 0.0001 \times 9{,}379) = 0.000516$.

## Components

| Module | Responsibility | Reads and writes |
|---|---|---|
| `nn.py` | The series' classes: `Layer_Dense`, `Activation_ReLU`, `Activation_Softmax_Loss_CategoricalCrossentropy`, `Layer_Dropout`, `Optimizer_Adam`, and `regularization_loss`. | Nothing; draws from NumPy's global generator. |
| `model.py` | `Network`: builds the layers in a fixed order, runs the forward and backward passes, counts parameters, fingerprints the weights, saves and loads them. | The weights file. |
| `data.py` | The record of the four dataset files (name, size, SHA-256), the ten class names in label order, the checksum check, the IDX parser, `load_fashion_mnist` and `load_test_set`. | The dataset directory, read only. |
| `download.py` | Fetches the four files over `https`, checks size and SHA-256, moves each into place. The only module that opens a network connection. | The dataset directory. |
| `train.py` | `fit`, the mini-batch loop for any network and data; `train`, the documented run; the command line. | Dataset in, weights out. |
| `evaluate.py` | The confusion matrix, the accuracies and the most-confused pairs computed from it, `score`, the printed report with class names, the command line with `--min-accuracy`. | Dataset and weights in. |

`scripts/download_fashion_mnist.py` is the entry point for the download and contains no logic of its own. `scripts/make_test_fixture.py` writes the tiny dataset under `tests/fixtures/` that the tests read. Nothing importable lives outside `src/fashion_mnist/`.

## Data flow

1. **Fetch.** `scripts/download_fashion_mnist.py` streams each file from the mirror to `fashion_mnist_cache/<name>.part`, stops if more bytes arrive than the record allows, compares size and SHA-256 with the record in `data.FASHION_MNIST_FILES`, and renames the file into place. A file that is already present and correct is left alone.
2. **Load.** `load_fashion_mnist` checks the SHA-256 of all four files, decompresses them, validates each IDX header against the length of its payload, and returns the images as `float32` in $[0, 1]$ with shape $(60{,}000, 784)$ and $(10{,}000, 784)$ and the labels as `int64`. The two image arrays occupy $70{,}000 \times 784 \times 4 = 219{,}520{,}000$ bytes.
3. **Train.** `train` seeds the generator, builds `Network()`, and calls `fit`. For each batch `fit` runs the forward pass with dropout on, adds the penalty to the data loss, backpropagates, and applies one Adam step to the three dense layers. Each epoch prints its mean loss, its training accuracy and the current learning rate. At the end the weights go to `fashion_mnist_weights.npz` and their fingerprint is printed.
4. **Evaluate.** `evaluate` builds a network, loads and checks the weights, loads the test set, runs one forward pass with dropout off, and prints the report. The exit status is 0, or 1 when `--min-accuracy` is given and not met.

## The report

Everything the evaluation prints after the loss is computed from one $10 \times 10$ confusion matrix, with the true class in rows and the predicted class in columns: the accuracy is the share of the counts on the diagonal, the accuracy of a class is its diagonal count over its row sum, and the most-confused pairs are the five largest counts off the diagonal, largest first, with equal counts ordered by true class and then by predicted class. The report is plain ASCII, so it can be redirected to a file under any console encoding.

![A ten-by-ten grid with the true clothing class in rows and the predicted class in columns, shaded along the diagonal, beside one horizontal bar per class for its accuracy and a list of the five most-confused pairs of classes; the counts drawn are illustrative, not measured.](diagrams/01-confusion-matrix.svg)

*The layout of the report: the matrix, the accuracy of each class, the most-confused pairs. The figure was drawn before the documented run and its numbers are illustrative; they are not the measured ones, including the overall accuracy it states. The measured matrix and every measured figure are in [EVALUATION.md](EVALUATION.md).*

## Reproducibility

A run is determined by the seed, the dataset bytes, the code and the arithmetic of the NumPy build. The first is an option with a default of 0, the second is fixed by checksum, and the last is fixed by pinning NumPy (record 0003). For the code, the order in which random numbers are drawn is a contract:

1. `np.random.seed(seed)`, once, at the start of `train`.
2. Three `randn` draws for the initial weights, in the order dense1, dense2, dense3.
3. In every epoch, one `permutation` of the sample indices; in every batch, one `binomial` mask for the first dropout layer and then one for the second.

Nothing else draws during training, and evaluation draws nothing. `Network.fingerprint` is a SHA-256 over the six parameter arrays; two runs with the same fingerprint have identical weights.

## Configuration

The commands take options and read nothing else: no environment variables, no configuration file, no secrets, so the project has no `.env.example`.

| Option | Command | Default | Checked for |
|---|---|---|---|
| `--epochs` | train | 20 | a whole number, at least 1 |
| `--batch-size` | train | 128 | a whole number, at least 1 |
| `--seed` | train | 0 | a whole number from 0 to 4,294,967,295 |
| `--data-dir` | all three | `fashion_mnist_cache` | the four files present with the recorded SHA-256 |
| `--weights` | train, evaluate | `fashion_mnist_weights.npz` | ends in `.npz`; for train, the directory exists; for evaluate, the content fits the network |
| `--min-accuracy` | evaluate | none | a fraction from 0 to 1 |
| `--base-url` | download | the mirror in `download.py` | starts with `https://` |

The architecture and the optimiser settings are constants in `model.py` and `train.py`, not options (record 0001).

## Failures and what the user sees

| Situation | Result |
|---|---|
| An option has an impossible value | A usage message naming the option and the reason; exit status 2; nothing is loaded. |
| The dataset directory or a file is missing | `error:` naming what is missing and the command `uv run python scripts/download_fashion_mnist.py`; exit status 1. |
| A dataset file is not the published one | `error:` with the SHA-256 found and the one expected; exit status 1. |
| The download fails, is cut short, or arrives altered | `error:` with the address and the reason; the partial file is removed; files already checked are kept; exit status 1. |
| The weights file is missing | `error:` naming the file and the command `uv run python -m fashion_mnist.train`; exit status 1. |
| The weights file is a pickle, is damaged, or belongs to another architecture | `error:` saying which check failed; exit status 1. |
| The machine runs out of memory | `error:` saying so and how much the images alone need; exit status 1. |
| `--min-accuracy` is not met | The full report, then `error:` with the measured and required accuracy; exit status 1. |

## Decisions

- [0001](adr/0001-fashion-mnist.md): the MNIST project's network and run unchanged, only the data changed, and a frozen baseline.
- [0002](adr/0002-weights-as-npz.md): weights as plain arrays in an `.npz` file, never a pickle.
- [0003](adr/0003-pin-numpy-for-bit-reproducibility.md): NumPy pinned exactly, and the order of random draws as part of the result.
- [0004](adr/0004-idx-files-with-checksums.md): the four original files, fetched by a script and checked by SHA-256.
