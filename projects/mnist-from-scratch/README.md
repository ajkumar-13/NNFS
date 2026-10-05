# MNIST from scratch

The series derives every layer, loss and optimiser on a toy spiral of 300 points, which leaves one question open: do the pieces compose into a model that works on real data? This project answers it. It assembles the classes the series builds, with their arithmetic unchanged, into a two-hidden-layer dense network ($784 \to 128 \to 128 \to 10$), trains it on the 60,000 MNIST training digits with Adam, L2 and dropout in mini-batches, and scores it on the 10,000 test digits. From the fixed seed the documented run classifies 9,800 of the 10,000 test images correctly, 98.00 percent, with 118,282 parameters and nothing but NumPy.

It is a reference model with a reproducible result, not a library and not a strong digit classifier: it has no convolution, no batch normalisation, no augmentation and no GPU code, on purpose.

## Who this is for

- Readers of Neural Networks from Scratch who have finished the post on mini-batching (`nn-032`) and want a first real result from the classes they wrote, with every number traceable to a command.
- Authors who need a dense baseline to measure against. The sibling series Convolutional Neural Networks from Scratch uses this run as that baseline in `cnn-011` and `cnn-012`.

## Quickstart

You need [uv](https://docs.astral.sh/uv/), which fetches Python 3.13 and the locked packages, and a network connection for the first two commands. From a clone of the series repository:

```bash
cd projects/mnist-from-scratch
uv sync --frozen
uv run python scripts/download_mnist.py
uv run python -m mnist_from_scratch.train
uv run python -m mnist_from_scratch.evaluate
```

`download_mnist.py` fetches the four MNIST files (11,594,722 bytes in all) into `mnist_cache/` and keeps a file only if its SHA-256 is the expected one. `train` runs 20 epochs from seed 0 and writes `mnist_weights.npz`; `evaluate` loads that file and prints the test loss, the accuracy, the accuracy of each digit and the confusion matrix. The dataset and the weights stay on your machine and are never committed.

To check the project itself:

```bash
uv run pytest --cov=src --cov-fail-under=95
uv run ruff check . && uv run ruff format --check .
```

These are the commands in `project.yaml`, and they were last run from a clean environment on the `verified` date there.

## What it does

- Trains the network from a fixed seed, so that two runs on the same machine give the same weights bit for bit. Both commands print a SHA-256 fingerprint of the weights, which is how you can tell.
- Scores the saved weights on the standard test split and reports per-digit accuracy and the full confusion matrix, not only the headline figure.
- Turns the headline into a check: `uv run python -m mnist_from_scratch.evaluate --min-accuracy 0.98` exits with status 1 if the test accuracy is lower.
- Refuses to start on a dataset that is not byte for byte the published one, and says which command fetches it.
- Stores weights as plain arrays in an `.npz` file, so loading weights never runs code from the file.

`--epochs`, `--batch-size`, `--seed`, `--data-dir` and `--weights` change a run; `--help` on any command lists them. The optimiser settings, the layer sizes, the L2 strength and the dropout rate are fixed in the code, because they are the baseline that other work cites.

## Architecture

![Four columns joined by arrows: the input of 784 pixels, two hidden blocks of Dense, ReLU and Dropout(0.1) with 100,480 and 16,512 parameters, and an output block of Dense and Softmax with 1,290, totalling 118,282 parameters.](docs/diagrams/architecture.svg)

Each image is flattened to 784 values in $[0, 1]$. Two hidden layers of 128 units follow, each a dense layer, a ReLU and a dropout layer with drop rate 0.1; a dense layer of 10 units and a softmax produce the class probabilities. The two hidden layers carry an L2 penalty of 0.0005 on their weights. Training is Adam with learning rate 0.001 and decay 0.0001 for 20 epochs, each one 469 mini-batches of at most 128 images drawn in a fresh random order.

The code is one package, `src/mnist_from_scratch/`, of six small modules: `nn.py` holds the series' classes, `model.py` assembles them and reads and writes the weights, `data.py` checks and parses the dataset, `download.py` fetches it, and `train.py` and `evaluate.py` are the two commands. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the components and the data flow, and the records under [docs/adr/](docs/adr/) give the reasons for the decisions a maintainer would otherwise have to rediscover.

## Results

Measured on 2026-10-05 with the quickstart commands, seed 0, on an Intel Core i7-9750H with 7.8 GB of memory, CPU only. [docs/EVALUATION.md](docs/EVALUATION.md) has the full output, the confusion matrix and the limits of the claim.

| Figure | Value |
|---|---|
| Test accuracy | 98.00 percent (9,800 of 10,000) |
| Test loss (mean cross-entropy) | 0.0628 |
| Misclassified test images | 200 |
| Parameters | 118,282 |
| Training accuracy in the last epoch, dropout active | 98.47 percent |
| Training time, 20 epochs | 85.3 s; other runs that day took up to 126.6 s on the same busy laptop |

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Test images | 980 | 1,135 | 1,032 | 1,010 | 982 | 892 | 958 | 1,028 | 974 | 1,009 |
| Correct | 969 | 1,125 | 1,018 | 992 | 971 | 871 | 940 | 1,005 | 949 | 960 |
| Accuracy (percent) | 98.88 | 99.12 | 98.64 | 98.22 | 98.88 | 97.65 | 98.12 | 97.76 | 97.43 | 95.14 |

In this run the digit 1 is the easiest and 9 the hardest, and the largest single confusion is 18 nines read as fours.

## Limits

- **One seed, one machine.** The headline is the seed 0 run. Seeds 1 and 2 gave 97.91 and 98.01 percent, so the three runs lie within 0.10 percentage points, and the hardest digit changed with the seed (9, then 2, then 9). Quote a per-digit figure with its seed.
- **Bit-for-bit reproduction needs the locked environment.** The result depends on the linear algebra library inside the NumPy wheel, so NumPy is pinned to 2.3.5. With another NumPy version or on another processor the last digits can differ; the fingerprint tells you whether your run is the documented one.
- **MNIST is an easy benchmark.** Centred, size-normalised digits flatter a dense network, and 98 percent here says little about harder data. The series' Fashion-MNIST project (`nn-p03`) runs the same network on clothing images.
- **No validation split.** The run trains on all 60,000 training images and uses the test set only to report. Anyone who changes a setting and compares test accuracy is tuning on the test set, the mistake `nn-029` describes; hold out part of the training data first.
- **A baseline, not the series' best practice.** The layers start from `0.01 * randn` weights, which `nn-033` shows is a poor choice for deeper networks. It is kept because this run is the documented baseline.
- **Weights only.** The `.npz` file holds the six parameter arrays and no optimiser state, so a run cannot be resumed. Pickle checkpoints written by the scripts before version 1.0.0 are not loaded.

## Built from

- `nn-016`: `Layer_Dense` and `Activation_ReLU`, each with its backward pass.
- `nn-019`: the combined softmax and categorical cross-entropy class and its one-line gradient.
- `nn-027`: `Optimizer_Adam`, with bias correction and learning-rate decay.
- `nn-030`: the L2 penalty in the dense layer's gradient and in the loss.
- `nn-031`: `Layer_Dropout` and the switch between training and evaluation.
- `nn-032`: the epoch and mini-batch loops of the trainer.

## Corrections

The series takes no pull requests or issues. If you find a mistake, send the correction through the series website.

## Licence

Code under MIT, prose and figures under CC BY 4.0, as stated in the series [LICENSE](../../LICENSE). MNIST itself is distributed under CC BY-SA 3.0.
