# Fashion-MNIST with the same network

A model that scores 98 percent on MNIST says little about harder images, and one accuracy figure hides which classes fail. This project takes the network of the series' MNIST project (`nn-p01`), a two-hidden-layer dense network ($784 \to 128 \to 128 \to 10$) with Adam, L2 and dropout, and changes nothing but the data: the 60,000 training and 10,000 test images of Fashion-MNIST, ten kinds of clothing in the same $28 \times 28$ grey-scale format. From the fixed seed the documented run classifies 8,712 of the 10,000 test images correctly, 87.12 percent, where the same network reaches 98.00 percent on the digits, and the report shows where the 1,288 errors are: 816 of them are confusions among the four upper-body garments (T-shirt/top, Pullover, Coat, Shirt).

It is a reference model with a reproducible result, not a strong clothing classifier. It has no convolution, no batch normalisation, no augmentation and no GPU code, on purpose, and its settings were not tuned for this dataset: they are the MNIST project's settings, kept so that the two results differ only in the data.

## Who this is for

- Readers of Neural Networks from Scratch who have trained the MNIST project and need to see that MNIST flatters a model, with every number traceable to a command.
- Authors who need a dense baseline on Fashion-MNIST with its errors broken down by class. The sibling series Convolutional Neural Networks from Scratch uses this run as that baseline.

## Quickstart

You need [uv](https://docs.astral.sh/uv/), which fetches Python 3.13 and the locked packages, and a network connection for the first two commands. From a clone of the series repository:

```bash
cd projects/fashion-mnist
uv sync --frozen
uv run python scripts/download_fashion_mnist.py
uv run python -m fashion_mnist.train
uv run python -m fashion_mnist.evaluate
```

`download_fashion_mnist.py` fetches the four Fashion-MNIST files (30,878,645 bytes in all) into `fashion_mnist_cache/` and keeps a file only if its SHA-256 is the expected one. `train` runs 20 epochs from seed 0 and writes `fashion_mnist_weights.npz`; `evaluate` loads that file and prints the test loss, the accuracy, the accuracy of each class, the confusion matrix and the five most-confused pairs of classes. The dataset and the weights stay on your machine and are never committed.

To check the project itself:

```bash
uv run pytest --cov=src --cov-fail-under=95
uv run ruff check . && uv run ruff format --check .
```

These are the commands in `project.yaml`, and they were last run from a clean environment on the `verified` date there.

## What it does

- Trains the network from a fixed seed, so that two runs on the same machine give the same weights bit for bit. Both commands print a SHA-256 fingerprint of the weights, which is how you can tell.
- Scores the saved weights on the standard test split and reports the accuracy of each class by name, the full confusion matrix and the most-confused pairs, not only the headline figure.
- Turns the headline into a check: `uv run python -m fashion_mnist.evaluate --min-accuracy 0.8712` exits with status 1 if the test accuracy is lower.
- Refuses to start on a dataset that is not byte for byte the published one, and says which command fetches it.
- Stores weights as plain arrays in an `.npz` file, so loading weights never runs code from the file.

`--epochs`, `--batch-size`, `--seed`, `--data-dir` and `--weights` change a run; `--help` on any command lists them. The optimiser settings, the layer sizes, the L2 strength and the dropout rate are fixed in the code, because they are the baseline that other work cites.

## Architecture

Each image is flattened to 784 values in $[0, 1]$. Two hidden layers of 128 units follow, each a dense layer, a ReLU and a dropout layer with drop rate 0.1; a dense layer of 10 units and a softmax produce the class probabilities. The two hidden layers carry an L2 penalty of 0.0005 on their weights. Training is Adam with learning rate 0.001 and decay 0.0001 for 20 epochs, each one 469 mini-batches of at most 128 images drawn in a fresh random order. The network has $100{,}480 + 16{,}512 + 1{,}290 = 118{,}282$ parameters. All of this is the MNIST project's network and training run; the modules `nn.py`, `model.py` and `train.py` differ from that project's only in the names of the package, the dataset and the weights file.

The code is one package, `src/fashion_mnist/`, of six small modules: `nn.py` holds the series' classes, `model.py` assembles them and reads and writes the weights, `data.py` checks and parses the dataset and names its ten classes, `download.py` fetches it, and `train.py` and `evaluate.py` are the two commands. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the components and the data flow, and the records under [docs/adr/](docs/adr/) give the reasons for the decisions a maintainer would otherwise have to rediscover.

## Results

Measured on 2026-10-05 with the quickstart commands, seed 0, on an Intel Core i7-9750H with 7.8 GB of memory, CPU only. [docs/EVALUATION.md](docs/EVALUATION.md) has the full output, the confusion matrix and the limits of the claim.

| Figure | Value |
|---|---|
| Test accuracy | 87.12 percent (8,712 of 10,000) |
| Test loss (mean cross-entropy) | 0.3419 |
| Misclassified test images | 1,288 |
| Parameters | 118,282 |
| Training accuracy in the last epoch, dropout active | 89.28 percent |
| Training time, 20 epochs | 116.1 s; other runs that day took up to 162.5 s on the same busy laptop |

| Class | T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |
|---|---|---|---|---|---|---|---|---|---|---|
| Label | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| Test images | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| Correct | 754 | 964 | 726 | 906 | 801 | 929 | 752 | 941 | 969 | 970 |
| Accuracy (percent) | 75.40 | 96.40 | 72.60 | 90.60 | 80.10 | 92.90 | 75.20 | 94.10 | 96.90 | 97.00 |

The five largest confusions of the run, true class first:

| True class | Predicted as | Test images |
|---|---|---|
| T-shirt/top | Shirt | 185 |
| Pullover | Shirt | 129 |
| Pullover | Coat | 121 |
| Coat | Shirt | 97 |
| Shirt | T-shirt/top | 80 |

In this run Ankle boot is the easiest class and Pullover the hardest, with Shirt and T-shirt/top, two images apart, just above it. Shirt is the class the network reaches for when it is unsure: it predicts Shirt 1,210 times for 1,000 shirts, and 458 of those predictions are wrong. Of the 1,288 errors, 816 are confusions among T-shirt/top, Pullover, Coat and Shirt and 157 are among Sandal, Sneaker and Ankle boot, so three errors in four stay inside one of those two groups.

The same network, trained the same way, makes 200 errors on the MNIST test set (`nn-p01`) and 1,288 here. Nothing changed but the images.

## Limits

- **One seed, one machine.** The headline is the seed 0 run. Seeds 1 and 2 gave 88.31 and 87.64 percent, so the three runs span 1.19 percentage points and seed 0 is the lowest of them. The hardest class changed with the seed (Pullover, then Shirt, then Shirt), and so did the largest confusion; Shirt itself scored 75.20, 69.80 and 67.70 percent. Quote a per-class figure with its seed.
- **Bit-for-bit reproduction needs the locked environment.** The result depends on the linear algebra library inside the NumPy wheel, so NumPy is pinned to 2.3.5. With another NumPy version or on another processor the last digits can differ; the fingerprint tells you whether your run is the documented one.
- **The settings are MNIST's, not tuned for this data.** The point of the project is the comparison, so nothing was adjusted. The figure is a baseline for this network, not the best a dense network can do on Fashion-MNIST.
- **No validation split.** The run trains on all 60,000 training images and uses the test set only to report. Anyone who changes a setting and compares test accuracy is tuning on the test set, the mistake `nn-029` describes; hold out part of the training data first.
- **A dense network ignores the layout of the pixels.** It sees 784 unordered numbers, so two garments with the same outline and a different collar or sleeve are close to identical to it. The breakdown by class shows the consequence; what a convolutional network does about it is the subject of the sibling series, not of this project.
- **A baseline, not the series' best practice.** The layers start from `0.01 * randn` weights, which `nn-033` shows is a poor choice for deeper networks. It is kept because this run is the documented baseline.
- **Weights only.** The `.npz` file holds the six parameter arrays and no optimiser state, so a run cannot be resumed. Pickle checkpoints written by the scripts before version 1.0.0 are not loaded.

## Built from

- `nn-016`: `Layer_Dense` and `Activation_ReLU`, each with its backward pass.
- `nn-019`: the combined softmax and categorical cross-entropy class and its one-line gradient.
- `nn-027`: `Optimizer_Adam`, with bias correction and learning-rate decay.
- `nn-028`: the separation of training data from test data, and reading a test figure as an estimate.
- `nn-030`: the L2 penalty in the dense layer's gradient and in the loss.
- `nn-031`: `Layer_Dropout` and the switch between training and evaluation.
- `nn-032`: the epoch and mini-batch loops of the trainer.
- `nn-033`: why the `0.01 * randn` initialisation this baseline keeps is not the one to choose for a new network.

## Corrections

The series takes no pull requests or issues. If you find a mistake, send the correction through the series website.

## Licence

Code under MIT, prose and figures under CC BY 4.0, as stated in the series [LICENSE](../../LICENSE). Fashion-MNIST itself is distributed by Zalando SE under the MIT licence.
