# 0001. Keep the MNIST project's network and run unchanged, and change only the data

- Status: accepted
- Date: 2026-10-05

## Context

The project exists to answer one question: what happens to the network of the series' MNIST project (`nn-p01`) when the images get harder and nothing else changes? That question is only answered if nothing else changes. A fall in accuracy can be attributed to the data only when the classes, the layer sizes, the penalty, the dropout rate, the optimiser, the schedule, the batch size and the seed are those of the MNIST run.

A second constraint is that the run is a reference point. The sibling series on convolutional networks quotes its test accuracy, its per-class figures and its confusions as the dense baseline on Fashion-MNIST. A baseline that drifts is worse than no baseline.

Four things look like mistakes to a maintainer who does not know this, and each is deliberate: the class names break Python naming conventions (`Layer_Dense`), the weights start at `0.01 * randn` although `nn-033` shows a better rule, the output layer carries no L2 penalty although the two hidden layers do, and no setting was tuned for Fashion-MNIST although tuning would raise the figure.

## Decision

1. `nn.py` is the MNIST project's `nn.py`, the same file byte for byte: the classes under the series' names with the series' arithmetic, plus argument checks in the constructors. Improvements to a class belong in the post that derives it first.
2. `model.py` and `train.py` are the MNIST project's modules with three names changed: the package, the dataset in the messages, and the default weights file. The model and its training configuration are constants, not options: layer sizes 784, 128, 128, 10; L2 strength 0.0005 on the two hidden layers' weights and none elsewhere; dropout rate 0.1; Adam with learning rate 0.001 and decay 0.0001; 20 epochs; mini-batches of 128; seed 0. The command line exposes the epoch count, the batch size and the seed for experiments, and no flag for the rest.
3. What this project owns is the data and the report: the record of the four Fashion-MNIST files and the ten class names in `data.py`, the mirror in `download.py`, and in `evaluate.py` the class names in the report and the list of most-confused pairs.
4. The code is copied, not imported from the MNIST project. The series' contract makes every project self-contained, with its own dependency file and tests, so that it can be extracted on its own. The cost is that a correction to a shared module has to be made in both projects; the test suites, which share their cases, are what catches a divergence.
5. Any change that alters the weights of the seed 0 run is a breaking change. It needs a new major version, a new evaluation, and a note to the series that cite the figures. A restructuring that claims to change nothing must show the same weight fingerprint for the documented run before it is merged.

## Consequences

- The naming rules of the linter are not enabled, and the classes are not rewritten in a more idiomatic style.
- The network keeps an initialisation and a penalty layout that the later posts of the series would not choose, and settings chosen for another dataset. The README says so under its limits.
- A reader who wants to try another width, rate or optimiser edits a constant and accepts that the result is no longer the documented one, and no longer comparable with the MNIST project's.
- If the MNIST project ever changes its baseline, this project has to follow in the same release or say that the two have parted, because the comparison is the reason it exists.
- `tests/test_model.py` pins the architecture, the parameter count, the penalty layout and the order of the initial draws; `tests/test_train.py` pins a short recorded run, whose figures are the ones the MNIST project's test records, because on identical bytes the two projects compute the same thing.
