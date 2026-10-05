# 0001. Assemble the series' classes as they are, and freeze the baseline

- Status: accepted
- Date: 2026-10-05

## Context

The project exists to answer one question: do the classes the series derives on a toy dataset compose into a model that works on real data? That question is only answered if the classes here are the series' classes. A second constraint arrived later. The run became a reference point: the sibling series on convolutional networks quotes its test accuracy, its error count and its parameter count as the dense baseline (`cnn-011`, `cnn-012`), and the series' Fashion-MNIST project reuses the network. A baseline that drifts is worse than no baseline.

Three things in the code look like mistakes to a maintainer who does not know this, and each is deliberate: the class names break Python naming conventions (`Layer_Dense`), the weights start at `0.01 * randn` although `nn-033` shows a better rule, and the output layer carries no L2 penalty although the two hidden layers do.

## Decision

1. `nn.py` holds the classes under the series' names with the series' arithmetic. The only additions are argument checks in the constructors. Improvements to a class belong in the post that derives it first.
2. The model and its training configuration are constants in `model.py` and `train.py`, not options: layer sizes 784, 128, 128, 10; L2 strength 0.0005 on the two hidden layers' weights and none elsewhere; dropout rate 0.1; Adam with learning rate 0.001 and decay 0.0001; 20 epochs; mini-batches of 128; seed 0. The command line exposes the epoch count, the batch size and the seed for experiments, and no flag for the rest.
3. Any change that alters the weights of the seed 0 run is a breaking change. It needs a new major version, a new evaluation, and a note to the series that cite the figures. A restructuring that claims to change nothing must show the same weight fingerprint for the documented run before it is merged.

## Consequences

- The naming rules of the linter are not enabled, and the classes are not rewritten in a more idiomatic style.
- The network keeps an initialisation and a penalty layout that the later posts of the series would not choose. The README says so under its limits.
- A reader who wants to try another width, rate or optimiser edits a constant and accepts that the result is no longer the documented one. This is a small cost, and it keeps every published figure tied to one configuration.
- `tests/test_model.py` pins the architecture, the parameter count, the penalty layout and the order of the initial draws; `tests/test_train.py` pins a short recorded run. A change that breaks the baseline fails a test before it reaches the evaluation.
