# 0001. One fixed network on two moons, and what counts as the baseline

- Status: accepted
- Date: 2026-10-05

## Context

The series derives the sigmoid with binary cross-entropy in `nn-034` and needs one place where that pair trains a model on data a reader can see. The project predates this record: the network, the optimiser, and the dataset were chosen when the project was added to the series on 2026-06-01, and its initialisation and noise level were changed on 2026-06-10 ([0004](0004-he-initialisation.md)). Posts of the series quote the class from this project and point readers to it, so its numbers are cited from outside this directory.

## Decision

The project trains exactly one configuration, and that configuration is the baseline every document here reports:

- Data: two moons, 1,000 points, noise 0.1, a seeded 80 to 20 split.
- Network: $2 \to 16 \to 16 \to 1$, ReLU in the hidden layers, one sigmoid output, 337 parameters.
- Loss: binary cross-entropy through the combined class, plus an L2 penalty of $10^{-4}$ on the weights of the two hidden layers.
- Optimiser: Adam, learning rate 0.01, no decay, full batch.
- Budget and seed: 2,000 epochs, seed 0.

A change to any item in this list is a change to the model, needs a new record that supersedes this one, and needs every figure in `docs/EVALUATION.md` and the README measured again. Restructuring the code is not such a change as long as the documented command prints the same figures; at version 1.0.0 the restructured code was checked to produce the same six weight arrays, bit for bit, as the code it replaced.

The layer sizes, the learning rate, and the penalty are constants in the code and not flags. The flags that exist (`--noise`, `--n-samples`, `--epochs`, `--seed`, `--init`) are there to measure the baseline against its neighbours, not to offer a general trainer.

## Consequences

- The result is small enough to be exact: 200 of 200 on the test set is a count, and anyone can check it in two seconds.
- The project is not a tool for other datasets. A reader who wants one copies `nn.py` and writes a loop, which is what the series teaches.
- Mini-batches, dropout, early stopping, and a validation set are absent because the baseline does not need them, and `docs/EVALUATION.md` states what their absence costs at noise 0.2.
