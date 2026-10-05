# 0005. Split first, fit the scaler on the training fold, and refuse to score weights on another split

- Status: accepted
- Date: 2026-10-05

## Context

`nn-029` gives the rule: any statistic estimated from data, a mean, a standard deviation, a vocabulary, is fitted on the training fold and then applied to the held-out fold. Fitting it on all the rows lets the test fold shape the training inputs, and the test figure stops being a measure of unseen data. This project is where the series applies that rule in code, so the code has to make the rule hard to break and the tests have to show that it holds.

Three facts about the project shape the decision.

- Both the eight features and the target are standardised. The target's statistics matter as much as the features': they define the units of the loss and the conversion back to dollars.
- The seed fixes the split. It did before version 1.0.0, when one `--seed` started both the generator that shuffles the rows into folds and the generator that initialises and trains the network, and that coupling is part of the documented run.
- Since version 1.0.0 the evaluation rebuilds the folds from the dataset instead of reading them from the checkpoint (record 0002). So the evaluation has to be told the seed, and a wrong seed would silently score the network on a test fold four fifths of which it was trained on.

## Decision

1. `load_california_housing` performs three steps in a fixed order: `split_indices`, then `standardise_fit` on the training rows, then `standardise_apply` on each fold. `standardise_fit` is called nowhere else, and `standardise_apply` takes a finished `Scaler` and fits nothing.
2. The scaler is four statistics, the mean and standard deviation of the features and of the target, all from the training fold. A small constant, $10^{-7}$, is added to each standard deviation so that a constant column cannot divide by zero.
3. The trainer stores the scaler in the weights file beside the parameters. The evaluation takes `--seed`, rebuilds the split, recomputes the scaler, and compares it with the stored one. If they differ by more than one part in a million, it stops and says that the weights were trained on another split, and scores nothing. Two different splits of this data differ by far more than that, and two computations of the same split differ by far less.
4. Scoring uses the recomputed scaler and the published dollar values of each fold, so nothing in the weights file can move the folds or the scale of the errors.
5. The seed keeps fixing the split, because separating the two would change the documented run. The option is named `--seed` on both commands, with the same default.

## Consequences

- `tests/test_data.py` tests the rule directly. The statistics are recomputed from the training rows alone and compared exactly. Every test row of a table is replaced by wildly different values, and the scaler and the training arrays must not move by a bit while the test arrays must. A training row is changed, and the scaler must move. The test fold must be scaled with the training statistics and must therefore not have zero mean and unit spread itself.
- `tests/test_evaluate.py` trains with one seed and evaluates with another, and requires the refusal.
- Evaluating weights from a run with another seed takes one more option than it would if the seed were stored in the file. The seed was kept out of the file so that the file stays ten arrays of floats and the check rests on the data, not on a number the file declares about itself.
- Because the seed moves the split, a comparison across seeds mixes two sources of variation, the fold and the initial weights. `docs/EVALUATION.md` says so beside its table of seeds. Separating them would need a second seed option, which is a change to the documented interface and a decision for a later version.
- There is no validation fold. The settings are fixed (record 0001), so nothing is tuned and nothing needs one; a reader who starts tuning must first hold out part of the training fold, as `nn-029` describes, and fit the scaler on what remains.
