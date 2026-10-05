# 0002. Store weights and scaler as plain arrays in an `.npz` file, never as a pickle

- Status: accepted
- Date: 2026-10-05

## Context

Before version 1.0.0 the trainer wrote one checkpoint with `pickle.dumps` and the evaluation read it with `pickle.loads`. Unpickling runs whatever code the file asks for, so a weights file handed from one person to another was a program, not data. The checkpoint also carried more than weights: the scaling statistics and both standardised folds, 782,237 bytes in all, so the evaluation never looked at the dataset and scored whatever arrays the file contained.

A regression model differs from the series' classifiers in one respect that matters here. Its weights alone cannot make a prediction: the inputs have to be scaled with the training fold's statistics first, and the output is in standardised units until the target's mean and standard deviation turn it back into dollars. The scaler is part of the model.

Two alternatives were weighed. Keeping pickle and documenting the risk would have preserved old files, but the mitigation is "only load files you wrote", which is no mitigation for a project whose users are told to share results. Storing the weights alone, as the MNIST project does, would have left the scaler to be recomputed from the data every time, with nothing to say whether the data and the split were the ones the weights were trained on.

## Decision

Weights are written with `numpy.savez` to one `.npz` archive of ten arrays: `dense1_weights`, `dense1_biases`, `dense2_weights`, `dense2_biases`, `dense3_weights`, `dense3_biases`, and the scaler `x_mean`, `x_std`, `y_mean`, `y_std`. They are read with `numpy.load(..., allow_pickle=False)`. `Network.load` checks, in this order: the file exists; the path ends in `.npz`; the archive holds exactly those ten arrays; every array's header declares the shape this network needs and a floating-point type, read before any array data; every value is finite; the two standard deviations are positive. Because the shapes are checked from the headers, a file cannot make the loader allocate more memory than the network itself occupies. The parameters are converted to 64-bit floats, the type the network trains in, and the feature statistics to 32-bit floats, the type the data is scaled in. A failed check raises `WeightsError` with the reason and leaves the network as it was. A path that does not end in `.npz` is refused without being opened.

The folds are not stored. The evaluation loads the dataset, splits it, and recomputes the scaler; the stored scaler is what it compares the result against (record 0005).

The file is written under a `.part` name and renamed, so an interrupted run never leaves a truncated file under the real name.

## Consequences

- Loading weights cannot execute code from the file. An object array inside an archive is refused, not unpickled.
- Pickle checkpoints written by the scripts before 1.0.0 (`cal_housing_weights.pkl`) no longer load, and there is no converter, because a converter would have to unpickle. Retraining takes about 20 seconds and, in the locked environment, gives the same weights.
- The evaluation now needs the dataset, which it did not before. In exchange its figures are computed from the checked archive and not from arrays of unknown origin.
- The archive stores a timestamp, so two files with identical weights differ as files. `Network.fingerprint`, a SHA-256 over the six parameter arrays, is the way to compare weights, and both commands print it. The scaler is left out of the fingerprint on purpose, so that it means the same thing as in the series' other projects; the scaler follows from the data and the seed, and is checked separately.
- The file holds parameters and scaler only. Resuming a run would need the optimiser's moment buffers and step count as well; that is out of scope until someone needs it, and would be a new file format with its own record.
