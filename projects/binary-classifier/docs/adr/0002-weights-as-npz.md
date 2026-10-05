# 0002. Store weights as plain arrays in an `.npz` file, never as a pickle

- Status: accepted
- Date: 2026-10-05

## Context

Until version 1.0.0 the training command wrote its weights with `pickle.dumps` and the evaluation command read them with `pickle.loads`. A pickle is a program: loading one runs whatever constructors the file names. A weights file is the one file of this project that a reader might receive from someone else, so the evaluation command could be made to run arbitrary code by handing it a file.

Everything in the file is a numeric array or a short string: six weight and bias arrays, four arrays of points and labels, and six scalars, which are the format version, the name of the initialisation, and four numbers of configuration.

## Decision

The weights file is a NumPy `.npz` archive, written with `numpy.savez` and read with `numpy.load(..., allow_pickle=False)`. The scalars are stored as zero-dimensional arrays and the name of the initialisation as a string array, so nothing in the file needs pickling and NumPy refuses any file that does.

`load_weights` trusts nothing it reads. It requires the sixteen arrays by name, the exact shape of each weight array, floating-point types and finite values, labels of 0 or 1, one label per point, non-empty sets, and `format_version` 1, and it reports the first thing that is wrong in a `WeightsError` that says what to do. The file is opened by the project and handed to NumPy, so that it is closed even when NumPy stops half-way through a damaged archive.

File names must end in `.npz`. `numpy.savez` appends `.npz` to any other name, which would write a file the user did not ask for, so other names are refused before training starts. A name ending in `.pkl` gets its own message.

Pickle files are not read at all. No converter is provided: a converter would have to unpickle, which is the thing being removed, and training again takes two seconds.

The option is `--weights` and the default name `moons_weights.npz`, the names the other projects of the series use.

## Consequences

- Weights written before 1.0.0 (`moons_weights.pkl`) no longer load. The changelog says so, and the error message says to train again.
- Loading a weights file cannot execute code from it. Two tests keep this true: a pickle renamed to `.npz` and an object array inside a valid archive are both refused, and neither is executed.
- A crafted archive can still claim arrays large enough to exhaust memory. `docs/SECURITY.md` records that as the remaining risk.
- The archive is not compressed. At 23 kB it does not need to be, and an uncompressed archive is simpler to inspect.
- The file holds more than weights: the train and test split and the configuration travel with them (record [0005](0005-generated-data-stored-split.md)). It holds no optimiser state, so a run cannot be resumed.
- `format_version` allows a later format to be told apart from this one. A version that changes the layout raises the number and decides then whether to read the old one.
