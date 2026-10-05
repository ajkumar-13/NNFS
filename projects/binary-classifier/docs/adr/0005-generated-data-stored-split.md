# 0005. The data are generated, checksummed, and stored with the weights

- Status: accepted
- Date: 2026-10-05

## Context

Two moons is usually taken from scikit-learn. The series is written in NumPy alone, and a project whose model is 337 parameters should not need a machine-learning library to make 1,000 points. But generated data raise a question that downloaded data do not: is the reader's dataset the one the documented figures were measured on? A downloaded file has a checksum. A generator has a seed, and a seed fixes the output only for one implementation of the generator.

## Decision

`make_moons` builds the dataset in NumPy: the noiseless construction scikit-learn uses, two half-circles of unit radius, with noise and shuffling from `numpy.random.default_rng(seed)`. Points are `float32` and labels `int64`. `train_test_split` uses its own `default_rng(seed)`. Nothing is downloaded, nothing is cached, and no data file is committed apart from an eight-point fixture for the tests.

The identity of the data is made checkable in three ways:

- Training prints `data sha256`, the SHA-256 of the points and labels as bytes, and `project.yaml` and `docs/EVALUATION.md` record the value for the documented run. A reader whose line matches has the same 1,000 points, bit for bit.
- The test suite compares the generator with the committed fixture to within $10^{-6}$, so a change in the generator's output on the locked NumPy fails the tests.
- The weights file stores the train and test split. Evaluation scores the points that training actually held out and never regenerates them, so a weights file is evaluated correctly even on a NumPy whose generator differs.

The same seed is used for the data, the split, and the weights. The weights come from NumPy's legacy global generator, seeded with `numpy.random.seed`, because the posts' `Layer_Dense` calls `numpy.random.randn` and the project keeps that class as it is ([0006](0006-series-classes-copied.md)). The legacy generator's stream is frozen by NumPy's compatibility policy; `default_rng` carries no such promise across NumPy versions, which is one reason NumPy is pinned ([0003](0003-pin-numpy-for-bit-reproducibility.md)) and why the checksum exists.

## Consequences

- No scikit-learn, no download script, no cache directory.
- The values are not scikit-learn's: the geometry is the same, the noise is not, and scikit-learn returns `float64`.
- The weights file is larger than the weights alone, 23 kB against roughly 3 kB of weights. That is the price of a self-contained evaluation.
- The data and the weights of a run cannot be seeded separately from the command line. The seed sweep therefore varies both together, and `docs/EVALUATION.md` says so.
- The default noise of `make_moons` is 0.1, the documented value. It was 0.2 before 1.0.0, a leftover from before [0004](0004-he-initialisation.md) that the training command overrode.
