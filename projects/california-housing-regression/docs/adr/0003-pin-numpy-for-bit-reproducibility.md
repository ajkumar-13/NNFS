# 0003. Pin NumPy exactly, and treat the order of random draws as part of the result

- Status: accepted
- Date: 2026-10-05

## Context

The documented result is one run from seed 0. For its figures to stay checkable, a rerun has to give the same weights, not merely similar ones. Three things decide whether it does.

The first is the stream that builds and trains the network. The series' classes draw from NumPy's global generator: `randn` for the initial weights, `permutation` for the shuffle. NumPy guarantees that this legacy generator produces the same stream for the same seed in every version, so these draws are stable as long as the code asks for them in the same order.

The second is the stream that splits the data. The split draws from `numpy.random.default_rng(seed)`, as it did before version 1.0.0. NumPy makes no such promise for this generator: a later version may give another permutation for the same seed, and then the folds, the scaler and every figure change.

The third is floating-point arithmetic. Every matrix product goes through the linear algebra library bundled in the NumPy wheel, and a different build can round the last bit of a sum differently. The series' MNIST project measured exactly that between two NumPy versions. Here, on 2026-10-05, the documented run was repeated on the same machine under NumPy 2.5.3 in place of 2.3.5 and gave the same weights bit for bit. That is one observation on one processor, with matrices of at most 256 by 64, and it is not a guarantee.

Under NumPy 2.3.5 the restructured package reproduces, bit for bit, the weights that the project's earlier scripts produced from seed 0.

## Decision

1. `pyproject.toml` requires `numpy==2.3.5`, not a range, and `uv.lock` records the wheel hashes. `.python-version` selects Python 3.13, the version the result was measured on. This is the same pin as the series' other projects.
2. The order of the random draws is a contract: `np.random.seed(seed)` once at the start of `train`; the split, from its own generator started from the same seed; then the three layers draw their weights in the order dense1, dense2, dense3; then every epoch draws one permutation of the training row numbers. Nothing else may draw from the global generator during a run. Evaluation draws only the split.
3. A change to the loop that is meant to be neutral has to keep the fingerprint of the documented run. The move from a shuffled copy of the training fold to indexing each batch through the permutation was accepted on that evidence: the batches are the same arrays and the weights are identical.
4. Both commands print `weights sha256`, the fingerprint of the six parameter arrays, so that anyone can tell an identical rerun from a merely close one.

## Consequences

- Upgrading NumPy is a deliberate act: change the pin, rerun the quickstart, compare the fingerprint and the figures with `docs/EVALUATION.md`, and if they moved, publish a new evaluation under a new major version.
- A security fix in a later NumPy does not arrive by itself. `docs/SECURITY.md` schedules the audit that would catch one.
- Bit-identity is claimed for the locked environment on the processor family it was measured on. Another processor can select other kernels inside the same wheel; the fingerprint then differs even though the code and the lock are the same, and the figures should be read with the limits in `docs/EVALUATION.md`.
- The global generator is kept although a local `numpy.random.Generator` is the modern practice, because moving to it would change every draw and with it every published figure (record 0001).
- `tests/test_model.py` checks the order of the initial draws against an independent generator. `tests/test_train.py` checks that training draws one permutation an epoch and nothing else, and replays a three-epoch run on the committed fixture whose losses and weights were recorded from the earlier scripts. `tests/test_data.py` pins the seed 0 split of that fixture row by row.
