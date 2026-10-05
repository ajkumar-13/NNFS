# 0003. Pin NumPy exactly, and treat the order of random draws as part of the result

- Status: accepted
- Date: 2026-10-05

## Context

The documented result is one run from seed 0, and other series quote its figures to the last digit. For those figures to stay checkable, a rerun has to give the same weights, not merely similar ones. Two things decide whether it does.

The first is the random stream. The series' classes draw from NumPy's global generator: `randn` for the initial weights, `permutation` for the shuffle, `binomial` for the dropout masks. NumPy guarantees that this legacy generator produces the same stream for the same seed in every version, so the draws are stable as long as the code asks for them in the same order.

The second is floating-point arithmetic. Every matrix product goes through the linear algebra library bundled in the NumPy wheel, and a different build can round the last bit of a sum differently. On 2026-10-05 the same code and seed were run for one epoch under NumPy 2.3.5 and under NumPy 2.5.3 on the same machine: the two sets of weights were not bit-identical. Over 9,380 updates such differences grow, so the final figures are tied to the build. The effect of the newer version on the 20-epoch accuracy was not measured.

Under NumPy 2.3.5 the restructured package reproduces, bit for bit, the weights that the project's earlier scripts produced from seed 0.

## Decision

1. `pyproject.toml` requires `numpy==2.3.5`, not a range, and `uv.lock` records the wheel hashes. `.python-version` selects Python 3.13, the version the result was measured on.
2. The order of the random draws is a contract: `np.random.seed(seed)` once at the start of `train`; then the three layers draw their weights in the order dense1, dense2, dense3; then every epoch draws one permutation of the 60,000 indices, and every batch draws one mask for each dropout layer, first then second. Nothing else may draw from the generator during a run. Evaluation draws nothing.
3. A change to the loop that is meant to be neutral has to keep the fingerprint of the documented run. The move from a shuffled copy of the dataset to indexing each batch through the permutation was accepted on that evidence: the batches are the same arrays, the weights are identical, and the peak memory falls by one copy of the training set.
4. Both commands print `weights sha256`, the fingerprint of the six arrays, so that anyone can tell an identical rerun from a merely close one.

## Consequences

- Upgrading NumPy is a deliberate act: change the pin, rerun the quickstart, compare the fingerprint and the figures with `docs/EVALUATION.md`, and if they moved, publish a new evaluation under a new major version and tell the series that cite the old figures.
- A security fix in a later NumPy does not arrive by itself. `docs/SECURITY.md` schedules the audit that would catch one.
- Bit-identity is claimed for the locked environment on the processor family it was measured on. Another processor can select other kernels inside the same wheel; the fingerprint then differs even though the code and the lock are the same, and the accuracy should be read with the limits in `docs/EVALUATION.md`.
- The global generator is kept although a local `numpy.random.Generator` is the modern practice, because moving to it would change every draw and with it every published figure (record 0001).
- `tests/test_model.py` checks the order of the initial draws against an independent generator, and `tests/test_train.py` replays a three-epoch run on the committed fixture whose losses and output biases were recorded from the earlier scripts.
