# 0003. Pin NumPy exactly, and treat the order of random draws as part of the result

- Status: accepted
- Date: 2026-10-05

## Context

Before 1.0.0 the project had a `requirements.txt` asking for `numpy>=1.23` and four scripts at the top of the directory that imported each other by file name. Nothing was installed and nothing was pinned, so the figures a reader got depended on whichever NumPy happened to be present.

The documented figures are floating-point results from seed 0, and two things decide whether a rerun gives the same weights. The first is the random stream: the weights come from NumPy's legacy global generator, whose stream NumPy keeps fixed across versions, and the data from `numpy.random.default_rng`, which carries no such promise. The second is arithmetic: every matrix product goes through the linear algebra library inside the NumPy wheel, and another build can round the last bit differently.

For this project the second effect was measured and did not appear: on 2026-10-05 the documented run gave the same six weight arrays, the same split and the same decision grid, bit for bit, under NumPy 2.3.5 with Python 3.13.5 and under NumPy 2.5.3 with Python 3.12.12, and both agree with the scripts the package replaced. The series' MNIST project (`nn-p01`) made the same comparison and found that its weights do differ between those two versions, which is why it pins NumPy 2.3.5.

## Decision

1. `pyproject.toml` requires `numpy==2.3.5`, not a range, and `uv.lock` records the wheel hashes. `.python-version` selects Python 3.13. This is the environment of every project of the series: a reader sets up one toolchain, and a figure of one project is never measured on a different NumPy from a figure of another.
2. The code is a package, `binary_classifier`, under `src/`, built with hatchling and installed into the project's environment by `uv sync --frozen`, which installs the locked versions or fails. `requirements.txt` is removed.
3. The order of the random draws is a contract. `make_moons` and `train_test_split` each use their own `default_rng(seed)`; then `np.random.seed(seed)` is called once and the three layers draw their weights in the order dense1, dense2, dense3. Nothing else draws during a run, and evaluation uses no random numbers for its figures.
4. Training prints `data sha256`, the fingerprint of the generated points and labels, so that anyone can tell whether their data are the documented ones.

## Consequences

- Upgrading NumPy is a deliberate act: change the pin, rerun the quickstart, and compare the output with `docs/EVALUATION.md`. The one upgrade tried so far, to 2.5.3, changed nothing here.
- A security fix in a later NumPy does not arrive by itself. `docs/SECURITY.md` schedules the audit that would catch one.
- A reader needs `uv`, which fetches Python 3.13 if it is missing.
- Bit-identity is claimed for the locked environment on the processor it was measured on. Another processor can select other kernels inside the same wheel, and the last digits of the losses can then differ.
- The build backend, hatchling, is fetched by `uv` at the first build and is not pinned by `uv.lock`; it does not take part in any computation.
- `tests/test_model.py` checks the order of the initial draws against an independent generator.
