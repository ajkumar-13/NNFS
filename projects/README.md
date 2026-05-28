# Projects

Hands-on builds that apply the neural network from `posts/` to real datasets and tasks. Each project is self-contained: its own directory, its own code files, its own runnable training loop.

The convention:

```
projects/
└── NN-slug/
    ├── README.md          ← writeup (what + why)
    ├── nn.py              ← from-scratch classes used by this project
    ├── data.py            ← dataset loading and preprocessing
    ├── train.py           ← training loop
    ├── evaluate.py        ← test-time evaluation
    └── requirements.txt   ← minimal Python dependencies
```

Numbering matches the order projects were added; the slug describes the project.

## Index

| # | Project | Dataset | Status |
|---|---|---|---|
| 1 | [MNIST from scratch](01-mnist-from-scratch/README.md) | MNIST (60k train, 10k test, 28×28 grayscale digits) | shipping |

## Adding a new project

1. Pick the next two-digit number and a kebab-case slug. Create `projects/NN-slug/`.
2. Copy the five-file skeleton above. Replace `nn.py` only if the project needs classes that don't already exist in `01-mnist-from-scratch/nn.py` — otherwise import from there.
3. Write `README.md` in the textbook voice used by the [lecture posts](../posts/): TL;DR, capability bullets, numbered sections, common pitfalls, further reading.
4. Add a row to the index table above.
5. Optional: add a `diagrams/` subfolder with hero SVGs in the [Standard visual language](../Standard/).
