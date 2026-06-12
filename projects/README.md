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

| # | Project | Task | Dataset | Hero result |
|---|---|---|---|---|
| 1 | [MNIST from scratch](01-mnist-from-scratch/README.md) | 10-class classification | MNIST (60k train, 10k test, 28×28 grayscale digits) | ~97% test accuracy |
| 2 | [Binary classifier on two-moons](02-binary-classifier/README.md) | binary classification | Two-moons synthetic 2-D dataset (1000 samples) | ~98.5% test accuracy, visualised decision boundary |
| 3 | [Fashion-MNIST](03-fashion-mnist/README.md) | 10-class classification | Fashion-MNIST (60k train, 10k test) | ~89% test accuracy, confusion matrix of the "shirt cluster" |
| 4 | [California housing regression](04-california-housing-regression/README.md) | regression | California housing (20 640 samples, 8 features) | R² ≈ 0.78, RMSE ~$58k |

## Adding a new project

1. Pick the next two-digit number and a kebab-case slug. Create `projects/NN-slug/`.
2. Copy the five-file skeleton above. Replace `nn.py` only if the project needs classes that don't already exist in `01-mnist-from-scratch/nn.py` — otherwise import from there.
3. Write `README.md` in the textbook voice used by the [lecture posts](../posts/): TL;DR, capability bullets, numbered sections, common pitfalls, further reading.
4. Add a row to the index table above.
5. Optional: add a `diagrams/` subfolder with hero SVGs matching the visual style of the existing project diagrams (e.g. [02-binary-classifier/diagrams/](02-binary-classifier/diagrams/)).
