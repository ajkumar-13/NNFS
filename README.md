# Neural Networks from Scratch

A 35-part blog series that builds a neural network by hand, plus 4 hands-on projects that put it to work. No frameworks, no black boxes. Every neuron, layer, activation function, loss function, backpropagation rule, optimiser, and regularisation technique is implemented from first principles using only Python and NumPy.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![NumPy only](https://img.shields.io/badge/dependencies-NumPy%20only-013243?logo=numpy&logoColor=white)
![Posts](https://img.shields.io/badge/posts-35-5B7FBF)
![Projects](https://img.shields.io/badge/projects-4-5C9E78)

---

## What you'll build

By the end of the series you will have a working multi-layer perceptron trained with Adam + L2 regularisation + Dropout, reaching ~97% test accuracy on MNIST, without touching Keras, PyTorch, or TensorFlow.

## What you'll understand

- How a neuron computes a weighted sum, and why that is all you need to start.
- Why matrix multiplication is the right abstraction for an entire layer.
- How the chain rule turns calculus into an algorithm: backpropagation.
- Why Adam outperforms vanilla gradient descent, and how each optimiser fixes the last one's flaw.
- How regularisation and dropout close the gap between training and test accuracy.

---

## Series structure

| Phase | Parts | Topic |
|---|:---:|---|
| 1 — Foundations | 1–5 | Neurons, NumPy, dot products, batches, broadcasting |
| 2 — Forward pass complete | 6–9 | ReLU, Softmax, the forward pass, cross-entropy, optimisation intro |
| 3 — Calculus | 10–11 | Derivatives, partial derivatives, the chain rule |
| 4 — Backpropagation | 12–21 | Full backward pass, all gradients, end-to-end pipeline |
| 5 — Optimisers | 22–27 | SGD → Decay → Momentum → AdaGrad → RMSProp → Adam |
| 6 — Generalisation & regularisation | 28–31 | Train/test split, validation, L1/L2, Dropout |
| 7 — Practical training & extensions | 32–35 | Mini-batching, weight init, sigmoid + BCE, what's next |

*(Phase numbering matches [INDEX.md](INDEX.md).)*

The complete annotated series listing with a guided learning path lives in [INDEX.md](INDEX.md).

---

## Requirements

```
Python 3.8+
NumPy
```

No deep-learning framework. No autograd. No automatic differentiation. Just arrays.

---

## Repository layout

```
posts/                       → 35 blog posts, one directory each
  NN-slug/
    index.md                 → the lecture
    diagrams/                → SVG diagrams shipped with the post

projects/                    → 4 applied projects, each a standalone README + code
  01-mnist-from-scratch/     → two-hidden-layer MLP, ~97% MNIST accuracy
  02-binary-classifier/      → two-moons, sigmoid + BCE
  03-fashion-mnist/          → 10-class image classification
  04-california-housing-regression/  → regression with MSE

cheatsheets/                 → four quick-reference sheets (Parts 1–5, 6–9, 10–21, 22–31)
dashboards/                  → optimiser and regularisation side-by-side comparison tables

cumulative_notebook.ipynb    → all series code in one runnable notebook
INDEX.md                     → series index and guided learning path
REFERENCES.md                → master bibliography for every post
glossary.md                  → plain-English definitions for every term
notation_guide.md            → symbols, tensor shapes, and optimiser variables decoded
exercises.md                 → hands-on experiments for every lecture
quizzes.md                   → multiple-choice comprehension checks
gradient_checking.md         → numerically verify your backprop implementation
common_pitfalls.md           → the most frequent mistakes and how to fix them
appendix_softmax_combined_backward.md  → full derivation of the combined backward pass
```

---

## Getting started

1. Open [INDEX.md](INDEX.md), the combined series listing and learning pathway.
2. Start at [Part 1 — Neurons and layers](posts/01-neurons-and-layers/index.md).
3. Work through each phase in order. Every post ends with a **What to read next** pointer.
4. Use the [cumulative notebook](cumulative_notebook.ipynb) to run any lecture's code interactively.
5. When something is unclear, consult the [concept dependency map](INDEX.md#concept-dependency-map) to find which earlier concept to revisit.

---

## Supplementary resources

| Resource | Purpose |
|---|---|
| [Glossary](glossary.md) | Plain-English definitions for every term in the series |
| [Notation Guide](notation_guide.md) | Symbols, tensor shapes, and optimiser variables in one place |
| [References](REFERENCES.md) | Master bibliography of every paper, book, and doc cited |
| [Exercises](exercises.md) | Hands-on experiments for every lecture |
| [Quizzes](quizzes.md) | Multiple-choice comprehension checks with explanations |
| [Gradient Checking](gradient_checking.md) | Numerically verify your backprop before trusting it |
| [Common Pitfalls](common_pitfalls.md) | The most frequent mistakes and exactly how to fix them |
| [Softmax Backward Appendix](appendix_softmax_combined_backward.md) | Full derivation of the combined softmax + cross-entropy backward pass |

### Cheat sheets

| Sheet | Covers |
|---|---|
| [Foundation](cheatsheets/01_Foundation.md) | Neurons, layers, NumPy, broadcasting (Parts 1–5) |
| [Activations & Loss](cheatsheets/02_Activations_Loss_Forward.md) | ReLU, Softmax, cross-entropy, forward pass (Parts 6–9) |
| [Calculus & Backprop](cheatsheets/03_Calculus_Backpropagation.md) | Derivatives, chain rule, full backward pass (Parts 10–21) |
| [Optimisers & Regularisation](cheatsheets/04_Optimizers_Regularization.md) | SGD through Adam, L1/L2, and Dropout (Parts 22–31) |

### Comparison dashboards

| Dashboard | Description |
|---|---|
| [Optimiser Comparison](dashboards/Optimizer_Comparison.md) | SGD vs Decay vs Momentum vs AdaGrad vs RMSProp vs Adam |
| [Regularisation Comparison](dashboards/Regularization_Comparison.md) | None vs L1 vs L2 vs Dropout vs combined strategies |

---

## Projects

| # | Project | Task | Result |
|:---:|---|---|---|
| 1 | [MNIST from scratch](projects/01-mnist-from-scratch/README.md) | 10-class classification | ~97% test accuracy |
| 2 | [Binary classifier on two-moons](projects/02-binary-classifier/README.md) | binary classification | ~98.5% test accuracy with sigmoid + BCE |
| 3 | [Fashion-MNIST](projects/03-fashion-mnist/README.md) | 10-class classification | ~89% test accuracy; the shirt cluster is hard |
| 4 | [California housing regression](projects/04-california-housing-regression/README.md) | regression | R² ≈ 0.78, RMSE ~$58k |

---

*If you can write a Python loop and do arithmetic, you can follow along. The mathematics is introduced exactly when it is needed, never before.*
