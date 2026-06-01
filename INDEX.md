# Neural Networks from Scratch — Series Index & Learning Pathway

*35 posts that build a neural network from first principles, plus 4 hands-on projects that put it to work. Use this page as your home base: the top half is a guided learning path with progress tracking; the bottom half is a reference index of every resource in the repo.*

---

## How to use this guide

1. **Work through phases in order** — each phase builds directly on the last.
2. **Check off each row** as you complete it: blog → quiz → exercise.
3. **Don't skip the exercises** — reading alone won't build the intuition.
4. **Use the prerequisite check** at the start of each phase before diving in.
5. **Revisit earlier phases** if a later concept feels unclear — the [Concept Dependency Map](concept_dependency_map.md) shows exactly what to review.

---

## Phase 1: Foundation (Parts 1–5)

> **Goal:** Understand what a neuron and layer compute, and become fluent in NumPy.

**Prerequisite check:** Can you write a Python list, call a function, and use a `for` loop? If yes, you're ready.

| # | Blog | Topic | Done |
|---|---|---|:---:|
| 1 | [Part 1 — Neurons and Layers](posts/01-neurons-and-layers/index.md) | Coding a neuron, a layer, loops, and the first NumPy version | ☐ |
| 2 | [Part 2 — NumPy and the Dot Product](posts/02-numpy-and-the-dot-product/index.md) | Dot products, matrix multiplication, and batch processing | ☐ |
| 3 | [Part 3 — Stacking Layers and the Forward Pass](posts/03-stacking-layers-and-the-forward-pass/index.md) | Chaining multiple layers together | ☐ |
| 4 | [Part 4 — Dense Layer Class and Spiral Data](posts/04-dense-layer-class-and-spiral-data/index.md) | OOP structure, reusable layers, and non-linear data | ☐ |
| 5 | [Part 5 — Array Summation, keepdims, and Broadcasting](posts/05-array-summation-keepdims-and-broadcasting/index.md) | Shapes, axes, and broadcasting rules | ☐ |

**Complete the quiz and exercises** for each part above, then check the milestones:

- [ ] Code a neuron from scratch without looking at notes
- [ ] Predict the output shape of `np.dot(A, B)` for any A, B
- [ ] Explain why `keepdims=True` matters in one sentence
- [ ] Build a `Layer_Dense` class from memory

---

## Phase 2: Forward Pass Complete (Parts 6–9)

> **Goal:** Build and run an end-to-end classification network.

**Prerequisite check:** Can you explain what `np.dot(X, W) + b` computes and predict its output shape?

| # | Blog | Topic | Done |
|---|---|---|:---:|
| 6 | [Part 6 — Activation Functions: ReLU and Softmax](posts/06-activation-functions-relu-and-softmax/index.md) | Why non-linearity matters and how logits become probabilities | ☐ |
| 7 | [Part 7 — Coding the Complete Forward Pass](posts/07-coding-the-complete-forward-pass/index.md) | End-to-end forward pass with classes | ☐ |
| — | **Milestone:** Run the complete forward pass yourself in a notebook | | ☐ |
| 8 | [Part 8 — Loss: Categorical Cross-Entropy](posts/08-loss-categorical-cross-entropy/index.md) | Measuring how wrong the network is | ☐ |
| 9 | [Part 9 — Introduction to Optimisation](posts/09-introduction-to-optimisation/index.md) | Why random search fails and why gradients matter | ☐ |

**Phase 2 milestones:**

- [ ] Explain why stacking linear layers without activation is useless
- [ ] Hand-compute softmax for a 3-element vector
- [ ] Compute cross-entropy loss for a single prediction
- [ ] Explain why random search fails for optimisation

---

## Phase 3: Calculus Foundation (Parts 10–11)

> **Goal:** Understand derivatives, partial derivatives, and the chain rule.

**Prerequisite check:** Can you evaluate `f(x) = 3x²` at x = 2? No prior calculus needed — we start from zero.

| # | Blog | Topic | Done |
|---|---|---|:---:|
| 10 | [Part 10 — Derivatives, Partial Derivatives, and Gradients](posts/10-derivatives-partial-derivatives-and-gradients/index.md) | Calculus foundations for backpropagation | ☐ |
| 11 | [Part 11 — The Chain Rule](posts/11-the-chain-rule/index.md) | The core rule that makes backprop work | ☐ |

**Phase 3 milestones:**

- [ ] Differentiate $f(x) = 5x^3 + 2x$ using the power rule
- [ ] Compute partial derivatives of $f(x,y) = x^2 y + 3y$
- [ ] Apply the chain rule to $f(g(x))$ where $g(x) = 2x+1$ and $f(g) = g^2$

---

## Phase 4: Backpropagation (Parts 12–21)

> **Goal:** Understand and implement the complete backward pass.

**Prerequisite check:** Can you state the chain rule in symbols? This is the hardest phase — take your time.

| # | Blog | Topic | Done |
|---|---|---|:---:|
| 12 | [Part 12 — Backprop Through a Single Neuron](posts/12-backprop-through-a-single-neuron/index.md) | The first complete backward pass | ☐ |
| 13 | [Part 13 — Backprop Through a Layer](posts/13-backprop-through-a-layer/index.md) | Extending gradients to a full layer | ☐ |
| 14 | [Part 14 — Matrices in Backpropagation](posts/14-matrices-in-backpropagation/index.md) | Matrix-form gradients and batching | ☐ |
| 15 | [Part 15 — Gradients with Respect to Inputs](posts/15-gradients-with-respect-to-inputs/index.md) | Why `dinputs` matters for chaining layers | ☐ |
| — | **Milestone:** Write down the 3 backprop formulas from memory | | ☐ |
| 16 | [Part 16 — Coding Backpropagation](posts/16-coding-backpropagation/index.md) | Implementing `backward()` for dense layers | ☐ |
| 17 | [Part 17 — Backprop Through Activation Functions](posts/17-backpropagation-through-activation-functions/index.md) | ReLU backward pass and Softmax preview | ☐ |
| 18 | [Part 18 — Backprop Through the Loss Function](posts/18-backpropagation-through-the-loss-function/index.md) | Cross-entropy backward pass | ☐ |
| 19 | [Part 19 — Softmax Derivatives and the Combined Backward Pass](posts/19-softmax-derivatives-and-the-combined-backward-pass/index.md) | The Softmax + cross-entropy shortcut | ☐ |
| 20 | [Part 20 — Assembling Full Backpropagation](posts/20-assembling-full-backpropagation/index.md) | Connecting every backward block | ☐ |
| 21 | [Part 21 — Coding the Full Backpropagation](posts/21-coding-the-full-backpropagation/index.md) | End-to-end code for the full backward pipeline | ☐ |
| — | **Milestone:** Run full forward + backward pass in notebook | | ☐ |
| — | Run the [Gradient Checker](gradient_checking.md) to verify your implementation | | ☐ |

**Phase 4 milestones:**

- [ ] Explain backpropagation in one paragraph without using code
- [ ] Write `backward()` for `Layer_Dense` from memory
- [ ] Explain why `dinputs` is the "glue" between layers
- [ ] State the combined Softmax + cross-entropy gradient: $\hat{y} - y$

---

## Phase 5: Optimisers (Parts 22–27)

> **Goal:** Build 5 progressively better optimisers and watch training actually work.

**Prerequisite check:** Can you run a forward + backward pass and inspect gradients? This phase is rewarding — you finally see the network learn.

| # | Blog | Topic | Done |
|---|---|---|:---:|
| 22 | [Part 22 — Gradient-Descent Optimiser](posts/22-gradient-descent-optimiser/index.md) | Vanilla SGD; the 67% plateau on spiral data | ☐ |
| — | **Milestone:** Train a network and watch loss decrease | | ☐ |
| 23 | [Part 23 — Learning-Rate Decay](posts/23-learning-rate-decay/index.md) | $\alpha(t) = \alpha_0/(1 + d \cdot t)$; the three-method optimiser contract | ☐ |
| 24 | [Part 24 — Momentum](posts/24-momentum/index.md) | Velocity, vector cancellation, $\beta = 0.9$ | ☐ |
| 25 | [Part 25 — AdaGrad](posts/25-adagrad/index.md) | Per-parameter cache; the dying-rate problem | ☐ |
| 26 | [Part 26 — RMSProp](posts/26-rmsprop/index.md) | EMA cache; bounded denominator | ☐ |
| 27 | [Part 27 — Adam](posts/27-adam-optimizer/index.md) | Momentum + RMSProp + bias correction; the modern default | ☐ |
| — | Review the [Optimiser Comparison Dashboard](dashboards/Optimizer_Comparison.md) | | ☐ |

**Phase 5 milestones:**

- [ ] Explain what problem each optimiser solves over the previous one
- [ ] State the Adam update rule from memory
- [ ] Train a network to >90% accuracy on spiral data

---

## Phase 6: Generalisation and Regularisation (Parts 28–31)

> **Goal:** Make models that work on unseen data.

**Prerequisite check:** Can you train a model with Adam and reach >90% accuracy on training data?

| # | Blog | Topic | Done |
|---|---|---|:---:|
| 28 | [Part 28 — Generalisation and Testing](posts/28-generalization-and-testing/index.md) | Train vs test, the overfitting gap | ☐ |
| 29 | [Part 29 — Validation and Hyperparameter Tuning](posts/29-validation-and-hyperparameter-tuning/index.md) | Three-way split, k-fold CV, data leakage | ☐ |
| 30 | [Part 30 — L1 and L2 Regularisation](posts/30-l1-and-l2-regularisation/index.md) | Weight-magnitude penalties; gradient shapes | ☐ |
| 31 | [Part 31 — Dropout](posts/31-dropout/index.md) | Random activation masking; train/test switch | ☐ |
| — | Review the [Regularisation Comparison Dashboard](dashboards/Regularization_Comparison.md) | | ☐ |

**Phase 6 milestones:**

- [ ] Explain overfitting to a non-technical person
- [ ] Add L2 regularisation to your network's forward and backward passes
- [ ] Implement Dropout with correct train/test behaviour

---

## Phase 7: Practical training and extensions (Parts 32–35)

> **Goal:** Fill in the topics the core series glossed over and chart what comes next.

**Prerequisite check:** Phases 1–6 complete. These four posts polish the foundation and point past it.

| # | Blog | Topic | Done |
|---|---|---|:---:|
| 32 | [Part 32 — Mini-batching](posts/32-mini-batching/index.md) | The two-loop training pattern: epoch × batch; why mini-batch beats full-batch and pure SGD | ☐ |
| 33 | [Part 33 — Weight initialisation](posts/33-weight-initialisation/index.md) | The `0.01 * randn` trap; Xavier / Glorot; He / Kaiming for ReLU | ☐ |
| 34 | [Part 34 — Sigmoid and binary cross-entropy](posts/34-sigmoid-and-binary-cross-entropy/index.md) | The binary counterpart to softmax + CCE; combined-derivative shortcut | ☐ |
| 35 | [Part 35 — What to read after this series](posts/35-whats-next/index.md) | Structured reading list: conv, RNN, transformers, batchnorm, RL, diffusion | ☐ |

**Phase 7 milestones:**

- [ ] Re-implement [Project 03 — Fashion-MNIST](projects/03-fashion-mnist/README.md) with `init="he"` and compare convergence
- [ ] Pick one item from Part 35's reading list and start it

---

## Projects

> **Goal:** Apply everything you've built to real datasets. Each project is self-contained: `posts/NN-slug/index.md` for the theory, `projects/NN-slug/` for the runnable code + README. See [projects/README.md](projects/README.md) for the convention.

| # | Project | Task | Hero result | Done |
|---|---|---|---|:---:|
| 1 | [MNIST from scratch](projects/01-mnist-from-scratch/README.md) | 10-class classification | ~97% test accuracy with Adam + L2 + Dropout | ☐ |
| 2 | [Binary classifier on two-moons](projects/02-binary-classifier/README.md) | binary classification | ~98.5% test accuracy with sigmoid + BCE | ☐ |
| 3 | [Fashion-MNIST](projects/03-fashion-mnist/README.md) | 10-class classification | ~89% test accuracy; the "shirt cluster" confusion | ☐ |
| 4 | [California housing regression](projects/04-california-housing-regression/README.md) | regression | R² ≈ 0.78, RMSE ~$58k with MSE loss | ☐ |

---

## Estimated time

| Phase | Parts | Estimate |
|---|:---:|---|
| 1. Foundation | 1–5 | One focused weekend |
| 2. Forward Pass | 6–9 | 1–2 days |
| 3. Calculus | 10–11 | 1 day |
| 4. Backpropagation | 12–21 | 1–2 weeks (the core) |
| 5. Optimisers | 22–27 | 3–4 days |
| 6. Regularisation | 28–31 | 2–3 days |
| 7. Practical training | 32–35 | 1–2 days |
| Projects | 4 hands-on builds (MNIST, moons, Fashion-MNIST, regression) | 3–5 days |

**Total:** ~3–4 weeks at a steady pace, or ~2 weeks intensive.

---

## Concept dependency map

*Use this when a topic feels shaky — trace back to its prerequisites.*

- **→** means "is required by" (read left to right)
- If concept B depends on concept A, master A before tackling B

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    PHASE 1: FOUNDATION                  │
                    │                                                         │
                    │   [Part 1: Neurons]──→[Part 2: NumPy Dot Product]       │
                    │        │                       │                        │
                    │        └───────┬───────────────┘                        │
                    │                ↓                                        │
                    │   [Part 3: Stacking Layers]                             │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 4: Dense Layer Class] ←── Python OOP basics     │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 5: Broadcasting & keepdims]                     │
                    └────────────────┼────────────────────────────────────────┘
                                     │
                    ┌────────────────↓────────────────────────────────────────┐
                    │              PHASE 2: FORWARD PASS                      │
                    │                                                         │
                    │   [Part 6: ReLU & Softmax] ←── Part 5 (keepdims)       │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 7: Complete Forward Pass]                       │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 8: Cross-Entropy Loss] ←── Part 5 (axis, log)  │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 9: Intro to Optimisation]                       │
                    └────────────────┼────────────────────────────────────────┘
                                     │
                    ┌────────────────↓────────────────────────────────────────┐
                    │           PHASE 3: CALCULUS                             │
                    │                                                         │
                    │   [Part 10: Derivatives & Gradients]                    │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 11: Chain Rule]                                 │
                    └────────────────┼────────────────────────────────────────┘
                                     │
          ┌──────────────────────────↓──────────────────────────────────────────────┐
          │                    PHASE 4: BACKPROPAGATION                              │
          │                                                                          │
          │   [Part 12: Backprop Single Neuron] ←── Parts 10-11 (chain rule)        │
          │                │                                                         │
          │                ↓                                                         │
          │   [Part 13: Backprop Through a Layer]                                    │
          │                │                                                         │
          │         ┌──────┴──────┐                                                  │
          │         ↓             ↓                                                  │
          │   [Part 14: dW]  [Part 15: dX]   ← Matrix forms of gradients            │
          │         │             │                                                  │
          │         └──────┬──────┘                                                  │
          │                ↓                                                         │
          │   [Part 16: Coding Backprop]                                             │
          │                │                                                         │
          │         ┌──────┴──────────────────┐                                      │
          │         ↓                         ↓                                      │
          │   [Part 17: Activation Backprop] [Part 18: Loss Backprop]                │
          │         │                         │                                      │
          │         └─────────┬───────────────┘                                      │
          │                   ↓                                                      │
          │   [Part 19: Softmax + CE Combined] ←── Part 6 (softmax forward)         │
          │                   │                                                      │
          │                   ↓                                                      │
          │   [Part 20: Assembling Full Backprop]                                    │
          │                   │                                                      │
          │                   ↓                                                      │
          │   [Part 21: Coding Full Backprop]                                        │
          └───────────────────┼──────────────────────────────────────────────────────┘
                              │
          ┌───────────────────↓──────────────────────────────────────────────────────┐
          │                    PHASE 5: OPTIMISERS                                    │
          │                                                                           │
          │   [Part 22: Gradient Descent] ←── Part 21 (full backprop code)           │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 23: Learning Rate Decay]                                          │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 24: Momentum]                                                     │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 25: AdaGrad] ←── new idea (per-parameter rates)                  │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 26: RMSProp] ←── fixes AdaGrad's problem                         │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 27: Adam] ←── combines Parts 24 + 26                             │
          └───────────────────┼──────────────────────────────────────────────────────┘
                              │
          ┌───────────────────↓──────────────────────────────────────────────────────┐
          │              PHASE 6: GENERALISATION & REGULARISATION                     │
          │                                                                           │
          │   [Part 28: Generalisation & Testing]                                     │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 29: Validation & K-Fold]                                          │
          │                │                                                          │
          │         ┌──────┴──────┐                                                   │
          │         ↓             ↓                                                   │
          │   [Part 30: L1/L2]  [Part 31: Dropout]                                   │
          └──────────────────────────────────────────────────────────────────────────┘
```

### Dependency lookup table

Find the part you're stuck on, then review what's listed.

| Stuck on… | Review these first |
|---|---|
| [Part 2](posts/02-numpy-and-the-dot-product/index.md) — Dot products | Part 1 (neurons, weighted sums) |
| [Part 3](posts/03-stacking-layers-and-the-forward-pass/index.md) — Stacking layers | Parts 1–2 (single layer, transpose) |
| [Part 5](posts/05-array-summation-keepdims-and-broadcasting/index.md) — Broadcasting | Part 2 (array shapes) |
| [Part 6](posts/06-activation-functions-relu-and-softmax/index.md) — Softmax | Part 5 (axis, keepdims, broadcasting) |
| [Part 8](posts/08-loss-categorical-cross-entropy/index.md) — Cross-entropy | Part 6 (softmax outputs), Part 5 (indexing) |
| [Part 12](posts/12-backprop-through-a-single-neuron/index.md) — Backprop | Parts 10–11 (derivatives, chain rule) |
| [Part 14](posts/14-matrices-in-backpropagation/index.md) — Matrix backprop | Parts 2, 12–13 (matrix math, single-neuron backprop) |
| [Part 16](posts/16-coding-backpropagation/index.md) — Coding backward | Parts 14–15 (formulas), Part 4 (class structure) |
| [Part 19](posts/19-softmax-derivatives-and-the-combined-backward-pass/index.md) — Softmax backward | Part 6 (softmax forward), Part 11 (chain rule) |
| [Part 22](posts/22-gradient-descent-optimiser/index.md) — Gradient descent | Part 21 (full backprop code) |
| [Part 24](posts/24-momentum/index.md) — Momentum | Parts 22–23 (basic SGD, learning rate decay) |
| [Part 27](posts/27-adam-optimizer/index.md) — Adam | Parts 24 (momentum) + 26 (RMSProp) |
| [Part 30](posts/30-l1-and-l2-regularisation/index.md) — L1/L2 | Part 16 (backward method), Part 28 (overfitting) |
| [Part 31](posts/31-dropout/index.md) — Dropout | Part 28 (overfitting), Part 5 (broadcasting) |

### Three pillars

The 31 parts teach three pillars that converge into a working network:

```
   FORWARD PASS          BACKWARD PASS           TRAINING
   (Parts 1–8)           (Parts 10–21)          (Parts 22–31)
        │                      │                      │
   Build the              Understand how         Make weights
   computation            to compute              better over
   pipeline               gradients               time
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                        Working Neural
                         Network ✓
```

---

## Reference index

### Supplementary resources

| Resource | When to use |
|---|---|
| [Glossary](glossary.md) | When you encounter an unfamiliar term |
| [Notation Guide](notation_guide.md) | When a symbol, shape, or optimiser variable is unclear |
| [Cumulative Notebook](cumulative_notebook.ipynb) | Running code from any lecture interactively |
| [Exercises](exercises.md) | Hands-on practice for every lecture |
| [Quizzes](quizzes.md) | Testing your comprehension with explanations |
| [Gradient Checking](gradient_checking.md) | Numerically verify your backprop implementation |
| [Common Pitfalls](common_pitfalls.md) | Most frequent mistakes and how to fix them |
| [Softmax Backward Appendix](appendix_softmax_combined_backward.md) | Full derivation of the combined backward pass |

### Cheat sheets

| Sheet | Covers |
|---|---|
| [Foundation](cheatsheets/01_Foundation.md) | Neurons, layers, NumPy, broadcasting (Parts 1–5) |
| [Activations & Loss](cheatsheets/02_Activations_Loss_Forward.md) | ReLU, Softmax, cross-entropy, forward pass (Parts 6–9) |
| [Calculus & Backprop](cheatsheets/03_Calculus_Backpropagation.md) | Derivatives, chain rule, and full backward pass (Parts 10–21) |
| [Optimisers & Regularisation](cheatsheets/04_Optimizers_Regularization.md) | SGD through Adam, L1/L2, and Dropout (Parts 22–31) |

### Comparison dashboards

| Dashboard | Description |
|---|---|
| [Optimiser Comparison](dashboards/Optimizer_Comparison.md) | SGD vs Decay vs Momentum vs AdaGrad vs RMSProp vs Adam |
| [Regularisation Comparison](dashboards/Regularization_Comparison.md) | None vs L1 vs L2 vs Dropout vs combined strategies |

---

*Each post stands alone while fitting into a coherent path built from first principles. Every lecture is `posts/NN-slug/index.md` with its hero diagram at `posts/NN-slug/diagrams/`.*
