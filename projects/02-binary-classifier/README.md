# Project 02 · Binary classifier on the two-moons dataset

> **TL;DR.** [Post 34](../../posts/34-sigmoid-and-binary-cross-entropy/) introduces **sigmoid + binary cross-entropy**, the binary counterpart to the softmax + categorical cross-entropy used through most of the series. This project applies it to a synthetic 2-D dataset (`make_moons`) that can be plotted on a single chart. A two-hidden-layer 16-unit network trained with Adam learns the decision boundary in ~2000 epochs, reaching > 98% test accuracy with a clean curve between the moons. The math change from softmax+CCE is mostly cosmetic — same chain-rule shortcut, same combined-class trick from [post 19](../../posts/19-softmax-derivatives-and-the-combined-backward-pass/) — but the geometry is dead obvious because the input space is 2-D.

---

## What this project demonstrates

- **Sigmoid + binary cross-entropy** as the natural 2-class analogue of softmax + CCE: one output neuron, one threshold, one log-loss term.
- The same combined-derivative trick from post 19 carries over: `d(loss)/d(logit) = (sigmoid(logit) − y) / N`, no division by the activation.
- A 2-D dataset means the trained decision boundary can be sampled on a grid and plotted directly — see `evaluate.py`, which dumps a 200 × 200 `decision_grid.npz` for any plotter to consume.

## File layout

```
02-binary-classifier/
├── README.md
├── requirements.txt    ← numpy only
├── nn.py               ← Dense, ReLU, Sigmoid, combined Sigmoid+BCE, Adam
├── data.py             ← make_moons (numpy-only) + train_test_split
├── train.py            ← full-batch training loop, 2000 epochs
└── evaluate.py         ← test metrics + 200x200 decision grid
```

## Quick start

```bash
cd projects/02-binary-classifier
pip install -r requirements.txt
python train.py            # ~5 sec on a modern laptop CPU
python evaluate.py         # writes decision_grid.npz alongside the checkpoint
```

---

## 1. Why two moons?

The spiral dataset from the lectures had three classes; for a *binary* problem we want exactly two. The classic choice in textbooks is **two moons**:

- Two interleaved half-rings in the (x, y) plane.
- Each ring is a different class.
- Light Gaussian noise around each ring so the data is not perfectly separable.

The dataset is 2-D (visualisable on a single chart), small (1000 samples by default), and non-linearly separable (no straight line can split the two moons cleanly). A linear classifier would fail; a small MLP succeeds.

`data.py` implements `make_moons` in pure NumPy so the project has no external dependency on scikit-learn. The output format matches `sklearn.datasets.make_moons` exactly so the two are interchangeable if you prefer the sklearn version.

## 2. From softmax+CCE to sigmoid+BCE

Multi-class with softmax + categorical cross-entropy (the series default) uses **K output neurons** for K classes:

```
logits  (N, K) → softmax → probabilities (N, K) → CCE against one-hot y
```

Binary with sigmoid + binary cross-entropy uses **1 output neuron** for 2 classes:

```
logits  (N, 1) → sigmoid → probability of class 1 (N, 1) → BCE against scalar y
```

The math equivalence is exact when K = 2: a two-class softmax is just a sigmoid in disguise. The reason to special-case it is **efficiency** (one neuron and one log term instead of two) and **clarity** (the sigmoid output is directly "probability of class 1", no argmax needed).

### 2.1 Sigmoid

The activation:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

`nn.Activation_Sigmoid` implements this with a numerically stable form that avoids overflow when `z` is large negative:

```python
out[z >= 0]  = 1 / (1 + exp(-z))           # safe; exp(-z) stays in [0, 1]
out[z <  0]  = exp(z) / (1 + exp(z))       # safe; exp(z) stays in [0, 1]
```

The derivative is the textbook $\sigma(z)(1 - \sigma(z))$.

### 2.2 Binary cross-entropy loss

For target $y \in \{0, 1\}$ and predicted probability $\hat{y} \in (0, 1)$:

$$L = -\left[\, y \log \hat{y} + (1 - y) \log(1 - \hat{y}) \,\right]$$

Averaging over the batch gives the scalar loss.

### 2.3 The combined trick

Computing the gradient of $L$ with respect to the logits $z$ separately involves a division by $\sigma(z)(1 - \sigma(z))$ that can blow up when the prediction is very confident. The combined sigmoid + BCE derivative simplifies the whole chain:

$$\frac{\partial L}{\partial z} = \sigma(z) - y$$

Identical in spirit to the softmax + CCE shortcut from [post 19](../../posts/19-softmax-derivatives-and-the-combined-backward-pass/). `nn.Activation_Sigmoid_Loss_BinaryCrossentropy` packages the forward and the simplified backward into a single class, exactly the same pattern as `Activation_Softmax_Loss_CategoricalCrossentropy`.

The training loop in `train.py` therefore just calls:

```python
loss = loss_act.forward(logits, y_batch)
loss_act.backward(loss_act.output, y_batch)
# loss_act.dinputs now has shape (N, 1) and contains (sigmoid(z) - y) / N
```

## 3. Architecture

```
input (2) → Dense(2, 16) → ReLU → Dense(16, 16) → ReLU → Dense(16, 1) → Sigmoid + BCE
```

Parameter count:

| Layer | Weights | Biases |
|---|:---:|:---:|
| Dense(2, 16)  | 32  | 16 |
| Dense(16, 16) | 256 | 16 |
| Dense(16, 1)  | 16  | 1  |
| **Total** |  | **337** |

337 parameters is tiny — roughly 350× smaller than project 01's MNIST model. The moons dataset is correspondingly simpler; a 16-unit hidden layer is enough to bend the decision boundary into the right shape.

## 4. Training and result

`train.py` runs 2000 epochs of full-batch Adam (lr=0.01) on 800 training points. The remaining 200 points form the test set. Mini-batching is unnecessary at this scale; a single matrix multiply per epoch is fast enough on CPU.

Representative final metrics:

| Split | Loss | Accuracy |
|---|:---:|:---:|
| Training (800) | ~0.02 | ~99.9% |
| Test (200) | ~0.04 | ~98.5% |

The 1.4-percentage-point train/test gap is the kind of mild overfit that small networks on small datasets exhibit. Adding the `weight_regularizer_l2` parameter to the `Layer_Dense` constructor (already set to 1e-4 in `train.py`) keeps the gap from blowing out.

## 5. The decision boundary

`evaluate.py` samples the model's probability on a 200 × 200 grid covering `x ∈ [-1.5, 2.5]` and `y ∈ [-1.0, 1.5]`, dumps it to `decision_grid.npz`, and prints a quick "fraction of cells in the boundary band" sanity check. To plot the boundary in a notebook:

```python
import numpy as np
import matplotlib.pyplot as plt

g = np.load("decision_grid.npz")
plt.contourf(g["XX"], g["YY"], g["probs"], levels=20, cmap="RdBu_r", alpha=0.5)
plt.contour(g["XX"], g["YY"], g["probs"], levels=[0.5], colors="black", linewidths=2)
plt.scatter(*g["X_train"].T, c=g["y_train"], cmap="RdBu_r", edgecolor="k")
plt.show()
```

The qualitative shape (an S-curve between the moons) is in this project's hero diagram. The boundary is the **locus of points where the network's sigmoid output equals 0.5** — every prediction above 0.5 is class 1, every prediction below 0.5 is class 0.

## 6. Stretch goals

| Goal | Difficulty | Hint |
|---|---|---|
| Plot the boundary in matplotlib | easy | The snippet in §5 |
| Higher noise (σ = 0.5) | easy | Pass `--noise 0.5` to `train.py`; observe where the boundary refuses to commit |
| Three-spiral classifier | medium | Reuse `spiral_data` from the lectures; replace sigmoid + BCE with softmax + CCE (back to project-01 style) |
| Add early stopping | medium | Track test loss every 50 epochs; halt when it plateaus |
| Mini-batch this with a 32-sample batch | easy | Loop over `range(0, len(X_train), 32)` inside the epoch loop |

## 7. Related lectures

| Lecture | Used here for |
|---|---|
| [Part 4 — Dense layer class](../../posts/04-dense-layer-class-and-spiral-data/) | `Layer_Dense` |
| [Part 6 — Activations](../../posts/06-activation-functions-relu-and-softmax/) | `Activation_ReLU` (sigmoid is the binary analogue of softmax) |
| [Part 19 — Softmax + cross-entropy combined](../../posts/19-softmax-derivatives-and-the-combined-backward-pass/) | The combined-derivative trick, applied to sigmoid + BCE here |
| [Part 34 — Sigmoid and binary cross-entropy](../../posts/34-sigmoid-and-binary-cross-entropy/) | `Activation_Sigmoid` and the combined sigmoid + BCE backward — the core technique of this project |
| [Part 27 — Adam](../../posts/27-adam-optimiser/) | `Optimizer_Adam` |
| [Part 30 — L1 / L2 regularisation](../../posts/30-l1-and-l2-regularisation/) | `weight_regularizer_l2` |

## 8. Common pitfalls

- **Treating `y` as one-hot.** Binary cross-entropy expects scalar `y ∈ {0, 1}`, not a 2-element one-hot vector. `nn.Activation_Sigmoid_Loss_BinaryCrossentropy` reshapes to `(N, 1)` internally.
- **Using softmax with a single output neuron.** Softmax over a 1-element vector always returns 1.0. Use sigmoid for 1-output binary tasks.
- **Non-numerically-stable sigmoid.** `1 / (1 + exp(-z))` overflows for very negative `z`. `nn.Activation_Sigmoid.forward` uses the conditional form that stays stable in both directions.
- **Threshold at 0.5 without checking class balance.** For balanced datasets (50/50 like moons), 0.5 is the right threshold. For imbalanced datasets, a different threshold often beats accuracy on the minority class.
- **Two separate sigmoid and BCE classes in the backward pass.** Use the combined class. Computing them separately invites the divide-by-σ(z)(1−σ(z)) numerical issue.

---

> *Project 02 of N. See [projects/README.md](../README.md) for the project index. The from-scratch series lives in [posts/](../../posts/).*
