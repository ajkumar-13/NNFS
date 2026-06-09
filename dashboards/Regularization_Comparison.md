# Regularization Comparison Dashboard

*Side-by-side comparison of every regularization technique built in this series.*

---

## The Techniques

| # | Method | Introduced In | Mechanism |
|---|---|---|---|
| 1 | No regularization | — | Baseline — unregularized model |
| 2 | L1 Regularization | [Part 30](../posts/30-l1-and-l2-regularisation/index.md) | Adds $\lambda \sum |w|$ to loss |
| 3 | L2 Regularization | [Part 30](../posts/30-l1-and-l2-regularisation/index.md) | Adds $\lambda \sum w^2$ to loss |
| 4 | L1 + L2 | [Part 30](../posts/30-l1-and-l2-regularisation/index.md) | Both penalties combined |
| 5 | Dropout | [Part 31](../posts/31-dropout/index.md) | Randomly zero out neurons during training |
| 6 | L2 + Dropout | Parts 30–31 | Combined approach |

---

## Test Setup

```python
# Same architecture for all experiments
# Dense(2, 64) → ReLU → [Dropout?] → Dense(64, 3) → Softmax
# Optimizer: Adam (lr=0.05, decay=1e-5); regularisation applied to dense1 only
# Train: spiral_data(samples=100, classes=3) → 300 samples
# Test:  spiral_data(samples=100, classes=3) → 300 samples (different seed)
# Epochs: 10,000
```

---

## Regularization Matrix

Verified by [`verify/regularization_results.py`](../verify/regularization_results.py) (Adam lr=0.05, decay=1e-5; regularisation on `dense1` weights **and** biases; 100 train + 100 fresh-test samples; 10k epochs; train accuracy measured with dropout off):

| Configuration | $\lambda_1$ (L1) | $\lambda_2$ (L2) | Dropout Rate | Train Acc | Test Acc | Gap |
|---|---|---|---|---|---|---|
| No reg | 0 | 0 | 0 | 95.3% | 78.7% | 16.6 |
| L1 only | 5e-4 | 0 | 0 | 96.3% | 80.3% | 16.0 |
| **L2 only** | 0 | 5e-4 | 0 | 95.3% | **84.0%** | 11.3 |
| L1 + L2 | 5e-4 | 5e-4 | 0 | 82.0% | 80.3% | **1.7** |
| Dropout only | 0 | 0 | 0.1 | 72.0% | 64.7% | 7.3 |
| L2 + Dropout | 0 | 5e-4 | 0.1 | 79.0% | 69.3% | 9.7 |

**The key lessons.** (1) **Weight-decay regularisation helps here:** L2 alone gives the best test accuracy (78.7% → 84.0%), and L1 and L1+L2 also lift it (~80%) — exactly the "improves the gap without harming training" behaviour Part 30 describes. (2) **L1+L2 (Elastic Net) gives the smallest gap (1.7) with strong test accuracy** — the best generalisation *balance*. (3) **Dropout over-regularises this small 64-neuron network** (test drops to ~65–69%): the layer is not wide enough, nor the dataset large enough, for dropping 10% of neurons every step to pay off. The lesson is to **match the regulariser to the problem** — light weight-decay suits this small spiral, while dropout shines on bigger, wider networks (Part 31 shows it working with 1000 samples/class).

---

## Visual Summary

### Generalization Gap (Train − Test)

*A small gap is good only if test accuracy is also high — compare with the chart below.*

```
Gap (%)
 17 ┤ ████████████████   No regularization
 16 ┤ ████████████████   L1 only
 11 ┤ ███████████        L2 only
 10 ┤ ██████████         L2 + Dropout
  7 ┤ ███████            Dropout only
  2 ┤ ██                 L1 + L2  ← smallest gap AND strong test acc
  0 ┤
    └──────────────────────────────────────────
```

### Test Accuracy (the metric that matters)

```
Test Accuracy (%)
 84 ┤ ████████████████████  L2 only  ← best
 80 ┤ ████████████████      L1 only / L1 + L2
 79 ┤ ███████████████       No regularization
 69 ┤ █████████             L2 + Dropout
 65 ┤ ███████               Dropout only
    └──────────────────────────────────────────
```

Weight-decay (L1, L2, Elastic Net) all beat the no-reg baseline on test accuracy; **L2 is best, and L1+L2 pairs a tiny gap with strong test accuracy**. The two dropout configs sit at the bottom — on a network this small, dropping 10% of neurons removes capacity the model needs.

---

## Detailed Analysis

### No Regularization (Baseline)
- **Train accuracy:** High (~95%) — the model fits the training data well
- **Test accuracy:** Lower (~79%)
- **Diagnosis:** *Mild* overfitting. The ~17-point gap shows some memorisation, but the model still generalises reasonably — this small spiral does not overfit catastrophically, which is why a light weight-decay penalty (L2) helps most while heavy dropout backfires below.

### L1 Regularization ($\lambda = 5 \times 10^{-4}$)
- **Effect:** Pushes weights toward exactly zero → creates sparsity
- **Result here:** Helps modestly — training stays high (~96%) and **test accuracy rises to ~80%** (from the no-reg ~79%). Its large train-test gap (~16) means it regularises less effectively than L2 here, but it does not hurt.
- **When to use:** When you suspect many features/neurons are irrelevant (feature selection effect).
- **Gradient:** $\partial R / \partial w = \lambda \cdot \text{sign}(w)$ — constant magnitude regardless of $w$

### L2 Regularization ($\lambda = 5 \times 10^{-4}$) — best test accuracy here
- **Effect:** Penalizes large weights quadratically → shrinks all weights toward zero
- **Result here:** Training stays ~95% and **test accuracy rises to ~84%** — the best of the six, and the largest single improvement over the no-reg baseline. Exactly the "improves the gap without harming training" behaviour Part 30 describes.
- **When to use:** Most common first choice. A gentle, well-behaved penalty.
- **Gradient:** $\partial R / \partial w = 2\lambda w$ — larger weights get stronger penalties

### L1 + L2 (Elastic Net) — best generalisation balance here
- **Combines:** Sparsity from L1 + shrinkage from L2
- **Result here:** Train ~82% / test ~80% — the **smallest train-test gap (~2)** of any config. Test accuracy is just below L2-alone, but the gap is far smaller, so it is the most balanced choice.
- **When to use:** When you want both effects and value a tight gap.

### Dropout (rate = 0.1)
- **Effect:** Randomly zeroes 10% of neuron outputs each forward pass
- **Result here:** Over-regularises this small network: training ~72%, **test ~65%**. The 64-neuron layer is not big enough to spare 10% every step on a dataset this small.
- **Key detail:** Inverted dropout scales outputs by $1/(1-p)$ during training, no scaling at test time
- **When to use:** Large networks with many parameters that overfit.

### L2 + Dropout (over-regularised here)
- **What happens:** L2 shrinks the weights and dropout zeroes neurons; on this small 64-neuron network the two together remove too much capacity.
- **Result here:** **Test accuracy drops to ~69%**, below the no-reg baseline — better than dropout-alone (~65%) but well short of L2-alone (~84%). This is Part 31's over-regularisation finding.
- **When to use:** When a model genuinely overfits (larger, wider networks; Part 31 shows this combination working at 1000 samples/class). For this small spiral, **light L2 alone** is the better choice.

---

## How to Choose

```
Start
  │
  ▼
Is your model overfitting?  ──No──→  No regularization needed
  │
  Yes
  │
  ▼
Try L2 first (λ = 1e-4 to 1e-3)
  │
  ▼
Still overfitting?  ──No──→  Done ✓
  │
  Yes
  │
  ▼
Add Dropout (0.1 to 0.3)
  │
  ▼
Still overfitting?  ──No──→  Done ✓
  │
  Yes
  │
  ▼
Reduce model size / Get more data
```

---

## Lambda Sweep Guide

| $\lambda$ | Effect | When |
|---|---|---|
| $0$ | No regularization | Not overfitting |
| $10^{-5}$ | Very mild | Large dataset, mild overfitting |
| $10^{-4}$ | Moderate | **Good starting point** |
| $10^{-3}$ | Strong | Small dataset, heavy overfitting |
| $10^{-2}$ | Very strong | Usually too aggressive (underfitting) |

**Tuning strategy:** Start at $10^{-4}$. If train-test gap > 10%, increase by 10×. If training accuracy is too low, decrease by 10×.

---

## Weight Distribution Comparison

```
No Regularization          L2 (λ=5e-4)             L1 (λ=5e-4)
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   ▁▃▅▇█▇▅▃▁     │    │    ▂▅▇█▇▅▂       │    │  █               │
│                  │    │                  │    │  █▃              │
│ spread: [-2, 2]  │    │ spread: [-0.5,0.5]│    │  █▃▁▁▁          │
│ |mean|: 0.45     │    │ |mean|: 0.15      │    │ many near 0     │
└──────────────────┘    └──────────────────┘    └──────────────────┘
  Wide, many large          Tight, small           Sparse, peaked
  weights                   weights                at zero
```

---

## Code: Run the Comparison

```python
import numpy as np
from nnfs.datasets import spiral_data

configs = [
    {"name": "No reg",      "l1": 0,    "l2": 0,    "dropout": 0},
    {"name": "L1",           "l1": 5e-4, "l2": 0,    "dropout": 0},
    {"name": "L2",           "l1": 0,    "l2": 5e-4, "dropout": 0},
    {"name": "L1+L2",        "l1": 5e-4, "l2": 5e-4, "dropout": 0},
    {"name": "Dropout",      "l1": 0,    "l2": 0,    "dropout": 0.1},
    {"name": "L2+Dropout",   "l1": 0,    "l2": 5e-4, "dropout": 0.1},
]

np.random.seed(0)
X_train, y_train = spiral_data(samples=100, classes=3)
np.random.seed(42)
X_test, y_test   = spiral_data(samples=100, classes=3)

for cfg in configs:
    np.random.seed(0)
    dense1 = Layer_Dense(2, 64,
                         weight_regularizer_l1=cfg["l1"],
                         weight_regularizer_l2=cfg["l2"])
    activation1 = Activation_ReLU()
    dropout1 = Layer_Dropout(cfg["dropout"]) if cfg["dropout"] > 0 else None
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.05, decay=1e-5)

    for epoch in range(10001):
        # Forward
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        if dropout1:
            dropout1.forward(activation1.output)
            dense2.forward(dropout1.output)
        else:
            dense2.forward(activation1.output)
        data_loss = loss_activation.forward(dense2.output, y_train)

        reg_loss = loss_activation.loss.regularization_loss(dense1) + \
                   loss_activation.loss.regularization_loss(dense2)
        loss = data_loss + reg_loss

        # Backward
        loss_activation.backward(loss_activation.output, y_train)
        dense2.backward(loss_activation.dinputs)
        if dropout1:
            dropout1.backward(dense2.dinputs)
            activation1.backward(dropout1.dinputs)
        else:
            activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    # Evaluate (dropout is disabled at inference: the eval path skips dropout1)
    def evaluate(Xe, ye):
        dense1.forward(Xe)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, ye)
        acc = np.mean(np.argmax(loss_activation.output, axis=1) == ye)
        return loss, acc

    train_loss, train_acc = evaluate(X_train, y_train)
    test_loss, test_acc = evaluate(X_test, y_test)
    print(f"{cfg['name']:15s} → train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")
```

---

*See also: [Optimizer Comparison](Optimizer_Comparison.md) | [Back to Index](../INDEX.md)*
