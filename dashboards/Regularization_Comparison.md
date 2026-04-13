# Regularization Comparison Dashboard

*Side-by-side comparison of every regularization technique built in this series.*

---

## The Techniques

| # | Method | Introduced In | Mechanism |
|---|---|---|---|
| 1 | No regularization | — | Baseline — unregularized model |
| 2 | L1 Regularization | [Part 30](../Lecture30/blog/Part30_L1_L2_Regularization.md) | Adds $\lambda \sum |w|$ to loss |
| 3 | L2 Regularization | [Part 30](../Lecture30/blog/Part30_L1_L2_Regularization.md) | Adds $\lambda \sum w^2$ to loss |
| 4 | L1 + L2 | [Part 30](../Lecture30/blog/Part30_L1_L2_Regularization.md) | Both penalties combined |
| 5 | Dropout | [Part 31](../Lecture31/blog/Part31_Dropout.md) | Randomly zero out neurons during training |
| 6 | L2 + Dropout | Parts 30–31 | Combined approach |

---

## Test Setup

```python
# Same architecture for all experiments
# Dense(2, 64) → ReLU → [Dropout?] → Dense(64, 3) → Softmax
# Optimizer: Adam (lr=0.02, decay=5e-7)
# Train: spiral_data(samples=100, classes=3) → 300 samples
# Test:  spiral_data(samples=100, classes=3) → 300 samples (different seed)
# Epochs: 10,000
```

---

## Regularization Matrix

| Configuration | $\lambda_1$ (L1) | $\lambda_2$ (L2) | Dropout Rate | Train Acc | Test Acc | Gap |
|---|---|---|---|---|---|---|
| **No reg** | 0 | 0 | 0 | ~98% | ~78% | ~20% |
| **L1 only** | 5e-4 | 0 | 0 | ~93% | ~82% | ~11% |
| **L2 only** | 0 | 5e-4 | 0 | ~94% | ~85% | ~9% |
| **L1 + L2** | 5e-4 | 5e-4 | 0 | ~91% | ~84% | ~7% |
| **Dropout only** | 0 | 0 | 0.1 | ~92% | ~84% | ~8% |
| **L2 + Dropout** | 0 | 5e-4 | 0.1 | ~90% | ~87% | ~3% |

---

## Visual Summary

### Generalization Gap (Train Accuracy − Test Accuracy)

```
Gap (%)
 20 ┤ ████████████████████  No regularization
    │
 11 ┤ ███████████           L1 only
    │
  9 ┤ █████████             L2 only
    │
  8 ┤ ████████              Dropout only
    │
  7 ┤ ███████               L1 + L2
    │
  3 ┤ ███                   L2 + Dropout ← best
    │
  0 ┤
    └──────────────────────────────────────────
```

### Test Accuracy

```
Test Accuracy (%)
 87 ┤                                  ████  L2 + Dropout ← best
 85 ┤                            ████        L2 only
 84 ┤                      ████  ████        L1+L2 / Dropout
 82 ┤                ████                    L1 only
 78 ┤ ████                                   No regularization
    └──────────────────────────────────────────
```

---

## Detailed Analysis

### No Regularization (Baseline)
- **Train accuracy:** Very high (~97-99%) — the model memorizes the training data
- **Test accuracy:** Significantly lower (~75-80%)
- **Diagnosis:** Classic overfitting. Train-test gap of ~20% tells us the model is too powerful for this dataset size.
- **Weight distribution:** Many large weights, wide spread

### L1 Regularization ($\lambda = 5 \times 10^{-4}$)
- **Effect:** Pushes weights toward exactly zero → creates sparsity
- **Result:** Some weights become zero or near-zero, reducing effective model capacity
- **When to use:** When you suspect many features/neurons are irrelevant (feature selection effect)
- **Gradient:** $\partial R / \partial w = \lambda \cdot \text{sign}(w)$ — constant magnitude regardless of $w$

### L2 Regularization ($\lambda = 5 \times 10^{-4}$)
- **Effect:** Penalizes large weights quadratically → shrinks all weights toward zero
- **Result:** Smoother weight distribution, no exact zeros
- **When to use:** Most common choice. Prevents any single weight from dominating
- **Gradient:** $\partial R / \partial w = 2\lambda w$ — larger weights get stronger penalties

### L1 + L2 (Elastic Net)
- **Combines:** Sparsity from L1 + shrinkage from L2
- **Result:** Good generalization with some weight pruning
- **When to use:** When you want the benefits of both — prune irrelevant connections while shrinking the rest

### Dropout (rate = 0.1)
- **Effect:** Randomly zeroes 10% of neuron outputs each forward pass
- **Result:** Forces neurons to be independently useful (no co-adaptation)
- **Key detail:** Inverted dropout scales outputs by $1/(1-p)$ during training, no scaling at test time
- **When to use:** Large networks with many parameters

### L2 + Dropout (Best Combination)
- **Why it works:** L2 keeps weights small globally; dropout prevents neuron co-adaptation locally
- **Result:** Smallest generalization gap (~3%), best test accuracy (~87%)
- **Recommended as default** for most from-scratch neural networks

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
    optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)

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

    # Evaluate
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    test_loss = loss_activation.forward(dense2.output, y_test)
    preds = np.argmax(loss_activation.output, axis=1)
    test_acc = np.mean(preds == y_test)

    train_preds = np.argmax(loss_activation.output, axis=1)
    print(f"{cfg['name']:15s} → train_loss: {loss:.4f}, test_acc: {test_acc:.4f}")
```

---

*See also: [Optimizer Comparison](Optimizer_Comparison.md) | [Back to Index](../INDEX.md)*
