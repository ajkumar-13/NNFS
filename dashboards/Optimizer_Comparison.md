# Optimizer Comparison Dashboard

*A side-by-side comparison of every optimizer built in this series, tested on the spiral dataset.*

---

## The Contenders

| # | Optimizer | Introduced In | Key Idea |
|---|---|---|---|
| 1 | SGD | [Part 22](../Lecture22/blog/Part22_Gradient_Descent_Optimizer.md) | Follow the negative gradient |
| 2 | SGD + Decay | [Part 23](../Lecture23/blog/Part23_Learning_Rate_Decay.md) | Reduce step size over time |
| 3 | SGD + Momentum | [Part 24](../Lecture24/blog/Part24_Momentum.md) | Accumulate velocity to smooth updates |
| 4 | AdaGrad | [Part 25](../Lecture25/blog/Part25_AdaGrad.md) | Per-parameter adaptive learning rate |
| 5 | RMSProp | [Part 26](../Lecture26/blog/Part26_RMSProp.md) | Fix AdaGrad's dying LR with EMA |
| 6 | Adam | [Part 27](../Lecture27/blog/Part27_Adam_Optimizer.md) | Momentum + RMSProp + bias correction |

---

## Test Setup

```python
import numpy as np
from nnfs.datasets import spiral_data

X, y = spiral_data(samples=100, classes=3)
# Architecture: Dense(2, 64) → ReLU → Dense(64, 3) → Softmax
# Loss: Categorical Cross-Entropy
# Epochs: 10,000
# Each optimizer uses its "best practice" defaults (see below)
```

---

## Hyperparameters Used

| Optimizer | Learning Rate | Decay | Momentum ($\beta$) | $\rho$ / $\beta_2$ | $\epsilon$ |
|---|---|---|---|---|---|
| SGD | 1.0 | — | — | — | — |
| SGD + Decay | 1.0 | 1e-3 | — | — | — |
| SGD + Momentum | 1.0 | 1e-3 | 0.9 | — | — |
| AdaGrad | 1.0 | 1e-4 | — | — | 1e-7 |
| RMSProp | 0.02 | 1e-5 | — | 0.999 | 1e-7 |
| Adam | 0.02 | 5e-7 | 0.9 | 0.999 | 1e-7 |

---

## Expected Results Summary

### Accuracy Over Training

```
Accuracy (%)
100 ┤
 95 ┤                          ╭──── Adam ≈ 97%
 90 ┤                     ╭────╯──── RMSProp ≈ 95%
 85 ┤               ╭─────╯───────── SGD+Mom ≈ 93%
 80 ┤          ╭────╯──────────────── SGD+Decay ≈ 88%
 75 ┤     ╭────╯
 70 ┤╭────╯
 65 ┤│
 60 ┤│
 55 ┤│──────────────────────────────── AdaGrad ≈ 86% (stalls)
 50 ┤│
 45 ┤│
 40 ┤│
 33 ┤├──── random baseline
    └┼────┼────┼────┼────┼────┼────┼────┼────┼────┼─
     0   1k   2k   3k   4k   5k   6k   7k   8k  10k
                          Epoch
```

### Loss Over Training

```
Loss
1.10 ┤├──── random baseline ≈ −ln(1/3) ≈ 1.099
1.00 ┤╲
0.90 ┤ ╲
0.80 ┤  ╲
0.70 ┤   ╲
0.60 ┤    ╲
0.50 ┤     ╲
0.40 ┤      ╲───── SGD (plateau ~0.4)
0.30 ┤       ╲──── SGD+Decay
0.20 ┤        ╲─── SGD+Momentum
0.15 ┤         ╲── AdaGrad (stalls ~0.25)
0.10 ┤          ╲─ RMSProp
0.05 ┤           ╲ Adam ≈ 0.05
0.00 ┤
     └┼────┼────┼────┼────┼────┼────┼────┼────┼────┼─
      0   1k   2k   3k   4k   5k   6k   7k   8k  10k
                          Epoch
```

---

## Detailed Analysis

### 1. Plain SGD
- **Pros:** Simple, no extra memory, easy to debug
- **Cons:** Sensitive to learning rate; oscillates in narrow valleys
- **When to use:** First baseline; educational clarity
- **Typical final accuracy:** ~85-88% at 10K epochs

### 2. SGD + Learning Rate Decay
- **Improvement over SGD:** Allows a high initial LR for fast early progress, then reduces oscillation
- **Sweet spot:** `decay=1e-3` halves the LR around epoch 1,000
- **Typical final accuracy:** ~88-90%

### 3. SGD + Momentum
- **Improvement over Decay:** Velocity accumulation smooths the path and helps escape shallow local minima
- **$\beta = 0.9$** is nearly universal — rarely needs tuning
- **Typical final accuracy:** ~91-94%

### 4. AdaGrad
- **Unique strength:** Per-parameter rates — frequent features get smaller updates, rare features get larger
- **Fatal flaw:** Cache monotonically grows → effective LR → 0 → learning stops
- **Typical behavior:** Fast early progress, then stalls around epoch 5-8K
- **Typical final accuracy:** ~85-88% (stalls before fully converging)

### 5. RMSProp
- **Fix over AdaGrad:** Exponential moving average prevents cache from growing without bound
- **Typical final accuracy:** ~93-96%
- **$\rho = 0.999$** provides a good balance of responsiveness and stability

### 6. Adam
- **Best of all worlds:** Combines momentum (direction) + RMSProp (magnitude) + bias correction (cold start)
- **Most forgiving:** Works well across a wide range of learning rates
- **Typical final accuracy:** ~95-97%
- **Go-to choice** for most practitioners when starting a new problem

---

## Head-to-Head Verdict

| Criterion | Winner | Runner-up |
|---|---|---|
| **Final accuracy** | Adam | RMSProp |
| **Convergence speed** | Adam | SGD+Momentum |
| **Simplicity** | SGD | SGD+Decay |
| **Robustness to LR** | Adam | RMSProp |
| **Memory efficiency** | SGD | SGD+Decay |
| **Sparse features** | AdaGrad | Adam |

---

## Recommended Starting Points

**Just learning?** Start with plain SGD to understand gradient descent intuitively, then add features one by one.

**Building a real model?** Use Adam with defaults: `lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7`.

**Fine-tuning?** Consider SGD + Momentum + Decay — often reaches better minima than Adam for long training runs.

---

## Code: Run the Comparison Yourself

```python
import numpy as np
from nnfs.datasets import spiral_data

np.random.seed(0)
X, y = spiral_data(samples=100, classes=3)

# --- Paste your Layer_Dense, Activation_ReLU, Softmax + CCE classes here ---
# --- Paste all 6 optimizer classes here ---

optimizers = {
    "SGD":          Optimizer_SGD(learning_rate=1.0),
    "SGD+Decay":    Optimizer_SGD(learning_rate=1.0, decay=1e-3),
    "SGD+Momentum": Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9),
    "AdaGrad":      Optimizer_Adagrad(learning_rate=1.0, decay=1e-4),
    "RMSProp":      Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999),
    "Adam":         Optimizer_Adam(learning_rate=0.02, decay=5e-7),
}

results = {}
for name, optimizer in optimizers.items():
    np.random.seed(0)  # same init for fair comparison
    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    history = {"loss": [], "acc": []}
    for epoch in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

        if epoch % 100 == 0:
            history["loss"].append(loss)
            history["acc"].append(accuracy)

    results[name] = history
    print(f"{name:20s} → final loss: {loss:.4f}, accuracy: {accuracy:.4f}")
```

---

*See also: [Regularization Comparison](Regularization_Comparison.md) | [Back to Index](../INDEX.md)*
