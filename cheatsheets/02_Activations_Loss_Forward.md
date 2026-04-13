# Cheat Sheet 2 — Activations, Loss & Forward Pass (Parts 6–9)

*Quick-reference flashcards for activation functions, loss computation, and the complete forward pipeline.*

---

## ⚡ ReLU (Rectified Linear Unit)

| Property | Value |
|---|---|
| Formula | $\text{ReLU}(x) = \max(0, x)$ |
| Output range | $[0, \infty)$ |
| Derivative | $1$ if $x > 0$, else $0$ |
| Dead neuron | A neuron stuck at 0 for all inputs (weights pushed too negative) |

```python
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
```

**Why ReLU?** Cheap to compute, introduces non-linearity, and avoids the vanishing-gradient problem that plagues sigmoid/tanh.

---

## 🌡️ Softmax

| Property | Value |
|---|---|
| Formula | $S_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ |
| Output range | $(0, 1)$ per element; sums to 1 |
| Use case | Final layer for multi-class classification |
| Trick | Subtract $\max(z)$ before exponentiation for numerical stability |

```python
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
```

**Temperature scaling:** Dividing inputs by $T$ before softmax controls sharpness:
- $T \to 0$: one-hot (confident)
- $T = 1$: standard
- $T \to \infty$: uniform (uncertain)

---

## 📉 Categorical Cross-Entropy Loss

| Property | Value |
|---|---|
| Formula | $L = -\sum_i y_i \ln(\hat{y}_i)$ (one-hot) |
| Simplified | $L = -\ln(\hat{y}_{\text{true class}})$ (integer labels) |
| Range | $[0, \infty)$; lower is better |
| Perfect prediction | $L = 0$ when $\hat{y}_k = 1$ for true class $k$ |
| Random guess (3 classes) | $L = -\ln(1/3) \approx 1.099$ |

```python
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Integer labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        # One-hot labels
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)
```

**Why clip?** `np.log(0)` = $-\infty$. Clipping to `[1e-7, 1-1e-7]` prevents numerical explosion.

---

## 🧮 Accuracy

```python
predictions = np.argmax(softmax_output, axis=1)
accuracy    = np.mean(predictions == y_true)
```

| Metric | Meaning |
|---|---|
| `np.argmax(..., axis=1)` | Index of highest probability per sample |
| Random accuracy (K classes) | $\approx 1/K$ |

---

## 🔗 Complete Forward Pass Pipeline

```
Input X                    (batch, 2)
  │
  ▼
Dense Layer 1              (batch, 64)     W₁: (2, 64),  b₁: (1, 64)
  │
  ▼
ReLU Activation            (batch, 64)     clips negatives to 0
  │
  ▼
Dense Layer 2              (batch, 3)      W₂: (64, 3),  b₂: (1, 3)
  │
  ▼
Softmax Activation         (batch, 3)      probabilities, sums to 1
  │
  ▼
Cross-Entropy Loss         scalar          −ln(ŷ_true_class)
```

**Shape tracking rule:** After `Dense(n_in, n_out)`, output shape changes from `(batch, n_in)` to `(batch, n_out)`. Activations preserve shape.

---

## 🧪 Optimization Preview

| Strategy | Description |
|---|---|
| Random search | Try many random weight sets, keep best — slow, doesn't scale |
| Random perturbation | Nudge weights randomly, keep if loss decreases — slightly better |
| Gradient descent | Follow the negative gradient — efficient, principled |

**Key insight:** The gradient tells you the direction of steepest *increase*. Negate it to decrease the loss.

---

## 🎯 Quick Self-Test

1. What is `softmax([100, 100, 100])`?
2. If your model has 5 output classes, what's the expected accuracy of random guessing?
3. Why can't we use ReLU as the output activation for classification?
4. What's the cross-entropy loss when the model predicts 50% confidence on the true class?

<details>
<summary>Answers</summary>

1. `[1/3, 1/3, 1/3]` — equal inputs → uniform probabilities
2. $1/5 = 20\%$
3. ReLU outputs are unbounded and don't sum to 1 — we need probabilities for classification
4. $-\ln(0.5) \approx 0.693$

</details>

---

[← Back to Index](../INDEX.md)
