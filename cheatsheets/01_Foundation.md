# Cheat Sheet 1 — Foundation (Parts 1–5)

*Quick-reference flashcards for neurons, layers, NumPy operations, and data representation.*

---

## 🧠 Single Neuron

| Concept | Formula / Code |
|---|---|
| Output | $z = \sum_{i} w_i x_i + b$ |
| Code | `output = np.dot(weights, inputs) + bias` |
| Inputs | Feature vector, e.g. `[1.0, 2.0, 3.0, 2.5]` |
| Weights | One per input, learned during training |
| Bias | One per neuron, shifts the decision boundary |

**Key intuition:** A neuron computes a *weighted sum* of its inputs plus a bias — essentially a linear function.

---

## 📐 Layer of Neurons

| Concept | Detail |
|---|---|
| Layer output | Each neuron produces one scalar → layer produces a vector |
| Weight matrix | Shape `(n_inputs, n_neurons)` |
| Bias vector | Shape `(1, n_neurons)` |
| Batch forward | `output = np.dot(X, W) + b` where `X` is `(batch, n_inputs)` |
| Output shape | `(batch_size, n_neurons)` |

**Remember:** Matrix multiply handles all neurons simultaneously — no loops needed.

---

## 🔢 NumPy Essentials

```python
# Dot product (1D)
np.dot([1, 2, 3], [4, 5, 6])        # → 32

# Matrix multiply (2D)
np.dot(X, W)                         # (N, in) × (in, out) → (N, out)

# Random init
weights = 0.01 * np.random.randn(n_in, n_out)
biases  = np.zeros((1, n_out))

# Shapes
X.shape        # → (batch, features)
W.shape        # → (features, neurons)
(X @ W).shape  # → (batch, neurons)
```

---

## 📊 Spiral Dataset

```python
# Generate 3-class spiral data
X, y = spiral_data(samples=100, classes=3)
# X.shape → (300, 2)   — 2D coordinates
# y.shape → (300,)     — class labels 0, 1, 2
```

| Parameter | Meaning |
|---|---|
| `samples` | Points per class |
| `classes` | Number of spirals |
| Total points | `samples × classes` |

---

## 🏗️ Dense Layer Class

```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs                        # save for backward
        self.output = np.dot(inputs, self.weights) + self.biases
```

**Why `0.01 *`?** Small initial weights prevent exploding outputs in the forward pass.

---

## 🔑 Summation & Broadcasting Rules

| Operation | Axis 0 (columns) | Axis 1 (rows) | No axis |
|---|---|---|---|
| `np.sum(X, axis=0)` | Sum each column → `(n_col,)` | — | — |
| `np.sum(X, axis=1)` | — | Sum each row → `(n_row,)` | — |
| `np.sum(X)` | — | — | Total scalar |
| `keepdims=True` | `(1, n_col)` | `(n_row, 1)` | `(1, 1)` |

**Broadcasting rule:** Shapes are compatible when, for each dimension, either they're equal or one of them is 1.

```
(5, 3) + (3,)   → ✅  (3,) broadcasts to (1, 3) → (5, 3)
(5, 3) + (5, 1) → ✅  (5, 1) broadcasts to (5, 3)
(5, 3) + (4,)   → ❌  3 ≠ 4
```

---

## 🎯 Quick Self-Test

1. What shape is the output of `np.dot(X, W) + b` if `X` is `(32, 4)`, `W` is `(4, 8)`, `b` is `(1, 8)`?
2. Why do we save `self.inputs` in `forward()`?
3. What happens if you initialize weights with `np.random.randn` (no `0.01` multiplier)?

<details>
<summary>Answers</summary>

1. `(32, 8)` — batch of 32, each with 8 neuron outputs
2. We need them for the backward pass to compute weight gradients: $\partial L / \partial \mathbf{W} = \mathbf{X}^{T} \cdot \text{dvalues}$
3. Outputs can be very large, causing softmax to produce extreme (near-0 or near-1) probabilities, leading to high loss and unstable training

</details>

---

[← Back to Index](../INDEX.md)
