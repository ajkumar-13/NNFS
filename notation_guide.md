# Notation Guide — Neural Networks from Scratch

*Use this when a symbol, shape, or optimizer variable starts to blur together.*

---

## 1. Forward-Pass Symbols

| Symbol | Meaning | Typical shape | Code name |
|---|---|---|---|
| $\mathbf{X}$ | Input batch | `(batch, features)` | `inputs`, `X` |
| $\mathbf{W}$ | Weights | `(features, neurons)` in the class-based code | `weights` |
| $\mathbf{b}$ | Biases | `(1, neurons)` | `biases` |
| $\mathbf{Z}$ | Pre-activation output | `(batch, neurons)` | `z`, dense layer output before activation |
| $\mathbf{A}$ | Activation output | `(batch, neurons)` | `output`, activation output |
| $\hat{\mathbf{y}}$ | Predicted probabilities | `(batch, classes)` | `y_pred`, `output`, `probs` |
| $\mathbf{y}$ | Ground-truth labels | `(batch,)` or `(batch, classes)` | `y_true`, `y` |
| $L$ | Loss | scalar | `loss` |

### Core forward equation

$$
\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}
$$

Then an activation is applied:

$$
\mathbf{A} = f(\mathbf{Z})
$$

---

## 2. Backward-Pass Symbols

| Symbol | Meaning | Code name |
|---|---|---|
| $\frac{\partial L}{\partial \mathbf{Z}}$ | Gradient of loss wrt pre-activation outputs | `dinputs` for an activation layer |
| $\frac{\partial L}{\partial \mathbf{W}}$ | Gradient of loss wrt weights | `dweights` |
| $\frac{\partial L}{\partial \mathbf{b}}$ | Gradient of loss wrt biases | `dbiases` |
| $\frac{\partial L}{\partial \mathbf{X}}$ | Gradient passed to previous layer | `dinputs` for a dense layer |
| `dvalues` | Incoming gradient from the next layer | `dvalues` |

### Dense-layer backward equations

For the class-based implementation used later in the series:

$$
\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \cdot \frac{\partial L}{\partial \mathbf{Z}}
$$

$$
\frac{\partial L}{\partial \mathbf{b}} = \sum_{\text{batch}} \frac{\partial L}{\partial \mathbf{Z}}
$$

$$
\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Z}} \cdot \mathbf{W}^T
$$

---

## 3. Shape Rules That Matter Most

### Matrix multiplication rule

$$
(m, n) \cdot (n, p) \rightarrow (m, p)
$$

Examples used throughout the course:

| Expression | Meaning | Result |
|---|---|---|
| `(batch, features) · (features, neurons)` | Forward pass | `(batch, neurons)` |
| `(features, batch) · (batch, neurons)` | `X.T · dvalues` | `(features, neurons)` |
| `(batch, neurons) · (neurons, features)` | `dvalues · W.T` | `(batch, features)` |

### Bias broadcasting

Biases are usually stored as `(1, neurons)` so NumPy can broadcast them across the batch.

$$
(batch, neurons) + (1, neurons) \rightarrow (batch, neurons)
$$

### `keepdims=True`

Use `keepdims=True` when you need a reduced result to stay two-dimensional for later broadcasting.

Example:

$$
\text{np.sum}(X, \text{axis}=1, \text{keepdims}=True)
$$

If `X` is `(3, 4)`, the result is `(3, 1)`, not `(3,)`.

---

## 4. One Important Convention Shift in This Repo

The early conceptual lectures sometimes present weights as:

$$
\text{weights shape} = (neurons, inputs)
$$

because that makes it easy to say, "one row of weights per neuron."

Later, in the reusable class-based implementation, the code stores weights as:

$$
\text{weights shape} = (inputs, neurons)
$$

so the forward pass becomes:

$$
\mathbf{X} \cdot \mathbf{W}
$$

Both conventions are valid. The important thing is to stay consistent within one implementation.

### Practical translation

| Presentation style | Weight shape |
|---|---|
| Early conceptual examples | `(neurons, inputs)` |
| Later class-based code | `(inputs, neurons)` |

If a formula looks "transposed" compared with an earlier lecture, this convention change is usually why.

---

## 5. Labels: Class Index vs One-Hot

There are two common ways to represent targets.

| Format | Example | Shape |
|---|---|---|
| Class indices | `[0, 2, 1]` | `(batch,)` |
| One-hot | `[[1,0,0],[0,0,1],[0,1,0]]` | `(batch, classes)` |

For one-hot labels, only one entry is `1` and the rest are `0`.

---

## 6. Optimizer Notation

| Symbol | Meaning | Common code name |
|---|---|---|
| $\alpha$ | Learning rate | `learning_rate`, `current_learning_rate` |
| decay | Learning-rate decay factor | `decay` |
| $\beta$ or $\beta_1$ | Momentum coefficient | `momentum`, `beta_1` |
| $\rho$ or $\beta_2$ | Exponential average coefficient for squared gradients | `rho`, `beta_2` |
| $\epsilon$ | Small constant for numerical stability | `epsilon` |
| $\lambda$ | Regularization strength | `lambda`, `weight_regularizer_l2`, etc. |
| cache | Running store of squared gradients | `weight_cache`, `bias_cache` |
| velocity / momentum term | Running average of updates | `weight_momentums`, `bias_momentums` |

### Quick intuition

- Learning rate: how big each step is.
- Decay: how the learning rate shrinks over time.
- Momentum: how much past directions influence the next step.
- RMSProp / Adam cache: a memory of gradient magnitudes.
- Epsilon: stops division by zero.

---

## 7. Common Confusions

### `axis=0` vs `axis=1`

- `axis=0`: reduce the batch/row dimension.
- `axis=1`: reduce across columns inside each row.

For a `(3, 4)` array:

- `np.sum(X, axis=0)` returns shape `(4,)`
- `np.sum(X, axis=1)` returns shape `(3,)`

### `dvalues` vs `dinputs`

- `dvalues` means: gradient coming *into* this layer.
- `dinputs` means: gradient going *out* of this layer toward the previous layer.

### Why does ReLU backward use a mask?

Because:

$$
\frac{d\,\text{ReLU}(z)}{dz} =
\begin{cases}
1 & z > 0 \\
0 & z \le 0
\end{cases}
$$

So gradients pass through positive inputs and are zeroed elsewhere.

---

## 8. Minimal Formula Map

| Step | Formula |
|---|---|
| Dense forward | $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$ |
| ReLU forward | $\mathbf{A} = \max(0, \mathbf{Z})$ |
| Softmax | $\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$ |
| Cross-entropy | $L = -\sum_i y_i \log(\hat{y}_i)$ |
| Dense backward weights | $\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \cdot \frac{\partial L}{\partial \mathbf{Z}}$ |
| Dense backward inputs | $\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Z}} \cdot \mathbf{W}^T$ |
| Softmax + CCE backward | $\hat{\mathbf{y}} - \mathbf{y}$ |

---

## 9. When to Use This Guide

Open this guide when:

- a shape mismatch error appears,
- a transpose seems to appear "out of nowhere,"
- the same symbol means slightly different things across parts,
- or optimizer variables like $\beta_1$, $\beta_2$, and $\rho$ start blending together.

---

*Suggested companions:* [Exercises](exercises.md), [Gradient Checking](gradient_checking.md), and [Softmax Backward Appendix](appendix_softmax_combined_backward.md).