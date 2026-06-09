# Cheat Sheet 3 — Calculus & Backpropagation (Parts 10–21)

*Quick-reference flashcards for derivatives, the chain rule, and the complete backward pass.*

---

## 📐 Derivatives Refresher

| Concept | Formula |
|---|---|
| Definition | $f'(x) = \lim_{h \to 0} \dfrac{f(x+h) - f(x)}{h}$ |
| Numerical approx | $f'(x) \approx \dfrac{f(x+h) - f(x-h)}{2h}$ (centered, more accurate) |
| Power rule | $\dfrac{d}{dx} x^n = n x^{n-1}$ |
| Sum rule | $(f + g)' = f' + g'$ |
| Product rule | $(fg)' = f'g + fg'$ |
| Partial derivative | $\dfrac{\partial f}{\partial x_i}$ — differentiate w.r.t. $x_i$, treat others as constants |
| Gradient | $\nabla f = \left[\dfrac{\partial f}{\partial x_1}, \dfrac{\partial f}{\partial x_2}, \ldots\right]$ |

---

## ⛓️ The Chain Rule

$$\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$$

For a chain $L = g(f(h(x)))$:

$$\frac{dL}{dx} = \frac{dL}{dg} \cdot \frac{dg}{df} \cdot \frac{df}{dh} \cdot \frac{dh}{dx}$$

**In neural networks:** Each layer receives $\frac{\partial L}{\partial \text{output}}$ (the upstream gradient) and computes:
- $\frac{\partial L}{\partial \text{input}}$ — passed to the previous layer
- $\frac{\partial L}{\partial \mathbf{W}}$, $\frac{\partial L}{\partial \mathbf{b}}$ — used to update parameters

---

## 🔙 Backprop Through a Single Neuron

For $z = w_1 x_1 + w_2 x_2 + b$:

| Gradient | Formula | Intuition |
|---|---|---|
| $\partial z / \partial w_1$ | $x_1$ | Weight gradient = its input |
| $\partial z / \partial x_1$ | $w_1$ | Input gradient = its weight |
| $\partial z / \partial b$ | $1$ | Bias gradient = always 1 |

Multiply each by the upstream gradient $\frac{\partial L}{\partial z}$ to get the full gradient (chain rule).

---

## 🏗️ Dense Layer Backward

```python
def backward(self, dvalues):
    # Gradients on parameters
    self.dweights = np.dot(self.inputs.T, dvalues)   # (n_in, n_out)
    self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)  # (1, n_out)
    # Gradient on inputs (to pass backward)
    self.dinputs  = np.dot(dvalues, self.weights.T)   # (batch, n_in)
```

| Gradient | Shape | Formula |
|---|---|---|
| `dweights` | `(n_in, n_out)` | $\mathbf{X}^{T} \cdot \text{dvalues}$ |
| `dbiases` | `(1, n_out)` | $\sum_{\text{batch}} \text{dvalues}$ |
| `dinputs` | `(batch, n_in)` | $\text{dvalues} \cdot \mathbf{W}^{T}$ |

**Why transpose?** The transpose "reverses" the matrix multiply from the forward pass.

---

## ⚡ ReLU Backward

```python
def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0   # zero out where input was ≤ 0
```

**Gate analogy:** ReLU is an open/closed gate. If the input was positive → pass the gradient through. If negative → block it (gradient = 0).

---

## 🌡️ Softmax Backward

The Jacobian of softmax output $S$ w.r.t. input $z$:

$$\frac{\partial S_i}{\partial z_j} = \begin{cases} S_i(1 - S_i) & \text{if } i = j \\\ -S_i S_j & \text{if } i \neq j \end{cases}$$

**In practice:** We almost never compute this alone. Use the combined shortcut instead ↓

---

## 🎯 Combined Softmax + Cross-Entropy Backward (The Shortcut)

$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

```python
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # Convert one-hot to integer labels if needed
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1   # ŷ - y
        self.dinputs /= samples                      # normalize by batch size
```

**Why this is beautiful:** The combined gradient is just `predictions - targets`. Simple, numerically stable, and efficient.

---

## 🔁 Full Backward Pipeline

```
Loss gradient        ← scalar 1.0 (seed)
  │
  ▼
Softmax + CCE        ← ŷ - y                    dinputs: (batch, 3)
  │
  ▼
Dense Layer 2        ← dvalues · W₂ᵀ           dinputs: (batch, 64)
  │                     dW₂ = X₂ᵀ · dvalues     dweights: (64, 3)
  │                     db₂ = Σ dvalues           dbiases: (1, 3)
  ▼
ReLU                 ← zero where input ≤ 0      dinputs: (batch, 64)
  │
  ▼
Dense Layer 1        ← dvalues · W₁ᵀ           dinputs: (batch, 2)
                       dW₁ = X₁ᵀ · dvalues     dweights: (2, 64)
                       db₁ = Σ dvalues           dbiases: (1, 64)
```

---

## 🧪 Gradient Checking Formula

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + h) - L(\theta - h)}{2h}$$

- Use $h \approx 10^{-4}$ or $10^{-5}$
- Compare to analytical gradient: should agree to $\sim 10^{-5}$ relative error
- Slow (one forward pass per parameter), but invaluable for debugging

---

## 🏋️ Training Loop Skeleton

```python
for epoch in range(10001):
    # Forward
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Backward
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update (simple SGD)
    for layer in [dense1, dense2]:
        layer.weights -= learning_rate * layer.dweights
        layer.biases  -= learning_rate * layer.dbiases
```

---

## 🎯 Quick Self-Test

1. What is $\frac{\partial}{\partial w}(wx + b)$?
2. If the upstream gradient is `[0.5, -0.3, 0.1]` and the ReLU input was `[2.0, -1.0, 0.5]`, what is the ReLU backward output?
3. What is the combined softmax + cross-entropy gradient for prediction `[0.7, 0.2, 0.1]` with true class 0?
4. Why do we divide `dinputs` by the number of samples in the combined backward?
5. What shape is `dweights` for a layer with 4 inputs and 8 neurons?

<details>
<summary>Answers</summary>

1. $x$ — the derivative of a linear function w.r.t. its weight is the input
2. `[0.5, 0.0, 0.1]` — the middle one is zeroed because its input was negative
3. `[0.7 - 1, 0.2 - 0, 0.1 - 0] = [-0.3, 0.2, 0.1]`
4. To average the gradient over the batch, ensuring the learning rate is consistent regardless of batch size
5. `(4, 8)` — same shape as the weight matrix

</details>

---

[← Back to Index](../INDEX.md)
