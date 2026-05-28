# Gradient Checking — Verify Your Backprop Implementation

*A practical exercise for numerically verifying that your analytical gradients are correct. This is the single most important debugging technique for neural network implementations.*

---

## Why Gradient Checking?

Backprop bugs are **silent killers** — the code runs, loss decreases *somewhat*, but the model never reaches its potential because some gradients are subtly wrong. Gradient checking catches these bugs by comparing your analytical gradients (from backprop) against numerical gradients (from the definition of a derivative).

---

## The Core Idea

The derivative of a function $f$ at point $\theta$ can be approximated numerically:

$$\frac{\partial f}{\partial \theta} \approx \frac{f(\theta + h) - f(\theta - h)}{2h}$$

This **centered difference** is accurate to $O(h^2)$, which with $h = 10^{-5}$ gives ~10 digits of accuracy.

**Compare this** to your analytical gradient from backprop. If they match to ~5+ digits, your backprop is correct.

---

## Step-by-Step Implementation

### Step 1: Build a Tiny Network

Use *small* dimensions so the check runs quickly:

```python
import numpy as np

np.random.seed(0)

# Tiny network: 3 inputs, 4 hidden neurons, 2 output classes
n_inputs = 3
n_hidden = 4
n_classes = 2
n_samples = 5

# Random data
X = np.random.randn(n_samples, n_inputs)
y = np.random.randint(0, n_classes, size=n_samples)

# Initialize parameters
W1 = 0.01 * np.random.randn(n_inputs, n_hidden)
b1 = np.zeros((1, n_hidden))
W2 = 0.01 * np.random.randn(n_hidden, n_classes)
b2 = np.zeros((1, n_classes))
```

### Step 2: Define the Forward + Loss Function

Pack the entire forward pass into one function that returns the scalar loss:

```python
def forward_and_loss(X, y, W1, b1, W2, b2):
    # Layer 1
    z1 = np.dot(X, W1) + b1
    # ReLU
    a1 = np.maximum(0, z1)
    # Layer 2
    z2 = np.dot(a1, W2) + b2
    # Softmax
    exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Cross-entropy loss
    correct_logprobs = -np.log(np.clip(probs[range(len(y)), y], 1e-7, 1 - 1e-7))
    loss = np.mean(correct_logprobs)
    return loss, probs, a1, z1
```

### Step 3: Compute Analytical Gradients (Your Backprop)

```python
# Forward pass
loss, probs, a1, z1 = forward_and_loss(X, y, W1, b1, W2, b2)

# Backward pass — softmax + cross-entropy combined
dscores = probs.copy()
dscores[range(n_samples), y] -= 1
dscores /= n_samples

# Dense layer 2
dW2 = np.dot(a1.T, dscores)
db2 = np.sum(dscores, axis=0, keepdims=True)
da1 = np.dot(dscores, W2.T)

# ReLU
dz1 = da1.copy()
dz1[z1 <= 0] = 0

# Dense layer 1
dW1 = np.dot(X.T, dz1)
db1 = np.sum(dz1, axis=0, keepdims=True)
```

### Step 4: Compute Numerical Gradients

```python
def numerical_gradient(f, param, h=1e-5):
    """Compute numerical gradient for every element of a parameter array."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_value = param[idx]

        # f(theta + h)
        param[idx] = old_value + h
        loss_plus = f()

        # f(theta - h)
        param[idx] = old_value - h
        loss_minus = f()

        # Centered difference
        grad[idx] = (loss_plus - loss_minus) / (2 * h)

        # Restore
        param[idx] = old_value
        it.iternext()
    return grad
```

### Step 5: Compare!

```python
# Helper: compute loss given current (possibly perturbed) parameters
def compute_loss():
    loss, _, _, _ = forward_and_loss(X, y, W1, b1, W2, b2)
    return loss

# Numerical gradients (slow but correct)
num_dW2 = numerical_gradient(compute_loss, W2)
num_db2 = numerical_gradient(compute_loss, b2)
num_dW1 = numerical_gradient(compute_loss, W1)
num_db1 = numerical_gradient(compute_loss, b1)

# Compare each parameter
for name, analytical, numerical in [
    ("dW2", dW2, num_dW2),
    ("db2", db2, num_db2),
    ("dW1", dW1, num_dW1),
    ("db1", db1, num_db1),
]:
    # Relative error
    diff = np.abs(analytical - numerical)
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
    relative_error = np.max(diff / denom)
    status = "✓ PASS" if relative_error < 1e-5 else "✗ FAIL"
    print(f"{name:5s} | max relative error: {relative_error:.2e} | {status}")
```

---

## Expected Output

```
dW2   | max relative error: 1.23e-08 | ✓ PASS
db2   | max relative error: 5.67e-10 | ✓ PASS
dW1   | max relative error: 2.34e-08 | ✓ PASS
db1   | max relative error: 8.91e-10 | ✓ PASS
```

If all relative errors are below $10^{-5}$, your backprop is correct.

---

## Interpreting Results

| Relative Error | Meaning |
|---|---|
| < $10^{-7}$ | Excellent — nearly perfect match |
| $10^{-7}$ to $10^{-5}$ | Good — acceptable for float64 |
| $10^{-5}$ to $10^{-3}$ | Suspicious — might be a bug, investigate |
| > $10^{-3}$ | Almost certainly a bug in your backward pass |

---

## Common Bugs This Catches

### 1. Forgetting to Divide by Batch Size

```python
# BUG: missing /= n_samples
dscores = probs.copy()
dscores[range(n_samples), y] -= 1
# dscores /= n_samples  ← forgot this line!
```

The numerical gradient will be `n_samples` times smaller than the analytical one.

### 2. Wrong Transpose

```python
# BUG: forgot .T
dW2 = np.dot(a1, dscores)       # ← wrong shape, or wrong values
# Should be:
dW2 = np.dot(a1.T, dscores)     # ← correct
```

### 3. ReLU Backward Bug

```python
# BUG: using output instead of input for the mask
dz1 = da1.copy()
dz1[a1 <= 0] = 0    # ← should use z1, not a1 (though for ReLU they're equivalent when a1 > 0)
```

### 4. Not Zeroing Gradients for Negative ReLU

```python
# BUG: forgetting the mask entirely
dz1 = da1.copy()
# Missing: dz1[z1 <= 0] = 0
```

The numerical gradient will show zeros where Z1 ≤ 0, but the analytical gradient won't.

---

## Gradient Checking with Regularization

When using L2 regularization, include the regularization term in the loss:

```python
def compute_loss_with_reg(lambda_reg=5e-4):
    loss, _, _, _ = forward_and_loss(X, y, W1, b1, W2, b2)
    loss += lambda_reg * (np.sum(W1**2) + np.sum(W2**2))
    return loss
```

And in the analytical gradient:

```python
dW1 += 2 * lambda_reg * W1
dW2 += 2 * lambda_reg * W2
```

---

## Complete Self-Contained Script

```python
"""
Gradient Checking for a 2-Layer Neural Network
Run this to verify your backprop implementation is correct.
"""
import numpy as np

np.random.seed(0)

# --- Data ---
n_inputs, n_hidden, n_classes, n_samples = 3, 4, 2, 5
X = np.random.randn(n_samples, n_inputs)
y = np.random.randint(0, n_classes, size=n_samples)

# --- Parameters ---
W1 = 0.01 * np.random.randn(n_inputs, n_hidden)
b1 = np.zeros((1, n_hidden))
W2 = 0.01 * np.random.randn(n_hidden, n_classes)
b2 = np.zeros((1, n_classes))


def forward_and_loss(X, y, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, W2) + b2
    exp_s = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    loss = np.mean(-np.log(np.clip(probs[range(len(y)), y], 1e-7, 1 - 1e-7)))
    return loss, probs, a1, z1


def numerical_gradient(f, param, h=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = param[idx]
        param[idx] = old + h
        lp = f()
        param[idx] = old - h
        lm = f()
        grad[idx] = (lp - lm) / (2 * h)
        param[idx] = old
        it.iternext()
    return grad


# --- Analytical gradients (backprop) ---
loss, probs, a1, z1 = forward_and_loss(X, y, W1, b1, W2, b2)

dscores = probs.copy()
dscores[range(n_samples), y] -= 1
dscores /= n_samples

dW2 = np.dot(a1.T, dscores)
db2 = np.sum(dscores, axis=0, keepdims=True)
da1 = np.dot(dscores, W2.T)

dz1 = da1.copy()
dz1[z1 <= 0] = 0

dW1 = np.dot(X.T, dz1)
db1 = np.sum(dz1, axis=0, keepdims=True)

# --- Numerical gradients ---
f = lambda: forward_and_loss(X, y, W1, b1, W2, b2)[0]

print("Gradient Check Results")
print("=" * 50)

for name, analytical, numerical in [
    ("dW2", dW2, numerical_gradient(f, W2)),
    ("db2", db2, numerical_gradient(f, b2)),
    ("dW1", dW1, numerical_gradient(f, W1)),
    ("db1", db1, numerical_gradient(f, b1)),
]:
    diff = np.abs(analytical - numerical)
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
    rel_err = np.max(diff / denom)
    status = "✓ PASS" if rel_err < 1e-5 else "✗ FAIL"
    print(f"  {name:5s} | max relative error: {rel_err:.2e} | {status}")

print("=" * 50)
print("If all PASS, your backprop is correct!")
```

---

## Exercises

1. **Introduce a bug intentionally.** Comment out `dscores /= n_samples` and re-run the check. What does the error look like?
2. **Try with ReLU bug.** Remove the `dz1[z1 <= 0] = 0` line. Which parameter fails?
3. **Add L2 regularization.** Add `lambda_reg * sum(W^2)` to the loss and `2 * lambda_reg * W` to each weight gradient. Verify the check still passes.
4. **Scale up.** Try with `n_hidden=64, n_samples=100`. The check should still pass but will be slower.
5. **Different $h$ values.** Try $h = 10^{-3}, 10^{-5}, 10^{-7}, 10^{-10}$. At what point does the numerical gradient become less accurate (due to floating-point cancellation)?

---

*See also: [Part 18 — Backprop Through the Loss](posts/18-backpropagation-through-the-loss-function/index.md) | [Exercises](exercises.md) | [Back to Index](INDEX.md)*
