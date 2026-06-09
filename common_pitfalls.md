# Common Pitfalls & Debugging Guide

*Stuck? This page covers the most frequent mistakes learners make — organized by where in the series they typically appear.*

---

## Phase 1: Foundation (Parts 1–5)

### Pitfall: Shape mismatch in `np.dot()`

**Symptom:** `ValueError: shapes (3,4) and (3,4) not aligned`

**Cause:** You forgot to transpose the weight matrix. When both inputs and weights are matrices with rows-per-sample and rows-per-neuron respectively, inner dimensions don't match.

**Fix:** Use `np.dot(inputs, weights.T)` or switch to the `(inputs × neurons)` weight convention from Part 4.

```python
# WRONG: (3×4) · (3×4) — inner dims 4 and 3 don't match
np.dot(inputs, weights)

# CORRECT: (3×4) · (4×3) — inner dims both 4 ✓
np.dot(inputs, weights.T)
```

---

### Pitfall: Broadcasting bug with missing `keepdims`

**Symptom:** Code runs without error but produces wrong results. Often subtle — the shape is `(3,)` instead of `(3,1)` so broadcasting goes in the wrong direction.

**Cause:** `np.sum(a, axis=1)` returns shape `(3,)` which broadcasts as a row `(1,3)`, not a column `(3,1)`.

**Fix:** Always use `keepdims=True` when you need per-row operations:
```python
# WRONG — broadcasts as row
row_sums = np.sum(a, axis=1)          # shape: (3,)

# CORRECT — broadcasts as column
row_sums = np.sum(a, axis=1, keepdims=True)  # shape: (3,1)
```

---

### Pitfall: Using Python lists instead of NumPy arrays

**Symptom:** `.T` gives `AttributeError`, or arithmetic doesn't work element-wise.

**Fix:** Convert with `np.array()`:
```python
weights = [[0.2, 0.8], [0.5, -0.9]]
np.array(weights).T  # now .T works
```

---

## Phase 2: Forward Pass (Parts 6–9)

### Pitfall: Softmax numerical overflow

**Symptom:** `RuntimeWarning: overflow encountered in exp` → output contains `nan` or `inf`.

**Cause:** `np.exp(1000)` overflows. Raw logits can be large positive numbers.

**Fix:** Subtract the max before exponentiating:
```python
# WRONG
exp_values = np.exp(inputs)

# CORRECT — shift so max is 0, prevents overflow
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
```

This is mathematically equivalent (the softmax probabilities are identical) but numerically stable.

---

### Pitfall: Cross-entropy loss returns `inf`

**Symptom:** Loss is `inf` on the very first iteration.

**Cause:** A predicted probability is exactly `0`, and `log(0) = -∞`.

**Fix:** Clip predictions to prevent log(0):
```python
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
loss = -np.log(y_pred_clipped[range(len(y_pred)), y_true])
```

---

### Pitfall: Confusing integer labels vs one-hot labels

**Symptom:** Indexing error or wrong loss values.

**Cause:** Your labels are `[0, 1, 2]` (integers) but your code expects `[[1,0,0], [0,1,0], [0,0,1]]` (one-hot), or vice versa.

**Fix:** Check `y.shape` and handle both cases:
```python
if len(y_true.shape) == 1:        # integer labels: [0, 1, 2]
    correct_confidences = y_pred[range(len(y_pred)), y_true]
elif len(y_true.shape) == 2:      # one-hot: [[1,0,0], ...]
    correct_confidences = np.sum(y_pred * y_true, axis=1)
```

---

## Phase 3: Calculus foundations (Parts 10–11)

### Pitfall: Stepping along the gradient instead of against it

**Symptom:** Loss goes *up* every iteration, not down.

**Cause:** The gradient points in the direction of steepest *increase*; adding it climbs the loss.

**Fix:** Subtract it: `w -= learning_rate * grad`. Gradient descent moves *opposite* the gradient.

---

### Pitfall: Dropping a factor in the chain rule

**Symptom:** A numerical gradient check fails by a constant factor (often an input value, or 2).

**Cause:** One local derivative was left out of the product — e.g. the input `x` at a multiply node, or the `2` from a squared-error loss.

**Fix:** Write the chain as an explicit product of every local derivative, one per operation, then multiply. Verify against a finite-difference check (see [gradient_checking.md](gradient_checking.md)).

---

## Phase 4: Backpropagation (Parts 12–21)

### Pitfall: Gradients have wrong shape

**Symptom:** `dweights` shape doesn't match `weights` shape.

**Cause:** Forgetting to transpose in the gradient formula.

**Fix:** The three formulas for a dense layer backward:
```python
self.dweights = np.dot(self.inputs.T, dvalues)   # (inputs×samples)·(samples×neurons) = (inputs×neurons)
self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)  # (1×neurons)
self.dinputs  = np.dot(dvalues, self.weights.T)   # (samples×neurons)·(neurons×inputs) = (samples×inputs)
```

**Verification:** `dweights.shape` must equal `self.weights.shape`. If not, something is transposed wrong.

---

### Pitfall: Modifying `dvalues` in-place (ReLU backward)

**Symptom:** Downstream layers get corrupted gradients.

**Cause:** `self.dinputs = dvalues` makes both point to the same array. Then `self.dinputs[mask] = 0` also zeros out the original `dvalues`.

**Fix:** Always copy first:
```python
self.dinputs = dvalues.copy()       # separate array
self.dinputs[self.inputs <= 0] = 0  # safe to modify
```

---

### Pitfall: Not caching inputs during forward pass

**Symptom:** `AttributeError: 'Layer_Dense' object has no attribute 'inputs'` during backward.

**Cause:** The forward method didn't save `self.inputs = inputs`.

**Fix:** Always cache inputs in `forward()`:
```python
def forward(self, inputs):
    self.inputs = inputs                           # ← cache for backward!
    self.output = np.dot(inputs, self.weights) + self.biases
```

---

### Pitfall: Forgetting to normalize loss by batch size

**Symptom:** Gradients scale with batch size, making the learning rate batch-size dependent.

**Cause:** The loss backward pass doesn't divide by number of samples.

**Fix:**
```python
self.dinputs = self.dinputs / samples   # normalize so gradients are batch-size independent
```

---

## Phase 5: Optimizers (Parts 22–27)

### Pitfall: Learning rate too high — loss oscillates or increases

**Symptom:** Loss goes up, or jumps around wildly.

**Fix:** Reduce learning rate by 10×. Start with `0.01` or `0.001` for Adam.

---

### Pitfall: Forgetting bias correction in Adam

**Symptom:** First few iterations behave strangely (too slow or too fast).

**Cause:** Momentum and cache are initialized to zero, biasing early estimates downward.

**Fix:** Apply bias correction:
```python
momentum_corrected = momentum / (1 - beta1 ** (step + 1))
cache_corrected    = cache    / (1 - beta2 ** (step + 1))
```

Note: `step` starts at 0, so `step + 1` starts at 1.

---

### Pitfall: AdaGrad learning rate goes to zero

**Symptom:** Training stalls after many epochs — loss stops decreasing.

**Cause:** AdaGrad's cache grows monotonically, shrinking the effective learning rate to near zero.

**Fix:** Switch to RMSProp (exponential moving average instead of cumulative sum) or Adam.

---

## Phase 6: Generalisation and regularisation (Parts 28–31)

### Pitfall: Dropout active during testing

**Symptom:** Test accuracy is anomalously low and non-deterministic.

**Cause:** Dropout is randomly zeroing neurons during evaluation.

**Fix:** Disable dropout during testing:
```python
# During training:
layer_dropout.forward(dense.output, training=True)

# During testing:
layer_dropout.forward(dense.output, training=False)  # no dropout!
```

---

### Pitfall: L2 regularization lambda too high

**Symptom:** Network barely learns — all weights shrink toward zero.

**Cause:** The penalty term dominates the actual loss, so the optimizer only cares about making weights small.

**Fix:** Reduce `lambda` by 10× or 100×. Typical range: `1e-4` to `5e-4`.

---

## Phase 7: Practical training and extensions (Parts 32–35)

### Pitfall: Re-creating the optimiser inside the batch loop

**Symptom:** Mini-batch training never converges, or Adam behaves like plain SGD.

**Cause:** Constructing `Optimizer_Adam(...)` inside the per-batch loop resets its momentum and cache every step.

**Fix:** Construct the optimiser once, outside both the epoch and batch loops. Only `pre_update_params` / `update_params` / `post_update_params` run per batch.

---

### Pitfall: Weights initialised at the wrong scale

**Symptom:** Activations (or gradients) vanish toward zero, or explode to `NaN`, within a few layers.

**Cause:** A fixed `0.01 * randn` does not scale with fan-in; deep stacks shrink or blow up the signal.

**Fix:** Scale the initial standard deviation by the layer's input count — He for ReLU, Xavier/Glorot otherwise. See Part 33.

---

### Pitfall: Wrong label shape for sigmoid + binary cross-entropy

**Symptom:** Broadcasting errors, or a loss that does not track accuracy.

**Cause:** Sigmoid + BCE uses **one** output neuron, so labels must be shape `(N, 1)` — not one-hot `(N, 2)` or a flat `(N,)`.

**Fix:** Reshape `y` to `(N, 1)`. Use sigmoid + BCE for 2 classes; softmax + CCE needs at least 2 output neurons. See Part 34.

---

## Universal Debugging Checklist

When something isn't working, check these in order:

1. **Print shapes** at every step. Shape mismatches cause 90% of bugs.
2. **Check loss on first iteration.** For 3-class problems, untrained loss should be ≈ $-\ln(1/3) \approx 1.099$.
3. **Verify gradient shapes.** `dweights.shape == weights.shape` and `dbiases.shape == biases.shape`.
4. **Use gradient checking.** See [gradient_checking.md](gradient_checking.md) — numerically verify your analytical gradients.
5. **Print gradient magnitudes.** If they're all zero → dead ReLU or broken backward. If they're huge → exploding gradients.
6. **Simplify.** Test with 1 layer, 1 sample, 2 features. If that works, scale up.

---

*Still stuck? Re-read the SVG diagrams for the relevant Part. They're designed to show exactly what's happening at each step.*
