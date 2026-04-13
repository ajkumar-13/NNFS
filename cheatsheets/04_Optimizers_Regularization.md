# Cheat Sheet 4 — Optimizers & Regularization (Parts 22–31)

*Quick-reference flashcards for every optimizer variant, generalization strategies, and regularization techniques.*

---

## 📊 Optimizer Comparison at a Glance

| Optimizer | Update Rule | Key Hyperparameters | Strengths | Weaknesses |
|---|---|---|---|---|
| **SGD** | $w \leftarrow w - \alpha \nabla L$ | $\alpha$ | Simple, interpretable | Slow, oscillates in ravines |
| **SGD + Decay** | $\alpha_t = \frac{\alpha_0}{1 + d \cdot t}$ | $\alpha_0$, $d$ | Fine-tunes late training | Need to tune decay rate |
| **SGD + Momentum** | $v \leftarrow \beta v + (1-\beta)\nabla L$; $w \leftarrow w - \alpha v$ | $\alpha$, $\beta$ | Smooths oscillations, escapes plateaus | Extra memory (velocity) |
| **AdaGrad** | $G \leftarrow G + (\nabla L)^2$; $w \leftarrow w - \frac{\alpha}{\sqrt{G + \epsilon}} \nabla L$ | $\alpha$, $\epsilon$ | Per-parameter adaptive rates | Cache grows forever → LR dies |
| **RMSProp** | $G \leftarrow \rho G + (1-\rho)(\nabla L)^2$ | $\alpha$, $\rho$, $\epsilon$ | Fixes AdaGrad's dying LR | One more hyperparameter |
| **Adam** | Momentum + RMSProp + bias correction | $\alpha$, $\beta_1$, $\beta_2$, $\epsilon$ | Best general-purpose | Most hyperparameters |

---

## 🏃 SGD (Stochastic Gradient Descent)

```python
layer.weights -= learning_rate * layer.dweights
layer.biases  -= learning_rate * layer.dbiases
```

**When to use:** Baseline, or when you want maximum control and simplicity.

---

## 📉 Learning Rate Decay

$$\alpha_t = \frac{\alpha_0}{1 + \text{decay} \cdot t}$$

```python
if self.decay:
    self.current_learning_rate = self.learning_rate / \
        (1. + self.decay * self.iterations)
```

| Decay value | Effect |
|---|---|
| `0` | Constant LR |
| `1e-4` | Gentle decay — LR halves around epoch 10,000 |
| `1e-3` | Moderate — LR halves around epoch 1,000 |
| `1e-2` | Aggressive — LR halves around epoch 100 |

**Rule of thumb:** Start with no decay, add it only if training oscillates in later epochs.

---

## 🌀 Momentum

$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \nabla L$$
$$w \leftarrow w - \alpha \cdot v_t$$

```python
if self.momentum:
    weight_updates = self.momentum * layer.weight_momentums - \
                     self.current_learning_rate * layer.dweights
    layer.weight_momentums = weight_updates
else:
    weight_updates = -self.current_learning_rate * layer.dweights
layer.weights += weight_updates
```

| $\beta$ | Behavior |
|---|---|
| `0.0` | No momentum (plain SGD) |
| `0.5` | Mild smoothing |
| `0.9` | **Typical default** — strong smoothing |
| `0.99` | Very heavy — may overshoot |

**Analogy:** A ball rolling downhill accumulates speed. Momentum helps push through small bumps (local minima) and reduces zig-zagging.

---

## 📈 AdaGrad

$$G_t = G_{t-1} + (\nabla L)^2$$
$$w \leftarrow w - \frac{\alpha}{\sqrt{G_t + \epsilon}} \cdot \nabla L$$

```python
layer.weight_cache += layer.dweights ** 2
layer.weights -= self.current_learning_rate * layer.dweights / \
    (np.sqrt(layer.weight_cache) + self.epsilon)
```

**Problem:** Cache $G$ only grows → effective learning rate → 0. Works well for sparse features, not for long training runs.

---

## ♻️ RMSProp

$$G_t = \rho \cdot G_{t-1} + (1 - \rho) \cdot (\nabla L)^2$$
$$w \leftarrow w - \frac{\alpha}{\sqrt{G_t + \epsilon}} \cdot \nabla L$$

```python
layer.weight_cache = self.rho * layer.weight_cache + \
    (1 - self.rho) * layer.dweights ** 2
layer.weights -= self.current_learning_rate * layer.dweights / \
    (np.sqrt(layer.weight_cache) + self.epsilon)
```

| $\rho$ | Effect |
|---|---|
| `0.9` | **Default** — remembers recent ~10 steps |
| `0.99` | Longer memory — slower adaptation |
| `0.5` | Short memory — very responsive |

**Fix over AdaGrad:** Exponential moving average means the cache doesn't grow without bound.

---

## 🏆 Adam (Adaptive Moment Estimation)

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L)^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w \leftarrow w - \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

```python
# Momentum
layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
    (1 - self.beta_1) * layer.dweights
# Cache
layer.weight_cache = self.beta_2 * layer.weight_cache + \
    (1 - self.beta_2) * layer.dweights ** 2
# Bias correction
weight_momentums_corrected = layer.weight_momentums / \
    (1 - self.beta_1 ** (self.iterations + 1))
weight_cache_corrected = layer.weight_cache / \
    (1 - self.beta_2 ** (self.iterations + 1))
# Update
layer.weights -= self.current_learning_rate * weight_momentums_corrected / \
    (np.sqrt(weight_cache_corrected) + self.epsilon)
```

**Common defaults:** $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$

**Bias correction:** Counters the zero-initialization of $m$ and $v$ — most impactful in early iterations.

---

## 🧪 Generalization & Testing

| Concept | Meaning |
|---|---|
| **Overfitting** | Model memorizes training data, fails on unseen data |
| **Underfitting** | Model too simple to capture the pattern |
| **Train/test split** | Evaluate on data never seen during training |
| **Validation set** | Used to tune hyperparameters without touching test set |
| **Generalization gap** | Difference between train and test performance |

```
All Data
├── Training set (70-80%)     ← model learns from this
├── Validation set (10-15%)   ← tune hyperparameters here
└── Test set (10-15%)         ← evaluate ONCE at the end
```

---

## 🛡️ L1 & L2 Regularization

| Type | Penalty term | Gradient addition | Effect on weights |
|---|---|---|---|
| **L1** | $\lambda \sum |w|$ | $\lambda \cdot \text{sign}(w)$ | Drives weights to exactly 0 (sparsity) |
| **L2** | $\lambda \sum w^2$ | $2 \lambda w$ | Shrinks weights toward 0 (decay) |
| **L1 + L2** | Both | Both | Sparsity + shrinkage |

```python
# In loss calculation
if layer.weight_regularizer_l1 > 0:
    regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
if layer.weight_regularizer_l2 > 0:
    regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights ** 2)

# In backward pass
if self.weight_regularizer_l1 > 0:
    dL1 = np.ones_like(self.weights)
    dL1[self.weights < 0] = -1
    self.dweights += self.weight_regularizer_l1 * dL1
if self.weight_regularizer_l2 > 0:
    self.dweights += 2 * self.weight_regularizer_l2 * self.weights
```

**Lambda ($\lambda$) tuning:** Start at $10^{-5}$ and increase by 10× until test accuracy peaks.

---

## 💧 Dropout

$$\text{output} = \text{input} \times \text{mask} \times \frac{1}{1 - p}$$

```python
class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate  # store SUCCESS rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) \
                           / self.rate  # inverted dropout
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
```

| Dropout rate | Effect |
|---|---|
| `0.0` | No dropout (all neurons active) |
| `0.1–0.2` | Mild regularization |
| `0.5` | Strong regularization (common in large networks) |
| `> 0.7` | Usually too aggressive |

**Key rules:**
1. **Only during training** — disable at test time
2. **Inverted dropout** — scale by $1/(1-p)$ during training so test-time needs no change
3. **Apply after activation**, before the next dense layer

---

## 🎯 Quick Self-Test

1. Which optimizer addresses AdaGrad's decaying learning rate?
2. What is Adam's bias correction formula for the first moment?
3. If $\lambda = 0.01$ and a weight is $w = 0.5$, what does L2 regularization add to its gradient?
4. With dropout rate 0.3, what fraction of neurons are active during training?
5. Name two differences between L1 and L2 regularization.
6. Why do we divide by $(1 - p)$ in inverted dropout?

<details>
<summary>Answers</summary>

1. **RMSProp** — uses exponential moving average instead of a sum for the cache
2. $\hat{m}_t = m_t / (1 - \beta_1^t)$
3. $2 \times 0.01 \times 0.5 = 0.01$
4. $70\%$ (1 − 0.3 = 0.7)
5. L1 creates sparsity (exact zeros), L2 shrinks toward zero; L1 gradient is $\pm\lambda$, L2 gradient is $2\lambda w$
6. To keep the expected output the same during training and testing, so we don't need to scale at test time

</details>

---

[← Back to Index](../INDEX.md)
