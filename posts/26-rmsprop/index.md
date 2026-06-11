---
slug: 26-rmsprop
title: "Part 26 · RMSProp"
date: 2026-05-29
tags: [neural-networks, from-scratch, optimisation, rmsprop, adaptive-rates, ema]
hero: diagrams/01-rmsprop-vs-adagrad-cache.svg
reading_time: 11
part: "Part VI — Optimisers"
---

# Part 26 · RMSProp

> **TL;DR.** RMSProp fixes AdaGrad's ever-growing cache with a one-line change: replace the cumulative sum of squared gradients with an exponential moving average, so the cache reflects recent gradients only and effective rates never drift to zero. This post derives that fix, implements `Optimizer_RMSprop` as a one-line patch to AdaGrad, and shows it holding ~90% accuracy on the spiral where AdaGrad would eventually stall.
>
> **Reading time:** ~11 minutes.
>
> **After reading this you will be able to:**
> - Explain why an exponential moving average over $g^2$ produces a bounded cache, and how the decay factor $\rho$ controls its memory horizon.
> - Implement `Optimizer_RMSprop` with per-layer caches and choose sensible defaults for $\alpha$, $\rho$, and $\epsilon$.
> - Read RMSProp as the bridge optimiser between AdaGrad (per-parameter rates without memory limit) and Adam (RMSProp with momentum).

![RMSProp vs AdaGrad cache growth: AdaGrad's sum grows linearly with iterations, RMSProp's EMA converges to a stable value.](diagrams/01-rmsprop-vs-adagrad-cache.svg)
*Same gradient magnitude at every step. AdaGrad's cache marches off to infinity; RMSProp's cache settles to a steady-state value determined by the recent gradient.*

---

## 1. What needs fixing

AdaGrad (Part 25) ended with one fatal flaw: the cache $G_t = \sum_{s \le t} g_s^2$ can only grow. After enough iterations, the effective learning rate $\alpha / \sqrt{G_t + \epsilon}$ becomes negligible regardless of what the gradient is saying. The parameter is "dead" not because its gradient is zero but because the denominator has grown so large that the update underflows in practice.

The structural cause is clear: **the cache has unbounded memory**. Every gradient ever seen still contributes to the sum, no matter how old. Even if the optimisation landscape changes (e.g. a parameter that was bouncing for the first thousand iterations and stabilised for the next ten thousand), the cache still remembers the early turbulence forever.

The fix is a cache that **forgets old gradients** as new ones arrive. Specifically:

- Recent gradients should dominate the cache (so the per-parameter scaling reflects current behaviour).
- Old gradients should fade away (so a parameter can recover its step size after a calm period).
- The cache should still grow when gradients are large and shrink when gradients are small (so the per-parameter idea is preserved).

The right tool for the job is the **exponential moving average** (EMA), already familiar from finance, signal processing, and (loosely) the momentum velocity from Part 24.

---

## 2. The exponential moving average

The EMA of a stream of values $x_1, x_2, x_3, \dots$ is defined recursively:

$$E_t = \rho \cdot E_{t-1} + (1 - \rho) \cdot x_t$$

where $\rho \in [0, 1)$ is the **decay factor** (sometimes called the "smoothing constant"). The EMA at step $t$ is a convex combination of the previous EMA and the new sample, that is, a weighted average whose weights $\rho$ and $1 - \rho$ are non-negative and sum to one.

For RMSProp's cache, the input stream is the squared gradient $g_t^2$:

$$G_t = \rho \cdot G_{t-1} + (1 - \rho) \cdot g_t^2$$

Two properties make this exactly the fix AdaGrad needed.

**The cache converges.** If $g_t^2$ is approximately constant at some value $\bar{g^2}$ over many iterations, the EMA approaches $\bar{g^2}$ as $t \to \infty$. This is not just plausible, it follows from the fixed point: at steady state $G_t = G_{t-1} = G$, so $G = \rho G + (1 - \rho)\bar{g^2}$, which rearranges to $G = \bar{g^2}$. There is no unbounded growth: the cache asymptotes to the recent steady-state value.

**Old values decay geometrically.** The contribution of $g_{t-k}^2$ to $G_t$ is proportional to $\rho^k (1 - \rho)$. With $\rho = 0.9$, a gradient from 10 steps ago contributes $0.9^{10} \approx 0.35$ of its weight; from 100 steps ago, $0.9^{100} \approx 2.7 \times 10^{-5}$. The cache effectively has a memory horizon of $\sim 1 / (1 - \rho)$ recent steps.

For $\rho = 0.9$ the horizon is about 10 steps; for $\rho = 0.999$ it is about 1000 steps. Larger $\rho$ means longer memory and smoother (slower-changing) cache.

---

## 3. AdaGrad versus RMSProp, side by side

Suppose the gradient magnitude is $|g| = 0.5$ at every iteration. The two caches evolve as:

| Iteration $t$ | AdaGrad $G_t$ | RMSProp $G_t$ ($\rho = 0.9$) |
|:---:|:---:|:---:|
| 1     | 0.250   | 0.025 |
| 2     | 0.500   | 0.048 |
| 10    | 2.500   | 0.163 |
| 100   | 25.0    | 0.249 |
| 1 000 | 250     | 0.250 |
| 10 000 | 2 500  | 0.250 |
| 100 000 | 25 000 | 0.250 |

AdaGrad's cache scales linearly with $t$; the resulting effective rate $\alpha / \sqrt{G_t} \sim 1 / \sqrt{t}$ decays without bound. RMSProp's cache converges to $\bar{g^2} = 0.25$ within a few hundred iterations and stays there. The effective rate stabilises at $\alpha / \sqrt{0.25} = 2\alpha$ and remains useful for as long as the gradient distribution holds.

The headline difference: **RMSProp's effective rate never dies as long as the gradient magnitude stays bounded**.

If the gradient magnitude *changes*, the cache tracks the change. A parameter that was seeing $|g| \approx 2$ for the first thousand iterations and then settles to $|g| \approx 0.1$ will see its cache decay from $\sim 4$ down to $\sim 0.01$ over the next thousand or so iterations, gradually restoring a larger effective step size. AdaGrad cannot do this; the cache it built during the noisy phase is permanent.

---

## 4. The update rule

The update step itself is identical to AdaGrad:

$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{G_t + \epsilon}} \cdot g_t$$

Only the *definition* of $G_t$ changes. This is intentional: RMSProp inherits AdaGrad's per-parameter scaling unchanged and only fixes the part of the algorithm that was broken.

Reading RMSProp as a composition of two pieces:

| Piece | Inherited from | What it does |
|---|---|---|
| Per-parameter rescaling $\alpha / \sqrt{G_t + \epsilon}$ | AdaGrad | Larger gradients → smaller effective steps |
| Bounded cache $G_t$ via EMA over $g^2$ | new in RMSProp | Cache reflects recent gradients only, never overflows |

The conceptual heritage is also clear: RMSProp uses an EMA over $g^2$ in exactly the way momentum uses a weighted sum over $g$. Different statistics of the gradient stream, but the same "let the past inform the present, but not dominate it forever" idea.

---

## 5. The optimiser class

```python
class Optimizer_RMSprop:

    def __init__(self, learning_rate=0.02, decay=0.0,
                 epsilon=1e-7, rho=0.9):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.epsilon               = epsilon
        self.rho                   = rho
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        # Lazy cache creation.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        # EMA of squared gradients (the only change vs AdaGrad).
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache   = self.rho * layer.bias_cache + \
                             (1 - self.rho) * layer.dbiases ** 2

        # Per-parameter update (same form as AdaGrad).
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
```

Three notes on the construction.

**The default learning rate is much smaller.** `Optimizer_SGD` defaulted to `1.0`; `Optimizer_RMSprop` defaults to `0.02`. The reason is that the EMA-based cache stabilises at a small value much faster than AdaGrad's cumulative sum, so the effective rate $\alpha / \sqrt{G_t}$ is much larger early in training. A `1.0` base rate that worked for SGD typically diverges immediately under RMSProp. Adaptive optimisers in general need smaller base rates.

**`rho` defaults to 0.9.** This gives a ~10-step memory horizon, which works well for the spiral problem and for most short-to-medium training runs. For longer runs (50k+ iterations) the original paper recommended `rho = 0.99` or `rho = 0.999`, which gives a much longer horizon and a smoother cache.

**The only line that changed vs AdaGrad** is the cache update. Everything else (the constructor signature, the `pre_update_params` decay, the `update_params` divide-by-root-cache step, the `post_update_params` counter) is identical. RMSProp is a one-line patch to AdaGrad.

---

## 6. The training loop

Identical in shape to all previous optimisers. Only the constructor differs:

```python
optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)

for epoch in range(10001):
    # Forward → accuracy → backward — unchanged.

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
```

Two implementation choices worth highlighting.

**`learning_rate = 0.02`.** Lower than SGD's 1.0, higher than Adam's typical 0.001. The "right" base rate for an adaptive optimiser depends on the cache statistics; experimentation matters.

**`rho = 0.999`** on the spiral example, not the 0.9 default. The long horizon smooths the cache more aggressively and stabilises training. With `rho = 0.9` the cache reacts very quickly to per-step noise and the loss curve becomes jittery.

---

## 7. Results on the spiral dataset

| Configuration | Final loss | Final accuracy |
|---|:---:|:---:|
| SGD + decay (Part 23) | 0.76 | 64.7% |
| AdaGrad (Part 25) | ~0.38 | ~84.0% |
| **RMSProp** | **~0.24** | **~90.0%** |
| SGD + momentum (Part 24) | 0.12 | 95.7% |

The numbers (from real runs of the optimiser classes on the spiral dataset (seed 0, 10k epochs)) tell two stories at once.

**Over 10 000 epochs, RMSProp edges ahead of AdaGrad** (~90% vs ~84%). RMSProp's bounded cache keeps its effective learning rate from collapsing the way AdaGrad's $1/\sqrt{t}$ rate slowly does, so it stays a little more effective even at this iteration count. The gap widens further over very long runs, where AdaGrad's dying-rate problem becomes severe (past ~50k iterations).

**Both still lose to momentum on this dense problem.** As discussed in §8 of Part 25, per-parameter scaling shines on sparse problems; for dense low-dimensional problems like the spiral, momentum's direction smoothing matters more than per-parameter scaling.

The natural conclusion: combine the two. Use momentum on the gradient (for direction smoothing) *and* an EMA cache on $g^2$ (for per-parameter scaling). That combination is exactly what Adam (Part 27) is, plus one bias-correction term.

---

## 8. Choosing $\rho$

The decay factor controls the memory horizon of the cache. The trade-offs are:

| $\rho$ | Memory horizon | Behaviour |
|:---:|:---:|---|
| 0.9 | ~10 steps | Cache reacts quickly to gradient changes; can be noisy on small-batch training |
| 0.99 | ~100 steps | Standard default; smooth cache; works well in most settings |
| 0.999 | ~1000 steps | Very smooth cache; recommended for long runs and small per-step gradient changes |

Production frameworks expose `rho` (or its equivalent) as a tunable hyperparameter. PyTorch's `RMSprop` defaults to `alpha=0.99` (its `alpha` is this post's `rho`); TensorFlow's `RMSprop` defaults to `rho=0.9`; JAX/Optax follows TensorFlow. Different frameworks reach different defaults because the right value depends on batch size, learning rate, and model scale.

For this series, the working default is `rho = 0.999` on the spiral example because the dataset is small and full-batch updates make the per-step gradients very stable.

---

## 9. RMSProp as the bridge to Adam

Two facts make RMSProp the natural last stop before Adam.

**RMSProp uses an EMA of $g^2$ for the cache.** Adam keeps this idea unchanged (it calls the cache the "second moment" $v_t$, the average of $g^2$, and uses an EMA factor $\beta_2$, typically 0.999).

**RMSProp does not use momentum.** Adam adds momentum: a separate EMA of $g$ itself (the "first moment" $m_t$, factor $\beta_1$, typically 0.9). The Adam update then scales $m_t$ by the same $1 / \sqrt{v_t + \epsilon}$ that RMSProp would apply to $g_t$ directly.

So Adam = RMSProp's cache-with-EMA + momentum's direction-with-EMA + a small bias correction. Each of the three pieces was already introduced separately:

| Adam piece | Introduced in |
|---|---|
| $\beta_1$-EMA of $g$ (momentum) | Part 24 |
| $\beta_2$-EMA of $g^2$ (cache) | Part 26 (this lecture) |
| Bias correction $(1 - \beta^t)$ | Part 27 |

Part 27 will spell out the bias-correction term and show how the combined Adam update outperforms every individual piece on the spiral dataset.

---

## 10. Anticipated questions

- **Is RMSProp better than AdaGrad in the short run?** Usually no. For up to ~10⁴ iterations the two are nearly indistinguishable. RMSProp's advantage is structural: it keeps working past where AdaGrad would have stalled.
- **What about Hinton's slide that introduced RMSProp?** It is unpublished, from his Coursera "Neural Networks for Machine Learning" course (2012). The lecture notes are widely cited as the original source; the algorithm has no formal paper.
- **Why is the default $\alpha$ so much smaller than for SGD?** Because the EMA-based cache stabilises at $\bar{g^2}$ quickly, so the effective rate $\alpha / \sqrt{G_t}$ is approximately $\alpha / |g|$ at steady state. With $|g| \sim 0.1$ that effective rate is $\sim 10 \alpha$; if $\alpha = 1$, the effective rate is $\sim 10$ and training diverges.
- **Can RMSProp be used with momentum?** Yes, and PyTorch's `RMSprop` exposes a `momentum` argument exactly for this. With momentum it is essentially Adam without bias correction. Most users go straight to Adam.
- **What is "Centered RMSProp"?** A variant that subtracts the EMA of $g$ (the first moment) before squaring, giving a variance estimate rather than a second-moment estimate. Sometimes more stable; rarely worth the complication outside research contexts.

---

## 11. Summary

| Concept | Takeaway |
|---|---|
| Cache update | $G_t = \rho G_{t-1} + (1 - \rho) g_t^2$ (EMA over $g^2$) |
| Parameter update | $\theta \mathrel{-}= \alpha \cdot g / (\sqrt{G_t} + \epsilon)$, same as AdaGrad |
| Why it fixes AdaGrad | Cache converges instead of growing; effective rate never dies |
| Default $\rho$ | 0.9 (short horizon) or 0.999 (long horizon); both common |
| Default $\alpha$ | 0.001–0.02; much smaller than SGD's 1.0 |
| Result on spiral | ~90% accuracy; above AdaGrad's ~84% at 10k epochs, with a bounded cache that holds up far better over very long runs |
| Position in the series | The bridge optimiser: AdaGrad → RMSProp → Adam |

---

## Common pitfalls

- **Using SGD's `learning_rate = 1.0` with RMSProp.** Divergence is almost guaranteed. Drop the base rate by one or two orders of magnitude when switching from SGD to any adaptive optimiser.
- **Setting $\rho$ to 1.0 exactly.** The cache stops updating: the new squared gradient gets weight $1 - \rho = 0$, so the cache is frozen at its *initial* value of zero. Every step then divides by $\sqrt{0} + \epsilon = \epsilon$, an enormous effective rate of $\alpha / \epsilon$ that blows up immediately. Always keep $\rho < 1$.
- **Choosing $\rho$ blindly.** A spiral-shaped toy run with full-batch updates wants a long horizon ($\rho = 0.999$); a stochastic mini-batch ImageNet run wants a shorter one ($\rho = 0.9$ or $0.99$). Match the horizon to the gradient noise scale.
- **Comparing RMSProp to AdaGrad in 10k-epoch runs only.** They look nearly identical. The structural advantage of RMSProp only shows up in long-horizon training (50k+ iterations).
- **Forgetting that the per-layer cache buffers are not reset between runs.** Calling `train()` twice on the same `layer` will continue the EMA from where it left off. That is sometimes the intent, but it can also produce subtle bugs.
- **Setting $\epsilon = 0$.** The first cache value is zero; the very first update divides by $\sqrt{\rho \cdot 0 + (1-\rho) g^2 + 0}$ which is well-defined, but later steps where $g \to 0$ can briefly trigger division by tiny numbers. Always keep $\epsilon$ small but positive.
- **Treating RMSProp's `decay` as a substitute for `rho`.** They control completely different things: `decay` controls the *global* learning rate over time; `rho` controls the *cache horizon*. Both can be tuned independently.

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning* — chapter 8.5.2 (RMSProp) (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python* — chapter 26 (2020).
- Ruder, S., *"An overview of gradient descent optimization algorithms"* (arXiv:1609.04747, 2016) — §4.5 (RMSProp).
- Tieleman, T. and Hinton, G., *Lecture 6.5 — RMSProp: Divide the gradient by a running average of its recent magnitude* (Coursera: Neural Networks for Machine Learning, 2012) — the original (unpublished) source.

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Part 27 — Adam](../27-adam-optimiser/index.md)** — add momentum's EMA-over-$g$ on top of RMSProp's EMA-over-$g^2$, plus bias correction. The modern default optimiser.

---

> **Try it yourself:** Hands-on exercises and quizzes for this lecture live in [Exercises](../../exercises.md) and [Quizzes](../../quizzes.md).
