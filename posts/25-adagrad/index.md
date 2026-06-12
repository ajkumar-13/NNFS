---
slug: 25-adagrad
title: "Part 25 · AdaGrad"
date: 2026-05-29
tags: [neural-networks, from-scratch, optimisation, adagrad, adaptive-rates]
hero: diagrams/01-per-parameter-rates.svg
reading_time: 12
part: "Part VI — Optimisers"
---

# Part 25 · AdaGrad

> **TL;DR.** **AdaGrad** (Duchi, Hazan, Singer, 2011) gives each parameter its own effective learning rate, scaled inversely by the square root of the running sum of its past squared gradients, so big-gradient parameters take small steps and small-gradient parameters keep their step size. This post implements `Optimizer_Adagrad` from scratch, shows why it lifts spiral accuracy above plain SGD, and explains the structural flaw (the cache only grows, so every rate eventually decays toward zero) that RMSProp and Adam exist to fix.
>
> **Reading time:** ~12 minutes.
>
> **After reading this you will be able to:**
> - Explain why a single global learning rate is fundamentally inadequate for parameters with different gradient scales.
> - Implement `Optimizer_Adagrad` with per-layer `weight_cache` and `bias_cache` buffers.
> - Predict, before running training, when AdaGrad will help and when its dying-rate problem will dominate.

![AdaGrad per-parameter rates: parameters with large historical gradients take small effective steps, parameters with small historical gradients keep large steps, and the cache grows monotonically so every rate eventually decays.](diagrams/01-per-parameter-rates.svg)
*The cache for each parameter is the running sum of squared gradients; the effective learning rate is the global $\alpha$ divided by $\sqrt{\text{cache} + \epsilon}$. Two parameters with different gradient histories end up with very different effective step sizes.*

---

## 1. The remaining failure mode

Vanilla SGD treated every parameter identically. So did SGD + decay, and so did SGD + momentum. All three share a single hyperparameter (the learning rate $\alpha$) and apply it uniformly to every weight in the network.

That uniformity has a hidden cost. Different parameters can have wildly different gradient magnitudes, and the optimum step size for each depends on its own magnitude, not on the global average.

Consider a stripped-down two-parameter loss:

$$L(W_1, W_2) = \frac{W_1^2}{100} + W_2^2$$

At the point $(W_1, W_2) = (1, 1)$:

| Parameter | $\partial L / \partial W$ | Magnitude |
|---|:---:|---|
| $W_1$ | $2 W_1 / 100 = 0.02$ | tiny |
| $W_2$ | $2 W_2 = 2.0$ | 100× larger |

With a single learning rate $\alpha = 0.1$:

- The step for $W_1$ is $0.1 \cdot 0.02 = 0.002$. Negligible. $W_1$ barely moves.
- The step for $W_2$ is $0.1 \cdot 2.0 = 0.2$. $W_2$ moves 100× faster. If $\alpha$ is large enough to make $W_1$ progress visibly, $W_2$ overshoots and oscillates.

This is a fundamental incompatibility. The right step for $W_1$ is the wrong step for $W_2$, and there is no global $\alpha$ that works for both. Either $\alpha$ is sized for the largest gradient (so the small-gradient parameters never learn) or for the smallest (so the large-gradient parameters explode).

In real networks the same problem appears at scale. Some weights see consistent large gradients (e.g. early layers receiving direct gradient signal from a deep stack); some see tiny gradients (e.g. late-layer weights inside saturated regions). A single $\alpha$ fights both at once.

---

## 2. The AdaGrad idea

The intuition is simple: **let each parameter scale its own step by what it has seen so far**.

Specifically, for each parameter $\theta$, maintain a running sum of its squared gradients:

$$G_t = G_{t-1} + g_t^2 \quad \text{(elementwise)}$$

then use $G_t$ to scale the update:

$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{G_t + \epsilon}} \cdot g_t$$

where $g_t = \partial L / \partial \theta$ is the current gradient, $\alpha$ is the global learning rate, and $\epsilon$ is a tiny constant (typically $10^{-7}$) preventing division by zero on the first step.

Three things to notice.

**$G_t$ has the same shape as $\theta$.** It is a tensor, not a scalar. Each weight has its own scalar cache value; large-gradient weights build up large caches, small-gradient weights build up small ones.

**The square root is what makes the scaling "right".** Without it, scaling by $1/G$ would over-correct: a parameter with cache 100 would have its step shrunk 100×. With $\sqrt{G}$, the scaling matches the *units* of the gradient (gradient magnitude grows as $\sqrt{\text{variance}}$), so the rescaling produces approximately unit-variance steps across parameters.

**Squaring removes the sign.** Negative and positive gradients both contribute positively to the cache. Without the square, a parameter that oscillated back and forth with equal-magnitude gradients would build a zero cache and never get rescaled, which is the opposite of the desired behaviour.

---

## 3. Worked example: the cache grows forever

Take a single parameter with gradient $g = 0.5$ at every iteration. The cache evolves as:

| Iteration $t$ | $g$ | $G_t$ | $\sqrt{G_t}$ | Effective rate $\alpha / \sqrt{G_t}$ (for $\alpha = 1$) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5 | 0.25 | 0.500 | 2.00 |
| 2 | 0.5 | 0.50 | 0.707 | 1.41 |
| 10 | 0.5 | 2.50 | 1.581 | 0.63 |
| 100 | 0.5 | 25.0 | 5.000 | 0.20 |
| 1 000 | 0.5 | 250  | 15.81 | 0.063 |
| 10 000 | 0.5 | 2500 | 50.00 | 0.020 |
| 100 000 | 0.5 | 25 000 | 158.1 | 0.0063 |

The cache grows linearly in $t$ (since $g^2$ is constant), so the effective learning rate shrinks as $1/\sqrt{t}$. This is similar to the decay schedule from Part 23, but with two crucial differences.

- It is **per-parameter**, not global. Parameters with small historical gradients keep their effective rate; only the noisy ones get pinned down.
- It is **implicit**. The decay arises from the optimiser's own bookkeeping; there is no separate `decay` knob to tune.

In the favourable case this is exactly the desired effect. A parameter that has been bouncing around (large $|g|$) gets calmed down. A parameter that has barely moved (small $|g|$) keeps its full step size and gets a chance to catch up.

---

## 4. The structural flaw: dying rates

The same property that makes AdaGrad work is the property that eventually kills it. **The cache is monotone non-decreasing.** $g^2 \ge 0$ for any $g$, so $G_t \ge G_{t-1}$ always. Once the cache has grown large, nothing in the algorithm can shrink it.

Three phases of an AdaGrad run:

| Phase | Cache size | Effective rate | Learning behaviour |
|---|---|---|---|
| **Early** ($t < 100$) | small | large | Optimiser learns fast; per-parameter scaling is beneficial |
| **Middle** ($t \approx 1000$) | moderate | moderate | Slow but useful learning; behaves like a tuned schedule |
| **Late** ($t > 10\,000$) | large | tiny | Updates become negligible; learning effectively stops |

The parameter is not "stuck in a local minimum" in the gradient-descent sense. The gradient might still be telling the optimiser exactly which way to go. The optimiser is simply ignoring it, because the cache-driven denominator has grown so large that the resulting step is smaller than the parameter's floating-point precision.

This is the AdaGrad analogue of "dead ReLU" neurons:

| Failure mode | Cause | Recoverable? |
|---|---|---|
| Dead ReLU neuron | Input drifts to ≤ 0; output and gradient permanently zero | Sometimes (if other neurons shift the input distribution) |
| Dead AdaGrad parameter | Cache grows; effective rate → 0 | **No** (cache can only grow) |

For tasks with a fixed, modest number of iterations (the original AdaGrad paper was about convex problems and short training runs), this is acceptable. For neural networks trained for 10⁴ to 10⁶ iterations, it is a deal-breaker. **RMSProp (Part 26) is the direct fix:** replace the cumulative sum with an exponential moving average so the cache can grow *and* shrink.

---

## 5. The optimiser class

A new class this time, not an extension of `Optimizer_SGD`, because the update rule is structurally different (no velocity, no momentum):

```python
class Optimizer_Adagrad:

    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.epsilon               = epsilon
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        # Lazy cache creation on first call.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        # Accumulate squared gradients.
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache   += layer.dbiases ** 2

        # Per-parameter update.
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
```

Three implementation notes.

**Same three-method contract as Part 23.** The training loop is unchanged: `pre_update_params → update_params per layer → post_update_params`. Decay is still optional and still works through the existing $1/(1 + d \cdot t)$ formula on top of the cache-based scaling.

**Buffers live on the layer.** Same convention as momentum's `weight_momentums`: lazy initialisation on first call, layer-local storage. A single optimiser instance can drive any number of layers.

**`epsilon` is added *outside* the square root, not inside.** Both forms appear in the literature. Outside means $\alpha / (\sqrt{G} + \epsilon)$; inside means $\alpha / \sqrt{G + \epsilon}$. The "outside" form is the one in the original AdaGrad paper and is what NumPy implementations use. The two behave identically when $G \gg \epsilon$, which is true after the very first iteration; the difference is only in the first-step boundary value.

---

## 6. The training loop

Same skeleton as Part 24. The only line that changes is the optimiser construction:

```python
optimizer = Optimizer_Adagrad(learning_rate=1.0, decay=1e-4)

for epoch in range(10001):
    # Forward, accuracy, backward: unchanged.

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
```

A small detail worth noting: the decay rate is `1e-4` here instead of `1e-3`. AdaGrad's cache already produces implicit decay; layering an explicit decay schedule on top requires a gentler $d$, or the rate dies even faster.

---

## 7. What happens when this is run

With $\alpha = 1.0$, $d = 10^{-4}$, $\epsilon = 10^{-7}$, and 10 000 epochs:

| Configuration | Final loss | Final accuracy |
|---|:---:|:---:|
| Vanilla SGD (Part 22) | 0.87 | 64.7% |
| SGD + decay (Part 23) | 0.76 | 64.7% |
| **AdaGrad** | **~0.38** | **~84.0%** |
| SGD + decay + momentum, $\beta = 0.9$ (Part 24) | 0.12 | 95.7% |

Two observations. (Figures from real runs of the optimiser classes on the spiral dataset (seed 0, 10k epochs).)

**AdaGrad clearly beats plain SGD and decay.** A roughly 19-point lift in accuracy over the ~65% of Parts 22–23, with no momentum and no tuning beyond `learning_rate` and `decay`, demonstrates that per-parameter scaling captures something real.

**It loses to momentum.** Momentum's accuracy on the spiral is still higher. There are two reasons: (a) momentum directly attacks the oscillation problem, which is the dominant failure mode in narrow valleys; (b) AdaGrad's late-phase rate dying limits how close it can get to the minimum once the cache is large.

The natural question is whether the two ideas can be combined. They can, and the result is RMSProp (Part 26, which fixes the cache-grows-forever problem) followed by Adam (Part 27, which adds momentum on top of RMSProp).

---

## 8. When AdaGrad is the right choice

Even though it loses to momentum on the spiral, AdaGrad still has its uses. The two settings where it shines:

**Sparse features.** When most parameters receive zero or near-zero gradients on most steps (e.g. word embeddings in NLP, where each step only touches the vocabulary present in the batch), AdaGrad's per-parameter rate ensures the rarely-updated parameters keep large effective rates and learn meaningfully when they do see a gradient. This was the original motivating use case in the 2011 paper.

**Short, convex problems.** AdaGrad has formal convergence guarantees on convex losses with bounded iteration counts. For logistic regression, linear SVMs, or other classical ML models trained for a few thousand iterations, the cache never grows large enough to matter and the per-parameter scaling is a strict improvement over fixed-rate SGD.

For the typical deep-network training regime (10⁴ to 10⁶ steps, non-convex loss, dense gradients), RMSProp or Adam is preferable. AdaGrad's value here is **conceptual**, not practical: it is the cleanest illustration of why per-parameter scaling matters, before the EMA refinement makes the implementation more opaque.

---

## 9. Anticipated questions

- **Why divide by $\sqrt{G}$ and not by $G$?** Variance-matching. The square root makes the rescaled gradient approximately unit-norm in expectation; dividing by $G$ would over-shrink the step for high-variance parameters.
- **What is the right $\epsilon$?** Anywhere from $10^{-8}$ to $10^{-6}$. The value rarely matters in practice; the cache is dominated by $g^2$ within the first few steps.
- **Can the cache be reset periodically to avoid the dying-rate problem?** Yes, and it is one of the early "fixes" historically tried. But every reset throws away the per-parameter adaptation that AdaGrad spent thousands of steps learning. The principled fix is RMSProp's EMA.
- **Does AdaGrad need decay at all?** Not really. The cache already provides implicit decay. The `decay` parameter is included for compatibility with the optimiser interface but is usually set to a small value or zero.
- **Is AdaGrad ever called "adaptive gradient"?** Yes. "AdaGrad" is shorthand for "Adaptive Gradient", and the paper's title is *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*.

---

## 10. Summary

| Concept | Takeaway |
|---|---|
| Cache | $G_t = G_{t-1} + g_t^2$ (one scalar per parameter) |
| Update | $\theta \mathrel{-}= \alpha \cdot g / (\sqrt{G_t} + \epsilon)$ |
| Per-parameter rate | Effective rate = $\alpha / \sqrt{G_t + \epsilon}$, different for every weight |
| Storage | Per-layer `weight_cache`, `bias_cache` buffers |
| Fatal flaw | Cache grows monotonically; effective rate $\to 0$; learning stops |
| Result on spiral | ~65% (decay) → 84% (AdaGrad); still below momentum (95.7%) |
| Best use case | Sparse features, short convex problems |
| Legacy | Foundation for RMSProp (Part 26) and Adam (Part 27) |

---

## Common pitfalls

- **Re-initialising the cache every epoch.** Once initialised to zero, the cache must persist for the entire training run. Re-initialising defeats the entire point.
- **Setting $\epsilon = 0$.** First-step division by zero. Either always use a small positive $\epsilon$ or add it inside the square root.
- **Using a large `decay` alongside AdaGrad.** Two decays compose multiplicatively. The implicit AdaGrad decay alone is usually enough; an extra explicit decay larger than $10^{-4}$ kills the rate well before convergence.
- **Treating cache size as a per-layer quantity.** It is per-*parameter*. Two weights in the same layer can have very different cache values.
- **Assuming AdaGrad will outperform momentum.** It rarely does on dense, deep networks. AdaGrad shines on sparse problems; momentum shines on dense ones. Spiral data is dense.
- **Forgetting to take the square root.** Dividing by $G$ instead of $\sqrt{G}$ over-shrinks the step and produces a much harsher decay than intended.
- **Using AdaGrad as a drop-in replacement for `Optimizer_SGD` without re-tuning.** AdaGrad's effective rate is implicitly smaller than $\alpha$ from very early on. A run that worked at $\alpha = 0.01$ with vanilla SGD probably wants $\alpha = 1.0$ or higher with AdaGrad.

---

## Further reading

- Duchi, J., Hazan, E., and Singer, Y., *"Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"* (Journal of Machine Learning Research, 2011). The original AdaGrad paper.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 8.5.1 (AdaGrad) (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 25 (2020).
- Ruder, S., *"An overview of gradient descent optimization algorithms"* (arXiv:1609.04747, 2016). §4.3 (AdaGrad).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Part 26 — RMSProp](../26-rmsprop/index.md)**: replace the cumulative sum with an exponential moving average so the cache can shrink. The dying-rate problem disappears.
- **[Part 27 — Adam](../27-adam-optimiser/index.md)**: RMSProp's per-parameter scaling plus momentum's velocity, with bias correction. The modern default.

---

> **Try it yourself:** Hands-on exercises and quizzes for this lecture live in [Exercises](../../exercises.md) and [Quizzes](../../quizzes.md).
