# Part 27 · Adam

> **TL;DR.** Adam (Kingma & Ba, 2014) combines momentum and an RMSProp cache with a bias correction, making it the optimiser most production neural networks ship with. This post builds `Optimizer_Adam` from those three ideas and reproduces the spiral result (~96% accuracy) with no per-problem tuning.
>
> **After reading this you will be able to:**
> - Decompose the Adam update into its three constituent ideas (first-moment EMA, second-moment EMA, and bias correction) and explain what each one contributes.
> - Implement `Optimizer_Adam` with two per-layer buffers (`weight_momentums`, `weight_cache`) and reproduce the spiral result.
> - Reason about when Adam is the wrong default (sharp-minima generalisation, sparse production training) and what the alternatives are.

![Adam update pipeline: first-moment EMA (from momentum) and second-moment EMA (from RMSProp), each bias-corrected, combined into a single per-parameter update.](diagrams/01-adam-pipeline.svg)
*The gradient enters on the left; two parallel EMAs maintain $m$ and $v$; bias correction applies the $1/(1 - \beta^t)$ amplifier so early steps are not damped; the corrected ratio drives the per-parameter update.*

---

## 1. The synthesis

Six lectures into the optimiser group, four independent ideas have been introduced. Each one fixed a different failure mode of vanilla SGD.

| Lecture | Idea | What it fixed |
|---|---|---|
| Part 22 | Vanilla SGD update | Baseline; nothing fixed yet |
| Part 23 | Learning-rate decay | Overshoot near the minimum |
| Part 24 | Momentum (EMA of $g$) | Direction noise, valley oscillations |
| Part 25 | AdaGrad (sum of $g^2$) | One-rate-fits-all problem |
| Part 26 | RMSProp (EMA of $g^2$) | AdaGrad's dying-rate problem |

The two strongest improvements act on **different parts** of the SGD update.

- **Momentum** smooths the *numerator* of $\theta \mathrel{-}= \alpha \cdot g$. Instead of the noisy current $g$, it uses an EMA $m \approx \mathbb{E}[g]$.
- **RMSProp** rescales the *denominator*. Instead of a single $\alpha$, it uses $\alpha / \sqrt{v}$ where $v \approx \mathbb{E}[g^2]$.

They are orthogonal in what they optimise. Combining them is the obvious next step: keep both EMAs, and form the update as $\alpha \cdot m / \sqrt{v}$. That is exactly what Adam does, plus one final wrinkle: a bias correction for the cold-start period before $m$ and $v$ have warmed up.

The name "Adam" comes from *adaptive moment estimation*: the first moment $m$ is an estimate of $\mathbb{E}[g]$, the second moment $v$ is an estimate of $\mathbb{E}[g^2]$. Both are maintained as EMAs.

---

## 2. The Adam update rule

The full update for a single parameter (applied elementwise across the tensor):

$$\begin{aligned}
m_t &= \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t \quad &\text{(first moment, momentum)}\\[2pt]
v_t &= \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2 \quad &\text{(second moment, RMSProp cache)}\\[2pt]
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}} \quad &\text{(bias-corrected first moment)}\\[2pt]
\hat{v}_t &= \frac{v_t}{1 - \beta_2^{\,t}} \quad &\text{(bias-corrected second moment)}\\[2pt]
\theta_t &= \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad &\text{(parameter update)}
\end{aligned}$$

Five lines, all elementwise. The first two come straight from Parts 24 and 26. The middle two are new. The last is a fusion of momentum's update ($\theta \mathrel{+}= v$) with RMSProp's per-parameter scaling ($\alpha / \sqrt{v + \epsilon}$).

### 2.1. Why bias correction is needed

Both $m_0$ and $v_0$ are initialised to zero. On the first iteration, $m_1 = (1 - \beta_1) \, g_1$, only 10% of the actual gradient when $\beta_1 = 0.9$, and only 0.1% when $\beta_2 = 0.999$. Without correction, the early steps are silently 10× to 1000× smaller than the gradient is asking for, and the optimiser wastes the first dozens of iterations crawling toward the warmed-up steady state.

The mathematical fix is exact. Take a constant gradient $g$ and trace the EMA:

$$m_t = (1 - \beta_1) \sum_{k=0}^{t-1} \beta_1^k \, g = (1 - \beta_1^t) \, g$$

So $\mathbb{E}[m_t] = (1 - \beta_1^t) \, g$, biased low by the factor $(1 - \beta_1^t)$. Dividing by that factor recovers the unbiased estimate $\hat{m}_t = g$ exactly. Same logic for $\hat{v}_t$. This derivation assumes a constant gradient for clarity, but the same factor is the right first-order correction when the gradient varies, which is why the correction is used unchanged in practice.

In effect: $\hat{m}_t / (1 - \beta_1^t)$ amplifies early estimates so they are not stuck near zero, and the amplifier shrinks to 1 as $\beta_1^t \to 0$. For $\beta_1 = 0.9$:

| Iteration $t$ | $\beta_1^t$ | $1 / (1 - \beta_1^t)$ |
|:---:|:---:|:---:|
| 1   | 0.900 | 10× |
| 5   | 0.590 | 2.44× |
| 10  | 0.349 | 1.54× |
| 20  | 0.122 | 1.14× |
| 50  | 0.005 | ≈ 1× |
| 100 | 0.00003 | ≈ 1× |

After a few dozen iterations the correction is identity. It only matters at the start.

### 2.2. Why $\beta_2 = 0.999$ rather than 0.9

$\beta_2$ controls how long the second-moment EMA remembers. The default is 0.999 because:

- $g^2$ is noisier than $g$. The square amplifies extremes, so a longer averaging window is needed to get a stable estimate.
- A long window means the per-parameter rate stays stable across many iterations, which matches the spirit of AdaGrad/RMSProp.

Setting $\beta_2 = 0.9$ instead of $0.999$ makes the cache react quickly to per-step gradient variation, which can be useful for non-stationary problems but often makes training jittery on standard supervised tasks.

---

## 3. The optimiser class

```python
class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0.0,
                 epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.epsilon               = epsilon
        self.beta_1                = beta_1
        self.beta_2                = beta_2
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        # Lazy buffer creation: one momentum + one cache per parameter tensor.
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache     = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)
            layer.bias_cache       = np.zeros_like(layer.biases)

        # 1) First moment: EMA of g.
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums   = self.beta_1 * layer.bias_momentums + \
                                 (1 - self.beta_1) * layer.dbiases

        # 2) Bias correction of m. Note (iterations + 1): step counter is
        #    incremented in post_update_params *after* this call.
        t = self.iterations + 1
        weight_m_hat = layer.weight_momentums / (1 - self.beta_1 ** t)
        bias_m_hat   = layer.bias_momentums   / (1 - self.beta_1 ** t)

        # 3) Second moment: EMA of g**2.
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache   = self.beta_2 * layer.bias_cache + \
                             (1 - self.beta_2) * layer.dbiases ** 2

        # 4) Bias correction of v.
        weight_v_hat = layer.weight_cache / (1 - self.beta_2 ** t)
        bias_v_hat   = layer.bias_cache   / (1 - self.beta_2 ** t)

        # 5) Parameter update.
        layer.weights -= self.current_learning_rate * weight_m_hat / \
                         (np.sqrt(weight_v_hat) + self.epsilon)
        layer.biases  -= self.current_learning_rate * bias_m_hat / \
                         (np.sqrt(bias_v_hat)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
```

Three implementation notes.

**Same three-method contract.** `pre_update_params → update_params per layer → post_update_params`. The training loop from Part 22 (or 23, 24, 25, 26) runs unchanged.

**Two buffers per parameter tensor, not one.** Where momentum used `weight_momentums` and RMSProp used `weight_cache`, Adam needs both. Memory cost doubles vs either single-buffer optimiser; for production models with millions of parameters this is the main reason to occasionally prefer SGD + momentum over Adam.

**The $t$ used in bias correction is `self.iterations + 1`, not `self.iterations`.** The counter is incremented *after* `update_params` returns, so by convention "iteration 1" is the first call to `update_params`. Off-by-one here would cause the first step's $\hat{m}_1$ to be undefined (division by zero from $1 - \beta_1^0 = 0$).

---

## 4. The training loop

Same as every other lecture in this group:

```python
optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)

for epoch in range(10001):
    # Forward, accuracy, backward: unchanged.

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
```

`learning_rate = 0.02` is a high choice for Adam (typical defaults sit at `0.001`). The spiral dataset is small and benefits from a more aggressive base rate. For larger models, drop to `0.001` or below.

---

## 5. What happens when this is run

The headline number, with $\alpha_0 = 0.02$, $d = 10^{-5}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ over 10 000 epochs:

| Configuration | Final loss | Final accuracy |
|---|:---:|:---:|
| Vanilla SGD (Part 22) | 0.87 | 64.7% |
| SGD + decay (Part 23) | 0.76 | 64.7% |
| AdaGrad (Part 25) | ~0.38 | 84.0% |
| RMSProp (Part 26) | ~0.24 | ~90.0% |
| SGD + decay + momentum, $\beta = 0.9$ (Part 24) | 0.12 | 95.7% |
| **Adam** | **~0.08** | **~96.3%** |

Three observations.

**Adam wins, but not by a lot.** On a small dense problem like the spiral, SGD + momentum is within striking distance. Adam's structural advantages (no need to tune the learning rate per parameter, no dying-rate problem) really shine at scale; on a 387-parameter toy network (dense1 maps 2 inputs to 64 units, dense2 maps 64 to 3, counting weights and biases) the gap to a well-tuned SGD+momentum is small.

**Adam needs minimal tuning.** The combination of momentum, RMSProp scaling, and bias correction makes the optimiser robust to the choice of learning rate within roughly an order of magnitude. SGD + momentum collapses if $\alpha$ is off by 5×; Adam usually keeps training.

**Adam reaches the same neighbourhood faster.** Even though the final accuracy is similar to SGD + momentum, Adam typically hits 90%+ accuracy hundreds of epochs sooner. The bias correction makes the early steps move at full magnitude rather than waiting for $m$ and $v$ to warm up.

---

## 6. Why Adam became the default

Three reasons it became the production standard around 2015–2018 and has stayed there since.

**Sensible per-parameter rates without tuning.** AdaGrad showed that per-parameter rates were a real win; RMSProp removed the dying-rate side effect. Adam inherits both, so a user who knows nothing about gradient statistics gets a reasonable per-parameter rate for free.

**Momentum without re-tuning the learning rate.** Adding plain momentum to vanilla SGD usually requires shrinking $\alpha$ to avoid divergence (Part 24, §9). Adam absorbs the momentum into the second-moment normalisation, so the effective step size stays bounded regardless of $\beta_1$. The reason is that $\hat{m}_t$ and $\sqrt{\hat{v}_t}$ both scale with the gradient, so their ratio is roughly order 1 and the step size is set mainly by $\alpha$ rather than by the momentum factor. The learning rate stays in a narrow band (`1e-4` to `1e-2`) across very different problems.

**Bias correction makes hyperparameter scans interpretable.** Without it, an early evaluation of "is this learning rate too small?" is confounded by the cold-start damping. With it, the early loss curve reflects the chosen rate's true behaviour, so hyperparameter sweeps converge faster.

---

## 7. When Adam is the wrong default

Not every problem wants Adam. Two well-known cases:

**Generalisation on image classification.** Wilson et al. (2017) showed that on certain image benchmarks (CIFAR-10/100 with ResNet, ImageNet with several architectures), well-tuned SGD + momentum *generalises better* than Adam, even when both reach similar training loss. The conjectured mechanism is that Adam's per-parameter scaling lets it find sharper minima that don't transfer as well to held-out data. For state-of-the-art vision results, many groups still tune SGD + momentum carefully rather than reach for Adam.

**Memory-constrained production training.** Adam keeps two extra buffers per parameter (vs zero for vanilla SGD and one for momentum). On a model with 100M parameters in fp32, that is 800 MB of extra optimiser state. For training-at-the-edge or very-large-model settings, SGD + momentum or memory-efficient variants like Adafactor are preferable.

The standard practical advice: start with Adam, validate against SGD + momentum if the model is large or vision-heavy or being prepared for production deployment.

---

## 8. Variants worth knowing

Several refinements of Adam appear in production code. Brief pointers:

- **AdamW** (Loshchilov & Hutter, 2017). Decouples L2 weight decay from the gradient. In standard Adam, L2 regularisation gets folded into the gradient before the second-moment scaling, which subtly damps the regularisation strength. AdamW applies decay directly to the parameters, fixing the bug. Default for most modern training pipelines (transformers, large vision models).
- **AMSGrad** (Reddi et al., 2018). Replaces $\hat{v}_t$ with $\max(\hat{v}_t, \hat{v}_{t-1})$ to ensure the effective rate is monotone non-increasing. Useful for problems where Adam fails to converge; rarely needed in practice.
- **NAdam** (Dozat, 2016). Adds a Nesterov-momentum twist to the first-moment EMA. Modest improvement over Adam in some settings.
- **Lion** (Chen et al., 2023). Drops the second moment entirely; uses just the sign of the first moment. Memory cost halves; comparable performance reported on large language models.

For this series, plain Adam is what is built. The variants share the same skeleton; they differ in which line of the update changes.

---

## 9. Anticipated questions

- **Is Adam's first moment the same as momentum's velocity?** Almost. Classical momentum (Part 24) used $v = \beta v - \alpha g$ and updated $\theta \mathrel{+}= v$; Adam's first moment is $m = \beta_1 m + (1 - \beta_1) g$ and is divided by $\sqrt{v}$ before scaling by $\alpha$. The $(1 - \beta_1)$ factor matters: Adam's $m$ is a true EMA of $g$, not the scaled "heavy ball" velocity from classical momentum.
- **Do I need both decay and Adam's adaptive behaviour?** Usually no. Adam already adapts the effective rate per parameter via the second moment; an additional $1/(1+dt)$ schedule on top is often redundant or harmful. The series includes `decay` for compatibility, but most production Adam configs leave it at zero and use a separate scheduler (cosine, step, warmup) when scheduling is needed.
- **Why default $\alpha = 0.001$ in production but $\alpha = 0.02$ here?** Small dataset, full-batch updates, narrow loss landscape. Larger or stochastic training wants smaller $\alpha$. Treat `0.001` as the default starting point and adjust by a few times in either direction.
- **What happens if $\beta_1 = \beta_2$?** Nothing crashes, but the optimiser loses its expressive power. The first moment and second moment have very different roles; sharing the EMA factor collapses the distinction.
- **Does the bias correction matter past iteration 100?** No. After about $5 / (1 - \beta)$ steps the correction factor is essentially 1. For $\beta_2 = 0.999$ that is ~5000 steps; the correction is meaningful longer than for $\beta_1 = 0.9$ but still vanishes in the long run.

---

## 10. Summary

| Concept | Takeaway |
|---|---|
| First moment | $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$  (momentum's EMA over $g$) |
| Second moment | $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$  (RMSProp's EMA over $g^2$) |
| Bias correction | $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$ |
| Update | $\theta \mathrel{-}= \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ |
| Defaults | $\alpha = 0.001$ (or $0.02$ for toy data), $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$ |
| Result on spiral | ~96.3% accuracy, ~0.08 loss; best of the series |
| Production status | Modern default; AdamW is the variant most widely used today |

---

## Common pitfalls

- **Forgetting to add 1 to `self.iterations` in the bias-correction denominator.** First step would divide by $1 - \beta^0 = 0$.
- **Re-using `Optimizer_SGD`'s `learning_rate = 1.0` for Adam.** Divergence on the first step. Default Adam $\alpha$ is two to three orders of magnitude smaller.
- **Combining Adam with an aggressive decay schedule.** Adam already adapts per parameter; an extra global decay on top often makes the rate die too fast. Default to `decay = 0` and add a scheduler only if profiling shows the rate plateauing.
- **Adding L2 weight decay through the gradient instead of through the parameter.** Standard Adam mixes the decay into the gradient, which interacts with the second-moment scaling and weakens the regularisation. Use AdamW for any setting where L2 matters (transformers, image classifiers).
- **Assuming Adam beats SGD + momentum on every problem.** It does not. For vision benchmarks where generalisation matters, well-tuned SGD + momentum often wins.
- **Treating $\beta_1$ and $\beta_2$ as the same hyperparameter.** They control different averaging horizons (10 steps vs 1000 steps for the defaults). Tuning them together is a frequent source of bugs.
- **Calling `update_params` for layers without `dweights` set.** Same trap as every previous optimiser: backward must run first.

---

## Further reading

- Chen, X. et al., *"Symbolic Discovery of Optimization Algorithms"* (NeurIPS, 2023). The paper that introduced Lion.
- Dozat, T., *"Incorporating Nesterov Momentum into Adam"* (ICLR Workshop, 2016). NAdam.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 8.5.3 (Adam) (MIT Press, 2016).
- Kingma, D. P. and Ba, J., *"Adam: A Method for Stochastic Optimization"* (ICLR, 2015). The original paper.
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 27 (2020).
- Loshchilov, I. and Hutter, F., *"Decoupled Weight Decay Regularization"* (ICLR, 2019). AdamW.
- Reddi, S. J., Kale, S., and Kumar, S., *"On the Convergence of Adam and Beyond"* (ICLR, 2018). AMSGrad and the convergence failure cases.
- Wilson, A. C., Roelofs, R., Stern, M., Srebro, N., and Recht, B., *"The Marginal Value of Adaptive Gradient Methods in Machine Learning"* (NeurIPS, 2017). The generalisation comparison vs SGD.

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

The optimiser group is done. The model can now reach 95%+ on the training data. The natural follow-up question is whether it actually *generalises* to data it has never seen.

- **[Part 28 — Generalisation and testing](../28-generalization-and-testing/index.md)**: splits train and test sets, measures the generalisation gap, and shows that high training accuracy alone is not the whole story.

---

> **Try it yourself:** Hands-on exercises and quizzes for this lecture live in [Exercises](../../exercises.md) and [Quizzes](../../quizzes.md).
