# 27 - Adam

> **TL;DR.** Adam keeps two exponential moving averages per parameter, one of the gradient (the idea of momentum) and one of the squared gradient (the cache of RMSProp), divides each by $1 - \beta^t$ to undo its start at zero, and steps by $\alpha \, \hat{m} / (\sqrt{\hat{v}} + \epsilon)$. On the documented spiral run, `Optimizer_Adam(learning_rate=0.02, decay=1e-5)` ends at a loss of 0.0806 and a training accuracy of 96.33 percent, the best of the six optimisers of Part VI on that seed. Over five seeds it ends between 81.3 and 96.3 percent and is the best of the six on one seed only, so the single run is a result about seed 0 and not a ranking.
>
> **Prerequisites:** [Post 24](../24-momentum/index.md), [Post 26](../26-rmsprop/index.md).
> **Safe to skip?** Skip it if the reader can already write the five lines of the Adam update from memory, say why $t$ starts at 1, and give the size of the first step.
>
> **After reading, you will be able to:**
>
> - Decompose the Adam update into its first-moment average, its second-moment average and its bias correction, and say what each contributes.
> - Implement Optimizer_Adam with two buffers per parameter array and reproduce the documented spiral run.
> - Reason about when Adam is the wrong default, from the measured seed spread and the published cases.

![A flow diagram of the Adam update for one parameter. The gradient g t forks into two lanes. Top, the first moment with beta 1 = 0.9, m t = beta 1 m t minus 1 plus (1 minus beta 1) g t, the averaging of momentum from post 24, then its correction, m hat t = m t over (1 minus beta 1 to the t). Bottom, the second moment with beta 2 = 0.999, the cache of RMSProp from post 26, then its correction v hat t = v t over (1 minus beta 2 to the t). Both feed the update theta t = theta t minus 1 minus alpha times m hat t over (root of v hat t plus epsilon), with epsilon = 10 to the minus 7. Under each line, the first update of a constant gradient 0.5: m 1 = 0.05 and v 1 = 0.00025, corrected to 0.5 and 0.25, a step of one learning rate, against 3.162 learning rates without the corrections.](diagrams/01-adam-update.svg)

*Two averages, two corrections, one update, with the numbers of the first update for a constant gradient of 0.5 (section 3).*

---

## 1. The question: can momentum and the RMSProp cache be used together?

Part VI has changed the update $\theta \leftarrow \theta - \alpha g$ of [post 22](../22-gradient-descent-optimiser/index.md) four times, where $\theta$ is one parameter, $g$ its gradient and $\alpha$ the learning rate.

| Post | Idea | What it changes in the update |
|---|---|---|
| Post 22 | Gradient descent | nothing: the baseline |
| Post 23 | Learning-rate decay | $\alpha$ shrinks as training proceeds |
| Post 24 | Momentum | $g$ is replaced by a velocity that remembers earlier gradients |
| Post 25 | AdaGrad | each parameter's step is divided by the root of its summed squared gradients |
| Post 26 | RMSProp | the sum becomes an exponential moving average, so the divisor stops growing |

Two of these act on different parts of the update.

- **Momentum** changes the numerator: in place of the current gradient it uses a running combination of recent ones.
- **RMSProp** changes the denominator: in place of one $\alpha$ for every parameter it uses $\alpha / (\sqrt{G} + \epsilon)$, where $G$ is that parameter's cache.

Neither interferes with the other, so the question is: **what happens when both are kept, one average for the numerator and one for the denominator, and what else does the combination need?** The answer is Adam, introduced by Kingma and Ba (posted in 2014, published at ICLR 2015). The name stands for adaptive moment estimation: the average of $g$ estimates its first moment, its mean, and the average of $g^2$ estimates its second moment. The one new ingredient is a correction for the fact that both averages start at zero.

---

## 2. The Adam update rule

For one parameter, with $g_t$ its gradient at update $t$, $t$ counting from 1, and $m_0 = v_0 = 0$:

$$\begin{aligned}
m_t &= \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t \quad &\text{(first moment)}\\
v_t &= \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2 \quad &\text{(second moment)}\\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}} \quad &\text{(bias-corrected first moment)}\\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^{\,t}} \quad &\text{(bias-corrected second moment)}\\
\theta_t &= \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad &\text{(parameter update)}
\end{aligned}$$

Five lines, all applied element by element to a whole weight or bias array; the figure at the top of the post draws them as two lanes that meet in the update. The second line is the cache of post 26 with $\rho$ renamed $\beta_2$; here $v$ is the second moment, not the velocity that post 24 called $v$. The last line has the form of the RMSProp update, $\alpha$ divided by $\sqrt{\cdot} + \epsilon$, with $\hat{m}_t$ where RMSProp has the raw gradient.

The first line is not the velocity of post 24. That velocity was $v \leftarrow \beta v - \alpha g$, and the parameter moved by it. Adam's $m$ carries the factor $(1 - \beta_1)$ and no learning rate, so it is an exponential moving average (EMA) of the gradient: a weighted mean, in the units of $g$, whose weights sum to $1 - \beta_1^t$. The learning rate enters once, in the last line.

---

## 3. Bias correction

### 3.1. What the zero start does to the averages

Both averages start at zero, so the first values are $m_1 = (1 - \beta_1) g_1$ and $v_1 = (1 - \beta_2) g_1^2$: a tenth of the gradient for $\beta_1 = 0.9$, and a thousandth of its square for $\beta_2 = 0.999$. Both are estimates of a mean, and both are far too small.

The size of the error is exact for a constant gradient $g$:

$$m_t = (1 - \beta_1) \sum_{k=0}^{t-1} \beta_1^k \, g = (1 - \beta_1^t) \, g$$

The geometric sum leaves $m_t$ short by the factor $1 - \beta_1^t$, and dividing by that factor returns $g$. The same holds for $v_t$ with $\beta_2$ and $g^2$. `snippets/bias_correction.py` traces $g = 0.5$ and prints $\hat{m}_t = 0.50000$ and $\hat{v}_t = 0.25000$ at each of the first five steps. When the gradient varies but its mean does not, the expectation is linear, so the mean of $m_t$ is $(1 - \beta_1^t)$ times the mean of $g$. Over 200,000 random sequences with mean 1 the script measures a mean $\hat{m}_t$ of 1.0003, 1.0033 and 1.0020 at $t = 1$, 3 and 10. When the mean itself drifts, the correction is approximate.

```text
     t   beta_1^t  factor for m   beta_2^t  factor for v
     1    0.90000        10.000    0.99900      1000.000
    10    0.34868         1.535    0.99004       100.451
   100    0.00003         1.000    0.90479        10.503
  1000    0.00000         1.000    0.36770         1.582
  5000    0.00000         1.000    0.00672         1.007
```

The correction of $m$ is gone after about 50 steps (the factor is 1.005 there). The correction of $v$ lasts a hundred times longer: it is still 10.5 at step 100 and 1.58 at step 1,000. These are factors on $v$; the step divides by $\sqrt{\hat{v}}$ and feels their square roots, 3.24 and 1.26. No schedule switches either of them off; $\beta^t$ goes to zero by itself.

### 3.2. What the correction does to the step

The step divides $\hat{m}_t$ by $\sqrt{\hat{v}_t}$, so the two corrections act against each other. Leaving $\epsilon$ aside, and for any sequence of gradients,

$$\frac{m_t}{\sqrt{v_t}} = \frac{1 - \beta_1^t}{\sqrt{1 - \beta_2^t}} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}}$$

At $t = 1$ the factor is $0.1 / \sqrt{0.001} = 3.162$. The denominator is short by more than the numerator, so an optimiser without the correction takes steps that are too large, not too small:

```text
     t  corrected  uncorrected = (1-b1^t) / sqrt(1-b2^t)
     1      1.000        3.162
    10      1.000        6.528
   100      1.000        3.241
  1000      1.000        1.258
largest uncorrected step: 6.569 learning rates at t = 12
uncorrected step still above 1.1 learning rates until t = 1750
```

The table is in units of $\alpha$ for a constant gradient, where the corrected step is exactly $\alpha$. Bias correction is therefore a brake on the early updates: compared with the uncorrected rule it multiplies the step by 0.316 at the first update and by as little as $1/6.569 = 0.152$ at the twelfth, and it releases the brake over the first two thousand updates. Kingma and Ba describe it the same way: the two averages are biased towards zero by their start, and with $\beta_2$ close to 1 a missing correction leads to initial steps that are much larger. Their paper also writes the corrected rule as the uncorrected one run at the rate $\alpha \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$, the reciprocal of the factor above. The figure draws the two tables of this section as curves over the first 10,000 updates.

![Two charts against the update t on a log axis from 1 to 10,000, for beta 1 = 0.9 and beta 2 = 0.999. Left, the correction factor 1 over (1 minus beta to the t) on a log scale: for m it falls from 10 to 1.535 at t = 10 and 1.000 at t = 100; for v from 1,000 to 100.451 at t = 10, 10.503 at t = 100, 1.582 at t = 1,000 and 1.007 at t = 5,000. Right, the step for a constant gradient in learning rates: 1 at every t with the correction; without it 3.162 at t = 1, 6.528 at t = 10, 3.241 at t = 100 and 1.258 at t = 1,000, largest at 6.569 at t = 12 and above 1.1 until t = 1,750.](diagrams/02-bias-correction.svg)

*The two corrections fade by themselves; without them the early steps rise to 6.569 learning rates before they settle.*

The first corrected update is the cleanest case. $\hat{m}_1 = g_1$ and $\hat{v}_1 = g_1^2$, so the step is $\alpha \, g_1 / (|g_1| + \epsilon)$: the learning rate, in the direction against the gradient, whatever the size of the gradient. The script hands `Optimizer_Adam()` four weight gradients and one bias gradient:

```text
gradients        [-1.e+04 -1.e+00  1.e-04  3.e+00] [1.e-08]
weights after    [ 0.001     0.001    -0.000999 -0.001   ]
bias after       [-9.09090909e-05]  = -0.001 * 1e-8 / (1e-8 + 1e-7)
```

Gradients eight orders of magnitude apart all move their weight by 0.001. Only a gradient as small as $\epsilon$ itself moves less: the bias, with a gradient of $10^{-8}$, moves by an eleventh of the learning rate.

`snippets/no_bias_correction.py` removes the two correction lines from the class and trains on the setup of section 6, with the same learning rate and decay, for five seeds. Final training accuracy, in percent:

| Seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Adam (`seeds_adam.py`) | 96.3 | 82.3 | 81.3 | 88.7 | 96.3 | 89.0 |
| Adam, uncorrected | 96.7 | 97.0 | 96.3 | 93.3 | 97.3 | 96.1 |

The uncorrected optimiser is not slower to start. Its loss at epoch 100 is between 0.4487 and 0.6775, against 0.7692 to 0.9988 with the correction, and it ends with the lower loss and the higher accuracy in every one of the five runs. At this learning rate the problem rewards larger early steps, and section 7 finds the same by raising the rate of the corrected optimiser. At a larger rate it does not. `snippets/no_bias_correction_high.py` runs the uncorrected optimiser at 0.1, where the corrected one ends between 89.3 and 98.7 percent (section 7): it ends between 41.7 and 52.0 percent, behind on all five seeds. Removing the correction multiplies the early steps by up to 6.6, and whether that helps depends on how far $\alpha$ lies below the largest rate the problem tolerates. Kingma and Ba report the failure on a variational autoencoder, where training without the correction was unstable in its first epochs when $\beta_2$ was close to 1. What the correction buys is that $\alpha$ is the size of the step from the first update, whatever $\beta_1$ and $\beta_2$ are. Without it, the early steps depend on the two decay rates: with $\beta_2 = 0.9999$ the first step would be $0.1 / \sqrt{0.0001} = 10$ learning rates.

### 3.3. What the defaults mean

The class defaults are $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\epsilon = 10^{-7}$. The first three are the values Kingma and Ba suggest; their $\epsilon$ is $10^{-8}$, and the series keeps the $10^{-7}$ of its other optimisers and of Kinsley and Kukieła.

An EMA with decay $\beta$ remembers about $1 / (1 - \beta)$ steps: the script prints that the last 10 steps carry 0.651 of the weight for $\beta_1 = 0.9$, and the last 1,000 carry 0.632 for $\beta_2 = 0.999$. The direction of the step therefore follows the last ten gradients, and its scale follows the last thousand. Kingma and Ba give sparse gradients as the reason for the long window: where a parameter's gradient is zero at most updates, a reliable estimate of its second moment has to average over many of them. A shorter window makes each parameter's rate follow recent gradients more closely; this post does not measure that setting.

The default $\alpha = 0.001$ is a starting point, not a constant of nature. Because the step is about $\alpha$ per parameter whatever the gradient's size, the same value is a sensible first try on very different problems, which is not true of gradient descent.

---

## 4. The optimiser class

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
        # Lazy buffer creation: one momentum and one cache per parameter array.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache     = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)
            layer.bias_cache       = np.zeros_like(layer.biases)

        # 1) First moment: moving average of the gradient.
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums   = self.beta_1 * layer.bias_momentums + \
                                 (1 - self.beta_1) * layer.dbiases

        # 2) Bias correction of the first moment. The step counter is
        #    incremented in post_update_params, after this call, hence the + 1.
        t = self.iterations + 1
        weight_m_hat = layer.weight_momentums / (1 - self.beta_1 ** t)
        bias_m_hat   = layer.bias_momentums   / (1 - self.beta_1 ** t)

        # 3) Second moment: moving average of the squared gradient.
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache   = self.beta_2 * layer.bias_cache + \
                             (1 - self.beta_2) * layer.dbiases ** 2

        # 4) Bias correction of the second moment.
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

**The contract is unchanged.** `pre_update_params` once, `update_params` once per layer, `post_update_params` once: the three methods of post 23. The constructor and the first and last methods are those of `Optimizer_RMSprop` with `rho` renamed `beta_2` and its default raised from 0.9 to 0.999, `beta_1` added, and a default learning rate of 0.001 in place of 0.02. Everything new is inside `update_params`.

**Two buffers per parameter array.** Momentum stored `weight_momentums` and RMSProp stored `weight_cache`; Adam stores both under the same names. The corrected values `weight_m_hat` and `weight_v_hat` are temporary: the buffers keep the uncorrected $m$ and $v$, because the next update must continue from those. For the network of this post the two layers hold 387 parameters and the optimiser 774 further numbers.

**$t$ is `self.iterations + 1`.** The counter starts at 0 and is raised after the layers have been updated, so the first call must use $t = 1$. With $t = 0$ both denominators are $1 - \beta^0 = 0$ (section 12).

**$\epsilon$ is outside the square root**, as in the classes of posts 25 and 26.

The released projects `nn-p01` to `nn-p04` all train with this class. Their copies have the same arguments, defaults and arithmetic; those of `nn-p01`, `nn-p03` and `nn-p04` also refuse, when the optimiser is built, a learning rate or an epsilon that is not positive, a negative decay, and a $\beta$ outside $[0, 1)$.

---

## 5. The training loop

The loop is that of every post of Part VI. Only the object changes:

```python
    optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)
    result = train(optimizer)
```

and inside `train`, after the backward pass:

```python
        # Update: the same three kinds of call as for every optimiser of Part VI.
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
```

A learning rate of 0.02 is twenty times the class default. It is the documented setting of this run, not a tuned one; section 7 tries 0.001 and 0.1 as well. The decay of $10^{-5}$ does little in 10,001 epochs: the script prints a learning rate of 0.018182 at the last epoch, $0.02 / 1.1$.

---

## 6. What happens when it runs

The setup is the shared one of Part VI, printed by `snippets/adam.py`: `nnfs.init()`, called once, which sets seed 0, makes float32 the default and makes `np.dot` return float32; `spiral_data(samples=100, classes=3)`; `Layer_Dense(2, 64)`, ReLU, `Layer_Dense(64, 3)`, softmax and cross-entropy; weights of `0.01 * randn`; 10,001 full-batch epochs. Loss and accuracy are measured on the 300 training points, in the forward pass of each epoch and so before that epoch's update.

```text
epoch     0  loss 1.0986  acc 0.3600
epoch   100  loss 0.7692  acc 0.6733
epoch  1000  loss 0.2382  acc 0.9133
epoch  2000  loss 0.1621  acc 0.9333
epoch  5000  loss 0.1097  acc 0.9467
epoch 10000  loss 0.0806  acc 0.9633

final loss 0.0806   final accuracy 0.9633 (289 of 300)
peak accuracy 0.9767 first reached at epoch 9309
first epoch with accuracy of 0.9 or more: 722
```

The loss starts at $\ln 3 = 1.0986$, the loss of a uniform guess, and ends at 0.0806 with 289 of the 300 points classified correctly. The accuracy is not monotone: it touches 97.67 percent at epoch 9,309 and is back at 96.33 at the end, a difference of four points out of 300. This is the 96.3 percent the series quotes for Adam, and it is the result of this seed.

---

## 7. Six optimisers, five seeds

`snippets/all_six.py` runs every optimiser of Part VI on the same setup and seed, each with the settings its own post documents:

```python
OPTIMISERS = {
    "SGD (post 22)":          lambda: Optimizer_SGD(learning_rate=1.0),
    "SGD, decay (post 23)":   lambda: Optimizer_SGD(learning_rate=1.0, decay=1e-3),
    "SGD, momentum (post 24)": lambda: Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9),
    "AdaGrad (post 25)":      lambda: Optimizer_Adagrad(learning_rate=1.0, decay=1e-4),
    "RMSProp (post 26)":      lambda: Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999),
    "Adam (post 27)":         lambda: Optimizer_Adam(learning_rate=0.02, decay=1e-5),
}
```

```text
optimiser                seed  loss@100  loss@1000  final loss  accuracy  peak acc  epoch of 90%
SGD (post 22)               0    1.0869     1.0623      0.8737    0.6467    0.6700         never
SGD, decay (post 23)        0    1.0882     1.0631      0.7612    0.6467    0.6900         never
SGD, momentum (post 24)     0    1.0535     0.4472      0.1209    0.9567    0.9567         1,526
AdaGrad (post 25)           0    1.0114     0.6704      0.3847    0.8400    0.8500         never
RMSProp (post 26)           0    1.0134     0.6373      0.2379    0.9000    0.9100         6,503
Adam (post 27)              0    0.7692     0.2382      0.0806    0.9633    0.9767           722
```

On seed 0 Adam has the lowest loss at every checkpoint and the highest final accuracy, and it passes 90 percent at epoch 722, where momentum needs 1,526 and RMSProp 6,503. One seed is one draw of the data and of the initial weights. `seeds_adam.py` and `seeds_rmsprop.py` repeat the last two rows for seeds 0 to 4, set with `np.random.seed` after a single `nnfs.init()`, so that both the data and the weights change. The first four rows of the table below are the five-seed runs of posts 22 to 25, made with the same setup and seeds; seed 0 of every row is the run above. Final training accuracy, in percent:

| Optimiser | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| SGD | 64.7 | 74.7 | 53.7 | 87.0 | 78.7 | 71.7 |
| SGD, decay | 64.7 | 69.3 | 58.0 | 67.0 | 63.7 | 64.5 |
| SGD, momentum | 95.7 | 68.7 | 73.0 | 81.7 | **98.0** | 83.4 |
| AdaGrad | 84.0 | 75.7 | 81.7 | **92.7** | 79.7 | 82.7 |
| RMSProp | 90.0 | **89.0** | **87.3** | 91.7 | 95.0 | 90.6 |
| Adam | **96.3** | 82.3 | 81.3 | 88.7 | 96.3 | 89.0 |

The best figure of each seed is in bold, and four different optimisers hold one. The figure draws each row as a band over the five seeds.

![A chart of the final training accuracy on the spiral after 10,001 epochs for the six optimisers of Part VI, one row each, a band from the lowest to the highest of seeds 0 to 4 with a tick per seed and a dot for seed 0, on an axis from 50 to 100 percent. Gradient descent, post 22: seed 0 64.7, mean 71.7, best on no seed. Learning-rate decay, post 23: 64.7, mean 64.5, none. Momentum, post 24: 95.7, mean 83.4, best on seed 4. AdaGrad, post 25: 84.0, mean 82.7, best on seed 3. RMSProp, post 26: 90.0, mean 90.6, best on seeds 1 and 2. Adam, this post, in blue: 96.3, mean 89.0, best on seed 0.](diagrams/04-six-optimisers.svg)

*The seed-0 dots alone put Adam first; the bands overlap, and the best of each seed is four different optimisers.*

What the five seeds support:

- **Adam is the best of the six on seed 0 and on no other seed.** It is second on seeds 1 and 4 and third on seeds 2 and 3. Its final accuracy runs from 81.3 to 96.3 percent and its final loss from 0.0806 to 0.4722.
- **On these five seeds RMSProp has the highest mean and the narrowest range**, 87.3 to 95.0 percent, and it ends above Adam on three. Five more seeds undo both leads. On seeds 5 to 9 post 26 measures 77.3, 89.0, 77.0, 85.7 and 85.3 percent for RMSProp, and `seeds_adam_more.py` 84.0, 79.7, 86.7, 82.0 and 95.7 for Adam. Over the ten seeds RMSProp runs from 77.0 to 95.0 percent with a mean of 86.7 and Adam from 79.7 to 96.3 with a mean of 87.3, and each ends above the other on five.
- **The last epoch is a steadier reading for Adam than for RMSProp.** Post 22 warned that a final epoch is one draw from the end of a jumpy run. In epochs 9,001 to 10,000 Adam's accuracy is never more than 4.7 points under its final figure on seeds 0 to 4, while RMSProp's falls to between 52.0 and 70.0 percent somewhere in that window on every one of them, the late spikes of post 26. The means over the window, which both scripts print, are within 1.6 points of the final figures for Adam and within 0.9 for RMSProp, and they put the two optimisers in the same order as the final figures on all five seeds.
- **Adam ends above momentum on four of the five seeds.** Momentum's range, 68.7 to 98.0 percent, is about twice as wide as Adam's.
- **Adam passes 90 percent on two of the five seeds**, at epochs 722 and 1,057. RMSProp passes it on four, and never before epoch 3,518.
- **Adam is ahead of RMSProp early.** Its loss at epoch 1,000 is the lower of the two on all five seeds, 0.2382 to 0.6623 against 0.5399 to 0.7692.
- **In final accuracy, no optimiser of Part VI improves on its predecessor on every seed.** Decay ends below the constant rate on three of the five, and AdaGrad below momentum on two. Momentum comes closest: it ends below decay on seed 1 only, by two of the 300 points, and post 24 shows its loss is the lower one on all five.

The ranking on seed 0 is therefore a property of that seed. With these settings and five seeds, the claim that Adam is the best optimiser on the spiral does not hold, and neither does any other ordering of the top four.

The learning rate was not tuned either. `rate_low.py` and `rate_high.py` rerun Adam on the same five seeds at 0.001, the class default, and at 0.1:

| Learning rate | Final accuracy, five seeds | mean | Final loss | Seeds that pass 90 percent |
|---|:---:|:---:|:---:|:---:|
| 0.001 | 88.0 to 93.3 | 90.6 | 0.1863 to 0.3404 | 3 |
| 0.02 (documented) | 81.3 to 96.3 | 89.0 | 0.0806 to 0.4722 | 2 |
| 0.1 | 89.3 to 98.7 | 94.8 | 0.0390 to 0.2655 | 5 |

Across a factor of 100 in the learning rate every one of the fifteen runs trains, to between 81.3 and 98.7 percent. The largest rate does best on four of the five seeds, which agrees with the uncorrected runs of section 3.2: this network on this data wants larger steps than 0.02 gives it. At a rate of 1.0, the setting of post 22, Adam fails on all five seeds (section 12). The figure puts the fifteen runs beside the ten uncorrected ones of section 3.2.

![A chart of Adam's final training accuracy on the spiral after 10,001 epochs, one row per setting, each a band from the lowest to the highest of seeds 0 to 4 with a tick per seed and a dot for seed 0, on an axis from 40 to 100 percent, with the mean and the number of seeds past 90 percent at the right. In blue, with the correction: learning rate 0.001, 88.0 to 93.3, mean 90.6, 3 of 5; 0.02, 81.3 to 96.3, mean 89.0, 2 of 5; 0.1, 89.3 to 98.7, mean 94.8, 5 of 5. In grey, without the correction: 0.02, 93.3 to 97.3, mean 96.1, 5 of 5; 0.1, 41.7 to 52.0, mean 46.7, 0 of 5.](diagrams/03-rates-and-correction.svg)

*Every corrected rate trains; the uncorrected optimiser is ahead at 0.02 on every seed and behind at 0.1 on every seed.*

---

## 8. Why Adam became the default

Adam is the optimiser most code reaches for first, and the four projects of this series do the same: `nn-p01` and `nn-p03` train with the default learning rate of 0.001, `nn-p02` and `nn-p04` with 0.01, all with the default $\beta_1$, $\beta_2$ and $\epsilon$. The reasons are in the update rule and not in a table of final accuracies.

**The learning rate has a meaning.** The step of each parameter is about $\alpha$ when its recent gradients agree in sign, and smaller when they disagree, because $m$ then averages towards zero while $v$ does not: for a gradient that flips between $+1$ and $-1$ at every update, `bias_correction.py` prints a step of 0.0526 learning rates. It can also be larger than $\alpha$: for a parameter whose gradient was zero for a long time and then appears, the step approaches $(1 - \beta_1) / \sqrt{1 - \beta_2} = 3.16$ learning rates, the upper bound Kingma and Ba give. The step does not grow with the size of the gradient (section 3.2). A rate chosen for one layer, or one problem, is therefore a reasonable guess for another. In gradient descent the step is $\alpha g$, and $g$ can differ by orders of magnitude between layers.

**The usable range is wide.** Section 7 measured it on this problem: rates from 0.001 to 0.1 all train.

**Each parameter gets its own rate, and the rate does not die.** This is inherited from posts 25 and 26.

**The start is controlled.** The bias correction keeps the first steps at $\alpha$ for any $\beta_1$ and $\beta_2$, so changing a decay rate does not silently change the early step size.

---

## 9. When Adam is the wrong default

**When a tuned alternative generalises better.** Wilson et al. (2017) compared adaptive methods with gradient descent and momentum on several deep-learning tasks and found that the adaptive methods, Adam among them, often generalised worse on held-out data, even where their training performance was better. This post measures training accuracy only; held-out data starts in post 28.

**When memory is the constraint.** Adam keeps two extra numbers per parameter, momentum one, plain gradient descent none. For a model of 100 million float32 parameters the two buffers are $2 \cdot 10^8 \cdot 4$ bytes, 800 MB of optimiser state.

**When the comparison has been run and says otherwise.** On the spiral, Adam and RMSProp each end above the other on five of ten seeds, and momentum ends above both on seed 4. The practical advice is to start with Adam and to compare it with momentum, over more than one seed, when the result matters.

---

## 10. Variants worth knowing

The variants keep the skeleton and change one line.

- **AdamW** (Loshchilov and Hutter, 2019). When an L2 penalty ([post 30](../30-l1-and-l2-regularisation/index.md)) is added to the gradient, Adam divides it by $\sqrt{\hat{v}}$ like the rest of the gradient, so parameters with a large gradient history are decayed less. AdamW takes the decay out of the gradient and applies it to the weights directly.
- **AMSGrad** (Reddi, Kale and Kumar, 2018). The authors gave examples on which Adam does not converge, and a repair: divide by the largest second-moment estimate seen so far, so that a parameter's rate cannot rise again.
- **NAdam** (Dozat, 2016). Adam with Nesterov momentum in the first moment.
- **Lion** (Chen et al., 2023). No second moment: the step is the sign of a momentum term, so one buffer per parameter is enough.

---

## 11. Make it run: Adam, its checks and the comparison

Every code block and every number of this post comes from the scripts in `snippets/`, run from the series root, for example:

```text
python posts/27-adam-optimiser/snippets/adam.py
```

| Script | What it runs | Time |
|---|---|---|
| `adam.py` | the classes, `Optimizer_Adam`, `train`, the documented run | about 10 s |
| `bias_correction.py` | the factors, the traces and the first update of section 3 | 1 s |
| `all_six.py` | six optimisers on seed 0 | about 45 s |
| `seeds_adam.py`, `seeds_rmsprop.py` | one optimiser each, seeds 0 to 4 | about 40 s each |
| `seeds_adam_more.py` | Adam, seeds 5 to 9 | about 40 s |
| `no_bias_correction.py`, `rate_low.py`, `rate_high.py` | Adam uncorrected, at 0.001 and at 0.1, seeds 0 to 4 | about 40 s each |
| `no_bias_correction_high.py` | Adam uncorrected at 0.1, seeds 0 to 4 | about 40 s |
| `what_can_go_wrong.py` | the mistakes of section 12 | about 5 s |

All need NumPy and the `nnfs` package. The network classes are those of posts 16 and 19, unchanged; `all_six.py` holds the optimiser classes of posts 22 to 26. Each spread covers five seeds, which is enough to show that an ordering is not stable and too few to rank optimisers that are close.

---

## 12. What can go wrong?

**The counter read without the `+ 1`.** With `t = self.iterations` the first update divides both moments by zero. NumPy warns and carries on:

```text
warnings: ['divide by zero encountered in divide', 'invalid value encountered in divide']
1 - beta_1 ** 0 = 0.0   NaN weights in dense1: 128 of 128
```

No exception is raised. Every weight of the layer is NaN after one call, and so is the loss of the next forward pass.

**The learning rate of gradient descent.** `Optimizer_Adam(learning_rate=1.0)` moves every parameter by about 1.0 in the first update, a hundred times the scale of the initial weights. Over five seeds and 1,001 epochs (not the 10,001 of the documented run):

```text
seed  loss@1  loss@1000  accuracy@1000  largest loss
   0   6.261     1.0724         0.3567        10.745
   2   6.390     0.8460         0.5167        10.567
   3   6.735     1.0986         0.3333         9.585
```

The loss jumps from 1.0986 to between 4.205 and 6.735 after one update, peaks near 10, and after 1,000 epochs the accuracy is between 33.3 and 51.7 percent on the five seeds.

**`post_update_params` left out.** The counter stays at 0, so $t$ is 1 for ever and the decay never acts. With a constant gradient, a rate of 0.02 and a decay of $10^{-3}$:

```text
update  step, loop complete  rate    step, call missing  rate
    10              0.01982  0.0198             0.04129  0.0200
 10000              0.00182  0.0018             0.00632  0.0200
```

The moments keep being divided by 0.1 and 0.001 after they have warmed up. The step is first too large and then settles at $10 / \sqrt{1000} = 0.316$ of the learning rate, and nothing is raised.

**A layer another optimiser has already updated.** The buffers of posts 24 to 26 share Adam's names. A layer that carries a `weight_cache` from AdaGrad or RMSProp skips the creation of the buffers, and the first-moment line fails:

```text
AttributeError: 'Layer_Dense' object has no attribute 'weight_momentums'
```

A change of optimiser needs fresh layers, or the old buffers deleted.

**The correction removed.** Nothing is raised. At the documented rate of 0.02 the spiral runs are better; at 0.1 all five end between 41.7 and 52.0 percent, where the corrected optimiser reaches 89.3 to 98.7 (section 3.2). The early steps are up to 6.6 times the learning rate that was asked for, and the multiple depends on $\beta_1$ and $\beta_2$.

---

## 13. Summary

| Concept | Takeaway |
|---|---|
| First moment | $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$, an EMA of the gradient over about 10 steps |
| Second moment | $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$, an EMA of its square over about 1,000 steps |
| Bias correction | $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$, with $t$ from 1 |
| Update | $\theta \leftarrow \theta - \alpha \, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$; the first step is $\alpha$ for any gradient much larger than $\epsilon$ |
| Effect of the correction | holds the early steps at $\alpha$; without it they are up to 6.569 times larger |
| Defaults | $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$ |
| Documented run | loss 0.0806, accuracy 96.33 percent, seed 0, rate 0.02, decay $10^{-5}$ |
| Five seeds | 81.3 to 96.3 percent, mean 89.0; best of the six on one seed. Ten seeds: 79.7 to 96.3, mean 87.3 |

---

## Common pitfalls

1. **Counting $t$ from 0.** `self.iterations + 1` in the two denominators, or the first update turns every parameter into NaN without an exception.
2. **Giving Adam the learning rate of `Optimizer_SGD`.** Its step is about $\alpha$ per parameter, not $\alpha g$; a rate of 1.0 wrecks the first update.
3. **Reading the correction as a speed-up.** It makes the early steps smaller than the uncorrected rule would, and it lasts about two thousand updates for $\beta_2 = 0.999$, not a few dozen.
4. **Ranking optimisers from one seed.** The order of Adam, RMSProp, momentum and AdaGrad on the spiral changes with the seed.
5. **Handing over a layer that still carries another optimiser's buffers.** The names are shared; the check for existing buffers is on `weight_cache`.

---

## Further reading

- Chen, X., et al., *"Symbolic Discovery of Optimization Algorithms"* (NeurIPS, 2023). Lion.
- Dozat, T., *"Incorporating Nesterov Momentum into Adam"* (ICLR Workshop, 2016). NAdam.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 8 (MIT Press, 2016).
- Kingma, D. P. and Ba, J., *"Adam: A Method for Stochastic Optimization"* (ICLR, 2015). The original paper.
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 10 (2020).
- Loshchilov, I. and Hutter, F., *"Decoupled Weight Decay Regularization"* (ICLR, 2019). AdamW.
- Reddi, S. J., Kale, S., and Kumar, S., *"On the Convergence of Adam and Beyond"* (ICLR, 2018). AMSGrad.
- Wilson, A. C., Roelofs, R., Stern, M., Srebro, N., and Recht, B., *"The Marginal Value of Adaptive Gradient Methods in Machine Learning"* (NeurIPS, 2017).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 28 - Generalization and testing](../28-generalization-and-testing/index.md):** the network trained here is run on spiral points it has never seen, and training accuracy stops being the whole story.
- **[Post 32 - Mini-batching](../32-mini-batching/index.md):** the same `Optimizer_Adam` updating once per batch instead of once per epoch, as the projects `nn-p01`, `nn-p03` and `nn-p04` use it.
