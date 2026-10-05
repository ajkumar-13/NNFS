# 26 - RMSProp

> **TL;DR.** RMSProp changes one statement of AdaGrad: the cache becomes an exponential moving average of the squared gradients, $G \leftarrow \rho G + (1 - \rho) g^2$, instead of their running sum, so it never exceeds the largest squared gradient seen and can fall again. On the shared spiral run AdaGrad's mean update shrinks about twelvefold between epochs 100 and 10,000, RMSProp's less than twofold. The price is noise: over ten seeds RMSProp ends at 77.00 to 95.00 percent accuracy against AdaGrad's 75.67 to 92.67 and is ahead in seven, but its loss spikes late in training, and in eight seeds its accuracy falls below 70 percent somewhere in the last 1,000 epochs.
>
> **Prerequisites:** [Post 25](../25-adagrad/index.md).
> **Safe to skip?** Skip it if the reader can already write the RMSProp cache update from memory, say why that cache is bounded where AdaGrad's sum is not, and say how far the first update moves a parameter when the cache starts at zero.
>
> **After reading, you will be able to:**
>
> - Explain why an exponential moving average of squared gradients gives a bounded cache and how the decay factor rho sets its memory horizon.
> - Implement Optimizer_RMSprop with per-layer caches and sensible defaults for the learning rate, rho and epsilon.
> - Place RMSProp between AdaGrad and Adam by naming what it keeps from the first and what the second adds to it.

![Two cache update rules side by side above a log-log chart of cache value against iteration, for a gradient of constant size 0.5. AdaGrad's cache climbs along a straight line from 0.25 to 2,500 at iteration 10,000. RMSProp's cache with rho 0.9 rises to 0.25 within the first hundred iterations and then stays flat.](diagrams/01-rmsprop-vs-adagrad-cache.svg)

*The same gradient feeds both rules. The sum has no ceiling; the average settles at the squared gradient. The side panel's RMSProp value at iteration 100 rounds to 0.250, not 0.249 (section 3), and the figure's "Part 25" is post 25.*

---

## 1. The question: what one-line change bounds AdaGrad's cache?

AdaGrad ([post 25](../25-adagrad/index.md)) ended with one flaw: its cache $G_t = \sum_{s \le t} g_s^2$ can only grow. The factor that scales each step, $\alpha / (\sqrt{G_t} + \epsilon)$, can therefore only shrink, whatever the gradient says later.

The structural cause is that **the cache has unbounded memory**. Every gradient ever seen still contributes to the sum with full weight, no matter how old. A parameter whose gradient was large for the first thousand updates and small for the next ten thousand is still scaled by the early turbulence.

The fix is a cache that **forgets old gradients** as new ones arrive:

- Recent gradients should dominate the cache, so that the per-parameter scaling reflects current behaviour.
- Old gradients should fade, so that a parameter can recover its step size after a calm period.
- The cache should still be large where gradients are large and small where they are small, so that AdaGrad's per-parameter idea survives.

The tool is the **exponential moving average** (EMA). The question of this post is how one changed statement turns AdaGrad's sum into such an average, and how its decay factor $\rho$ sets the length of the memory.

---

## 2. The exponential moving average

The EMA of a stream of values $x_1, x_2, x_3, \dots$ is defined recursively:

$$E_t = \rho \, E_{t-1} + (1 - \rho) \, x_t$$

where $\rho \in [0, 1)$ is the **decay factor**. The EMA at step $t$ is a weighted average of the previous EMA and the new sample, with weights $\rho$ and $1 - \rho$ that are non-negative and sum to one.

For RMSProp's cache the stream is the squared gradient, and the cache starts at $G_0 = 0$:

$$G_t = \rho \, G_{t-1} + (1 - \rho) \, g_t^2$$

Three properties make this the fix AdaGrad needed.

**The cache is bounded.** Suppose no squared gradient exceeds some number $M$. If $G_{t-1} \le M$, then $G_t \le \rho M + (1 - \rho) M = M$, and $G_0 = 0 \le M$, so by induction the cache never exceeds the largest squared gradient it has been fed. AdaGrad's sum has no such ceiling. The scale $\alpha / (\sqrt{G_t} + \epsilon)$ is then never smaller than $\alpha / (\sqrt{M} + \epsilon)$.

**The cache follows the gradient.** If $g_t^2$ stays at a constant value $c$, subtracting $c$ from both sides of the recursion gives $G_t - c = \rho \, (G_{t-1} - c)$: the gap to $c$ shrinks by the factor $\rho$ at every step. From the zero start, $G_t = c \, (1 - \rho^t)$, which approaches $c$ and never passes it.

**Old values fade geometrically.** Unrolling the recursion gives $G_t = (1 - \rho) \sum_{k=0}^{t-1} \rho^k g_{t-k}^2$: the squared gradient from $k$ steps ago carries the weight $(1 - \rho) \rho^k$, which is $\rho^k$ times the weight of the newest one. `snippets/ema_cache.py` prints how fast that is:

```text
rho     1/(1-rho)  rho^10      rho^100     rho^1000    rho^(1/(1-rho))  weight of the newest 1/(1-rho)
0.9     10         0.3487      2.66e-05    1.75e-46    0.3487           0.6513
0.99    100        0.9044      3.66e-01    4.32e-05    0.3660           0.6340
0.999   1000       0.9900      9.05e-01    3.68e-01    0.3677           0.6323
1/e = 0.3679
```

With $\rho = 0.9$ a gradient from 10 steps ago still counts 0.35 times as much as the newest, and one from 100 steps ago $2.7 \times 10^{-5}$ times as much. The usual summary is that the average has a **memory horizon** of about $1 / (1 - \rho)$ steps: 10 for $\rho = 0.9$, 100 for 0.99, 1,000 for 0.999. That is a closed-form rule of thumb, not a cut-off. At an age of $1 / (1 - \rho)$ steps the weight has fallen to roughly $1/e$ of the newest, and the newest $1 / (1 - \rho)$ samples carry about 63 to 65 percent of the total weight. A larger $\rho$ means a longer memory and a cache that changes more slowly.

---

## 3. AdaGrad against RMSProp, side by side

Suppose the gradient has size $|g| = 0.5$ at every step, so $g^2 = 0.25$. The script feeds that stream to both caches and prints them with the factor $1 / \sqrt{G}$ by which each divides the step:

| Step $t$ | AdaGrad $G_t$ | AdaGrad $1/\sqrt{G_t}$ | RMSProp $G_t$ ($\rho = 0.9$) | RMSProp $1/\sqrt{G_t}$ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.250 | 2.0000 | 0.0250 | 6.3246 |
| 2 | 0.500 | 1.4142 | 0.0475 | 4.5883 |
| 10 | 2.500 | 0.6325 | 0.1628 | 2.4782 |
| 100 | 25.000 | 0.2000 | 0.2500 | 2.0000 |
| 1,000 | 250.000 | 0.0632 | 0.2500 | 2.0000 |
| 10,000 | 2,500.000 | 0.0200 | 0.2500 | 2.0000 |
| 100,000 | 25,000.000 | 0.0063 | 0.2500 | 2.0000 |

AdaGrad's cache is $0.25 \, t$, so its factor falls as $1 / \sqrt{t}$ without end. RMSProp's cache is $0.25 \, (1 - 0.9^t)$, the closed form of section 2; it is within 1 percent of 0.25 from step 44 on and stays there. The factor settles at $1 / \sqrt{0.25} = 2$ and remains useful for as long as the gradient keeps its size.

If the size of the gradient changes, the cache tracks the change. In the script's second stream the gradient is 2 for 1,000 steps and 0.1 afterwards:

```text
     t   AdaGrad G   RMSProp 0.9   RMSProp 0.999
  1000     4000.00        4.0000          2.5292
  1050     4000.50        0.0306          2.4063
  1100     4001.00        0.0101          2.2894
  2000     4010.00        0.0100          0.9363
 11000     4100.00        0.0100          0.0101
rho 0.9: within 1 percent of the new level 0.01 after 101 steps
rho 0.999: within 1 percent of the new level 0.01 after 10,130 steps
AdaGrad: 1/sqrt(G) is 0.0158 at t = 1,000 and 0.0154 at t = 21,000; RMSProp 0.9: 0.5000 and 10.0000
```

With $\rho = 0.9$ the cache falls from 4 to the new level 0.01 in about a hundred steps, and the factor $1 / \sqrt{G}$ rises from 0.5 to 10. With $\rho = 0.999$ the same recovery takes about ten thousand steps, and at step 1,000 the cache has reached only 2.5292 of the 4 it is heading for, because of its zero start. AdaGrad cannot recover at all: the 4,000 it accumulated during the turbulent phase is permanent, and its factor stays near 0.016.

---

## 4. The update rule

The step itself is the one AdaGrad takes:

$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{G_t} + \epsilon} \, g_t$$

Only the definition of $G_t$ changes. As in post 25 and in the code, $\epsilon$ stands outside the square root.

| Piece | Comes from | What it does |
|---|---|---|
| Per-parameter scale $\alpha / (\sqrt{G_t} + \epsilon)$ | AdaGrad | a larger gradient history means a smaller step |
| Cache $G_t$ as an EMA of $g^2$ | new in RMSProp | the history is recent, bounded, and can shrink |

One consequence decides how the learning rate must be chosen. While a parameter's gradient keeps a steady size, the cache settles at $g^2$, so $\sqrt{G}$ is close to $|g|$ and the step $\alpha \, g / (\sqrt{G} + \epsilon)$ is close to $\alpha$ in size, **whatever the size of the gradient**. Under RMSProp the learning rate is roughly the distance a parameter moves per update, not a multiplier on the gradient. A parameter with a gradient of 0.001 and one with a gradient of 10 take steps of about the same length.

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
        # Lazy cache creation on first call.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        # Moving average of squared gradients: the two statements that differ from AdaGrad.
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache   = self.rho * layer.bias_cache + \
                             (1 - self.rho) * layer.dbiases ** 2

        # Per-parameter update, as in AdaGrad.
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
```

**What differs from `Optimizer_Adagrad`.** Three things, and nothing else. The constructor takes one more argument, `rho`, and stores it. The default `learning_rate` is `0.02` where AdaGrad's is `1.0`. And the cache statement, once for the weights and once for the biases, is an assignment of the moving average where AdaGrad's is `+=` of the squared gradient. The decay in `pre_update_params`, the lazy creation of `weight_cache` and `bias_cache` on the layer, the division by the root of the cache and the counter are those of post 25, so the class obeys the same three-call contract.

**The default learning rate is small because it is a distance.** By section 4 a parameter moves about `learning_rate` per update. A rate of `1.0`, the default of `Optimizer_SGD`, would move every weight by about 1 at each step, in a network whose weights start near 0.01. Section 11 measures what that does.

**`rho` defaults to 0.9**, a horizon of about 10 steps, which is also the value in the lecture that introduced the method. RMSProp was never published as a paper: it comes from lecture 6.5 of Geoffrey Hinton's 2012 course Neural Networks for Machine Learning, and is cited as Tieleman and Hinton (2012).

---

## 6. The training loop

The loop is the one of post 22, and the update is the three-call contract that every optimiser class of Part VI shares:

```python
        # Update: the three-call contract.
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
```

Only the constructor differs. `snippets/rmsprop.py` wraps the shared setup of Part VI in a function `train(optimizer, seed=0)` and runs it twice, once with post 25's AdaGrad settings and once with this post's:

```python
    documented_run("AdaGrad", Optimizer_Adagrad(learning_rate=1.0, decay=1e-4))
    documented_run("RMSProp", Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999))
```

The setup is printed by the script: `nnfs.init()` (float32, seed 0), `spiral_data(samples=100, classes=3)`, `Layer_Dense(2, 64)`, ReLU, `Layer_Dense(64, 3)`, the combined softmax and loss, and 10,001 full-batch epochs. The network has 387 parameters. With `decay=1e-5` the rate falls from 0.02 to 0.01818 over the run, so nearly all of the change in step size comes from the cache. The documented run uses `rho=0.999`, not the default; section 8 measures the difference.

---

## 7. Results on the spiral

### 7.1 The documented run and its caches

At seven epochs the script reads the caches of all 387 parameters straight after the update. `scale` is `current_learning_rate / (sqrt(cache) + epsilon)`, the number each gradient is multiplied by, and `|update|` is the distance the parameter moved. The last two lines of each run set apart the parameters of dead hidden neurons (post 17): their gradient is exactly zero, so they do not move, whatever their scale.

```text
== AdaGrad: Optimizer_Adagrad, learning_rate 1.0, decay 0.0001
 epoch   loss     accuracy   rate      median cache   largest cache   median scale   mean |update|
     0   1.0986   0.3600     1.00000   8.740e-09      1.985e-06        10685.163     9.94e-01
     1   6.2613   0.3433     0.99990   2.946e-03      7.291e-01           18.423     9.42e-01
    10   1.1331   0.3600     0.99900   6.706e-02      8.155e+00            3.858     9.08e-03
   100   1.0114   0.4533     0.99010   8.206e-02      8.614e+00            3.456     1.18e-02
  1000   0.6704   0.6900     0.90909   1.242e-01      1.439e+01            2.580     5.63e-03
  5000   0.4585   0.8133     0.66667   1.589e-01      4.931e+01            1.673     2.92e-03
 10000   0.3847   0.8400     0.50000   1.725e-01      1.229e+02            1.204     9.42e-04
caches smaller at epoch 10,000 than at epoch 1,000: 0 of 387
dead hidden neurons at epoch 10,000: 34 of 64 (the same as at epoch 1,000); their 204 parameters: 204 zero gradients, 0 caches smaller
the other 183 parameters: 0 caches smaller; median scale 1.319 at epoch 1,000 and 0.409 at epoch 10,000

== RMSProp: Optimizer_RMSprop, learning_rate 0.02, decay 1e-05, rho 0.999
 epoch   loss     accuracy   rate      median cache   largest cache   median scale   mean |update|
     0   1.0986   0.3600     0.02000   8.740e-12      1.985e-09         6543.771     5.84e-01
     1   2.8219   0.3167     0.02000   1.664e-06      2.261e-04           15.504     5.69e-01
    10   1.2039   0.3300     0.02000   2.458e-05      2.816e-03            4.034     2.89e-02
   100   1.0134   0.4667     0.01998   2.422e-05      2.587e-03            4.060     1.05e-02
  1000   0.6373   0.7300     0.01980   2.722e-05      1.277e-02            3.796     1.12e-02
  5000   0.3125   0.8633     0.01905   1.857e-06      4.841e-02           13.977     6.80e-03
 10000   0.2379   0.9000     0.01818   1.040e-07      6.369e-02           56.371     6.08e-03
caches smaller at epoch 10,000 than at epoch 1,000: 259 of 387
dead hidden neurons at epoch 10,000: 31 of 64 (the same as at epoch 1,000); their 186 parameters: 186 zero gradients, 186 caches smaller
the other 201 parameters: 73 caches smaller; median scale 1.266 at epoch 1,000 and 0.920 at epoch 10,000
```

On this seed AdaGrad ends at a loss of 0.3847 and 84.00 percent accuracy, the figures of post 25, and RMSProp at 0.2379 and 90.00 percent. The caches behave as sections 2 and 3 predict.

**AdaGrad's caches only grow.** The median cache rises at every watched epoch, from 0.0671 at epoch 10 to 0.1725 at epoch 10,000, the largest from 8.155 to 122.9, and not one of the 387 is smaller at epoch 10,000 than at epoch 1,000. The median scale falls from 3.858 to 1.204, partly through the cache and partly through the decay, which halves the rate; that median includes the 204 frozen parameters of the dead neurons, and among the other 183 it falls from 1.319 at epoch 1,000 to 0.409. The mean update shrinks from $1.18 \times 10^{-2}$ at epoch 100 to $9.42 \times 10^{-4}$ at epoch 10,000, a factor of 12.5. The steps have not died in 10,001 epochs, but they only ever get shorter.

**RMSProp's caches go both ways, but the count needs splitting.** 259 of the 387 caches are smaller at epoch 10,000 than at epoch 1,000, and the median scale rises from 3.796 to 56.371. Most of that is the 31 dead neurons: their 186 gradients are zero, so their caches shrink by the factor 0.999 at every epoch, and the scale that grows with them multiplies zero. The evidence is in the other 201 parameters: 73 of their caches are smaller, where under AdaGrad none of 183 is, and their median scale falls from 1.266 to 0.920 against AdaGrad's 1.319 to 0.409. The mean update over all parameters shrinks by a factor of 1.7 between epochs 100 and 10,000, and at the end it is 6.5 times AdaGrad's.

**The two learning rates are 50 times apart and the steps are not.** At epochs 10 and 100 the median scale is 3.858 and 3.456 under AdaGrad and 4.034 and 4.060 under RMSProp; among the parameters that still learn it is 1.319 and 1.266 at epoch 1,000. AdaGrad divides a rate of 1.0 by the root of a sum; RMSProp divides a rate of 0.02 by the root of an average, which is far smaller.

### 7.2 Ten seeds

One seed is one draw. `seeds_adagrad.py`, `seeds_rmsprop.py` and their `_more` companions repeat both runs for seeds 0 to 9, calling `np.random.seed(seed)` after `nnfs.init()` so that the data and the weights are redrawn. Accuracies are in percent; "late" means the last 1,000 epochs.

| Seed | AdaGrad final loss | AdaGrad final accuracy | AdaGrad lowest late accuracy | AdaGrad highest late loss | RMSProp final loss | RMSProp final accuracy | RMSProp lowest late accuracy | RMSProp highest late loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.3847 | 84.00 | 83.67 | 0.3933 | 0.2379 | 90.00 | 59.33 | 1.6829 |
| 1 | 0.5319 | 75.67 | 75.33 | 0.5384 | 0.2663 | 89.00 | 61.67 | 1.8076 |
| 2 | 0.3792 | 81.67 | 80.33 | 0.3820 | 0.3148 | 87.33 | 66.67 | 1.5053 |
| 3 | 0.2191 | 92.67 | 92.33 | 0.2283 | 0.2226 | 91.67 | 52.00 | 3.6640 |
| 4 | 0.4048 | 79.67 | 78.67 | 0.4137 | 0.1252 | 95.00 | 70.00 | 1.6136 |
| 5 | 0.3265 | 84.67 | 83.67 | 0.3329 | 0.4882 | 77.33 | 66.33 | 0.7458 |
| 6 | 0.3408 | 84.67 | 84.33 | 0.3506 | 0.2488 | 89.00 | 45.67 | 4.7996 |
| 7 | 0.2987 | 89.67 | 89.00 | 0.3104 | 0.5061 | 77.00 | 75.67 | 0.5483 |
| 8 | 0.3778 | 83.67 | 83.67 | 0.3849 | 0.3252 | 85.67 | 57.33 | 1.5976 |
| 9 | 0.3508 | 82.00 | 81.67 | 0.3571 | 0.3555 | 85.33 | 48.00 | 2.9183 |

**RMSProp usually finishes higher, not always.** Its final accuracy runs from 77.00 to 95.00 percent and AdaGrad's from 75.67 to 92.67. RMSProp is ahead in seven of the ten seeds and behind in seeds 3, 5 and 7; the mean accuracy over the last 1,000 epochs, which the scripts also print, gives the same seven and the same three (75.35 to 94.91 percent for RMSProp, 75.94 to 92.57 for AdaGrad). The 90.00 against 84.00 percent of seed 0 is a typical draw, not a guaranteed margin.

**RMSProp's last epoch is a noisy reading.** AdaGrad's curve is smooth at the end: in no seed is its lowest late accuracy more than 1.34 points under its final one, or its highest late loss more than 0.012 above its final loss. RMSProp's loss spikes. In eight of the ten seeds its loss exceeds 1.5 somewhere in the last 1,000 epochs, against a final loss between 0.13 and 0.36 in those seeds, and in eight, not the same eight, its accuracy falls below 70 percent. Section 11 shows one spike epoch by epoch. The bounded cache keeps the steps long enough to go on learning, and steps that stay long can also overshoot.

Whether momentum or per-parameter scaling matters more on the spiral is a comparison this post does not run; post 27 puts the optimisers of Part VI side by side.

---

## 8. Choosing $\rho$

The decay factor sets the memory horizon of section 2:

| $\rho$ | Horizon $1/(1 - \rho)$ | Cache |
|:---:|:---:|---|
| 0.9 | 10 steps | reacts within tens of steps to a change in the gradient |
| 0.99 | 100 steps | in between |
| 0.999 | 1,000 steps | smooth, and slow to follow a change in either direction |

![A chart of the weight a past gradient still carries against its age in steps, on a logarithmic age axis. AdaGrad is a flat line at one. Three RMSProp curves fall away at different ages: rho 0.9 crosses the 1/e level near 10 steps, rho 0.99 near 100 and rho 0.999 near 1,000. A side card lists the three horizons.](diagrams/02-memory-horizon.svg)

*The horizon $1/(1 - \rho)$ is the age at which a gradient's weight has fallen to about $1/e$. The card's "noisy" for 0.9, its "usual default" and the library defaults it names are conventions, not measurements of this post.*

At the lower end, $\rho = 0$ makes the cache the current squared gradient, and the update becomes $\alpha \, g / (|g| + \epsilon)$: every parameter moves by $\alpha$ against the sign of its gradient and the size of the gradient is ignored. Libraries do not agree on a default; 0.9 and 0.99 are both in use.

A long memory does not mean calm training here. `seeds_rho.py` repeats the documented run with `rho=0.9` on seeds 0 to 4:

| Seed | $\rho = 0.9$ final accuracy | $\rho = 0.9$ highest late loss | $\rho = 0.999$ final accuracy | $\rho = 0.999$ highest late loss |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 84.00 | 0.4172 | 90.00 | 1.6829 |
| 1 | 87.00 | 0.3947 | 89.00 | 1.8076 |
| 2 | 71.33 | 0.7240 | 87.33 | 1.5053 |
| 3 | 80.00 | 0.7170 | 91.67 | 3.6640 |
| 4 | 88.67 | 0.3980 | 95.00 | 1.6136 |

In all five seeds $\rho = 0.999$ ends at the higher accuracy, by 2.00 to 16.00 points, and in all five its largest late spike is the bigger one: the highest late loss is 0.39 to 0.72 with $\rho = 0.9$ and 1.51 to 3.66 with $\rho = 0.999$. By the mean accuracy of the last 1,000 epochs, the steadier reading of section 7.2, which both scripts print, it is ahead in four seeds and behind on seed 0, with 89.19 against 91.34 percent. Section 3 offers a reading, which is an interpretation and not a measurement: a cache with a horizon of 1,000 steps takes hundreds of steps to register that a gradient has grown, and until it does the steps stay long. On this problem and these five seeds the long memory buys accuracy and pays in spikes.

---

## 9. RMSProp as the bridge to Adam

**What RMSProp keeps from AdaGrad** is the per-parameter division by the root of a cache of squared gradients. **What it changes** is how the cache is filled. **What it still lacks** is twofold.

It has no memory of direction. The cache averages $g^2$, which has lost the sign; the step uses the current gradient alone. Momentum (post 24) is the piece that averages $g$ itself.

It starts its cache at zero and does nothing about it. After the first update the cache is $(1 - \rho) g^2$, so the first step has the size $\alpha / \sqrt{1 - \rho}$ for every parameter, whatever its gradient. The script prints the factor: 3.16 for $\rho = 0.9$, 10.00 for 0.99 and 31.62 for 0.999. With the documented settings that is $0.02 \cdot 31.62 = 0.632$ per parameter, in a network initialised at a scale of 0.01. Section 7.1 shows it: the mean first update is 0.584, a little under 0.632 because $\epsilon$ is not negligible beside the smallest gradients, and the loss jumps from 1.0986 to 2.8219 at epoch 1 before it recovers.

Adam (post 27) supplies both missing pieces:

| Adam piece | Introduced in |
|---|---|
| EMA of $g$ with factor $\beta_1$ (momentum) | post 24, in a different form |
| EMA of $g^2$ with factor $\beta_2$ (the cache) | this post, where the factor is $\rho$ |
| Bias correction, a division by $1 - \beta^t$ | post 27 |

The correction divides the cache by $1 - \beta_2^t$, which is $1 - \rho$ at the first step and cancels the factor above, so Adam's first step has size $\alpha$. RMSProp with a momentum term added is close to Adam without that correction.

---

## 10. Make it run: eight scripts

Every code block and every number of this post comes from these scripts, run from the series root. All but the first need the `nnfs` package for the spiral data.

```text
python posts/26-rmsprop/snippets/ema_cache.py
python posts/26-rmsprop/snippets/rmsprop.py
python posts/26-rmsprop/snippets/seeds_adagrad.py
python posts/26-rmsprop/snippets/seeds_rmsprop.py
python posts/26-rmsprop/snippets/seeds_adagrad_more.py
python posts/26-rmsprop/snippets/seeds_rmsprop_more.py
python posts/26-rmsprop/snippets/seeds_rho.py
python posts/26-rmsprop/snippets/what_can_go_wrong.py
```

`ema_cache.py` holds the hand-made streams of sections 2, 3 and 9 and finishes at once. `rmsprop.py` defines the classes of posts 16, 19 and 25 unchanged, `Optimizer_RMSprop`, and `train`, and prints section 7.1 in about 15 seconds. It calls `nnfs.init()` once, when it is loaded, and reseeds with `np.random.seed` for every run. The five `seeds_` scripts import from it and make five full runs each, about 35 seconds apiece; the ten seeds are split over two files per optimiser to keep each script short. `what_can_go_wrong.py` runs section 11 in under 10 seconds.

---

## 11. What can go wrong?

**The last epoch read as the result.** In the documented run the loss is above twice its late median on 19 of the last 1,000 epochs. `rmsprop.py` prints the largest excursion, as epoch, loss and accuracy:

```text
around the highest late loss, epoch 9130:
  9115   0.3033   0.8667
  9121   0.6545   0.7467
  9127   0.8526   0.7267
  9130   1.6829   0.6133
  9133   1.1322   0.6500
  9136   0.2873   0.8700
  9139   0.2423   0.8900
```

Within 15 epochs the loss climbs from 0.30 to 1.68 and the accuracy drops from 86.67 to 61.33 percent; six epochs later both are back. Over the last 1,000 epochs the accuracy ranges from 59.33 to 91.00 percent. A run stopped at epoch 9,130 would report 61.33 percent for a network that scored 86.67 percent fifteen epochs earlier and scores 89.00 percent nine epochs later. The remedy is to judge an RMSProp run by more than its last epoch, for instance by the mean over a late window, as section 7.2 does.

**The learning rate of plain gradient descent.** With `learning_rate=1.0` every parameter moves by about 1 per update, and by 31.6 on the first:

```text
seed   rate 1.0: loss at epoch 1   loss at 1,000   accuracy at 1,000   |   rate 0.02: the same three
   0    10.7153   0.9072   0.5833   |     2.8219   0.6373   0.7300
   1     9.6171   0.9001   0.5500   |     2.3801   0.6649   0.6533
   2    10.3693   1.2062   0.4033   |     3.4394   0.7620   0.6900
   3    10.3156   0.9747   0.5033   |     3.2581   0.7692   0.6367
   4     9.4022   0.9872   0.5600   |     1.9202   0.5399   0.7467
```

Nothing overflows, and nothing is learned well either. After 1,001 epochs the runs with a rate of 1.0 stand at 40.33 to 58.33 percent, and those with 0.02 at 63.67 to 74.67 percent, ahead in every one of the five seeds.

**`rho=1.0`.** The new squared gradient gets the weight $1 - \rho = 0$, so the cache stays at its initial zero and every update divides by $\epsilon$ alone, a scale of $0.02 / 10^{-7} = 200{,}000$:

```text
loss at epochs 0 to 5: 1.0986 10.6379 10.7454 10.7454 10.7454 10.7454
loss at epoch 10: 10.7454   accuracy: 0.3333   largest cache entry over the run: 0.0
```

The network is thrown to a point where it predicts one class with certainty, and stays there: 10.7454 is two thirds of $-\ln 10^{-7}$, the clipped loss of the two classes it gets wrong. $\rho$ must stay below 1.

**`epsilon=0`.** A parameter whose gradient is exactly zero at its first update, as for the weights of a neuron that is dead from the start, has a cache of zero, and the update divides zero by zero:

```text
epsilon 1e-07: weight after one update 0.5, cache 0.0, warning: none
epsilon 0.0: weight after one update nan, cache 0.0, warning: invalid value encountered in divide
```

NumPy warns and stores `nan` in the weight, and the next forward pass spreads it through the network. With the default $\epsilon$ the same update leaves the weight where it was.

**A cache left on the layer.** The caches live on the layer, and `update_params` creates them only when `weight_cache` is missing. A layer that another optimiser has already trained hands its cache to a new `Optimizer_RMSprop`:

```text
no cache on the layer: first step 0.632436
cache 122.9 on the layer: first step 0.000180; cache within 1 percent of g^2 = 0.01 after 14,015 updates
```

With a cache of 122.9, the largest that AdaGrad left in section 7.1, a gradient of 0.1 moves the weight by 0.00018 in place of 0.632, and with $\rho = 0.999$ the stale value needs about 14,000 updates to fade. A new optimiser on a trained layer needs the two cache attributes deleted first, unless continuing the old average is the intent.

---

## 12. Summary

| Concept | Takeaway |
|---|---|
| Cache | $G_t = \rho G_{t-1} + (1 - \rho) g_t^2$, an EMA of $g^2$ started at zero |
| Update | $\theta \leftarrow \theta - \alpha g / (\sqrt{G_t} + \epsilon)$, the step of AdaGrad |
| Why the cache is bounded | a weighted average never exceeds the largest value averaged |
| Horizon | about $1/(1 - \rho)$ steps: 10 for 0.9, 1,000 for 0.999 |
| Step size | about $\alpha$ per update while the gradient is steady; $\alpha / \sqrt{1 - \rho}$ on the first |
| Class defaults | `learning_rate=0.02, decay=0.0, epsilon=1e-7, rho=0.9` |
| Measured cache, seed 0 | smaller at epoch 10,000 than at epoch 1,000, among the parameters that still learn: 0 of 183 caches under AdaGrad, 73 of 201 under RMSProp |
| Measured result, ten seeds | RMSProp 77.00 to 95.00 percent, AdaGrad 75.67 to 92.67; RMSProp ahead in seven, with late loss spikes |
| Place in Part VI | AdaGrad's scaling with a forgetting cache; Adam adds momentum and a bias correction |

---

## Common pitfalls

1. **Reporting the last epoch of an RMSProp run.** The loss spikes late in training; the final epoch can sit on a spike or beside one. Report a late-window mean, or the spread over seeds.
2. **Carrying a learning rate over from another optimiser.** Under RMSProp the rate is roughly the distance moved per update. The 1.0 of gradient descent and AdaGrad is 50 times the 0.02 used here.
3. **Setting `rho` to 1, or `epsilon` to 0.** The first freezes the cache at zero and divides every gradient by $\epsilon$; the second turns a zero gradient into `nan`.
4. **Confusing `decay` with `rho`.** `decay` lowers the global learning rate over time; `rho` sets how long the cache remembers. They are tuned independently.
5. **Expecting a long memory to give a calm run.** On the spiral, $\rho = 0.999$ ended higher than 0.9 in five of five seeds, in four of five by the late-window mean, and produced the larger loss spikes in all five.
6. **Reusing a trained layer with a new optimiser.** The old `weight_cache` and `bias_cache` are still on the layer and are continued, not reset.

---

## Further reading

- Tieleman, T. and Hinton, G., *"Lecture 6.5: RMSProp"* (Coursera: Neural Networks for Machine Learning, 2012). The lecture that introduced the method; there is no paper.
- Ruder, S., *"An Overview of Gradient Descent Optimization Algorithms"* (arXiv:1609.04747, 2016), the section on RMSprop.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 8, the section on RMSProp (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 10 (2020).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 27 - Adam](../27-adam-optimiser/index.md):** momentum's average of the gradient on top of RMSProp's average of its square, with the bias correction that tames the first step.
- **[Post 24 - Momentum](../24-momentum/index.md):** the memory of direction that RMSProp lacks, and the other half of Adam.
