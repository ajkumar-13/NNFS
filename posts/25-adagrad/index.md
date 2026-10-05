# 25 - AdaGrad

> **TL;DR.** AdaGrad (Duchi, Hazan and Singer, 2011) keeps a cache for every parameter, the running sum of that parameter's squared gradients, and divides the parameter's step by the square root of it, so a gradient 100 times larger no longer means a step 100 times larger. The cache never shrinks, so every effective learning rate can only fall; under a constant gradient it falls as $1/\sqrt{t}$. On the spiral data `Optimizer_Adagrad(learning_rate=1.0, decay=1e-4)` ends at a loss of 0.3847 and an accuracy of 84.0 percent on seed 0, and over five seeds its accuracy is above plain gradient descent on all five and above momentum on three.
>
> **Prerequisites:** [Post 24](../24-momentum/index.md).
> **Safe to skip?** Skip it if the reader can already write the AdaGrad update with its cache from memory, say what the first step of every parameter is, and explain why the effective learning rate of a parameter can never rise.
>
> **After reading, you will be able to:**
>
> - Explain why one global learning rate cannot suit parameters whose gradients differ in scale.
> - Implement Optimizer_Adagrad with a per-layer weight_cache and bias_cache.
> - Predict when AdaGrad helps and when its shrinking effective learning rates dominate.

![The AdaGrad update rule above a log-log chart of the effective learning rate over 10,000 iterations for two parameters. Parameter A, with gradients of size 2.0, falls from 0.5 to 0.005. Parameter B, with gradients of size 0.2, falls from 5 to 0.05. A side panel gives the caches at iteration 1,000 as 4,000 and 40.](diagrams/01-per-parameter-rates.svg)

*Two parameters, two caches, two effective learning rates. The lines are parallel because both rates fall as $1/\sqrt{t}$, and B's stays ten times higher because its gradients are ten times smaller. The figure's "Part 26" is post 26.*

---

## 1. The question: what does one learning rate cost?

Gradient descent ([post 22](../22-gradient-descent-optimiser/index.md)), its decay schedule ([post 23](../23-learning-rate-decay/index.md)) and momentum ([post 24](../24-momentum/index.md)) all multiply every gradient of the network by the same number, the learning rate $\alpha$. The question here is: **what does that one number cost when the gradients of different parameters differ in scale, and what can replace it?**

A loss of two weights shows the cost:

$$L(w_1, w_2) = \frac{w_1^2}{100} + w_2^2$$

Its minimum is at $(0, 0)$. At $(w_1, w_2) = (1, 1)$ the partial derivatives are $2 w_1 / 100 = 0.02$ and $2 w_2 = 2.0$, a factor of 100 apart. `snippets/adagrad.py` runs gradient descent on this loss from that point, with the `Optimizer_SGD` of post 24:

```text
gradient descent, alpha =  0.1: first steps 0.002 and 0.2; after 100 steps w1 = 0.8186, w2 = 2.037e-10
gradient descent, alpha =  0.5: first steps 0.01 and 1; after 100 steps w1 = 0.3660, w2 = 0
gradient descent, alpha = 10.0: first steps 0.2 and 20; after   5 steps w1 = 0.3277, w2 = -2.476e+06
```

With $\alpha = 0.1$, $w_2$ has arrived after 100 steps and $w_1$ has covered 18 percent of its way. $\alpha = 0.5$ is the best possible rate for $w_2$, which lands on its minimum in one step, and $w_1$ is still at 0.366 after 100. $\alpha = 10$ gives $w_1$ a sensible first step of 0.2 and throws $w_2$ from 1 to $-19$; five steps later it is at $-2.5$ million.

The reason is in the update. One step multiplies $w_2$ by $1 - 2\alpha$ and $w_1$ by $1 - \alpha/50$. The first factor is below 1 in size only for $\alpha < 1$, and for every such $\alpha$ the second factor is above 0.98: $w_1$ loses at most 2 percent of its value per step. A rate sized for the large gradient leaves the small-gradient weight crawling, and a rate sized for the small gradient makes the other one diverge.

![Two cards. The left gives the loss W1 squared over 100 plus W2 squared and its two gradients at the point 1, 1: 0.02 and 2.0, a hundred times apart. The right works two learning rates: at 0.1, W1 moves 0.002 and W2 moves 0.2; at 10, W1 moves 0.2 and W2 moves 20 and diverges. A band states that AdaGrad makes the two steps equal.](diagrams/02-one-rate-two-params.svg)

*The first steps of the two rates in the output above. The figure writes the weights as $W_1$ and $W_2$.*

A network has the same problem with more parameters. In the first backward pass of the spiral classifier of this post, the 387 gradients range in size from $3.4 \times 10^{-8}$ to $1.4 \times 10^{-3}$ (section 7), a factor of about 40,000.

---

## 2. The AdaGrad idea

AdaGrad, short for adaptive gradient, scales the step of each parameter by what that parameter has seen so far. For a parameter $\theta$ whose gradient at update $t$ is $g_t = \partial L / \partial \theta$, it keeps a **cache** $G$, the running sum of the squared gradients, starting from $G_0 = 0$:

$$G_t = G_{t-1} + g_t^2$$

The step is then divided by the square root of the cache:

$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{G_t} + \epsilon} \, g_t$$

Here $\alpha$ is the global learning rate and $\epsilon$ is a small constant, $10^{-7}$ in the class of section 5, that keeps the denominator above zero. The factor $\alpha / (\sqrt{G_t} + \epsilon)$ is the **effective learning rate** of that one parameter. Three properties follow from the two formulas.

**The cache has the shape of the parameters.** Every weight and every bias has its own entry of $G$, so every one has its own effective learning rate.

**Squaring removes the sign.** Positive and negative gradients both add to the cache. A sum of the plain gradients would stay near zero for a parameter that oscillates, which is the parameter that most needs a smaller step.

**The square root makes the step independent of the gradient's scale.** If every gradient of a parameter is multiplied by a constant $c$, its cache is multiplied by $c^2$, the root of the cache by $c$, and the step $\alpha g_t / \sqrt{G_t}$ does not change. In particular the first step is $\alpha g_1 / \lvert g_1 \rvert$: it has the size $\alpha$ whatever the gradient is. Dividing by $G_t$ itself would overcorrect, because the step would then shrink by $c$ when the gradient grows by $c$.

The same two-weight loss, under the `Optimizer_Adagrad` of section 5 with $\alpha = 0.1$:

```text
AdaGrad, alpha = 0.1: after   1 steps w1 = 0.900000, w2 = 0.900000, caches 0.0004 and 4
AdaGrad, alpha = 0.1: after  10 steps w1 = 0.553732, w2 = 0.553731, caches 0.002272 and 22.72
AdaGrad, alpha = 0.1: after 100 steps w1 = 0.027387, w2 = 0.027387, caches 0.004011 and 40.11
```

The two weights travel together, to within the effect of $\epsilon$. The caches differ by a factor of 10,000, their roots by 100, and that cancels the factor of 100 between the gradients. What remains of each gradient is its sign and its size relative to its own history.

---

## 3. A worked example: the cache grows forever

A single parameter receives the gradient $g = 0.5$ at every update, with $\alpha = 1$. The script follows it through the class:

```text
      t        G     sqrt(G)   alpha/sqrt(G)   distance moved
      1     0.25     0.500       2.0000           1.00
      2     0.50     0.707       1.4142           1.71
     10     2.50     1.581       0.6325           5.02
    100    25.00     5.000       0.2000          18.59
   1000   250.00    15.811       0.0632          61.80
  10000  2500.00    50.000       0.0200         198.54
```

The cache is $G_t = 0.25\,t$, so the effective learning rate is $2/\sqrt{t}$ and the step at update $t$, the rate times the gradient, is $1/\sqrt{t}$. The first step is 1, the size of $\alpha$, and the step at update 10,000 is 0.01. The last column is the sum of those steps, which approaches $2\sqrt{t}$: a hundred times the updates buy ten times the distance. Gradient descent with the same $\alpha$ and the same gradient moves 0.5 at every step, 5,000 in all.

This resembles the schedule of post 23, with two differences. It is **per parameter**: the rate of each parameter falls according to its own gradients. The two parameters of the hero figure, with gradients of size 2.0 and 0.2, hold caches of 4,000 and 40 after 1,000 updates and effective rates of 0.0158 and 0.158. And it is **implicit**: no `decay` argument sets it, the bookkeeping of the optimiser produces it.

---

## 4. The structural flaw: rates that only fall

A squared gradient is never negative, so $G_t \ge G_{t-1}$ at every update, and nothing in the algorithm lowers a cache. The effective learning rate of a parameter can therefore fall or stay where it is, and never rise.

How fast it falls can be predicted. After $t$ updates the cache is $t$ times the mean of the squared gradients. With $g_\text{rms}$ for the root of that mean, and leaving $\epsilon$ and any decay aside:

$$\frac{\alpha}{\sqrt{G_t}} = \frac{\alpha}{g_\text{rms} \sqrt{t}}$$

Halving a rate takes four times as many updates. With $\alpha = 1$, a parameter whose gradients have a typical size of 1 is down to a rate of 0.01 after 10,000 updates, and one whose gradients have a typical size of 0.01 is still at 1. Large gradients and long runs are where the shrinking dominates, and the gradients that count are all of them: a parameter that saw large gradients early keeps them in its cache when it later needs to move again.

The result is a parameter that still receives a gradient and barely follows it. That differs from the dead neuron of [post 17](../17-backpropagation-through-activation-functions/index.md), whose gradient is zero, but neither condition repairs itself. [Post 26](../26-rmsprop/index.md) replaces the sum by an exponential moving average, a cache that forgets and can therefore shrink.

---

## 5. The optimiser class

The update rule has no velocity, so AdaGrad is a new class beside `Optimizer_SGD`, not an extension of it. The constructor, `pre_update_params` and `post_update_params` are those of post 24 with `momentum` replaced by `epsilon`:

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

**The contract is unchanged.** One `pre_update_params`, one `update_params` per layer, one `post_update_params`, as in post 23. The decay schedule is still there, and it multiplies the cache's own shrinking: the $\alpha$ of the formulas is `current_learning_rate`.

**The caches live on the layer.** Like the `weight_momentums` of post 24, `weight_cache` and `bias_cache` are created on the first call, with the shapes of `weights` and `biases`, and one optimiser object serves any number of layers. They are also never reset: a second optimiser object handed the same layers finds the old caches and continues from them.

**The order inside `update_params` matters.** The cache is updated first, so the current gradient is already in the denominator of its own step. That is what makes the first step $\alpha$ in size; with the old cache it would be $\alpha g / \epsilon$, ten million times the gradient.

**$\epsilon$ is added to the root, not under it.** The code computes $\sqrt{G} + \epsilon$, as the formula of section 2 does. The other form in use, $\sqrt{G + \epsilon}$, gives the same step only while $\sqrt{G}$ is well above $\sqrt{\epsilon} = 3.2 \times 10^{-4}$; section 10 measures the difference.

---

## 6. The training loop

The loop is the one of post 24. `snippets/adagrad.py` wraps it in a function, so that the same lines serve every run of this post; `dead` counts the hidden neurons whose output is zero for all 300 samples, and `watch` is a hook for printing.

```python
def train(optimizer, seed=0, epochs=10001, watch=None):
    """Train the shared network of Part VI; return the last loss, the accuracy and the dead neurons."""
    np.random.seed(seed)                        # the data and the weights both follow from the seed
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    for epoch in range(epochs):
        # Forward pass.
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)
        dead = int(np.sum(np.max(activation1.output, axis=0) == 0))   # hidden neurons at 0 on every sample

        # Backward pass.
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update.
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

        if watch is not None:
            watch(epoch, loss, accuracy, dead, (dense1, dense2), optimizer)

    return float(loss), float(accuracy), dead
```

The documented run is `train(Optimizer_Adagrad(learning_rate=1.0, decay=1e-4))`. The script calls `nnfs.init()` once at the top, which makes the arrays float32 and patches `np.dot` (post 04), and seed 0 is the seed that call sets, so `seed=0` reproduces the series' usual run.

The decay is $10^{-4}$ where post 24 used $10^{-3}$. Over 10,001 updates the schedule alone leaves $1/(1 + 10^{-4} \cdot 10{,}000) = 0.5$ of the learning rate, against 0.0909 for $10^{-3}$, and that factor multiplies rates the cache is already lowering. Whether the heavier schedule hurts depends on the seed. `snippets/seeds_decay.py` runs `decay=1e-3` on seeds 0 to 4 and ends at losses of 0.5104, 0.4801, 0.3580, 0.5444 and 0.5412; the documented setting gives 0.3847, 0.5319, 0.3792, 0.2191 and 0.4048 (section 7). The heavier decay is worse on seeds 0, 3 and 4 and better on seeds 1 and 2.

---

## 7. What happens when it runs

The setup is the shared one of Part VI: `nnfs.init()`, `spiral_data(samples=100, classes=3)`, `Layer_Dense(2, 64)`, ReLU, `Layer_Dense(64, 3)`, the combined softmax and loss, weights of scale 0.01, and 10,001 full-batch epochs. The network has $2 \cdot 64 + 64 + 64 \cdot 3 + 3 = 387$ parameters. `adagrad.py` prints the first update on seed 0 on its own line:

```text
first update: |gradient| from 3.4e-08 to 1.4e-03 (median 9.3e-05); |step| from 0.254 to 1.000 (median 0.999)
```

This is section 2 on a real network. The gradients span more than four orders of magnitude and almost every parameter moves by about 1, the size of $\alpha$. The few smaller steps belong to gradients that are not large against $\epsilon$. For weights that start at a scale of 0.01, a step of 1 is a jump, and the table that follows shows what it does. Each row gives the loss, the accuracy and the dead neurons of that epoch's forward pass, and then the state after that epoch's update: the scheduled learning rate, the cache $G$ and the effective learning rate over all 387 parameters, how many of the rates are below 1, and the mean size of the step.

```text
 epoch    loss   acc    dead   lr      G median   G max     rate median  rate min  below 1   mean |step|
     0  1.0986  0.360    0   1.0000  8.74e-09  1.98e-06    10685.16   709.779       0   9.94e-01
     1  6.2613  0.343    0   0.9999  2.95e-03  7.29e-01       18.42     1.171       0   9.42e-01
     2  9.9083  0.303   13   0.9998  3.09e-02  2.48e+00        5.69     0.634      20   5.31e-01
    10  1.1331  0.360   32   0.9990  6.71e-02  8.16e+00        3.86     0.350      46   9.08e-03
   100  1.0114  0.453   34   0.9901  8.21e-02  8.61e+00        3.46     0.337      48   1.18e-02
  1000  0.6704  0.690   34   0.9091  1.24e-01  1.44e+01        2.58     0.240      88   5.63e-03
  2000  0.5555  0.787   34   0.8333  1.30e-01  2.05e+01        2.32     0.184      99   4.59e-03
  5000  0.4585  0.813   34   0.6667  1.59e-01  4.93e+01        1.67     0.095     136   2.92e-03
  9000  0.3932  0.840   34   0.5263  1.69e-01  1.15e+02        1.28     0.049     170   1.20e-03
 10000  0.3847  0.840   34   0.5000  1.72e-01  1.23e+02        1.20     0.045     177   9.42e-04
final: loss 0.3847, accuracy 0.8400, dead neurons 34 of 64
smallest change of any cache entry between two consecutive epochs: 0.0e+00
```

**The start is violent.** The loss begins at 1.0986, which is $\ln 3$, and the first two updates take it to 6.2613 and 9.9083. The gradients after such a jump are large, so the median cache rises from $8.74 \times 10^{-9}$ to 0.0309 within two further updates and the median rate collapses from 10,685 to 5.69. By epoch 10 the loss is back at 1.1331, and 32 of the 64 hidden neurons are dead; at epoch 100 and in every later row it is 34. The remaining 30 neurons do all the learning that follows.

**The caches only grow.** No cache entry fell between two consecutive epochs in the whole run: the smallest change is exactly zero, which belongs to parameters with a zero gradient, the parameters of the dead neurons among them. The largest cache grows from 8.16 at epoch 10 to 123 at the end, and the median from 0.0671 to 0.172.

**The effective rates fall, from two causes.** Between epoch 10 and epoch 10,000 the median rate goes from 3.86 to 1.20 and the smallest from 0.350 to 0.045. The schedule accounts for a factor of 2.0 in both, since `current_learning_rate` goes from 0.9990 to 0.5. The rest is the cache: a factor of $\sqrt{0.172/0.0671} = 1.6$ at the median and $\sqrt{123/8.16} = 3.9$ between the largest caches of the two epochs, which belong to two different parameters. The smallest rate is the scheduled rate over the root of the largest cache, $0.5/\sqrt{123} = 0.045$. The number of parameters whose rate is below 1 rises from 46 to 177, and the mean step shrinks about tenfold, from $9.08 \times 10^{-3}$ to $9.42 \times 10^{-4}$.

**The rates have shrunk, not died, and the medians above flatter them.** A dead neuron passes no gradient to its two incoming weights, its bias and its three outgoing weights, so the caches of those six parameters stop growing and only the schedule lowers their rates. With 34 neurons dead that is 204 of the 387 parameters, and they hold 175 of the 210 rates that are still above 1 at the end. The last two lines of the script's output split the parameters:

```text
epoch   100: 204 parameters of dead neurons, rate median 7.34, 197 above 1; the other 183, rate median 2.36, 142 above 1
epoch 10000: 204 parameters of dead neurons, rate median 3.71, 175 above 1; the other 183, rate median 0.41, 35 above 1
```

Among the 183 parameters that still learn, the median rate falls from 2.36 to 0.41, a factor of 5.8, of which the schedule is 2.0 and the cache 2.9, and 148 of them end below 1. The loss is still falling at the end, by 0.0085 over the last 1,000 epochs, against 0.1149 between epochs 1,000 and 2,000. The run does not show how much of that slowdown is the shrinking rates and how much the loss surface.

One seed is not a result. `seeds_adagrad.py`, `seeds_momentum.py` and `seeds_sgd.py` repeat the run for seeds 0 to 4 with the optimiser of this post and with the documented settings of posts 24 and 22. Each cell holds the final loss, the accuracy, and the number of dead neurons of 64:

| Seed | `Optimizer_SGD(learning_rate=1.0)` | `Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)` | `Optimizer_Adagrad(learning_rate=1.0, decay=1e-4)` |
|:---:|:---:|:---:|:---:|
| 0 | 0.8737, 0.6467, 3 | 0.1209, 0.9567, 13 | 0.3847, 0.8400, 34 |
| 1 | 0.5091, 0.7467, 6 | 0.5986, 0.6867, 32 | 0.5319, 0.7567, 42 |
| 2 | 0.9906, 0.5367, 11 | 0.5271, 0.7300, 48 | 0.3792, 0.8167, 34 |
| 3 | 0.3943, 0.8700, 2 | 0.3687, 0.8167, 37 | 0.2191, 0.9267, 22 |
| 4 | 0.4843, 0.7867, 3 | 0.0865, 0.9800, 23 | 0.4048, 0.7967, 38 |

Seed 0 reproduces the figures of posts 22 and 24: 0.87 and 64.7 percent, 0.12 and 95.7 percent.

**Against plain gradient descent, AdaGrad is ahead on accuracy in all five seeds**, by 1.0 to 28.0 percentage points, and on loss in four of the five (on seed 1 its loss is 0.5319 against 0.5091).

**Against momentum there is no winner.** Momentum is ahead on seeds 0 and 4, with 95.7 and 98.0 percent against 84.0 and 79.7. AdaGrad is ahead on seeds 1, 2 and 3, with 75.7, 81.7 and 92.7 percent against 68.7, 73.0 and 81.7, and the losses order the same way. Seed 0 alone would say that AdaGrad loses to momentum by 11.7 points; five seeds do not support that. AdaGrad's accuracies lie between 75.7 and 92.7 percent and momentum's between 68.7 and 98.0.

**Momentum and AdaGrad both pay in dead neurons.** In every seed each of them ends with more dead neurons than plain gradient descent: 13 to 48 for momentum and 22 to 42 for AdaGrad, against 2 to 11. For AdaGrad the first table of this section shows when it happens on seed 0: in the first ten updates, whose steps have the size of the learning rate.

---

## 8. When AdaGrad is the right choice

The case AdaGrad was designed for is **sparse gradients**: parameters that receive a gradient only rarely, as the weights of a rare input feature do. A single learning rate treats them like every other parameter, and a global schedule has already lowered their rate by the time their gradient arrives. A cache that counts only a parameter's own gradients does not. The script gives one parameter a gradient of 1 at every update and another the same gradient at every hundredth update:

```text
after 1,000 steps: caches 1000 and 10, effective rates 0.0316 and 0.3162, ratio 10.0
```

The rare parameter has seen ten gradients and still steps ten times as far on its next one. This is the setting of the original paper, which is also where the guarantees of the method are proved: for convex problems, not for a network.

The rule of section 4 says when the price is low. The rate of a parameter is $\alpha / (g_\text{rms} \sqrt{t})$, so short runs and small gradients leave the rates large. Section 7 is the example: after 10,001 updates the parameters that still learn have a median rate of 0.41, where gradients of a typical size of 1 would have left $0.5/\sqrt{10{,}001} = 0.005$. Long runs with gradients arriving at every update are the opposite case, and they are the usual case in deep learning. There the methods of posts 26 and 27, which bound the cache, are the common choice, and AdaGrad is the clearest place to see what dividing by a gradient history does.

---

## 9. Make it run: six scripts

Every code block and every number of this post comes from a script in `snippets/`. All need NumPy and the `nnfs` package and are seeded:

```text
python posts/25-adagrad/snippets/adagrad.py
python posts/25-adagrad/snippets/seeds_adagrad.py
python posts/25-adagrad/snippets/seeds_momentum.py
python posts/25-adagrad/snippets/seeds_sgd.py
python posts/25-adagrad/snippets/seeds_decay.py
python posts/25-adagrad/snippets/what_can_go_wrong.py
```

`adagrad.py` holds the classes, the `train` function, the small examples of sections 1, 3 and 8, and the documented run; it takes about 7 seconds. The layer, activation and loss classes are those of posts 16 and 19 and `Optimizer_SGD` is that of post 24, all unchanged. The four `seeds_` scripts import from it and make five full runs each, 35 to 50 seconds apiece. `what_can_go_wrong.py` runs the mistakes of section 10 in about 10 seconds.

---

## 10. What can go wrong?

Each mistake is a subclass in `what_can_go_wrong.py` that changes `update_params` only. None of them raises an error.

**$\epsilon$ moved inside the root.** Writing `np.sqrt(layer.weight_cache + self.epsilon)` looks like a detail, and it changes the first step of every parameter with a small gradient:

```text
first gradient          1e-01     1e-03     1e-04     1e-05
step, epsilon outside   1.0000    0.9999    0.9990    0.9901
step, epsilon inside    1.0000    0.9535    0.3015    0.0316
the two agree while |g| is well above sqrt(epsilon) = 3.16e-04
spiral run, seed 0, epsilon outside: loss 1.0986 before the first update, 6.2613 after it
spiral run, seed 0, epsilon inside : loss 1.0986 before the first update, 1.6884 after it
```

With $\epsilon$ inside, a gradient below $\sqrt{\epsilon} = 3.2 \times 10^{-4}$ no longer gets a full first step. The median first gradient of the spiral network is $9.3 \times 10^{-5}$, so the two versions part at the first update and are different training runs from then on. Neither is the wrong formula; the mistake is to mix them, or to compare numbers across them.

**$\epsilon = 0$.** A parameter whose gradient has been exactly zero since the start has a cache of zero, and its step becomes $0/0$:

```text
steps for the gradients 0.5 and 0.0: [ 1. nan]   warning: invalid value encountered in divide
with epsilon = 1e-7:                [ 0.9999998 -0.       ]
```

NumPy warns and writes `nan` into the weight, and the next forward pass spreads it. A positive $\epsilon$ turns that step into zero. This is its only job in the class.

**The cache zeroed on every call.** Creating the cache without the `hasattr` test leaves one squared gradient in it, so every step is $\alpha g / \lvert g \rvert$: the learning rate with a sign.

```text
epoch    10: loss 4.9378, accuracy 0.3433, dead neurons 54 of 64
epoch   100: loss 2.8459, accuracy 0.3600, dead neurons 60 of 64
epoch 10000: loss 1.0986, accuracy 0.3333, dead neurons 64 of 64
```

Every update is as large as the first one of section 7. On seed 0 all 64 hidden neurons are dead at the end and the loss is back at $\ln 3$.

**The square root left out.** Dividing by the cache itself makes the first step $g / (g^2 + \epsilon)$ for $\alpha = 1$. That is 10 for a gradient of 0.1 and 909 for a gradient of 0.001: far above $\alpha$, and larger for the smaller gradient until $\epsilon$ takes over.

```text
first step for the gradients 0.1, 0.001 and 0.00001: 10 909.1 99.9
epoch     0: loss 1.0986, accuracy 0.3600, largest |weight| after its update 1.58e+03
epoch     1: loss 10.9066, accuracy 0.3233, largest |weight| after its update 1.58e+03
epoch   100: loss 10.9066, accuracy 0.3233, largest |weight| after its update 1.58e+03
```

One update takes the weights of the spiral network from a scale of 0.01 to as much as 1,580, and the loss does not move in the 100 epochs that follow.

---

## 11. Summary

| Concept | Takeaway |
|---|---|
| Cache | $G_t = G_{t-1} + g_t^2$, one entry per parameter, in `weight_cache` and `bias_cache` on the layer |
| Update | $\theta_t = \theta_{t-1} - \alpha g_t / (\sqrt{G_t} + \epsilon)$, with $\epsilon$ outside the root |
| First step | has the size $\alpha$ for every parameter, whatever its gradient |
| Effective learning rate | $\alpha / (g_\text{rms} \sqrt{t})$; it never rises, and halving it takes four times the updates |
| Spiral, seed 0 | loss 0.3847, accuracy 84.0 percent, smallest rate 0.045, 34 of 64 neurons dead |
| Five seeds | accuracy above plain gradient descent on five, above momentum on three |
| Best case | sparse gradients; short runs or small gradients |
| What comes next | a cache that forgets (post 26), then that cache with momentum (post 27) |

---

## Common pitfalls

1. **Reading $\alpha$ as in gradient descent.** In AdaGrad the first step of every parameter has the size $\alpha$ itself. With `learning_rate=1.0` and weights of scale 0.01, that is the jump of section 7.
2. **Treating the cache as one number per layer.** It has the shape of the parameters. At the end of the documented run the median cache is 0.172 and the largest is 123.
3. **Expecting a fresh optimiser to mean fresh rates.** The caches are stored on the layers and survive the optimiser object.
4. **Moving $\epsilon$ under the root without saying so.** The two forms give different first steps for small gradients, so results from one do not carry over to the other.
5. **Judging AdaGrad against momentum from one seed.** On the spiral the order of the two changes with the seed (section 7).

---

## Further reading

- Duchi, J., Hazan, E., and Singer, Y., *"Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"* (Journal of Machine Learning Research, 2011). The original AdaGrad paper.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 8 (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 10 (2020).
- Ruder, S., *"An Overview of Gradient Descent Optimization Algorithms"* (arXiv:1609.04747, 2016).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 26 - RMSProp](../26-rmsprop/index.md):** the sum of squared gradients becomes an exponential moving average, so the cache can shrink.
- **[Post 27 - Adam](../27-adam-optimiser/index.md):** that cache combined with the velocity of post 24, with a correction for the first steps.
