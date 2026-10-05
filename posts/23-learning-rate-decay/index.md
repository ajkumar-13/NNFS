# 23 - Learning-rate decay

> **TL;DR.** A constant learning rate of 1.0 trains the spiral classifier of post 22 in jumps: over five seeds the loss goes up on 4,154 to 4,765 of the 10,000 steps, by as much as 1.10 in a single step. The schedule $\alpha_t = \alpha_0 / (1 + d \cdot t)$, added to `Optimizer_SGD` as two short methods and a counter, removes the jumps at $d = 10^{-3}$ (no rise above 0.0152), and it pays for that with distance: the decayed rates add up to 24 percent of the constant ones. On the documented seed the loss ends at 0.7612 against 0.8737 with the accuracy unchanged at 64.7 percent, but the end value of a curve that jumps is a draw: by the mean loss of the last 1,000 epochs decay is ahead on one of five seeds only, and a decay of $10^{-2}$ stalls every seed near a loss of 1.06.
>
> **Prerequisites:** [Post 22](../22-gradient-descent-optimiser/index.md).
> **Safe to skip?** Skip it if the reader can already write an optimiser whose learning rate follows $\alpha_0 / (1 + d \cdot t)$ with a counter that advances once per update, and can tell from a training log whether a flat loss comes from the rate or from the gradient.
>
> **After reading, you will be able to:**
>
> - Derive and explain the decay schedule alpha(t) = alpha_0 / (1 + d t) and choose a sensible decay rate d.
> - Extend Optimizer_SGD with pre_update_params, post_update_params and an iterations counter.
> - Diagnose whether a plateau is a learning-rate issue or a local-minimum issue.

![Two panels. Left: the learning rate over 10,000 iterations, flat at 1.0 for d = 0, falling to about 0.09 for d = 1e-3, and already near 0.09 after 1,000 iterations for d = 1e-2. Right: three sketched loss curves over 10,000 epochs, ending at 0.87 with no decay, 0.76 with d = 1e-3 and 1.07 with d = 1e-2.](diagrams/01-decay-schedule-and-result.svg)

*The left panel is the schedule itself. The right panel is a sketch: its three end values are the losses measured on seed 0, the shapes of its curves are not drawn from data, and its label "winner" holds for that seed only (section 6).*

---

## 1. The question: can one learning rate serve the whole run?

[Post 22](../22-gradient-descent-optimiser/index.md) trained the spiral classifier with `Optimizer_SGD(learning_rate=1.0)`, and the rate stayed at 1.0 for all 10,001 epochs. `snippets/seed_spread.py` repeats that run: on the documented seed it ends at a loss of 0.8737 and an accuracy of 64.7 percent. It also counts something that a log of every 100th or 1,000th epoch hides: on 4,628 of the 10,000 steps the loss went up.

A learning rate has two jobs, and they pull in opposite directions.

| Phase | What the optimiser needs | What a constant $\alpha$ gives |
|---|---|---|
| Early: far from any good region | large steps, or the first thousand epochs are spent crawling | fine if $\alpha$ is large |
| Late: inside a narrow valley | small steps that stay inside it | too large if $\alpha$ suited the early phase: the step overshoots and the loss rises |

A constant has to be chosen for one row and is then wrong for the other. The question of this post is: **can a rate that shrinks during the run do both jobs, and what does the shrinking cost?**

---

## 2. The decay formula

The schedule of this series makes the rate a function of the number of updates already made:

$$\alpha_t = \frac{\alpha_0}{1 + d \cdot t}$$

- $\alpha_0$ is the **initial learning rate**, the value given to the optimiser (`learning_rate`).
- $d$ is the **decay rate** (`decay`), a small number that is zero or positive.
- $t$ is the **update counter** (`iterations`): 0 for the first update, 1 for the second, and so on.

The formula comes from one rule: at every update the reciprocal of the rate grows by the same amount, $d / \alpha_0$. The reciprocal starts at $1/\alpha_0$, so after $t$ updates it is $(1 + d \cdot t)/\alpha_0$, and turning it over gives the formula. A rate of 1.0 with $d = 10^{-3}$ has the reciprocals 1, 1.001, 1.002 and so on. Three properties follow for every $d > 0$.

**It only ever decreases.** The denominator grows with $t$, so $\alpha_{t+1} < \alpha_t$ for every $t$.

**It reaches a given fraction at a time that can be read off.** Solving $\alpha_t = \alpha_0 / k$ gives $t = (k - 1)/d$. The rate is halved at $t = 1/d$, quartered at $3/d$, and at an eighth at $7/d$: each halving takes twice as long as the one before.

**It tends to zero and never reaches it.** No update is ever switched off.

### 2.1. What different values of $d$ look like

`snippets/decay_schedule.py` reads the rate off the class of section 4 at chosen updates, with $\alpha_0 = 1.0$:

```text
     t   d=0        d=0.0001   d=0.001    d=0.01     d=0.1
     0     1.0000     1.0000     1.0000     1.0000     1.0000
     1     1.0000     0.9999     0.9990     0.9901     0.9091
   100     1.0000     0.9901     0.9091     0.5000     0.0909
  1000     1.0000     0.9091     0.5000     0.0909     0.0099
  5000     1.0000     0.6667     0.1667     0.0196     0.0020
 10000     1.0000     0.5000     0.0909     0.0099     0.0010
```

The columns are the same curve at different speeds: $d = 10^{-2}$ at update 100 stands where $d = 10^{-3}$ stands at update 1,000, because only the product $d \cdot t$ enters the formula.

### 2.2. Choosing $d$

The second property turns the choice of $d$ into a choice that is easier to make: the fraction $1/k$ of the initial rate that should be left at the last update $T$. Then

$$d = \frac{k - 1}{T}.$$

For the 10,001 epochs of this series $T = 10{,}000$. Keeping a tenth of the rate at the end gives $d = 9 \times 10^{-4}$, which the series rounds to $10^{-3}$; keeping a half gives $d = 10^{-4}$; keeping a hundredth gives $d = 9.9 \times 10^{-3}$, close to $10^{-2}$. A tenth is a common starting point and no more than that: section 6 measures all three. The initial rate is tuned first, with `decay=0`, and $d$ afterwards, because changing both at once leaves no way to tell which of the two changed the result.

---

## 3. Why $1/(1 + d \cdot t)$ and not something else

Three other schedules are in common use.

**Step decay** multiplies the rate by a fixed factor at fixed updates, for example by 0.5 at every 1,000th. It is simple, and the positions of the steps are further hyperparameters.

**Exponential decay** is $\alpha_t = \alpha_0 \gamma^t$ with $\gamma$ slightly below 1: the logarithm of the rate, not its reciprocal, changes by the same amount at every update. Its half-life is constant, $\ln 2 / \ln(1/\gamma)$ updates, wherever the run is.

**Cosine decay** follows half a cosine wave from $\alpha_0$ down to a small floor over a number of updates fixed in advance (Loshchilov and Hutter, 2017).

The difference shows when three schedules are made to agree at update 1,000:

```text
gamma = 0.999307
     t   inverse time   exponential   step
     0         1.0000        1.0000   1.0000
   500         0.6667        0.7071   1.0000
  1000         0.5000        0.5000   0.5000
  3000         0.2500        0.1250   0.1250
  7000         0.1250        0.0078   0.0078
 10000         0.0909        0.0010   0.0010
```

The exponential and the step schedule halve again every 1,000 updates and are at a thousandth by the end. The schedule of section 2, often called inverse-time decay, halves at 1,000, 3,000 and 7,000 and still has 9 percent left. With a careless $\gamma$ the exponential is gone much sooner: $\gamma = 0.99$ has a half-life of 69.0 updates, and $0.99^{10000} = 2.2 \times 10^{-44}$.

A second way to compare schedules is to add up all the rates of a run. Each update moves the parameters by the rate times the gradient, so the sum of the rates is the distance the run could cover if the gradient had length 1 throughout:

```text
d=0       sum   10001.0   share of the constant rate 100.0%
d=0.0001  sum    6932.2   share of the constant rate  69.3%
d=0.001   sum    2398.4   share of the constant rate  24.0%
d=0.01    sum     462.0   share of the constant rate   4.6%
d=0.1     sum      69.6   share of the constant rate   0.7%
exponential, gamma above: sum    1441.8
```

This is the price of decay, known before any training: $d = 10^{-3}$ leaves 24 percent of the constant rate's budget, $d = 10^{-2}$ under 5 percent.

The sum also explains the form. For noisy gradients, two conditions on the step sizes, usually named after Robbins and Monro (1951), are the classical route to a convergence proof: $\sum_t \alpha_t = \infty$, so the steps can still cover any distance, and $\sum_t \alpha_t^2 < \infty$, so the noise is averaged away. Rates that fall like $1/t$ meet both, and an exponential schedule fails the first, because its sum is finite. Bottou (2012) recommends a schedule of this inverse-time form for stochastic gradient descent. Two cautions apply here. The conditions come with assumptions that a neural network's loss does not satisfy, so no schedule guarantees the best minimum. And the full-batch gradient of this post has no sampling noise at all; the shrinking rate is used against overshooting, not against noise.

---

## 4. The updated optimiser class

The update of post 22 is one line per parameter array, called once per layer. Decay needs two more moments in every training step: one before the layers are updated, to compute the rate, and one after all of them are, to count the update. That gives the class three methods that the training loop calls, two of them new:

```python
class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases  -= self.current_learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1
```

Against the class of post 22, the additions are the `decay` argument, the attributes `current_learning_rate`, `decay` and `iterations`, and the methods `pre_update_params` and `post_update_params`. The two existing lines change in one name each: `update_params` reads `current_learning_rate` where it read `learning_rate`.

**Two rate attributes.** `learning_rate` is $\alpha_0$ and is never overwritten. `current_learning_rate` is $\alpha_t$, written by `pre_update_params` and read by `update_params`. It is an attribute and not a return value because `update_params` is called once per layer and every layer of one step must see the same rate.

**The default is no decay.** With `decay=0.0` the guard `if self.decay:` is false, the division is skipped, and `current_learning_rate` keeps its initial value. A loop that calls `update_params` alone, as the loop of post 22 does, behaves exactly as before.

**The counter has its own method.** `update_params` runs once per layer, so a counter incremented there would advance twice per step in this network (section 10 measures it). `post_update_params` runs once per step.

**The counter starts at 0 and is read before it is incremented.** The first update uses $\alpha_0$ exactly; the update with index $t$ uses $\alpha_t$. The script confirms it on the class:

```text
decay=1e-3, first update: iterations 0, current_learning_rate 1.0
decay=1e-3, second update: iterations 1, current_learning_rate 0.999001
largest gap between the class and 1 / (1 + d t): 0.0e+00
```

After a run of 10,001 updates the counter stands at 10,001, and the last update, index 10,000, used $1/(1 + 10^{-3} \cdot 10{,}000) = 1/11 = 0.0909$.

---

## 5. The training loop with decay

The setup is the one of post 22: `nnfs.init()`, which sets the seed to 0 and the dtype to float32, then `spiral_data(samples=100, classes=3)`, `Layer_Dense(2, 64)`, a ReLU, `Layer_Dense(64, 3)` and the combined softmax and cross-entropy class, with weights of 0.01 times standard normal draws, biases of zero, and 10,001 full-batch epochs. Against that post's loop, the optimiser gets its `decay` argument, the two new calls enclose the updates, and the log prints the rate.

```python
optimizer = Optimizer_SGD(learning_rate=1.0, decay=1e-3)

for epoch in range(10001):
    # Forward.
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Accuracy.
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy    = np.mean(predictions == y)

    # Backward.
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update with decay.
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    if epoch % 1000 == 0:
        print(f'epoch {epoch:5d}  loss {loss:.4f}  '
              f'acc {accuracy:.4f}  lr {optimizer.current_learning_rate:.4f}')
```

`snippets/train_with_decay.py` prints:

```text
epoch     0  loss 1.0986  acc 0.3600  lr 1.0000
epoch  1000  loss 1.0631  acc 0.4433  lr 0.5000
epoch  2000  loss 0.9996  acc 0.4900  lr 0.3333
epoch  3000  loss 0.9771  acc 0.4867  lr 0.2500
epoch  4000  loss 0.9319  acc 0.5367  lr 0.2000
epoch  5000  loss 0.8935  acc 0.5633  lr 0.1667
epoch  6000  loss 0.8626  acc 0.5867  lr 0.1429
epoch  7000  loss 0.8357  acc 0.5933  lr 0.1250
epoch  8000  loss 0.8063  acc 0.5933  lr 0.1111
epoch  9000  loss 0.7820  acc 0.6333  lr 0.1000
epoch 10000  loss 0.7612  acc 0.6467  lr 0.0909
```

The `lr` column is the rate that the update of that epoch used, and it follows the table of section 2.1. The loss in a row is computed before that epoch's update. The first 1,000 epochs, at rates between 1.0 and 0.5, move the loss from 1.0986 to 1.0631 only: the network starts in a flat region, and the largest rates of the schedule are spent leaving it.

---

## 6. What happens when it runs

### 6.1. Five seeds, with and without decay

One run is one draw of the data and of the initial weights. `snippets/seed_spread.py` trains the network ten times: with the constant rate of post 22 and with `decay=1e-3`, each for seeds 0 to 4, where seed 0 is the documented run. `rises` counts the steps on which the loss went up, `largest` is the largest single rise, the next column is the lowest and the highest loss of the last 1,000 epochs, and the last two columns are the mean loss and the mean accuracy of those 1,000 epochs.

```text
seed  setting     loss    acc     rises  largest  loss in the last 1,000 epochs   |gradient|  mean loss  mean acc
   0  constant    0.8737  0.6467   4628   0.0553  0.835 to 0.933                  0.6569      0.8703     0.6132
   0  decay 1e-3  0.7612  0.6467   3973   0.0152  0.761 to 0.782                  0.3898      0.7714     0.6570
   1  constant    0.5091  0.7467   4665   0.2879  0.483 to 1.231                  0.4293      0.5880     0.7203
   1  decay 1e-3  0.7335  0.6933    925   0.0002  0.733 to 0.763                  0.3323      0.7486     0.6819
   2  constant    0.9906  0.5367   4154   0.2551  0.793 to 1.326                  0.8130      0.8905     0.5619
   2  decay 1e-3  0.9107  0.5800      0   0.0000  0.911 to 0.934                  0.0254      0.9222     0.5714
   3  constant    0.3943  0.8700   4501   1.1043  0.352 to 2.636                  0.5946      0.4159     0.8454
   3  decay 1e-3  0.7310  0.6700   3252   0.0099  0.731 to 0.766                  0.6955      0.7500     0.6869
   4  constant    0.4843  0.7867   4765   0.3507  0.483 to 1.461                  0.0581      0.5842     0.7262
   4  decay 1e-3  0.7750  0.6367   4356   0.0017  0.775 to 0.797                  0.2408      0.7867     0.6255

constant    final loss 0.3943 to 0.9906   final accuracy 0.5367 to 0.8700   largest rise 0.0553 to 1.1043   width of the last band 0.098 to 2.283
decay 1e-3  final loss 0.7310 to 0.9107   final accuracy 0.5800 to 0.6933   largest rise 0.0000 to 0.0152   width of the last band 0.021 to 0.035
decay ends with the lower loss on seeds [0, 2], with the higher accuracy on seeds [2], with the same accuracy on seeds [0]

the same comparison on the means of the last 1,000 epochs, which do not depend on the stopping epoch
constant    mean loss 0.4159 to 0.8905   mean accuracy 0.5619 to 0.8454
decay 1e-3  mean loss 0.7486 to 0.9222   mean accuracy 0.5714 to 0.6869
decay has the lower mean loss on seeds [0], the higher mean accuracy on seeds [0, 2]
```

**The documented seed.** On seed 0 decay ends at a loss of 0.7612 where the constant rate ends at 0.8737, and both runs classify 194 of the 300 points correctly, 64.7 percent.

**The jumps disappear, on every seed.** With the constant rate the loss rises on 4,154 to 4,765 of the 10,000 steps, and its largest single rise is between 0.0553 and 1.1043. With decay the largest rise is between 0 and 0.0152. The loss still goes up on many steps in four of the five decayed runs, but by far less: the largest rise is about a quarter of the constant rate's on seed 0 and under a hundredth of it on the other four.

**The end of a constant-rate run is a draw.** In its last 1,000 epochs the constant-rate loss moves inside a band that is 0.098 to 2.283 wide, and the reported final loss is wherever in that band epoch 10,000 happens to fall. On seed 3 the band reaches from 0.352 to 2.636. The decayed runs stay within a band of 0.021 to 0.035, so stopping an epoch earlier or later reports nearly the same loss.

**Decay does not end lower on most seeds.** By the final loss the decayed run is lower on seeds 0 and 2 and higher on seeds 1, 3 and 4. A final value of the constant-rate run is one draw from its band, though, so the comparison that does not depend on the stopping epoch is the mean over the last 1,000 epochs, and it is less kind to decay: the decayed mean loss is the lower one on seed 0 only (0.7714 against 0.8703), and on seed 2 the order turns round (0.9222 against 0.8905). The mean accuracy is higher with decay on seeds 0 and 2, by 4 points and by 1, and lower on seeds 1, 3 and 4, by 4 to 16 points. The comparison on seed 0 alone would have suggested a clear gain; the spread shows a trade. Decay buys a steady, repeatable descent and pays with the 76 percent of the step budget it gives up (section 3), and inside 10,001 epochs that budget was worth more than the steadiness on four of these five seeds.

### 6.2. A smaller and a larger decay

`snippets/decay_too_small.py` and `snippets/decay_too_large.py` run the same five seeds with $d = 10^{-4}$ and with $d = 10^{-2}$:

```text
decay 1e-4  final loss 0.4610 to 1.0095   final accuracy 0.4633 to 0.8233   rises 4090 to 4719   largest rise 0.0425 to 0.0722
decay 1e-2  final loss 1.0457 to 1.0725   final accuracy 0.3967 to 0.4633   largest rise 0.000000 to 0.000000
```

**$d = 10^{-4}$ changes little.** The rate is still 0.5 at the end. The loss rises on 4,090 to 4,719 steps, about as often as with the constant rate, and the final losses cover the same wide range. The largest rises are smaller than the constant rate's on four of the five seeds.

**$d = 10^{-2}$ stalls every seed.** The rate is at 0.5 after 100 updates and below 0.1 after 1,000, before the network has left the flat region of its start. All five runs end between 1.0457 and 1.0725, barely below the 1.0986 they started from, with 39.7 to 46.3 percent accuracy. That is the highest loss of the four settings on every seed, and the lowest accuracy on every seed but seed 2, where $d = 10^{-4}$ ends at the same 46.3 percent. The curve is smooth, with no step that raises the loss by as much as a millionth, and smoothness is not the goal.

On the documented seed the four settings end at 0.8737 ($d = 0$), 0.9710 ($10^{-4}$), 0.7612 ($10^{-3}$) and 1.0725 ($10^{-2}$), with accuracies of 64.7, 59.0, 64.7 and 39.7 percent.

---

## 7. What decay solves, and what it does not

**Decay solves overshooting.** A loss that rises after an update means the step went past the bottom of the valley it was in. A smaller rate makes that step shorter, and section 6.1 shows the effect on every seed.

**Decay does not make training faster.** Every decayed step is shorter than the constant-rate step from the same point would be. Where the constant-rate run was making progress between its jumps, decay slows that progress down, which is what four of the five seeds show. Getting further in the same number of epochs needs steps that are better aimed, not smaller; that is the subject of [post 24](../24-momentum/index.md).

**Decay cannot move an optimiser out of a point where the gradient is zero.** At a local minimum the update is the rate times a gradient of zero, whatever the rate is.

### 7.1. Reading a plateau

A loss that has stopped falling has two possible causes, since each update is $\alpha_t$ times the gradient: the rate has become too small, or the gradient has. The log shows which signs go with which cause.

| What the log shows | Cause | Does decay help? |
|---|---|---|
| The loss goes up and down from step to step | The rate is too large for the valley | Yes |
| The loss falls on every step, by less and less, and `current_learning_rate` is a small fraction of its start | The rate has run out | No: more decay makes it worse; a smaller $d$ or a restored rate helps |
| The loss is flat, and stays flat when the rate is restored | A minimum, or a region too flat for any stable rate: the gradient is near zero | No schedule helps |

The second and third rows cannot be told apart by the length of the gradient alone. `snippets/decay_too_large.py` shows it on the stalled runs of $d = 10^{-2}$:

```text
seed  loss    acc     last rate  |gradient|  fall in the last 1,000 epochs   loss after  acc after
   0  1.0725  0.3967  0.0099     0.0058      0.00031                         0.9548      0.5333
   1  1.0539  0.4400  0.0099     0.0042      0.00009                         0.8449      0.5467
   2  1.0636  0.4633  0.0099     0.0052      0.00019                         0.9019      0.5667
   3  1.0636  0.4200  0.0099     0.0030      0.00004                         0.7486      0.5967
   4  1.0457  0.4633  0.0099     0.0094      0.00081                         0.7384      0.6367
```

`|gradient|` is the length of the whole gradient, the square root of the sum of the squares of all 387 entries of the four gradient arrays. At these plateaus it is 0.0030 to 0.0094, about a hundredth of the 0.33 to 0.81 at which seven of the ten runs of section 6.1 end, and the loss fell by less than 0.001 in the last 1,000 epochs. Both numbers look like a minimum. The test is to restore the rate: the last two columns are the same networks after 3,000 further epochs with `Optimizer_SGD(learning_rate=1.0)`. On all five seeds the loss falls, to between 0.7384 and 0.9548, and the accuracy rises by 10 to 18 points. The plateau was the rate.

A plateau is a learning-rate issue when restoring the rate, with a new optimiser object or by setting `optimizer.iterations = 0`, makes the loss fall again. It is a minimum, or a region too flat to cross, when the loss stays where it is at every rate that does not make it jump. None of the 20 runs of this post is of that kind: the loss was still moving at the end of fifteen of them, and the five that looked flat passed the test.

---

## 8. Why three methods instead of one

Two extra methods are a lot of structure for a division and an increment. They are written here because every optimiser of Part VI uses the same three calls, and only what happens inside them differs. In the table $g$ is the gradient of a parameter $\theta$.

| Optimiser | `pre_update_params` | `update_params` | `post_update_params` |
|---|---|---|---|
| Gradient descent (post 22) | not present | $\theta \leftarrow \theta - \alpha g$ | not present |
| With decay (this post) | recompute $\alpha_t$ from $t$ | $\theta \leftarrow \theta - \alpha_t g$ | $t \leftarrow t + 1$ |
| Momentum (post 24) | the same | $v \leftarrow \beta v - \alpha_t g$, then $\theta \leftarrow \theta + v$ | the same |
| AdaGrad (post 25) | the same | $G \leftarrow G + g^2$, then $\theta \leftarrow \theta - \alpha_t g / (\sqrt{G} + \epsilon)$ | the same |
| RMSProp (post 26) | the same | $G \leftarrow \rho G + (1 - \rho) g^2$, then the AdaGrad step | the same |
| Adam (post 27) | the same | moving averages of $g$ and $g^2$, corrected for their zero start, then the step | the same |

![A table of six optimisers as rows, gradient descent, decay, momentum, AdaGrad, RMSProp and Adam, against three columns for pre_update_params, update_params and post_update_params. From the second row down the first column always says recompute alpha and the last always says t += 1, and only the middle column changes.](diagrams/02-three-hook-contract.svg)

*Two of the three columns stop changing here. The figure writes $\epsilon$ inside the square root for AdaGrad and RMSProp; the series' code adds it outside, as the table above does.*

The four update lines of section 5 are therefore written once. Replacing the optimiser object replaces the middle column and leaves the loop alone, and the decay of this post comes with every later optimiser at no further cost.

---

## 9. Make it run: schedule, training, seeds

Six scripts hold every code block and every printed number of this post. They run from the series root:

```text
python posts/23-learning-rate-decay/snippets/decay_schedule.py
python posts/23-learning-rate-decay/snippets/train_with_decay.py
python posts/23-learning-rate-decay/snippets/seed_spread.py
python posts/23-learning-rate-decay/snippets/decay_too_small.py
python posts/23-learning-rate-decay/snippets/decay_too_large.py
python posts/23-learning-rate-decay/snippets/what_can_go_wrong.py
```

`decay_schedule.py` and `what_can_go_wrong.py` need only NumPy, train nothing and finish in under a second. The other four also need the `nnfs` package for the spiral data and for `nnfs.init()`. `train_with_decay.py` is the documented run and takes about 7 seconds. `seed_spread.py` trains ten networks for 10,001 epochs and takes a little over a minute; `decay_too_small.py` and `decay_too_large.py` train five each and take 40 to 50 seconds. These three call `nnfs.init()` once and `np.random.seed(seed)` before each run, so that each seed draws its own data and its own weights. The dense layer, the ReLU and the combined softmax and loss class are those of posts 16 and 19, unchanged.

---

## 10. What can go wrong?

None of the first four mistakes raises an error. `snippets/what_can_go_wrong.py` runs each for 10,001 updates on two layers whose gradients are set to 1, so that the distance a weight has moved equals the sum of the rates used.

```text
== Reference: the class of section 4, decay=1e-3, two layers, 10,001 updates
iterations 10001   last rate 0.0909   distance moved 2398.4
```

**The counter incremented in `update_params`.** The method runs once per layer, so with two layers the counter advances by two per step and the schedule runs at twice the intended decay:

```text
iterations 20002   last rate 0.0476   distance moved 1522.8
```

The last rate is $1/(1 + 2 \times 10^{-3} \cdot 10{,}000) = 1/21$. A network with ten layers would decay ten times too fast.

**A hook that is never called.** Without `pre_update_params` the rate is never recomputed, and without `post_update_params` the counter never moves. Either way the run is the constant-rate run of post 22, whatever `decay` says:

```text
no pre_update_params : iterations 10001   last rate 1.0000   distance moved 10001.0
no post_update_params: iterations     0   last rate 1.0000   distance moved 10001.0
```

**The update reads `learning_rate`.** If `update_params` multiplies by `self.learning_rate`, the log prints a falling `current_learning_rate` while every step is taken at the base rate:

```text
iterations 10001   last rate 0.0909   distance moved 10001.0
```

The printed rate is not evidence that the rate was applied. Only the weights are.

**One optimiser object used for a second run.** The counter belongs to the object and carries on:

```text
first rate of the second run: 0.0909   iterations 10001
after iterations = 0:         1.0000
```

A second network trained with the same object starts at 9 percent of the intended rate. Each run gets a new optimiser, or the counter is set back to 0.

**A negative decay.** The class accepts it. The rate then grows, and with `decay=-1e-3` the denominator is zero at $t = 1/\lvert d \rvert = 1{,}000$:

```text
rate at updates 0, 500, 900, 999: 1.0, 2.0, 10.0, 1000.0
update 1000: ZeroDivisionError: float division by zero
```

Where $1/\lvert d \rvert$ is not a whole number the denominator steps over zero and nothing is raised: with `decay=-3e-4` the rate is 10,000 at update 3,333 and $-5{,}000$ at update 3,334, and from there on every step goes up the gradient.

A decay that is too large raises nothing either; its stalled runs are measured in sections 6.2 and 7.1.

---

## 11. Summary

| Concept | Takeaway |
|---|---|
| Schedule | $\alpha_t = \alpha_0 / (1 + d \cdot t)$; $t$ starts at 0, so the first update uses $\alpha_0$ |
| Reading $d$ | the rate is at $\alpha_0 / k$ when $t = (k - 1)/d$; for a tenth left after $T$ updates, $d = 9/T$ |
| Cost | the rates of 10,001 updates at $d = 10^{-3}$ add up to 24 percent of the constant rate's |
| Class | `pre_update_params` computes the rate, `update_params` applies it per layer, `post_update_params` counts the step |
| Measured on five seeds | decay $10^{-3}$ removes the jumps on all five; its mean loss over the last 1,000 epochs is the lower one on one |
| Too much decay | $d = 10^{-2}$ stalls all five seeds near a loss of 1.06 |
| Plateau test | restore the rate; if the loss falls again, the plateau was the rate |

---

## Common pitfalls

1. **Counting updates per layer.** The counter belongs in `post_update_params`, which runs once per step; in `update_params` it multiplies the decay by the number of layers.
2. **Trusting the printed rate.** `update_params` must read `current_learning_rate`. A log can show the rate falling while the weights move at the base rate.
3. **Judging decay from one seed.** On seed 0 decay lowers the final loss by 0.11; on four of the five seeds its mean loss over the last 1,000 epochs is the higher one.
4. **Taking a smooth curve for a good one.** The run with $d = 10^{-2}$ never rises and never learns. A rate that has fallen below a hundredth of its start is the first thing to check on a flat loss.
5. **Reusing an optimiser.** `iterations` carries over to the next run. A fresh object starts the schedule again.

---

## Further reading

- Bottou, L., *"Stochastic Gradient Descent Tricks"* (Neural Networks: Tricks of the Trade, 2012).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 8 (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 10 (2020).
- Loshchilov, I. and Hutter, F., *"SGDR: Stochastic Gradient Descent with Warm Restarts"* (ICLR, 2017).
- Robbins, H. and Monro, S., *"A Stochastic Approximation Method"* (Annals of Mathematical Statistics, 1951).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 24 - Momentum](../24-momentum/index.md):** a velocity that carries the direction of earlier steps, so that the steps are better aimed and not only smaller.
- **[Post 32 - Mini-batching](../32-mini-batching/index.md):** an epoch then holds many updates, and `iterations` counts updates, so a decay chosen for 10,001 full-batch epochs has to be chosen again.
