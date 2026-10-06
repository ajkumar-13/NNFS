# 24 - Momentum

> **TL;DR.** Gradient descent uses the current gradient and nothing else, so it bounces between the walls of a narrow valley and crawls across flat ground. **Momentum** keeps a velocity for every parameter, $v \leftarrow \beta v - \alpha g$, and steps by it: parts of consecutive steps that oppose each other cancel, and parts that agree add up to as much as $1/(1 - \beta)$ times one step. On the spiral run of post 23, adding `momentum=0.9` takes the final loss from 0.7612 to 0.1209 and the accuracy from 64.7 to 95.7 percent on seed 0. Over five seeds the loss is lower every time, the accuracy is higher in four, and the accuracy ranges from 68.7 to 98.0 percent.
>
> **Prerequisites:** [Post 22](../22-gradient-descent-optimiser/index.md), [Post 23](../23-learning-rate-decay/index.md).
> **Safe to skip?** Skip it if the reader can write the velocity update from memory, say where the learning rate enters it in each of the two common conventions, and add it to an optimiser class as one buffer per parameter array.
>
> **After reading, you will be able to:**
>
> - Explain momentum as vector cancellation across steps: opposing parts of consecutive steps cancel and agreeing parts add up.
> - Implement momentum inside Optimizer_SGD with a per-layer weight_momentums buffer.
> - Translate between the two sign conventions of the velocity: learning rate inside and subtracted, or gradient added and the rate applied at the step.

![Two charts of the ravine L = (x squared + 100 y squared)/2 with elliptical contours, each a 100-step path from (10, 1) with learning rate 0.019. Without momentum the path zig-zags from wall to wall while it creeps along the floor and ends at x = 1.4686, loss 1.0784. With momentum 0.9 it holds still across on every second step, runs past the minimum to x = -2.84, comes back and ends at x = 0.0523, loss 0.0027. Travel across is 18.999 against 18.902, travel along 8.531 against 17.834.](diagrams/01-ravine-paths.svg)

*The ravine of section 1, 100 steps of `ravine.py` each. With momentum the path crosses the ravine about as much, travels twice as far along the floor, and ends with a loss 400 times lower.*

---

## 1. The question: what does gradient descent lose by forgetting?

The optimiser of [post 22](../22-gradient-descent-optimiser/index.md) takes one piece of information at each step, the current gradient, moves against it by the learning rate, and forgets it. [Post 23](../23-learning-rate-decay/index.md) made the learning rate shrink over time and changed nothing else. Two failures follow from having no memory.

**Narrow valleys produce zig-zags.** Picture a loss surface shaped like a long ravine: steep walls on two sides and a gentle slope along the floor. Almost everywhere the gradient is dominated by the walls, so a step moves mostly across the ravine and only a little along it. If the step is long enough to cross the floor, the next gradient points back at the wall just left, and the optimiser bounces from side to side.

**Flat ground stalls.** Where the gradient is small the step is small, however many steps before it pointed the same way. A decay schedule makes this worse late in training, because the learning rate has shrunk as well.

Both can be measured on the smallest surface that has a ravine, $L(x, y) = (x^2 + 100 y^2)/2$. Its gradient is $(x, 100y)$, its minimum is at the origin, $x$ runs along the floor and $y$ across it. `snippets/ravine.py` starts at $(10, 1)$, where the loss is 100, with a learning rate of 0.019:

```text
step   no momentum: step, position             momentum 0.9: step, position
   1   (-0.1900, -1.9000) ( 9.8100, -0.9000)   (-0.1900, -1.9000) ( 9.8100, -0.9000)
   2   (-0.1864,  1.7100) ( 9.6236,  0.8100)   (-0.3574,  0.0000) ( 9.4526, -0.9000)
   3   (-0.1828, -1.5390) ( 9.4408, -0.7290)   (-0.5013,  1.7100) ( 8.9514,  0.8100)
   4   (-0.1794,  1.3851) ( 9.2614,  0.6561)   (-0.6212,  0.0000) ( 8.3302,  0.8100)
```

The left half is gradient descent. Each step multiplies $y$ by $1 - 0.019 \cdot 100 = -0.9$, so the point jumps to the opposite wall every time, and multiplies $x$ by $1 - 0.019 = 0.981$, so it creeps along the floor. The across part of a step is about ten times its along part. A smaller learning rate stops the bounce and slows the creep further; a larger one, above $2/100$, makes the bounce grow. The question of this post is: **what can an optimiser do with the gradients it has already seen?**

---

## 2. The vector-cancellation intuition

Take the first two steps of the left half and split each into its part across the ravine and its part along it. The across parts are $-1.9$ and $+1.71$: they disagree. The along parts are $-0.19$ and $-0.1864$: they agree. The script adds the two steps:

```text
steps 1 and 2 without momentum, added: (-0.3764, -0.1900)
```

Added, the across parts nearly cancel and the along parts double. Gradient descent never adds them: it takes each step in full, so the across motion is paid for twice and undone once.

Momentum is the rule that does the adding. The optimiser keeps a **velocity**, a running sum of its past steps in which older steps count for less, and moves by the velocity instead of by the latest gradient step alone. The right half of the output above is the same start with a momentum coefficient of 0.9. Step 1 is identical, because the velocity starts at zero. Step 2 is 0.9 times step 1 plus the new gradient step:

```text
step 2 with momentum = 0.9 * step 1 - alpha * gradient 2 = (-0.1710, -1.7100) + (-0.1864,  1.7100) = (-0.3574,  0.0000)
```

The across parts cancel, here exactly, and the along part is nearly twice a single step. Steps 3 and 4 repeat the pattern while the along part keeps growing: $-0.19$, $-0.36$, $-0.50$, $-0.62$. The figure below sets the two additions side by side.

![Two vector diagrams with tables of the parts along and across. Left, without momentum, step 1 (-0.1900, -1.9000) and step 2 (-0.1864, 1.7100) form a V, and their sum (-0.3764, -0.1900) is nearly flat. Right, with momentum 0.9, 0.9 times step 1 (-0.1710, -1.7100) plus the same gradient step gives step 2, (-0.3574, 0.0000).](diagrams/02-steps-added.svg)

*Along the floor is drawn at four times the scale across. The gradient step is the same in both panels; with momentum the addition happens inside step 2.*

After 100 steps:

```text
beta    x      |y|        loss      travel across  travel along  across steps that reverse
0.00   1.4686  2.66e-05     1.0784         18.999         8.531                         99
0.90   0.0523  5.15e-03     0.0027         18.902        17.834                         49
```

The loss is 400 times lower with momentum (0.0027 against 1.0784), and all of that gain is along the floor: $x$ has gone from 10 to 0.0523 instead of 1.4686. The bounce has not vanished. The across step reverses 49 times instead of 99, but the distance travelled across the ravine is almost the same (18.902 against 18.999), and the remaining $|y|$ is larger with momentum. What the velocity does on this surface is hold the across motion still on every second step and use the agreement along the floor to go twice as far. The figure at the top of the post draws the two paths.

Flat ground works the same way. Where the current gradient is tiny, the velocity inherited from many steps in one direction is not, and the optimiser keeps moving. Section 3 puts a number on it.

---

## 3. The momentum formula

$$v_t = \beta \, v_{t-1} - \alpha \, g_t, \qquad \theta_t = \theta_{t-1} + v_t$$

Here $\theta$ is any trainable parameter, a weight or a bias, and $g_t = \partial L / \partial \theta$ is its gradient at $\theta_{t-1}$.

- $v_t$ is the **velocity**: one number per parameter, so the velocity of an array has the shape of that array.
- $\beta$ is the **momentum coefficient**, a number in $[0, 1)$, `momentum` in the code. It is the fraction of the previous velocity that survives a step.
- $\alpha$ is the **current** learning rate of post 23, so a decay schedule still applies.
- The velocity before the first step is zero, so the first step is $-\alpha g_1$, a plain gradient step, whatever $\beta$ is.

**With $\beta = 0$** the rule is $v_t = -\alpha g_t$ and $\theta_t = \theta_{t-1} - \alpha g_t$: the update of post 22. `ravine.py` runs the class of section 5 against the two lines above written out, and with `momentum=0.0` against `theta -= alpha * g`:

```text
beta 0.9: largest gap between class and formula over 100 steps 0.0e+00
momentum 0.0 against theta -= alpha * g: largest gap 0.0e+00
```

**With a gradient that never changes**, the velocity is a geometric series. After $n$ steps its size is $\alpha g (1 + \beta + \dots + \beta^{n-1}) = \alpha g \, (1 - \beta^n)/(1 - \beta)$, which approaches $\alpha g / (1 - \beta)$:

```text
beta 0.90  t=1: 1.000  t=2: 1.900  t=3: 2.710  t=10: 6.513  t=50: 9.948  t=1000: 10.000   1 / (1 - beta) = 10
```

This is the speed that builds up along a consistent direction: with $\beta = 0.9$ the step grows to ten times a plain gradient step, and with 0.99 to a hundred times. Unrolled, $v_t = -\alpha \sum_{k \ge 0} \beta^k g_{t-k}$ at a constant rate: a weighted sum of all past gradients in which a gradient $k$ steps old counts $\beta^k$. The weights add up to $1/(1 - \beta)$, which is why that number is quoted as the horizon of the velocity; with $\beta = 0.9$ the latest ten gradients carry 0.651 of the total weight. It is a sum and not an average: there is no factor $(1 - \beta)$ in front of the gradient, as there will be in Adam (post 27). The figure below draws both the growth and the weights.

![Left, the size of the velocity under a gradient that never changes, against the step on logarithmic axes, for beta 0.5, 0.9 and 0.99: it levels off at 2, 10 and 100 plain steps, and the values printed at t = 1, 2, 3, 10, 50 and 1,000 are marked. Right, stems for the weight 0.9 to the k of a gradient k steps old, k from 0 to 29: the latest ten carry 0.651 of a total weight of 10.](diagrams/03-velocity-horizon.svg)

*Both panels are computed from the formulas above; the marks on the left are the values `ravine.py` prints.*

**With $\beta = 1$** nothing is ever forgotten, and on the ravine the path never settles (section 10).

The idea is older than deep learning. Polyak's heavy-ball method of 1964 writes the same step as $\theta_{t+1} = \theta_t - \alpha g + \beta (\theta_t - \theta_{t-1})$, which is the rule above because $v_t = \theta_t - \theta_{t-1}$. Rumelhart, Hinton and Williams used such a term in their 1986 backpropagation paper, and Sutskever and co-authors (2013), who write the rule in the form used here, studied how much it matters for training deep networks. Their paper also treats Nesterov's variant, which evaluates the gradient at the point the velocity is about to reach; this series does not implement it.

### 3.1. The two sign conventions

Two forms of the rule are in common use. They differ in the sign of the velocity and in where the learning rate enters.

| Convention | Velocity | Parameter update | Buffer holds |
|---|---|---|---|
| Learning rate inside (this series) | $v_t = \beta v_{t-1} - \alpha g_t$ | $\theta \leftarrow \theta + v_t$ | the step itself |
| Gradient added | $u_t = \beta u_{t-1} + g_t$ | $\theta \leftarrow \theta - \alpha u_t$ | a sum of gradients |

With a constant learning rate the two are the same algorithm, and the translation is $v_t = -\alpha u_t$: a change of sign and of scale. `ravine.py` runs both:

```text
decay 0.0: largest gap between the two paths 8.9e-16   loss after 100 steps 0.002698 against 0.002698
decay 0.01: largest gap between the two paths 3.8e-01   loss after 100 steps 0.003174 against 0.001451
after 3 steps at a constant rate: v = (-0.5013,  1.7100)   -alpha * u = (-0.5013,  1.7100)
```

With a learning rate that changes they are two different algorithms, and the second line shows it. In the first form every past gradient was scaled by the rate of its own step before it entered the buffer. In the second the whole history is rescaled by today's rate. Under decay the first form therefore carries more of its old steps forward. A buffer moved from one convention to the other needs the factor $-\alpha$, while $\alpha$ and $\beta$ are the same numbers in both, and under a schedule the two will still not follow the same path.

---

## 4. Where the velocity lives

The optimiser of post 23 held only numbers of its own: the learning rate, the decay, and the step counter. Momentum adds state **per parameter**: one velocity array for each weight array and one for each bias array.

This series attaches that state to the layer, as `layer.weight_momentums` and `layer.bias_momentums`, each with the shape of the parameter it belongs to. The optimiser creates the two arrays the first time it meets a layer and reads and replaces them on every later call. Two things follow:

- **One optimiser object serves any number of layers** without a table of buffers indexed by layer.
- **The buffer travels with the layer.** A layer that is saved, or reused in a second run, carries its velocity along, which section 10 shows can be a trap.

Keeping the buffers inside the optimiser, keyed by parameter, is the other common design. The bookkeeping is the same either way: one velocity per parameter array.

---

## 5. The optimiser class with momentum

The class of post 23 gains one constructor argument, one attribute, and a new body for `update_params`. `pre_update_params` and `post_update_params` are unchanged.

```python
class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.momentum              = momentum
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:
            # The velocity buffers are created on the first call, full of zeros.
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)

            # New velocity = beta * old velocity - current learning rate * gradient.
            weight_updates = self.momentum * layer.weight_momentums \
                           - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums \
                         - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # No momentum: the update of post 23.
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbiases

        # The step is the velocity, added.
        layer.weights += weight_updates
        layer.biases  += bias_updates

    def post_update_params(self):
        self.iterations += 1
```

Sign by sign, the code is the formula of section 3: the old velocity times `self.momentum`, **minus** the current learning rate times the gradient, and the result **added** to the parameter.

**The buffers are created lazily.** The `hasattr` test means a layer needs to know nothing about the optimiser that will train it. On the first call the buffers are zeros, so the first step is a plain gradient step.

**The buffer always holds the latest velocity.** The line that computes `weight_updates` reads the velocity of the previous step, and the next line stores the new one under the same name. The same array object is then added to the weights, which is harmless because nothing modifies it afterwards.

**The `if self.momentum:` guard keeps the behaviour of post 23.** With the default `momentum=0.0` no buffer is created and the update is `-current_learning_rate * gradient`, as before.

**Each call handles one layer completely.** `update_params(dense1)` computes and applies both updates of `dense1` before `dense2` is touched. That is safe for the reason post 20 gave: by the time the optimiser runs, every gradient of the backward pass is already stored.

---

## 6. The training loop

The loop is the one of post 23. Only the constructor call differs: `Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)`.

```python
    for epoch in range(epochs):
        # Forward, accuracy, backward: unchanged from post 23.
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update: the three calls of post 23.
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
```

The three-call contract of post 23 absorbs the new feature without a change to the loop, which is what the contract is for. In `snippets/momentum_sgd.py` the loop sits inside a function, `train`, so that the same lines serve every run of this post; the function also records, for each step, whether it points against the step before.

---

## 7. What happens when it runs

The setup is the shared one of Part VI, printed by every script: `nnfs.init()` (float32), `spiral_data(samples=100, classes=3)`, `Layer_Dense(2, 64)`, ReLU, `Layer_Dense(64, 3)`, the combined softmax and loss, weights of scale 0.01, 10,001 full-batch epochs, and seed 0, the seed `nnfs.init()` sets. `momentum_sgd.py` runs it twice, with the optimiser of post 23 and with momentum added:

```text
decay only (post 23)
  epoch  1000  loss 1.0631  acc 0.4433
  epoch 10000  loss 0.7612  acc 0.6467

decay and momentum 0.9
  epoch  1000  loss 0.4472  acc 0.8367
  epoch 10000  loss 0.1209  acc 0.9567
```

On this seed momentum takes the final loss from 0.7612 to 0.1209 and the accuracy from 64.7 to 95.7 percent, with the same learning rate, the same decay, and the same number of epochs. The first 1,000 epochs show the flat ground of section 1. Both runs start at a loss of 1.0986, which is $\ln 3$; the run without momentum has moved to 1.0631 and the run with it to 0.4472.

The second table of the script is about the path:

```text
                          final loss  accuracy  peak loss (epoch)  loss rose  step against the last
decay only (post 23)          0.7612    0.6467      1.099 (    0)      3,973                  8,188
decay and momentum 0.9        0.1209    0.9567      4.690 ( 1765)      2,856                  1,128
```

"Step against the last" counts the steps whose dot product with the previous step, over all 387 parameters, is negative. Without momentum that is 8,188 of 10,000 pairs: for most of the run the optimiser undoes part of its previous step, which is the zig-zag of section 1 on a real network. With momentum it is 1,128.

One seed is not a result. `snippets/seed_spread.py` repeats both runs for seeds 0 to 4, set with `np.random.seed` after `nnfs.init()`, so the data and the initial weights both change:

```text
seed  optimiser      loss@1000  final loss  accuracy  peak loss  loss rose  step against the last
   0  decay only        1.0631      0.7612    0.6467      1.099      3,973                  8,188
   0  momentum 0.9      0.4472      0.1209    0.9567      4.690      2,856                  1,128
   1  decay only        1.0511      0.7335    0.6933      1.099        925                  5,715
   1  momentum 0.9      0.6630      0.5986    0.6867      1.559      4,111                    194
   2  decay only        1.0604      0.9107    0.5800      1.099          0                    542
   2  momentum 0.9      0.5771      0.5271    0.7300      2.178      4,011                    584
   3  decay only        1.0608      0.7310    0.6700      1.099      3,252                  6,884
   3  momentum 0.9      0.5403      0.3687    0.8167      1.476      3,722                    102
   4  decay only        1.0303      0.7750    0.6367      1.099      4,356                  8,942
   4  momentum 0.9      0.3697      0.0865    0.9800      3.017      3,587                    371
```

What holds in every one of the five seeds: momentum ends with the lower loss (0.0865 to 0.5986 against 0.7310 to 0.9107), and it is far ahead after 1,000 epochs (0.3697 to 0.6630 against 1.0303 to 1.0631). What does not hold in every seed is the size of the gain. The accuracy with momentum ranges from 68.7 to 98.0 percent, against 58.0 to 69.3 without, and on seed 1 it is slightly lower with momentum (68.7 against 69.3). Seed 0, the documented run, is one of the two best of the five; the mean accuracy is 83.4 percent against 64.5.

The path measures are mixed in the same honest way. Momentum has fewer steps against the last in four seeds of five; in the fifth, seed 2, the run without momentum hardly zig-zags at all (542 pairs) and still ends with the highest loss of the table. The number of epochs in which the loss rose is lower with momentum in two seeds and higher in three, so momentum does not make the loss curve smoother. And its peak loss is above the starting loss in every seed, which section 10 returns to.

---

## 8. Choosing $\beta$

On the ravine of section 1, the four coefficients after 100 steps:

```text
beta    x      |y|        loss      travel across  travel along  across steps that reverse
0.00   1.4686  2.66e-05     1.0784         18.999         8.531                         99
0.50   0.1834  4.60e-16     0.0168          4.872         9.817                         36
0.90   0.0523  5.15e-03     0.0027         18.902        17.834                         49
0.99   1.6948  6.78e-01    24.4354         96.638        69.025                         48
```

A coefficient of 0.5 damps the bounce best (the travel across falls from 18.999 to 4.872), 0.9 gives the lowest loss because it gains most along the floor, and 0.99 ends with about 23 times the loss of no momentum at all: it travels 96.638 across a ravine whose walls started 1 away.

On the spiral, `snippets/beta_sweep.py` runs 0.5 and 0.99 over the same five seeds, with the same decay; the rows for 0 and 0.9 are those of section 7.

| $\beta$ | Horizon $1/(1 - \beta)$ | Final loss, five seeds | Accuracy, five seeds | Mean accuracy | Steps against the last |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 (decay only) | 1 | 0.7310 to 0.9107 | 0.5800 to 0.6933 | 0.6453 | 542 to 8,942 |
| 0.5 | 2 | 0.3661 to 0.7216 | 0.7133 to 0.8567 | 0.8020 | 8,459 to 9,500 |
| 0.9 | 10 | 0.0865 to 0.5986 | 0.6867 to 0.9800 | 0.8340 | 102 to 1,128 |
| 0.99 | 100 | 0.5612 to 1.0237 | 0.3900 to 0.6767 | 0.5447 | 17 to 214 |

The figure below draws every run of the table, one tick per seed. Three readings follow.

![Two dot charts with one row per momentum coefficient, 0, 0.5, 0.9 and 0.99, each row a band over five seeds with one tick per seed and a dot for seed 0. Final loss: 0.7310 to 0.9107, 0.3661 to 0.7216, 0.0865 to 0.5986 and 0.5612 to 1.0237. Accuracy: 58.0 to 69.3, 71.3 to 85.7, 68.7 to 98.0 and 39.0 to 67.7 percent, with means of 64.5, 80.2, 83.4 and 54.5 percent.](diagrams/04-beta-over-seeds.svg)

*The rows for 0 and 0.9 are the runs of `seed_spread.py`, the rows for 0.5 and 0.99 those of `beta_sweep.py`; all use learning rate 1.0 with decay 0.001.*

**A coefficient of 0.5 helps in every seed.** Its loss is lower and its accuracy higher than without momentum in all five, although its steps still point against each other almost every time. Half of the previous step is not enough to stop the zig-zag, and it is enough to lengthen the net step.

**Between 0.5 and 0.9 these runs do not decide.** On seed 0 the accuracies are 78.0 and 95.7 percent, which is the comparison usually quoted. Over five seeds 0.9 has the higher accuracy and the lower loss in three and 0.5 in two, and 0.9 has both the best run and a run with a lower accuracy than every run at 0.5.

**A coefficient of 0.99 does not help.** Its accuracy is below the decay-only run in all five seeds, although its final loss is the lower of the two in three of them (seeds 0, 1 and 2). The likely reason is the size of the step: with a horizon of 100 it can grow to a hundred times a plain gradient step, and the learning rate of 1.0 was chosen for a step of one. The sweep does not test that reading, because it never lowers the learning rate.

That last point is the practical rule. The size a consistent step grows to is $\alpha / (1 - \beta)$, so the two hyperparameters are not independent: raising $\beta$ at a fixed $\alpha$ raises the step, and a larger $\beta$ usually needs a smaller $\alpha$. A coefficient of 0.9 is the customary starting value, and Adam's default first-moment rate is the same number; the table says it is a starting value and not a guarantee.

---

## 9. Make it run: five scripts

```text
python posts/24-momentum/snippets/ravine.py
python posts/24-momentum/snippets/momentum_sgd.py
python posts/24-momentum/snippets/seed_spread.py
python posts/24-momentum/snippets/beta_sweep.py
python posts/24-momentum/snippets/what_can_go_wrong.py
```

`momentum_sgd.py` holds the classes, the `train` function and the documented run, and needs NumPy and the `nnfs` package; it takes about 15 seconds. `seed_spread.py` and `beta_sweep.py` import from it and make ten full runs each, about 70 seconds apiece. `ravine.py` and `what_can_go_wrong.py` work on the two-parameter ravine in float64, are not random, and finish in about a second. The network classes are those of posts 16 and 19, unchanged.

---

## 10. What can go wrong?

**The loss shoots up before it comes down.** A velocity that has built up keeps going when the surface turns. In `seed_spread.py` the run without momentum never exceeds its starting loss of 1.0986, and the run with `momentum=0.9` peaks between 1.476 and 4.690 across the five seeds; on seed 0 the peak of 4.690 comes at epoch 1,765, long after the loss had fallen below 0.5. Every one of those runs recovered. A log that shows a sudden spike under momentum is not by itself a bug, and a spike that does not recover points to a step limit $\alpha / (1 - \beta)$ that is too large.

The other four are measured on the ravine by `snippets/what_can_go_wrong.py`, with a learning rate of 0.019 and $\beta = 0.9$.

**The buffer recreated on every step.** Zeroing the buffers in each call, instead of once, throws the history away:

```text
loss after 100 steps: correct 0.0027   recreated 1.0784   no momentum 1.0784
largest gap between the recreated path and the path without momentum: 0.0e+00
```

Nothing fails. The optimiser is exactly gradient descent with an unused argument.

**The velocity line of one convention with the update line of the other.** Writing `v = beta * v + g` and keeping `weights += v` adds the gradient: the step goes uphill, and the learning rate is gone as well.

```text
loss at the start and after 1, 2 and 3 steps: 100  510,250  5,295,235,250  54,974,113,399,900
```

**A layer that still carries an old buffer.** The `hasattr` test finds the buffer of an earlier run and keeps it. Three steps are taken, the weights are reset to the start, and a new optimiser is created:

```text
velocity left on the layer after 3 steps: (-0.5013,  1.7100)
first step of the second run: (-0.6411, -0.3610)   on a fresh layer: (-0.1900, -1.9000)
```

The first step of the second run is not a gradient step: it is 0.9 times a velocity that belongs to another run, plus the gradient step. Continuing a run needs the buffer; a fresh start needs new layers or deleted buffers.

**A coefficient of 1.**

```text
momentum 0.9: loss after 100 steps    0.0027   lowest and highest loss over steps 1,000 to 2,000: 1.55e-90 and 1.32e-44
momentum 1.0: loss after 100 steps   23.2446   lowest and highest loss over steps 1,000 to 2,000: 1.36e-01 and 1.45e+02
```

With nothing forgotten there is no friction. Between steps 1,000 and 2,000 the loss still swings between 0.136 and 145, above the 100 it started at.

---

## 11. Summary

| Concept | Takeaway |
|---|---|
| Velocity | $v \leftarrow \beta v - \alpha g$, same shape as the parameter, zero at the start |
| Update | $\theta \leftarrow \theta + v$ |
| Cancellation | opposing parts of consecutive steps cancel; agreeing parts grow towards $\alpha g/(1 - \beta)$ |
| Other convention | $u \leftarrow \beta u + g$, $\theta \leftarrow \theta - \alpha u$; equal to this one, with $v = -\alpha u$, only at a constant rate |
| Storage | `weight_momentums` and `bias_momentums` on each layer |
| Spiral, five seeds | final loss 0.7310 to 0.9107 without, 0.0865 to 0.5986 with `momentum=0.9`; accuracy 58.0 to 69.3 against 68.7 to 98.0 percent |
| Choice of $\beta$ | 0.5 and 0.9 both end with a lower loss than none in all five seeds; 0.99 at the same learning rate ends with a lower accuracy than none in all five |

---

## Common pitfalls

1. **Recreating the buffer every step.** The velocity is the history; the `hasattr` guard exists so that it is zeroed once.
2. **Mixing the conventions.** `v = beta * v + g` belongs with `weights -= learning_rate * v`. Paired with `weights += v` it climbs.
3. **Raising $\beta$ without lowering $\alpha$.** The step can grow to $\alpha/(1 - \beta)$: ten times at 0.9, a hundred times at 0.99.
4. **Forgetting that decay still applies.** The velocity line reads `current_learning_rate`, the decayed rate, and `pre_update_params` must still be called first.
5. **Reusing layers across runs.** The buffer stays on the layer and the next run starts with a velocity that is not its own.
6. **Reading Adam's $\beta_1$ as this $\beta$.** Adam multiplies the gradient by $(1 - \beta_1)$ before adding it; classical momentum does not (post 27).

---

## Further reading

- Polyak, B. T., *"Some Methods of Speeding up the Convergence of Iteration Methods"* (USSR Computational Mathematics and Mathematical Physics, 1964). The heavy-ball method.
- Rumelhart, D., Hinton, G., and Williams, R., *"Learning Representations by Back-Propagating Errors"* (Nature, 1986).
- Sutskever, I., Martens, J., Dahl, G., and Hinton, G., *"On the Importance of Initialization and Momentum in Deep Learning"* (ICML, 2013).
- Qian, N., *"On the Momentum Term in Gradient Descent Learning Algorithms"* (Neural Networks, 1999).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 8 (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 10 (2020).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 25 - AdaGrad](../25-adagrad/index.md):** one learning rate per parameter, scaled by the history of its squared gradients.
- **[Post 27 - Adam](../27-adam-optimiser/index.md):** the velocity of this post, as a moving average, joined to per-parameter scaling.
