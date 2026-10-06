# 30 - L1 and L2 regularisation

> **TL;DR.** A penalty on weight size is one more term in the loss, $\lambda \sum |w|$ for L1 or $\lambda \sum w^2$ for L2, and one more term in each weight gradient, $\lambda \, \text{sign}(w)$ or $2 \lambda w$. `Layer_Dense` gains four arguments and a few lines in `backward`, `Loss` gains one method, and the training loop gains one sum. On the spiral network of this post, L2 at $\lambda = 5 \times 10^{-4}$ raised test accuracy on each of five seeds, from 73.0 to 82.3 percent without a penalty to 81.0 to 86.7 percent with it, and narrowed the gap between training and test accuracy on each. L1 at the same strength did the same and narrowed the gap by less; no weight in any run ended at exactly zero, and among neurons that still fire only L1 left weights within 0.001 of zero.
>
> **Prerequisites:** [Post 16](../16-coding-backpropagation/index.md), [Post 28](../28-generalization-and-testing/index.md).
> **Safe to skip?** Skip it if the reader can already write both penalties and their gradients from memory, add them to a dense layer's backward pass, and say why a plain gradient step on an L1 penalty does not leave weights at exactly zero.
>
> **After reading, you will be able to:**
>
> - Write the L1 and L2 penalties and derive the gradient of each.
> - Explain why L1 induces sparsity and L2 shrinkage, from the size of each penalty's gradient.
> - Extend Layer_Dense with weight_regularizer_l1 and weight_regularizer_l2 that hook into the existing forward and backward passes.
> - Choose a sensible regularisation strength from a sweep read on held-out data over several seeds.

![Two charts over one weight w from −1.5 to 1.5 at a strength of 0.01. Left, the penalty: L1, lambda times the absolute value of w, is a V with its corner at 0 and reaches 0.015 at the ends; L2, lambda times w squared, is a parabola and reaches 0.0225. Right, the gradient of each penalty as the dense layer computes it: L1 is −0.01 for every negative weight and +0.01 for every positive one and for w = 0; L2 is the line 2 lambda w from −0.03 to 0.03. The two gradients are equal at w = 0.5 and w = −0.5.](diagrams/01-penalty-and-gradient.svg)

*One weight under the two penalties at $\lambda = 0.01$. The L1 gradient has the same size at every weight; the L2 gradient grows with the weight and passes it at $|w| = 0.5$.*

---

## 1. The question: what does a penalty on weight size change?

[Post 28](../28-generalization-and-testing/index.md) measured the gap between training accuracy and test accuracy and named the levers that narrow it. One of them is to constrain the weights. The question of this post is: **how does a penalty on weight magnitude change the loss, the gradient, and the weights the optimiser prefers?**

The reason to look at weight size is geometric. The output of a dense, ReLU, dense network changes with its input no faster than the weights allow: the steepness of the logits is bounded by the product of the sizes of the two weight matrices. A decision boundary that bends sharply around a handful of training points therefore needs large weights, and a network that fits noise tends to have them. On the runs of section 7 the 128 weights of the first layer, which start at a scale of 0.01, end with a sum of squares between 1,936 and 4,184 when nothing holds them back.

There are two ways to act on that. More data makes memorising harder than learning the pattern, and is often not available. A constraint on the weights is available in every training run. It takes the form of a second term in the loss:

$$L_{\text{total}} = L_{\text{data}} + L_{\text{reg}}$$

$L_{\text{data}}$ is the loss of the earlier posts, here the mean cross-entropy. $L_{\text{reg}}$ depends on the weights alone, not on the data, and grows with their size. The optimiser minimises the sum, so a weight may grow only if that lowers the data loss by more than it raises the penalty. This is **regularisation**, and the two standard penalties are named after the vector norms they use.

---

## 2. The two penalties and their gradients

Both penalties are sums over the weights of one layer, written $w_m$ with $m$ running over all entries of the weight array. The **regularisation strength** $\lambda \ge 0$ is a hyperparameter.

**L1** is the sum of the absolute values:

$$L_{\text{reg}}^{\text{L1}} = \lambda \sum_m |w_m|$$

Only one term of the sum contains $w_m$. For $w_m > 0$ that term is $\lambda w_m$ with slope $\lambda$, and for $w_m < 0$ it is $-\lambda w_m$ with slope $-\lambda$:

$$\frac{\partial L_{\text{reg}}^{\text{L1}}}{\partial w_m} = \lambda \, \text{sign}(w_m) \qquad (w_m \ne 0)$$

At $w_m = 0$ the absolute value has a corner and no derivative: the slope is $-\lambda$ on one side and $+\lambda$ on the other. Code has to pick a value there, and section 5 shows which one this series' class picks.

**L2** is the sum of the squares:

$$L_{\text{reg}}^{\text{L2}} = \lambda \sum_m w_m^2 \qquad \frac{\partial L_{\text{reg}}^{\text{L2}}}{\partial w_m} = 2 \lambda \, w_m$$

The factor 2 comes from $\frac{d}{dw} w^2 = 2w$. Some texts write the penalty as $\frac{\lambda}{2} \sum w^2$, whose gradient is $\lambda w$; the two conventions differ only in what $\lambda$ means, by a factor of 2. This series uses $\lambda \sum w^2$ and $2 \lambda w$.

The two gradients differ in one respect. The L1 gradient has the same size, $\lambda$, at every non-zero weight. The L2 gradient is proportional to the weight. `snippets/network.py` prints both at $\lambda = 0.01$:

| Weight $w$ | L1 gradient $\lambda \, \text{sign}(w)$ | L2 gradient $2 \lambda w$ |
|:---:|:---:|:---:|
| 0.01 | 0.01 | 0.0002 |
| 0.1 | 0.01 | 0.002 |
| 0.5 | 0.01 | 0.01 |
| 1 | 0.01 | 0.02 |
| 10 | 0.01 | 0.2 |
| 100 | 0.01 | 2 |

The figure at the top of the post draws both gradients at this strength. With the same $\lambda$ on both, they cross where $\lambda = 2 \lambda |w|$, at $|w| = 0.5$ whatever $\lambda$ is. Below $|w| = 0.5$ the L1 gradient is the larger one, and above it the L2 gradient is.

---

## 3. What each gradient does to a weight

The difference shows most clearly on a weight that the data loss does not care about, so that the penalty is the only gradient. `snippets/single_weight.py` takes one weight of 0.0105 and applies 1,000 plain gradient steps, `weights -= learning_rate * dweights`, with a learning rate $\alpha = 0.1$ and $\lambda = 0.01$:

```text
step      L1 weight    L2 weight
   1       0.009500     0.010479
  10       0.000500     0.010292
  11      -0.000500     0.010271
  12       0.000500     0.010251
 100       0.000500     0.008595
1000       0.000500     0.001418
L1 after step 10: |w| stays at 0.000500 to 0.000500; steps with w == 0: 0 of 1000
L2 factor per step: 1 - 2 * learning rate * lambda = 0.998; start * factor^1000 = 0.001418
L1 from a weight of exactly 0: -0.001  0  -0.001  0
```

The figure below draws both paths from the same steps, the first 20 on the left and all 1,000 on the right.

![Two charts of the weight against the step, from 0.0105, under each penalty alone with plain steps at a learning rate of 0.1 and lambda 0.01. Left, steps 0 to 20: the L1 weight falls by 0.001 per step to 0.0005 at step 10 and then jumps between 0.0005 and −0.0005, never 0, while the L2 weight barely moves. Right, steps 0 to 1,000: the L2 weight follows 0.0105 times 0.998 to the power t, 0.008595 at step 100 and 0.001418 at step 1,000; the L1 weight stays in a band from −0.0005 to 0.0005 from step 10 on.](diagrams/02-one-weight-plain-steps.svg)

*The same 1,000 steps on two time scales. L1 covers the distance to zero at a constant speed and then oversteps it every step; L2 loses 0.2 percent of what is left each step.*

**L2 shrinks.** Each step multiplies the weight by $1 - 2 \alpha \lambda = 0.998$, so the weight decays geometrically: after 1,000 steps it is $0.0105 \cdot 0.998^{1000} = 0.001418$. The step is proportional to what is left, so a large weight loses a lot, a small one loses little, and the weight never arrives at zero.

**L1 subtracts.** Each step moves the weight by the same amount, $\alpha \lambda = 0.001$, towards zero. The weight reaches the neighbourhood of zero in a finite number of steps, ten here, however small it has become. This constant pull is why L1 is called sparsity-inducing: among weights the data loss barely needs, L1 removes all of them at the same rate, where L2 spares the small ones.

**A plain gradient step does not stop at zero.** From step 10 on, the L1 weight jumps between $0.0005$ and $-0.0005$: a step of fixed size $\alpha \lambda$ oversteps zero unless the weight is an exact multiple of it. Even a weight that is exactly zero does not stay, because the class of section 5 uses a sign of $+1$ there and the next step moves the weight to $-0.001$. So the result of L1 with plain gradient steps is weights within $\alpha \lambda$ of zero, not weights equal to zero. Exact zeros belong to the minimiser of an L1-penalised loss. In one dimension, $\frac{1}{2}(w - a)^2 + \lambda |w|$ is smallest at $w = 0$ whenever $|a| \le \lambda$. Methods that are built to return that minimiser handle the corner explicitly; the gradient step of this series does not. This is the setting of the lasso of Tibshirani (1996), which put the L1 penalty on linear regression to select variables.

| Property | L1 | L2 |
|---|---|---|
| Penalty | $\lambda \sum \lvert w \rvert$, a V | $\lambda \sum w^2$, a parabola |
| Gradient | $\lambda \, \text{sign}(w)$, constant size | $2 \lambda w$, proportional to $w$ |
| One plain step, penalty alone | subtracts $\alpha \lambda$ | multiplies by $1 - 2 \alpha \lambda$ |
| Small weights | pulled as hard as large ones | barely touched |
| Large weights | pulled no harder than small ones | pulled hardest |
| At the minimiser | some weights can be exactly zero | weights are small, rarely zero |

---

## 4. The forward pass: the penalty joins the reported loss

The penalty is computed per layer. It is a new method of the `Loss` base class of [post 19](../19-softmax-derivatives-and-the-combined-backward-pass/index.md), which takes a layer and reads the strengths stored on it. `calculate` is unchanged, and so is every other class of posts 16 and 19 except `Layer_Dense`. These are the added lines:

```python
    # Added in post 30: the penalty of one layer, zero when no strength is set.
    def regularization_loss(self, layer):
        regularization_loss = 0.0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights ** 2)
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases ** 2)

        return regularization_loss
```

The combined class of post 19 already holds a `Loss_CategoricalCrossentropy` in its attribute `loss`, so the training loop reaches the method through `loss_activation.loss`. One call per dense layer, added up:

```python
data_loss = loss_activation.forward(dense2.output, y)

regularization_loss = (loss_activation.loss.regularization_loss(dense1) +
                       loss_activation.loss.regularization_loss(dense2))
loss = data_loss + regularization_loss
```

A layer with no strength set contributes exactly 0.0, so the sum may run over every dense layer. For a layer with weights $0.5, -2, 0$ and $1, 0.1, -0.2$ and both weight strengths at 0.01, `snippets/network.py` prints $\sum |w| = 3.8$, $\sum w^2 = 5.3$ and a penalty of $0.01 \cdot 3.8 + 0.01 \cdot 5.3 = 0.091$.

Three points about the term.

**The penalty is a training quantity.** `loss` is what the optimiser minimises. A test or validation pass scores predictions, so it reports `data_loss` alone, and a comparison of training loss with test loss uses the data loss on both sides. Every loss in the tables of this post is a data loss, and the penalty is given beside it.

**The penalty does not depend on the batch.** The data loss is a mean over $N$ samples; the penalty is not divided by anything. The same $\lambda$ therefore weighs more against a data loss that is itself small.

**Biases have their own two strengths.** A bias shifts a neuron's threshold and does not steepen the output, so it is often left unpenalised. The method supports both, and the runs of section 7 penalise the biases of the first layer as well as its weights.

---

## 5. The backward pass: extra terms in `Layer_Dense`

The total loss is a sum, so its gradient is the sum of the two gradients (post 10). The data gradient of a weight array is what `backward` of [post 16](../16-coding-backpropagation/index.md) already stores in `dweights`. The penalty gradient of section 2 is added to it, in the layer that owns the weights. The constructor gains four arguments, all 0.0 by default, and stores them:

```python
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0.0, weight_regularizer_l2=0.0,
                 bias_regularizer_l1=0.0, bias_regularizer_l2=0.0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Added in post 30: one strength per penalty and per parameter array.
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
```

`forward` is unchanged. `backward` keeps its three lines of post 16 and gains the block between the second and the third:

```python
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)          # shape of weights
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)   # shape of biases

        # Added in post 30: the gradient of each penalty, added to the data gradient.
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)          # shape of inputs
```

**The data gradient is assigned and the penalty is added to it.** `dweights` is still written afresh on every call (post 16), and the `+=` lines add to that fresh array within the same call. Nothing accumulates across calls. `dinputs` gets no extra term, because the penalty does not depend on the layer's inputs.

**The sign at zero is $+1$.** `dL1` starts as an array of ones and only the entries of negative weights are flipped, so a weight of exactly zero receives $+\lambda$. Any value between $-\lambda$ and $+\lambda$ is defensible at the corner; $+1$ is the choice of this class. For the layer of section 4 the script prints the penalty terms on their own, and the zero weight in the top right gets 0.01:

```text
dweights, penalty terms only:
[[ 0.02  -0.05   0.01 ]
 [ 0.03   0.012 -0.014]]
```

The entry for the weight $-2$ is $-0.01 + 2 \cdot 0.01 \cdot (-2) = -0.05$: both penalties were on, and their gradients add.

**The strengths live on the layer.** With all four at their default the class behaves exactly as the class of post 16, and the optimisers of Part VI need no change: they read `dweights` and `dbiases`, which now hold the gradient of the total loss. Each layer can have its own strengths:

```python
dense1 = Layer_Dense(2, 64,
                     weight_regularizer_l1=l1, weight_regularizer_l2=l2,
                     bias_regularizer_l1=l1, bias_regularizer_l2=l2)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
```

---

## 6. Checking the new terms

A wrong penalty gradient raises no error, so it is checked the way post 21 checked the whole network: a central difference with $h = 10^{-5}$ in float64 on every parameter, with the penalty included in the loss that is differenced. `snippets/gradient_check.py` does this on the small network of post 21 (two inputs, three ReLU neurons, three classes) with penalties on all four parameter arrays. It does not call `nnfs.init()`, since post 21 measured that the `np.dot` installed by that call returns float32 even for float64 inputs.

```text
h = 1e-05, pass mark 1e-07; X float64, weights float64, np.dot returns float64

== Seed 0, both penalties at 0.01 on all four arrays
gradient         largest relative error   largest absolute gap
dense2.dweights  3.9e-10                  1.9e-11   pass
dense2.dbiases   8.0e-11                  1.8e-11   pass
dense1.dweights  4.4e-10                  1.6e-11   pass
dense1.dbiases   1.7e-10                  4.9e-12   pass

== Seeds 0 to 9: the largest relative error over the four arrays
penalties             largest relative error   largest absolute gap   seeds above the pass mark
L2 = 0.01             3.2e-07                  6.9e-11                [6]
L1 = 0.01             1.4e-08                  6.4e-11                []
L1 = L2 = 0.01        1.8e-08                  8.6e-11                []
```

Twenty-nine of the thirty checks are below the pass mark of $10^{-7}$. The one above it, seed 6 under L2, has an absolute gap of at most $6.9 \times 10^{-11}$: it is the tiny-gradient case of post 21, where rounding in the difference is divided by a very small gradient, and not a wrong formula.

The corner of L1 is the one place where the check cannot pass:

```text
== A weight of exactly zero under L1 = 0.01 (seed 0, dense2.weights[0, 0] set to 0)
dense2.dweights against the loss with the penalty: largest absolute gap 0.010000
the other three arrays: largest relative error 5.8e-10
the code's sign at w = 0: +1, so it adds 0.01; central difference of 0.01 * |w| at 0: 0
```

The central difference straddles the corner, sees $|h| = |-h|$, and measures a slope of 0 for the penalty. The class adds $\lambda = 0.01$. The gap is exactly $\lambda$, on that one weight, and it reports a function without a derivative, not a mistake in the code. A weight drawn by `np.random.randn` is not exactly zero, so the case arises only when weights are set by hand.

---

## 7. Results on the spiral

The setup, printed by every training script of this post: `nnfs.init()` once (seed 0, float32, and its own `np.dot`); then for each seed $s$, `np.random.seed(s)`, the training set `spiral_data(samples=100, classes=3)`, the test set by a second identical call, `Layer_Dense(2, 64)` with the penalties, ReLU, `Layer_Dense(64, 3)` without penalties, the combined softmax and loss, `Optimizer_Adam(learning_rate=0.05, decay=1e-5)`, and 10,001 full-batch epochs. `spiral_data` draws from NumPy's global stream, so the test set is the 300 points that follow the training set in the stream, and the weights are drawn after both. The penalties sit on the weights and the biases of `dense1` only. Two things differ from the run of posts 27 and 28: the learning rate, 0.05 here against 0.02 there, and the test set, drawn here before the weights and there after them, so a seed gives other initial weights. The unpenalised figures below are therefore not post 28's (96.33 and 82.33 percent on seed 0), and every comparison in this post is with its own unpenalised run. Seed 0 is the seed `nnfs.init()` sets.

Each row is a range over seeds 0 to 4, measured forward-only after the last update. Accuracies are in percent, the gap is training minus test accuracy in points, and the test loss is the data loss.

| Penalty on `dense1` | Script | Training accuracy | Test accuracy | Gap | Test data loss |
|---|---|:---:|:---:|:---:|:---:|
| none | `seeds_none.py` | 84.67 to 97.67 | 73.00 to 82.33 | 11.67 to 16.67 | 0.758 to 1.073 |
| L2, $5 \times 10^{-4}$ | `seeds_l2.py` | 92.67 to 96.33 | 81.00 to 86.67 | 7.00 to 12.33 | 0.455 to 0.636 |
| L1, $5 \times 10^{-4}$ | `seeds_l1.py` | 95.00 to 97.33 | 80.33 to 86.67 | 9.33 to 16.00 | 0.485 to 0.775 |
| L1 and L2, both $5 \times 10^{-4}$ | `seeds_l1_l2.py` | 82.00 to 96.33 | 77.33 to 88.00 | 1.67 to 11.33 | 0.440 to 0.682 |

The figure below draws every run of the first three rows seed by seed, each as a line from its test accuracy to its training accuracy, so that the length of the line is the gap.

![One row per run for seeds 0 to 4, three runs per seed: no penalty, L2 and L1, both at 5 times 10 to the minus 4, on an accuracy axis from 70 to 100 percent. Each run is a line from its test accuracy, a solid mark, to its training accuracy, a hollow mark. On every seed the L2 and L1 lines start further right than the line without a penalty and are shorter: test accuracy 78.67, 84.00 and 80.33 percent on seed 0, 73.00, 85.67 and 86.67 on seed 1, 77.33, 81.00 and 83.00 on seed 2, 81.33, 84.67 and 83.33 on seed 3, 82.33, 86.67 and 83.33 on seed 4.](diagrams/03-train-and-test-per-seed.svg)

*The ranges of the table overlap; seed by seed, each penalty has the higher test accuracy and the shorter line.*

**L2 helped on each of the five seeds.** Seed by seed, test accuracy rose by 3.33 to 12.67 points, the gap narrowed by 3.00 to 5.67 points, and the test data loss fell, in all five runs. On seed 0 the figures are 95.33 percent in training with and without the penalty, and 78.67 against 84.00 percent on the test set. The ranges of the two rows overlap, so a single run of each on different seeds could show the opposite; the claim rests on the paired comparison.

**L1 raised test accuracy on each of the five seeds too, and narrowed the gap by less.** Test accuracy rose by 1.00 to 13.67 points and the gap narrowed by 0.67 to 3.33 points, against 3.00 to 5.67 under L2, because L1 kept the higher training accuracy on all five seeds. A rise of 1.00 point is 3 test points of 300, where one test accuracy scatters by about 2.2 points (post 28, section 4): on such a seed the run shows that L1 did no harm, and no more. Between L1 and L2 at this strength the runs give no order in test accuracy: L2 is higher on three seeds and L1 on two.

**Five further seeds keep the direction and not the "every".** Given the arguments `5 6 7 8 9`, the scripts run a second set of seeds. On it L2 raised test accuracy on three seeds, by 2.67 to 7.00 points, left it unchanged on one and lowered it by 1.00 point on one; L1 raised it on all five, by 0.33 to 7.33 points. Each narrowed the gap on four seeds, widened it on seed 5, and lowered the test data loss on all five. Over the ten seeds L2 has the higher test accuracy on four and L1 on six. With both penalties, seed 7 ends at 56.67 percent on the test set, 24.00 points below its unpenalised run.

**Both together is not reliably better.** The gap of 1.67 points on seed 0 is the smallest in the table, and it comes from training accuracy falling to 82.00 percent, not from test accuracy rising. On the other four seeds the gap is 8.00 to 11.33 points, and on seed 2 the test accuracy equals that of the run without a penalty.

What the penalties did to the 128 weights of `dense1`, over the same five seeds:

| Penalty on `dense1` | $\sum \lvert w \rvert$ | $\sum w^2$ | Dead neurons of 64 | Weights with $\lvert w \rvert < 0.001$ | Of these, in live neurons | Weights equal to 0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| none | 321.91 to 504.13 | 1,935.66 to 4,183.64 | 12 to 23 | 0 | 0 | 0 |
| L2, $5 \times 10^{-4}$ | 86.79 to 109.45 | 99.64 to 141.04 | 0 to 11 | 0 to 18 | 0 | 0 |
| L1, $5 \times 10^{-4}$ | 124.11 to 144.36 | 345.70 to 435.02 | 4 to 7 | 6 to 22 | 4 to 16 | 0 |
| L1 and L2, both $5 \times 10^{-4}$ | 64.21 to 81.17 | 74.72 to 95.38 | 11 to 18 | 4 to 22 | 0 to 8 | 0 |

**Both penalties shrink the weights a great deal.** The sum of squares falls by a factor of roughly 14 to 39 under L2 and 5 to 11 under L1.

**No weight is exactly zero in any of the twenty runs,** as section 3 predicts for gradient steps. With a threshold of 0.001 for "near zero", L1 leaves 6 to 22 of the 128 weights there and L2 leaves 0 to 18, which reads as no contrast. The count mixes two kinds of weight. A dead neuron (post 06), with zero output on every training point, passes no data gradient to its two incoming weights, so they move under the penalty alone, towards zero under either one. Without a penalty 12 to 23 of the 64 neurons end dead and their weights stay where they were. Under L2 every near-zero weight belongs to a dead neuron, on all five seeds; among live neurons L2 leaves no weight below the threshold and L1 leaves 4 to 16. The textbook contrast, sparse under L1 and dense under L2, therefore holds on these runs among the weights the network uses, and the raw count hides it. The figure below splits the near-zero weights of each L2 and L1 run by the neuron they belong to.

![Two stacked bar charts of the weights of the first layer below 0.001 after training, one bar per seed, 0 to 4. Left, L2 at 5 times 10 to the minus 4: 18, 0, 4, 12 and 6 weights, all of them in dead neurons. Right, L1 at the same strength: 7 and 9, 2 and 4, 3 and 11, 6 and 16, 3 and 8 weights in dead and in live neurons, totals 16, 6, 14, 22 and 11.](diagrams/04-near-zero-weights.svg)

*Grey bars are weights that only the penalty moves; the coloured part, under L1 alone, is near-zero weights of neurons that still fire.*

`snippets/single_weight.py` follows a weight that only the penalty moves. Under `Optimizer_Adam(learning_rate=0.05)` and the L1 penalty alone, a weight starting at 1 crosses zero at step 21 and then keeps moving within 0.0002 to 0.0049 of it, below the threshold on 213 of the last 1,000 steps and never at zero. Over 10,001 steps with the decay of the runs it is below the threshold on 214 of the last 1,000 steps under L1 and on 940 under L2, and in the trained networks 21 of the 52 weights of dead neurons are near zero under L1 against 40 of 44 under L2. For a weight the data has let go, Adam brings L2 closer to zero than L1, the reverse of the plain steps of section 3.

**With ten times the data the gap was already small.** `snippets/jobs/more_data.py` repeats the first two rows with 1,000 samples per class in both sets. Without a penalty the gap is 0.90 to 3.00 points over the five seeds, against 11.67 to 16.67 points with 100 samples per class: the data closed most of it. With L2 the gap is 1.13 to 1.77 points, narrower than without on two seeds and wider on three. Test accuracy is 87.50 to 89.23 percent with L2 and 86.50 to 87.93 percent without on four seeds, a difference of $-0.43$ to $+1.30$ points seed by seed. On the fifth, seed 2, the unpenalised run stalled at 64.43 percent training accuracy and 63.43 percent on the test set, and the run with L2 did not. On these runs, then, L2 adds little to a gap that more data has already narrowed.

---

## 8. Choosing $\lambda$

$\lambda$ is a hyperparameter, and [post 29](../29-validation-and-hyperparameter-tuning/index.md) gives the procedure: candidates are compared on validation data and the test set is read once, for the winner. The table below reads the test set for every candidate, which is legitimate only because nothing is chosen from it here; it shows the shape of the dependence. The strength $5 \times 10^{-4}$ of section 7 was fixed before this sweep was run and is not a pick from it. The rows for 0 and $5 \times 10^{-4}$ are those of section 7.

| L2 strength | Script | Training accuracy | Test accuracy (mean) | Gap | $\sum w^2$ in `dense1` |
|---|---|:---:|:---:|:---:|:---:|
| 0 | `seeds_none.py` | 84.67 to 97.67 | 73.00 to 82.33 (78.53) | 11.67 to 16.67 | 1,935.66 to 4,183.64 |
| $10^{-4}$ | `l2_weak.py` | 95.00 to 98.67 | 78.33 to 88.33 (84.67) | 6.67 to 16.67 | 347.29 to 463.01 |
| $5 \times 10^{-4}$ | `seeds_l2.py` | 92.67 to 96.33 | 81.00 to 86.67 (84.40) | 7.00 to 12.33 | 99.64 to 141.04 |
| $10^{-3}$ | `l2_strong.py` | 93.33 to 95.67 | 81.00 to 86.67 (83.53) | 7.00 to 13.67 | 61.57 to 72.44 |
| $10^{-2}$ | `l2_too_strong.py` | 81.00 to 89.00 | 76.00 to 79.67 (77.33) | 3.33 to 13.00 | 7.98 to 11.52 |

The figure below draws the two accuracy columns seed by seed, with a tick for every run.

![Two dot charts, one row per L2 strength, 0, 10 to the minus 4, 5 times 10 to the minus 4, 10 to the minus 3 and 10 to the minus 2, each row a band over seeds 0 to 4 with a tick per seed and a dot at the mean. Left, training accuracy from 80 to 100 percent: the three middle strengths sit between 92.67 and 98.67, and 10 to the minus 2 falls to 81.00 to 89.00. Right, test accuracy from 70 to 95 percent, with mean test accuracies of 78.53, 84.67, 84.40, 83.53 and 77.33: the bands of the three middle strengths overlap almost entirely, and their means lie 5.0 to 6.1 points above that of no penalty.](diagrams/05-strength-sweep.svg)

*No penalty in grey. Between $10^{-4}$ and $10^{-3}$ the spread over seeds is wider than the differences between strengths; at $10^{-2}$ both accuracies fall.*

**The weights shrink steadily with $\lambda$.** The sum of squares falls at every step of the sweep, with no overlap between neighbouring rows.

**Between $10^{-4}$ and $10^{-3}$ the test accuracy does not separate.** The three means lie within 1.2 points of each other and the ranges overlap almost completely; the spread between seeds is several times the difference between strengths. Five seeds do not justify calling $5 \times 10^{-4}$ better than its neighbours. They do put the whole decade ahead of no penalty, by 5.0 to 6.1 points in the mean.

**At $10^{-2}$ the network underfits.** Training accuracy falls to 81.00 to 89.00 percent, and test accuracy is below that of the unpenalised run on four of the five seeds. The gap is the smallest of the sweep on average, which is the reason a small gap alone is not the goal: a model that fits nothing has no gap either.

| Reading on held-out data | Likely state of $\lambda$ |
|---|---|
| Training and validation accuracy both low | too high: the penalty outweighs the data loss |
| Training accuracy high, validation accuracy well below it | too low, or the penalty is not the lever that is missing |
| Validation accuracy at its best, training accuracy somewhat above it | in the useful range |

A sweep therefore goes by powers of ten first, such as $10^{-5}$ to $10^{-2}$, reads validation results over several seeds, and refines only where the difference between candidates exceeds the spread between seeds. L1 and L2 are tuned separately, because the same $\lambda$ multiplies different sums and so does not mean the same strength: at $5 \times 10^{-4}$ on seed 0 the L1 penalty ends at 0.0890 and the L2 penalty at 0.0637.

---

## 9. Weight decay and priors

**L2 with plain gradient descent is weight decay.** Substituting the gradient of the total loss into the update of post 22 gives

$$w \leftarrow w - \alpha \left( \frac{\partial L_{\text{data}}}{\partial w} + 2 \lambda w \right) = (1 - 2 \alpha \lambda) \, w - \alpha \frac{\partial L_{\text{data}}}{\partial w}$$

Every step first multiplies each weight by a factor slightly below 1 and then takes the ordinary data step. That multiplication is weight decay, and it is the factor 0.998 of section 3. Krogh and Hertz (1992) analysed why this decay can improve the generalisation of a neural network.

**With Adam the two are not the same thing.** Adam divides each gradient by a running size of that gradient (post 27), and the L2 term is inside the gradient it divides. `snippets/single_weight.py` hands one weight of 1 with the L2 penalty alone to `Optimizer_Adam(learning_rate=0.001)` for 100 steps:

```text
lambda    Adam, L2 in the gradient    w * (1 - 2 * 0.001 * lambda)^100
0.0001    0.901793                    0.999980
0.01      0.901744                    0.998002
1         0.901744                    0.818567
```

The right column is weight decay by the factor above. The left column barely moves while $\lambda$ changes by four orders of magnitude: Adam's normalisation turns a gradient of any constant size into a step of about the learning rate, so the weight loses close to $100 \cdot 0.001$ whatever $\lambda$ is. In a real run the data gradient is in the same average and the effect is less extreme, but the shrinkage of a weight is no longer a fixed factor per step. Loshchilov and Hutter (2019) made this point and proposed applying the decay to the weights directly, outside the adaptive step. The runs of sections 7 and 8 use the L2 term in the gradient, as the class of section 5 computes it, and their results are results for that combination.

**The penalties are priors.** If the data loss is a negative log-likelihood, minimising the total loss is maximum a posteriori estimation under a prior on the weights. The negative log of a zero-mean Gaussian density with variance $\sigma^2$ is $w^2 / (2 \sigma^2)$ plus a constant, so the L2 penalty $\lambda w^2$ is a Gaussian prior with $\sigma^2 = 1 / (2 \lambda)$ when the data term is the summed loss, and with $\sigma^2 = 1 / (2 N \lambda)$ when it is the mean over $N$ samples, as in this series. The L1 penalty corresponds in the same way to a Laplace prior, whose density has a peak at zero. Neither view changes the code.

---

## 10. Make it run: the scripts

Every code block and every number of this post comes from a script in `snippets/`, run from the series root, for example:

```text
python posts/30-l1-and-l2-regularisation/snippets/seeds_l2.py
```

| Script | Contents | Time |
|---|---|---|
| `network.py` | the classes, the loop, the setup; prints the table of section 2 and the layer of sections 4 and 5 | 1 s |
| `single_weight.py` | one weight under each penalty, plain steps and Adam (sections 3, 7 and 9) | 1 s |
| `gradient_check.py` | the central-difference check (section 6) | 1 s |
| `what_can_go_wrong.py` | the mistakes of section 11 | 1 s |
| `seeds_none.py`, `seeds_l2.py`, `seeds_l1.py`, `seeds_l1_l2.py` | five seeds each (section 7) | 30 to 50 s each |
| `l2_weak.py`, `l2_strong.py`, `l2_too_strong.py` | five seeds each (section 8) | 30 to 50 s each |
| `jobs/more_data.py` | ten runs on 3,000 points (section 7) | 10 to 20 min |

`jobs/more_data.py` is too slow for the two-minute limit of the other scripts and is run on its own. The training scripts take other seeds as arguments, as in `seeds_l2.py 5 6 7 8 9`. All need NumPy and the `nnfs` package, which supplies `spiral_data` and `nnfs.init()`. The first four run in float64 and do not call `nnfs.init()`; the training scripts call it once and reseed with `np.random.seed` for each run.

---

## 11. What can go wrong?

`snippets/what_can_go_wrong.py` runs the gradient check of section 6 with L2 at 0.01 on every array and one thing wrong in `backward`:

```text
gradient         correct              no term in backward   term without the 2
dense2.dweights  3.9e-09 (1.8e-11)    1.1e+00 (4.5e-02)     1.3e+00 (2.3e-02)
dense2.dbiases   1.2e-10 (2.0e-11)    1.4e-01 (5.3e-02)     7.2e-02 (2.7e-02)
dense1.dweights  5.9e-09 (1.8e-11)    1.0e+00 (1.4e-02)     5.0e-01 (7.2e-03)
dense1.dbiases   2.0e-10 (1.1e-11)    1.0e+00 (1.8e-02)     5.0e-01 (8.9e-03)
largest |w| in dense2: 2.2567; 2 * 0.01 * that: 0.0451; 0.01 * that: 0.0226
```

Each cell is the largest relative error with the largest absolute gap in brackets.

**The penalty in the loss, and no term in `backward`.** The reported loss rises by the penalty, the gradients are those of the data loss, and the weights never feel the penalty. Nothing is raised. The check shows relative errors between 0.14 and 1.1, and the absolute gap on `dense2.dweights`, 0.045, is $2 \lambda$ times the largest weight: the missing term itself.

**The L2 term without its 2.** The gap is half as large, 0.023, and the relative error is still between 0.07 and 1.3. Training would proceed with half the intended strength and nothing would show it.

**A strength with the wrong sign.** The tests are `> 0`, so a negative strength is skipped without a word:

```text
weight_regularizer_l2 = -0.01: regularization_loss 0.0  largest |dweights| 0.0
```

Three of the four released projects of this series (`nn-p01`, `nn-p03`, `nn-p04`) add a check to the constructor that rejects negative strengths; the class of this post does not have it.

**The penalty counted on one side of a comparison.** On seed 0 with L2, `seeds_l2.py` prints a training data loss of 0.1396, a penalty of 0.0637 and a test data loss of 0.4823. The loop's `loss` is $0.1396 + 0.0637 = 0.2033$. Set against the test data loss it understates the gap in loss by the penalty; adding the penalty to the test loss instead inflates a number that is supposed to score predictions.

**Judging $\lambda$ by the training loss.** Averaged over the five seeds, the loop's loss, data loss plus penalty, rises with every step of the sweep of section 8, from 0.1603 without a penalty to 0.5161 at $10^{-2}$, while test accuracy first rises and then falls. The training loss does not say which strength generalises. The reading that counts is on held-out data.

---

## 12. Summary

| Concept | Takeaway |
|---|---|
| Regularisation | $L_{\text{total}} = L_{\text{data}} + L_{\text{reg}}$; the optimiser minimises the sum |
| L1 | $\lambda \sum \lvert w \rvert$; gradient $\lambda \, \text{sign}(w)$; a plain step subtracts $\alpha \lambda$ |
| L2 | $\lambda \sum w^2$; gradient $2 \lambda w$; a plain step multiplies by $1 - 2 \alpha \lambda$ |
| In the code | four strengths on `Layer_Dense`, extra terms in its `backward`, `regularization_loss` on `Loss` |
| Reported loss | data loss plus penalty in training; data loss alone on test data |
| Measured, five seeds | L2 and L1 at $5 \times 10^{-4}$ each raised test accuracy and narrowed the gap on all five, and on most of five further seeds; no weight ended at exactly zero; near-zero weights of live neurons under L1 only |
| Strength | $10^{-4}$ to $10^{-3}$ indistinguishable on these runs; $10^{-2}$ underfits |
| Under Adam | L2 in the gradient is not weight decay by a fixed factor |

---

## Common pitfalls

1. **Adding the penalty to the loss and not to the gradient.** The loss changes and training does not. A gradient check against the loss with the penalty catches it.
2. **Reporting the penalty in a test loss.** Test and validation passes report the data loss; the penalty belongs to the training objective.
3. **Expecting exact zeros from L1.** Gradient steps leave weights near zero, not at zero. Sparsity is counted with a stated threshold, and apart from the weights of dead neurons, which either penalty pulls to zero.
4. **Giving L1 and L2 the same $\lambda$ and calling them equally strong.** The two multiply different sums. Each is tuned on its own.
5. **Picking $\lambda$ from one run.** The spread between seeds was larger than the difference between neighbouring strengths. Candidates are compared over several seeds, on validation data.
6. **Reading a small gap as success.** $\lambda = 10^{-2}$ gave the smallest mean gap and the worst test accuracy of the sweep.

---

## Further reading

- Bishop, C. M., *Pattern Recognition and Machine Learning*, chapter 3 (Springer, 2006). Regularised least squares and its Bayesian reading.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 7.1 (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 14 (2020).
- Krogh, A. and Hertz, J. A., *"A Simple Weight Decay Can Improve Generalization"* (NeurIPS, 1992).
- Loshchilov, I. and Hutter, F., *"Decoupled Weight Decay Regularization"* (ICLR, 2019).
- Tibshirani, R., *"Regression Shrinkage and Selection via the Lasso"* (Journal of the Royal Statistical Society, 1996).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 31 - Dropout](../31-dropout/index.md):** the other regulariser of the series, which acts on activations instead of weights.
- **[Post 29 - Validation and hyperparameter tuning](../29-validation-and-hyperparameter-tuning/index.md):** the procedure by which a strength is chosen without spending the test set.
