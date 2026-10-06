# 34 - Sigmoid and binary cross-entropy

> **TL;DR.** With two classes, one output neuron with a **sigmoid** and **binary cross-entropy** is the same model as two softmax outputs: $\sigma(z_2 - z_1)$ equals the softmax probability of class 1, and the two losses agree to the last printed digit. The gradient of the mean loss with respect to a logit is $(\sigma(z) - y)/N$, the prediction-minus-label form of post 19, and the combined class returns it without dividing by the prediction. `Activation_Sigmoid` gets a forward pass that never overflows, and the loss keeps the clip of post 08, which removes `inf` and `nan` and caps the reported loss at 16.118. On two moons the one-output and the two-output network each classified 197 to 200 of 200 test points over ten seeds, with equal counts on nine of them.
>
> **Prerequisites:** [Post 19](../19-softmax-derivatives-and-the-combined-backward-pass/index.md).
> **Safe to skip?** Skip it if the reader can already derive the gradient of binary cross-entropy with respect to the logit, write a sigmoid that does not overflow, and say why one sigmoid output and two softmax outputs describe the same classifier.
>
> **After reading, you will be able to:**
>
> - Derive the gradient of the mean binary cross-entropy with respect to a logit, the sigmoid output minus the label over the batch size.
> - Implement a numerically stable Activation_Sigmoid and the combined class Activation_Sigmoid_Loss_BinaryCrossentropy.
> - Choose between sigmoid plus binary cross-entropy and softmax plus categorical cross-entropy for a binary problem.

![A pipeline: a logit z enters a sigmoid box that outputs a probability p, which enters a binary cross-entropy box with a target y and gives a loss. A dashed arrow runs back from the loss to the logit, past the sigmoid, labelled p minus y over N. On the right, the sigmoid curve and the algebra of the cancellation.](diagrams/01-sigmoid-bce-pipeline.svg)

*Forward: logit, sigmoid, loss. Backward: one subtraction. The figure writes $p$ for the prediction $\hat{y}$; its note "no division anywhere" means no division by the prediction, since the division by $N$ remains.*

---

## 1. The question: what replaces softmax when there are two classes?

[Post 19](../19-softmax-derivatives-and-the-combined-backward-pass/index.md) completed the classification head of the series: $K$ logits per sample, softmax, categorical cross-entropy, and a combined backward pass that returns $(\hat{\mathbf{y}} - \mathbf{y})/N$. That head treats two classes as it treats ten: two output neurons, two probabilities that sum to 1, and the logarithm of the one that belongs to the true class.

With two classes the second probability carries no information, because $\hat{y}_1 + \hat{y}_2 = 1$ and one fixes the other. A single output neuron is enough. Its output $\hat{y} \in (0, 1)$ is read as the probability of class 1, $1 - \hat{y}$ is the probability of class 0, and a label is one number $y \in \{0, 1\}$. The question of this post is: **which activation and which loss belong to that single output, and does the shortcut of post 19 survive?**

The answer is the sigmoid, binary cross-entropy, and yes.

| | Softmax and categorical cross-entropy, $K = 2$ | Sigmoid and binary cross-entropy |
|---|:---:|:---:|
| Output neurons | 2 | 1 |
| Logits | $(N, 2)$ | $(N, 1)$ |
| Labels | integers $(N,)$ or one-hot $(N, 2)$ | 0 or 1, $(N,)$ or $(N, 1)$ |
| Last layer after 16 hidden neurons | 34 parameters | 17 parameters |
| Gradient at the logits | $(\hat{\mathbf{y}} - \mathbf{y})/N$ | $(\hat{y} - y)/N$ |

The saving is one column of the last layer and nothing more. Section 7 shows that the two heads are the same model and measures them side by side. The one-output form is the one the released project `nn-p02` trains, and this post is its theory.

---

## 2. The sigmoid

The activation is the sigmoid of [post 17](../17-backpropagation-through-activation-functions/index.md), applied to the single logit $z$ of a sample:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Three properties make it an output activation. `snippets/binary_classes.py` prints them at five logits:

```text
z                    [-5.0000 -2.0000  0.0000  2.0000  5.0000]
sigma(z)             [0.0067 0.1192 0.5000 0.8808 0.9933]
sigma(z) + sigma(-z) [1.0000 1.0000 1.0000 1.0000 1.0000]
slope                [0.0066 0.1050 0.2500 0.1050 0.0066]
```

**Range.** $\sigma(z)$ lies strictly between 0 and 1, rises with $z$, and equals 0.5 at $z = 0$. A raw score becomes a number that can be read as a probability.

**Symmetry.** $\sigma(-z) = 1 - \sigma(z)$. The probability of class 0 is the sigmoid of the negated logit, so no second logit is needed.

**Slope.** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, derived in post 17, section 3.1. It peaks at 0.25 and falls towards zero on both sides: the saturation of post 06.

### 2.1. The overflow trap

Post 17 wrote the forward pass in one line and said the class would return here:

```python
def sigmoid_naive(z):
    return 1.0 / (1.0 + np.exp(-z))          # the forward of post 17
```

For a large negative $z$ the argument of `np.exp` is large and positive. `snippets/sigmoid_overflow.py` runs this form, the same function written as $e^z / (1 + e^z)$, and the form this post adopts:

```text
z              -1000        -720         -40           0          40         720        1000
naive              0           0   4.248e-18         0.5           1           1           1   warnings: ['overflow encountered in exp']
other              0  2.032e-313   4.248e-18         0.5           1         nan         nan   warnings: ['invalid value encountered in divide', 'overflow encountered in exp']
stable             0  2.032e-313   4.248e-18         0.5           1           1           1   warnings: none
```

```text
float64: np.exp overflows above 709.78, so the naive form warns for z below -709.78
float32: np.exp overflows above 88.72, so the naive form warns for z below -88.72
```

The damage of the naive form is smaller than its warning suggests. In float64, $e^{720}$ overflows to `inf` and $1/(1 + \infty)$ is 0, which is the right answer to within $4.5 \times 10^{-309}$: the two forms differ only for $z$ between $-710$ and $-744$, and between $-700$ and $700$ they agree to $5.6 \times 10^{-17}$. What the naive form costs is a `RuntimeWarning` on a correct result, and a `FloatingPointError` in a program that has asked NumPy to raise on overflow. The other one-line form is the dangerous one: at $z = 720$ it computes $\infty / \infty$ and returns `nan`.

The stable form uses each expression where it is safe, $1/(1 + e^{-z})$ for $z \ge 0$ and $e^z/(1 + e^z)$ for $z < 0$, so `np.exp` never receives a positive argument. It is the principle of post 06, where the largest logit was subtracted before the softmax exponentials. The class of post 17 changes in `forward` only:

```python
class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        # Changed in post 34: two forms, chosen by sign, so that np.exp
        # never receives a positive argument.
        out = np.empty_like(inputs, dtype=np.float64)
        pos = inputs >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-inputs[pos]))     # exp of a value <= 0
        neg = ~pos
        ex = np.exp(inputs[neg])                           # exp of a value < 0
        out[neg] = ex / (1.0 + ex)
        self.output = out

    def backward(self, dvalues):
        # f'(z) = sigma(z) * (1 - sigma(z)), read from the cached output
        self.dinputs = dvalues * self.output * (1 - self.output)
```

`backward` is the line of post 17, unchanged. One side effect of the new `forward`: the output array is created as float64, so float32 logits, which are what `nnfs.init()` produces, give a float64 output whose values were computed in float32.

Saturation in floating point is lopsided, and the next section depends on it. The script finds that in float64 $\sigma(z)$ is exactly 1.0 from $z = 37$ on, because $1 - 8.5 \times 10^{-17}$ rounds to 1, while $\sigma(-z)$ is exactly 0 only from $z = 745$ on. With float32 logits, which a network produces after `nnfs.init()`, the thresholds are 17 and 104; the post's other numbers are float64 unless marked.

---

## 3. Binary cross-entropy

For one sample with label $y \in \{0, 1\}$ and prediction $\hat{y} = \sigma(z)$, the loss is

$$L_i = -\bigl[\, y \log \hat{y} + (1 - y) \log(1 - \hat{y}) \,\bigr]$$

and the loss of a batch is the mean, $L = \frac{1}{N} \sum_{i=1}^{N} L_i$.

**One term is active.** For $y = 1$ the second term vanishes and the loss is $-\log \hat{y}$; for $y = 0$ the first vanishes and the loss is $-\log(1 - \hat{y})$. Either way it is minus the logarithm of the probability given to the true class, the reading of post 08. A prediction of 0.99 costs 0.0101 when the label is 1 and 4.6052 when it is 0; a prediction of 0.5 costs $\log 2 = 0.6931$ for either label.

**It is categorical cross-entropy with $K = 2$.** Writing the two class probabilities as $1 - \hat{y}$ and $\hat{y}$ and the one-hot label as $(1 - y, y)$, the sum $-\sum_k y_k \log \hat{y}_k$ of post 08 is the expression above, term for term.

The worked batch of this post has four samples. `snippets/binary_classes.py` prints:

```text
z              [ 2.0000 -1.0000  0.5000 -3.0000]
y              [1 0 0 1]
y_hat          [0.8808 0.2689 0.6225 0.0474]
sample losses  [0.1269 0.3133 0.9741 3.0486]
mean loss      1.1157
```

The fourth sample is confidently wrong, 0.0474 for a label of 1, and contributes most of the loss.

### 3.1. What the clip protects

Post 08 clipped predictions to $[10^{-7}, 1 - 10^{-7}]$ before the logarithm. Here both $\hat{y}$ and $1 - \hat{y}$ enter a logarithm, so both bounds matter. `snippets/loss_forms.py` computes the loss three ways:

```text
z                            -40        -17         -5          0          5         17         40
y = 0  unclipped               0   4.14e-08   0.006715     0.6931      5.007         17        inf   warnings: ['divide by zero encountered in log']
y = 0  clipped             1e-07      1e-07   0.006715     0.6931      5.007      16.12      16.12
y = 0  from logits     4.248e-18   4.14e-08   0.006715     0.6931      5.007         17         40
y = 1  unclipped              40         17      5.007     0.6931   0.006715   4.14e-08        nan   warnings: ['divide by zero encountered in log', 'invalid value encountered in multiply']
y = 1  clipped             16.12      16.12      5.007     0.6931   0.006715      1e-07      1e-07
y = 1  from logits            40         17      5.007     0.6931   0.006715   4.14e-08  4.248e-18
```

**Without the clip a confident correct prediction returns `nan`.** At $z = 40$ the sigmoid is exactly 1 and $\log(1 - \hat{y})$ is $-\infty$. For $y = 0$ the loss is `inf`. For $y = 1$ the inactive term is $0 \cdot (-\infty)$, which floating point evaluates to `nan`, although the prediction is right. One such sample turns the batch mean into `nan`. At $z = -40$ nothing fails, since $\sigma(-40)$ is $4.2 \times 10^{-18}$ and not 0: the lopsided saturation of section 2.1.

**The clip removes both, and caps the loss.** A prediction is changed whenever $|z|$ exceeds $\log\bigl((1 - 10^{-7})/10^{-7}\bigr) = 16.118$. Beyond that the reported loss of a wrong prediction stays at $-\log(10^{-7}) = 16.118$ however wrong the logit is, and the loss of a right one stays at $10^{-7}$. On 100,000 logits drawn uniformly from $[-40, 40]$ with random labels the clipped mean loss is 6.4622 where the exact one is 10.0339. The clip protects the arithmetic and not the value: a loss near 16 means "at least 16".

**The loss can be computed from the logit.** Substituting $\hat{y} = \sigma(z)$ and simplifying gives $L_i = \log(1 + e^{z}) - y z$, which the script evaluates without overflow as

```python
def bce_from_logits(z, y):
    # max(z, 0) - z * y + log(1 + exp(-|z|)): no probability, no clip
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
```

This is the third row of each block: 40 at $z = 40$, where the clipped form says 16.12. On 100,000 logits in $[-16, 16]$, where the clip is idle, the two forms agree to $1.4 \times 10^{-9}$; the remainder is the rounding of $1 - \hat{y}$ near 1 in the probability route. The class of section 5 keeps the clipped form, as post 19 kept it for softmax and as `nn-p02` does, so that the numbers of the series stay comparable.

---

## 4. The combined gradient

The optimiser needs $\partial L / \partial z$, the slope of the loss with respect to the logit. The chain rule goes through the prediction, in two factors.

**The loss with respect to the prediction.** Differentiating $L_i$ term by term, with $\frac{d}{dx} \log x = 1/x$ and the inner derivative $-1$ of $1 - \hat{y}$:

$$\frac{\partial L_i}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}} = \frac{-y(1 - \hat{y}) + (1 - y)\hat{y}}{\hat{y}(1 - \hat{y})} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

**The prediction with respect to the logit.** The slope of section 2:

$$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$$

**The product.** The denominator of the first factor is the second factor:

$$\frac{\partial L_i}{\partial z} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \cdot \hat{y}(1 - \hat{y}) = \hat{y} - y$$

**The batch.** $L$ is the mean of the $L_i$, and logit $z_i$ enters $L_i$ alone, so the factor $1/N$ carries over as in post 19:

$$\frac{\partial L}{\partial z_i} = \frac{\hat{y}_i - y_i}{N} = \frac{\sigma(z_i) - y_i}{N}$$

On the worked batch, $\hat{\mathbf{y}} - \mathbf{y} = (-0.1192, 0.2689, 0.6225, -0.9526)$ and division by 4 gives $(-0.0298, 0.0672, 0.1556, -0.2381)$.

**Prediction minus label.** The gradient is positive where the prediction is above the label and negative where it is below, so a descent step moves each logit towards its label, by an amount proportional to the error.

**No division by the prediction.** The first factor divides by $\hat{y}(1 - \hat{y})$, which is 0 as soon as the sigmoid has rounded to 1. The product needs no division except the one by $N$. Section 6 runs both routes on saturated logits.

**A wrong output keeps its gradient.** At $z = -10$ with $y = 1$ the slope sent to the logit is $-0.999955$, where a squared error, which keeps the factor $\hat{y}(1 - \hat{y})$, would send $-0.000091$ (`snippets/gradient_check.py`). The gradient saturates only when the prediction is right.

---

## 5. The combined class

The class mirrors `Activation_Softmax_Loss_CategoricalCrossentropy` of post 19: it owns its activation, `forward` returns the mean loss, and `backward` receives the cached output and the labels.

```python
class Activation_Sigmoid_Loss_BinaryCrossentropy:

    def __init__(self):
        self.activation = Activation_Sigmoid()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output               # (N, 1) probabilities

        # Labels of shape (N,) or (N, 1) become a float column.
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        y_pred_clipped = np.clip(self.output, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        return float(np.mean(sample_losses))

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)

        # Two steps: subtract the label, normalise.
        self.dinputs = (dvalues - y_true) / samples
```

It is called as the softmax class is:

```python
    loss_activation = Activation_Sigmoid_Loss_BinaryCrossentropy()
    loss = loss_activation.forward(logits, y)
    loss_activation.backward(loss_activation.output, y)
```

On the worked batch this returns the loss 1.1157 and stores `dinputs` $(-0.0298, 0.0672, 0.1556, -0.2381)$ with shape $(4, 1)$, the numbers of sections 3 and 4.

**`dvalues` is the sigmoid output.** As in post 19, the argument keeps the name of the backward convention and holds the predictions. `backward` stores `dinputs`, returns nothing, and the layer below receives `loss_activation.dinputs`.

**The clip is in the loss and not in the gradient.** `backward` reads the unclipped output. $\hat{y} - y$ contains no logarithm and no division by $\hat{y}$, so there is nothing to protect.

**Labels are reshaped inside the class.** Labels of shape $(N,)$ or $(N, 1)$, integer or float, become a float column, so `dvalues - y_true` subtracts row by row. The script prints the same loss and a `dinputs` of shape $(4, 1)$ for all three. Without the reshape, a column minus a flat array broadcasts to $(N, N)$ (post 05).

**The divisions.** `forward` takes `np.mean` over the $N$ sample losses. `backward` divides by `samples`, the same $N$, once, and nothing downstream divides again (post 19). Neither method divides by a number of outputs, because there is one output. The class is written for one: with $J$ output columns `np.mean` would divide by $N \cdot J$ and `backward` by $N$, and the stored gradient would be $J$ times the slope of the reported loss.

**Predictions.** The predicted class is 1 where $\hat{y} \ge 0.5$, which is where $z \ge 0$: `(loss_activation.output >= 0.5)`. On the worked batch that gives $(1, 0, 1, 0)$ against labels $(1, 0, 0, 1)$, an accuracy of 0.50. The threshold 0.5 picks the more probable class. Where the two kinds of error cost differently another threshold can be chosen afterwards, on validation data (post 29), and the loss and the gradient are unaffected.

The class of `nn-p02` has the same arithmetic line for line, the subtraction `dvalues - y_true` and the single division by `samples` included. It differs in three names (the sigmoid attribute, `logits` for the first argument of `forward`, the clipped prediction) and in carrying a docstring.

---

## 6. Checking the gradient

The check is the one of posts 10, 16 and 21: a central difference with $h = 10^{-5}$ in float64, relative error, pass mark $10^{-7}$, in a process that has not called `nnfs.init()`. `snippets/gradient_check.py` prints:

```text
== The worked batch: dinputs against the measured slope of the loss
dinputs            [-0.029801  0.067235  0.155615 -0.238144]
central difference [-0.029801  0.067235  0.155615 -0.238144]
largest relative error 1.1e-10, largest absolute gap 8.1e-12
two-step route, largest |two-step - combined|: 2.8e-17

== Seeds 0 to 9: 8 logits drawn as 3 * randn, labels drawn 0 or 1
dinputs: largest relative error 1.7e-07, largest absolute gap 4.8e-10
checks above the pass mark: ['seed 0: z = 6.72, y = 1, dinputs -1.5e-04, absolute gap 2.6e-11']

== Seeds 0 to 9: one Layer_Dense(2, 1) in front of the head, 8 samples
dweights  largest relative error 3.6e-09, largest absolute gap 1.3e-11
dbiases   largest relative error 2.7e-10, largest absolute gap 3.9e-11
checks above the pass mark: []
```

The formula holds, the division by $N$ included, since the measured slope is the slope of the mean loss. The one check above the pass mark is the tiny-gradient case of post 16: a correct, confident prediction with a gradient of $-1.5 \times 10^{-4}$ and an absolute gap of $2.6 \times 10^{-11}$. The third block is a dense layer feeding the head directly, which is logistic regression; its `dweights` and `dbiases` pass on all ten seeds.

The two-step route, the loss gradient $-(y/\hat{y} - (1 - y)/(1 - \hat{y}))/N$ handed to `Activation_Sigmoid.backward`, agrees with the combined class to $2.8 \times 10^{-17}$ on the worked batch. On saturated logits it does not:

```text
z                        [  40.000000   40.000000  -40.000000 -800.000000]
y                        [0 1 1 1]
two-step                 [      nan       nan -0.250000       nan]
two-step, y_hat clipped  [ 0.000000 -0.000000 -0.000000 -0.000000]
combined                 [ 0.250000  0.000000 -0.250000 -0.250000]
exact (sigma(z) - y) / 4 [ 0.250000  0.000000 -0.250000 -0.250000]
```

Unclipped, the route divides by zero and returns `nan` on three of the four samples. Clipping $\hat{y}$ first, the usual repair, removes the `nan` and returns 0 for every sample, the three confidently wrong ones included: a large finite quotient times a sigmoid slope that is 0 or below $10^{-17}$. The combined class returns the exact values. This is the practical reason for the class, more than the saved arithmetic.

One place remains where the check cannot pass, and it is the clip:

```text
z =    10, y = 0: loss 10.0000  dinputs +0.999955  central difference +0.999955
z =    20, y = 0: loss 16.1181  dinputs +1.000000  central difference +0.000000
z =   -20, y = 1: loss 16.1181  dinputs -1.000000  central difference +0.000000
```

Beyond $|z| = 16.118$ the reported loss is flat, so its measured slope is 0, while `dinputs` is the slope of the unclipped loss. The gradient is the right one and training continues; a gradient check on such a logit reports a mismatch that is not a bug.

---

## 7. One sigmoid output or two softmax outputs?

**They are the same function.** For two logits $z_1$ (class 0) and $z_2$ (class 1), dividing the numerator and the denominator of the softmax by $e^{z_2}$ gives

$$\hat{y}_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{-(z_2 - z_1)}} = \sigma(z_2 - z_1)$$

and likewise $\hat{y}_1 = \sigma(z_1 - z_2)$. Only the difference of the two logits matters. `snippets/softmax_equivalence.py` feeds six random logit pairs to the combined class of post 19 and their differences $z = z_2 - z_1$ to the class of this post:

```text
largest |softmax class 1 - sigmoid(z_2 - z_1)|: 0.0e+00
largest |softmax class 0 - sigmoid(z_1 - z_2)|: 1.1e-16
categorical cross-entropy 1.346182   binary cross-entropy 1.346182   gap 0.0e+00
```

```text
largest |column z_2 - sigmoid dinputs|: 0.0e+00
largest |column z_1 + column z_2|:      2.8e-17
```

The probabilities, the losses and the gradients agree. The softmax gradient at $z_2$ is the sigmoid gradient, and the gradient at $z_1$ is its negative, so the two softmax logits are pushed apart by equal and opposite amounts. Adding 100 to both logits changes the softmax output by $8.9 \times 10^{-16}$: one direction of the two-logit space does nothing, and the sigmoid head is the softmax head with that direction removed.

**Trained, the two classify alike here.** The same function does not guarantee the same training run, since the two heads have different parameters and Adam scales each one separately. `snippets/seeds.py` trains the network of `nn-p02` with each head. The setup, printed by the script: no `nnfs.init()`, float64 weights; for each seed $s$, 1,000 two-moons points at noise 0.1 from `np.random.default_rng(s)`, split 800 to 200; `np.random.seed(s)`; `Layer_Dense(2, 16)`, ReLU, `Layer_Dense(16, 16)`, ReLU, `Layer_Dense(16, 1)` or `Layer_Dense(16, 2)`; the He initialisation of post 33 through `init="he"`; the L2 penalty of post 30 at $10^{-4}$ on the two hidden weight arrays; `Optimizer_Adam(learning_rate=0.01)` of post 27; 2,000 full-batch epochs. A seed fixes the data, the split and the weights, and the two heads share the initial weights of the hidden layers.

| Head | Parameters | Training points correct, of 800 | Test points correct, of 200 | Seeds with 200 of 200 | Test data loss |
|---|:---:|:---:|:---:|:---:|:---:|
| sigmoid, 1 output | 337 | 800 on all ten seeds | 197 to 200 | 8 of 10 | 0.0004 to 0.0333 |
| softmax, 2 outputs | 354 | 800 on all ten seeds | 197 to 200 | 7 of 10 | 0.0003 to 0.0433 |

Seed by seed the test counts are equal on nine seeds and the sigmoid head has one point more on seed 3; the test loss is lower with the sigmoid head on five seeds and with the softmax head on five. One test point of 200 on one seed is no evidence for either. The training data loss does differ: it is lower with the softmax head on eight of the ten seeds. The sigmoid counts are, seed by seed, the ones `nn-p02` publishes for this setup, 200 of 200 on seed 0 with a test loss of 0.0045.

**Which head for which task.**

| Task | Output | Activation and loss |
|---|---|---|
| Two classes | 1 neuron | sigmoid and binary cross-entropy (this post) |
| $K \ge 3$ classes, one per sample | $K$ neurons | softmax and categorical cross-entropy (post 19) |
| $K$ labels, any number per sample | $K$ neurons | one sigmoid and one binary cross-entropy per output |
| A real number | 1 or more neurons | no activation, mean squared error (`nn-p04`) |

![Four cards for binary, multi-class, multi-label and regression tasks, each listing output size, activation, loss and target: one sigmoid neuron with BCE; K softmax neurons with CCE; K sigmoid neurons with summed BCE; no activation with MSE. The multi-label card is marked as the trap.](diagrams/02-choosing-the-head.svg)

*Multi-label sits beside multi-class because that is where the mistake happens: the same neuron count and a different activation. The cards say "Part" where this series now says "post".*

For two classes either head is correct, and the choice is one of convenience: one output and scalar labels, or the same code path as a multi-class problem. The choice matters for **multi-label** data: softmax makes the outputs sum to 1 and so asserts exactly one label, and the class of section 5 does not cover several outputs as written (section 9). A dense layer followed directly by the sigmoid head, with no hidden layer, is **logistic regression**.

---

## 8. Make it run: the scripts

Every code block and every number of this post comes from a script in `snippets/`, run from the series root, for example:

```text
python posts/34-sigmoid-and-binary-cross-entropy/snippets/seeds.py
```

| Script | Contents | Time |
|---|---|---|
| `binary_classes.py` | the two classes, the worked batch | 1 s |
| `sigmoid_overflow.py` | section 2.1 | 1 s |
| `loss_forms.py` | section 3.1 | 1 s |
| `gradient_check.py` | sections 4 and 6 | 1 s |
| `softmax_equivalence.py` | section 7 | 1 s |
| `network.py` | the earlier posts' classes, the two-moons network, seed 0 | 3 to 10 s |
| `seeds.py` | ten seeds with each head | 15 to 40 s |
| `what_can_go_wrong.py` | section 9 | 1 s |

All need only NumPy, run in float64, and do not call `nnfs.init()`. `network.py` holds `Layer_Dense` with the arguments of posts 30 and 33, and the classes of posts 16, 19 and 27 unchanged.

---

## 9. What can go wrong?

`snippets/what_can_go_wrong.py` runs each mistake.

**Accuracy from a column and a flat array.** The class reshapes labels for itself; code outside it does not. With eight samples that are all classified correctly:

```text
predictions (8, 1) == y (8,): shape (8, 8), mean 0.5312
predictions.ravel() == y: shape (8,), mean 1.0000
```

The comparison broadcasts to every prediction against every label and reports 53 percent for a perfect classifier. Nothing is raised.

**The wrong array, or no division, in `backward`.** On the worked batch:

```text
central difference         [-0.0298  0.0672  0.1556 -0.2381]
correct                    [-0.0298  0.0672  0.1556 -0.2381]  relative error 1.1e-10
logits handed to backward  [ 0.2500 -0.2500  0.1250 -1.0000]  relative error 1.3e+00
no division by samples     [-0.1192  0.2689  0.6225 -0.9526]  relative error 7.5e-01
```

Handing `backward` the logits computes $(z - y)/N$, with the wrong sign on the first two samples. Leaving out the division gives 4 times the slope at $N = 4$, which under plain gradient descent acts as a learning rate multiplied by the batch size. Neither raises an error, and the gradient check catches both.

**Labels $-1$ and $+1$.** Some texts code the two classes that way. The class accepts them without complaint:

```text
worked batch: loss 0.9907 (with labels 0 and 1: 1.1157)
one sample, z = -20, y = -1, a correct prediction: loss -16.1181, dinputs +1.0000
```

With $y = -1$ the factor $1 - y$ is 2 and the factor $y$ is negative, so the loss can fall below zero and the gradient $(\hat{y} + 1)/N$ never vanishes: the logit is pushed down without end. Labels are converted to 0 and 1 before training.

**Two output neurons on the sigmoid head.**

```text
logits (4, 2), labels (4,): loss 1.0532, dinputs shape (4, 2), no error
logits (4, 2), one-hot labels (4, 2): ValueError: operands could not be broadcast together with shapes (8,1) (4,2) 
```

With flat labels the column of labels broadcasts across both logit columns: each output is trained as its own predictor of the same label, with the mismatched divisions of section 5. With one label per output the reshape fails.

**A sigmoid output with the categorical loss of post 19.**

```text
integer labels: IndexError: index 1 is out of bounds for axis 1 with size 1
one-hot labels: loss 1.2407, no error; mean of -log(y_hat) over all four samples: 1.2407
```

Integer labels fail as soon as a label is 1. One-hot labels do not fail: the single column is multiplied into both label columns and the loss becomes $-\log \hat{y}$ for every sample whatever its label, a loss that is minimised by predicting class 1 always.

---

## 10. Summary

| Concept | Takeaway |
|---|---|
| Sigmoid | $\sigma(z) = 1/(1 + e^{-z})$, in $(0, 1)$, with $\sigma(-z) = 1 - \sigma(z)$ |
| Stable forward | $1/(1 + e^{-z})$ for $z \ge 0$, $e^z/(1 + e^z)$ for $z < 0$; no overflow, no `nan` |
| Binary cross-entropy | $L_i = -[y \log \hat{y} + (1 - y) \log(1 - \hat{y})]$; categorical cross-entropy at $K = 2$ |
| The clip | removes `inf` and `nan` once the sigmoid rounds to 1 (from $z = 37$ in float64, 17 in float32); caps the reported loss at 16.118 |
| Combined gradient | $\partial L / \partial z_i = (\sigma(z_i) - y_i)/N$; one division, by $N$, in `backward` |
| Combined class | `Activation_Sigmoid_Loss_BinaryCrossentropy`, one output, labels reshaped to $(N, 1)$ |
| Against softmax | $\hat{y}_2 = \sigma(z_2 - z_1)$; same loss and gradient; 197 to 200 of 200 on two moons with either head |
| Not covered by the class | several labels per sample |

---

## Common pitfalls

1. **Comparing a column of predictions with flat labels.** The result has shape $(N, N)$ and its mean is not an accuracy. Predictions are flattened first.
2. **Taking the logarithm without the clip.** From $z = 37$ in float64 (17 in float32) the sigmoid is exactly 1, and the loss is `inf` for a wrong prediction and `nan` for a right one.
3. **Reading a clipped loss as exact.** 16.118 is a cap, not a value (sections 3.1 and 6).
4. **Running the loss backward and the sigmoid backward separately.** Correct on paper; `nan` on saturated logits, or 0 when the prediction is clipped first.
5. **Softmax for labels that are not exclusive.** Softmax asserts one label per sample. Several labels need one sigmoid per output.
6. **Labels other than 0 and 1.** The class does not check them, and $-1$ and $+1$ give a loss that falls below zero.

---

## Further reading

- Bishop, C. M., *Pattern Recognition and Machine Learning*, section 4.3.2 (Springer, 2006). Logistic regression.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.2.2.2 (MIT Press, 2016). Sigmoid units for a yes-or-no output.
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 16 (2020).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 35 - What to read after this series](../35-whats-next/index.md):** where the dense stack built here leads, architecture by architecture.
- **[Post 19 - Softmax derivatives and the combined backward pass](../19-softmax-derivatives-and-the-combined-backward-pass/index.md):** the $K$-class cancellation that this post repeats for one output.
