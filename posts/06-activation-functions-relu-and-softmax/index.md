# 06 - Activation functions: ReLU and Softmax

> **TL;DR.** Without a non-linear function between its layers, a deep network computes nothing that a single linear layer cannot. This post adds the two activations that fix that for the rest of the series: **ReLU**, `max(0, z)`, after every hidden layer, and **softmax** after the output layer of a classifier, where it turns each row of raw scores into probabilities that sum to 1. Softmax subtracts each row's largest score before exponentiating, which leaves the probabilities unchanged and keeps `np.exp` from overflowing.
>
> **Prerequisites:** [Post 03](../03-stacking-layers-and-the-forward-pass/index.md), [Post 05](../05-array-summation-keepdims-and-broadcasting/index.md).
> **Safe to skip?** Skip it if the reader can already say why two dense layers with nothing between them are one layer, write `Activation_ReLU` and an overflow-safe `Activation_Softmax` from memory, and name the activation that belongs on each layer of a classifier.
>
> **After reading, you will be able to:**
>
> - State in one sentence why dense layers need a non-linear activation between them.
> - Implement an Activation_ReLU class whose forward method follows the Layer_Dense pattern.
> - Implement an Activation_Softmax class that turns each row of logits into probabilities that sum to 1.
> - Explain why softmax subtracts the per-row maximum before exponentiating, and what goes wrong without that step.
> - Explain why hidden layers use ReLU while the output layer of a classifier uses softmax.

![Two panels plot y against x for the same two dense layers. Left, without an activation: the formula Z = (X W1 + b1) W2 + b2 collapses to one linear map, X (W1 W2) + (b1 W2 + b2), and the network's single straight line cannot follow a dashed zigzag target. Right, with ReLU between the layers: the output is piecewise linear, with a marked kink wherever a hidden neuron crosses zero, and it follows the same zigzag.](diagrams/01-why-nonlinearity.svg)

*The same two layers, without and with a ReLU between them. The straight line on the left is everything a stack of linear layers can produce; every kink on the right comes from one hidden neuron.*

---

## 1. The question: what has to sit between two layers?

[Post 03](../03-stacking-layers-and-the-forward-pass/index.md) ended with a quiet warning: a stack of dense layers without an activation between them is itself a dense layer. In the weight layout of [post 04](../04-dense-layer-class-and-spiral-data/index.md), where $\mathbf{W}$ has shape $(n_\text{inputs}, n_\text{neurons})$ and the forward call needs no transpose, the two-layer forward pass

$$\mathbf{Z}_2 = (\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

expands to a single linear map $\mathbf{X}(\mathbf{W}_1 \mathbf{W}_2) + (\mathbf{b}_1 \mathbf{W}_2 + \mathbf{b}_2)$. Two layers, one effective layer. Fifty layers, still one effective layer. Depth without a non-linearity is a mirage.

The question of this post therefore has two halves. What has to sit between two dense layers for the second one to add anything? And why is the answer different for the hidden layers of a classifier and for its output layer?

The fix is small and decisive. An **activation function** is a function applied to a layer's weighted sums before they are handed on, either to each entry on its own or to each row. Insert one after each dense layer:

$$\mathbf{A}_2 = f_2(f_1(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2).$$

Here, as in the [notation guide](../../notation_guide.md), $\mathbf{Z}$ stands for a layer's weighted sums and $\mathbf{A} = f(\mathbf{Z})$ for what the activation makes of them. If $f_1$ is a **non-linearity**, a function that is not of the form $az + b$, the substitution that merged the two layers no longer goes through: $f_1$ stands between $\mathbf{W}_1$ and $\mathbf{W}_2$, and the two matrices cannot be multiplied together.

### 1.1. One network, run both ways

The script `snippets/why_nonlinearity.py` shows the difference on a network small enough to check by hand: one input feature, three hidden neurons, and one output neuron, with every weight and bias written out.

```python
# One input feature, three hidden neurons, one output neuron.
# Weights are stored as (n_inputs, n_neurons), the layout of post 04.
W1 = np.array([[1.0, 1.0, 1.0]])            # (1, 3)
b1 = np.array([[0.0, -1.0, -2.0]])          # (1, 3)
W2 = np.array([[1.0], [-2.0], [2.0]])       # (3, 1)
b2 = np.array([[0.0]])                      # (1, 1)

X = np.arange(-1.0, 3.5, 0.5).reshape(-1, 1)    # nine inputs from -1 to 3, shape (9, 1)

# Without an activation: two dense layers, one after the other.
Z1        = np.dot(X, W1) + b1
no_activ  = np.dot(Z1, W2) + b2

# The single layer the two collapse into.
W_star    = np.dot(W1, W2)                  # (1, 1)
b_star    = np.dot(b1, W2) + b2             # (1, 1)
one_layer = np.dot(X, W_star) + b_star

# With ReLU between the same two layers.
A1        = np.maximum(0, Z1)
with_relu = np.dot(A1, W2) + b2
```

The first computation is the stack of post 03. The second is the single layer it collapses into, with $\mathbf{W}_\ast = \mathbf{W}_1 \mathbf{W}_2$ and $\mathbf{b}_\ast = \mathbf{b}_1 \mathbf{W}_2 + \mathbf{b}_2$. The third puts `np.maximum(0, Z1)`, the activation of section 2, between the same two layers. The script prints all three for the nine inputs:

```text
W_star = [1.]  b_star = [-2.]
two layers without an activation equal the one layer: True

    x   no activation   one layer   with ReLU
 -1.0            -3.0        -3.0         0.0
 -0.5            -2.5        -2.5         0.0
  0.0            -2.0        -2.0         0.0
  0.5            -1.5        -1.5         0.5
  1.0            -1.0        -1.0         1.0
  1.5            -0.5        -0.5         0.5
  2.0             0.0         0.0         0.0
  2.5             0.5         0.5         0.5
  3.0             1.0         1.0         1.0

slopes without an activation: [1.]
slopes with ReLU:             [-1.  0.  1.]
with ReLU: f(0) = 0.0  f(2) = 0.0  their midpoint = 0.0  but f(1) = 1.0
```

Without an activation, the two layers and their 10 parameters are the straight line $x - 2$, a layer with one weight and one bias, and the two columns agree entry for entry. With the activation, the same 10 parameters produce a zigzag: flat up to $x = 0$, rising to 1 at $x = 1$, falling back to 0 at $x = 2$, and rising again. Its slope takes three values, 0, 1, and $-1$, where the straight line has one. The bends sit at $x = 0$, $1$, and $2$, one per hidden neuron: each neuron computes $x$ plus its own bias, the biases are 0, $-1$, and $-2$, and a neuron switches on where its weighted sum crosses zero.

No single dense layer can reproduce the last column. A straight line through the outputs at $x = 0$ and $x = 2$, both 0, would have to give 0 at $x = 1$ as well, and the network gives 1. This is what the figure at the top of the post draws: one straight line on the left, and on the right a piecewise-linear curve with one kink per hidden neuron.

### 1.2. How far the activation reaches

The universal-approximation theorems say how far this goes. Cybenko (1989) proved that a network with one hidden layer of sigmoid-shaped units can, given enough of them, approximate any continuous function on a closed, bounded region as closely as desired. Hornik (1991) extended the result to a much wider class of activations, and Leshno, Lin, Pinkus, and Schocken (1993) showed that it holds for every continuous activation that is not a polynomial, which includes the one this post adopts. The activation is the source of that power; without it, the depth does nothing. The theorems state that suitable weights exist. They do not say how to find them, which is the work of posts 09 to 27.

This post introduces the two activations that carry the rest of the series:

| Activation | Where it lives | What it does |
|---|---|---|
| **ReLU** | every hidden layer | injects non-linearity; cheap, simple, gradient-friendly |
| **Softmax** | the output layer (for classification) | turns raw scores into a probability distribution |

---

## 2. ReLU, formally

The rectified linear unit (**ReLU**) is the function:

$$\text{ReLU}(z) = \max(0, z).$$

Negative inputs become zero. Positive inputs pass through unchanged. The plot is two straight lines meeting at the origin, with a sharp kink at $z = 0$.

![A plot of f(x) = max(0, x): flat along the horizontal axis for negative inputs, labelled negative in gives 0 out, a marked kink at x = 0, and a straight diagonal for positive inputs, labelled positive in passes unchanged. Four cards beside the plot name what ReLU is not: not smooth, not zero-centred, not safe from dying, and not a probability.](diagrams/04-relu-anatomy.svg)

*Two straight lines meeting at the origin; the figure writes the input as $x$ where the text writes $z$. Every bend a network makes traces back to this one kink, and the four cards are the limits that section 2.2 goes through.*

That kink is everything. A single ReLU adds one bend to the function the network represents. Giving many neurons different weights and biases shifts and scales these bends, as the three neurons of section 1.1 did. Putting them in two or three layers lets the network compose hundreds of bends into a close approximation of any continuous shape: spirals, decision boundaries, image edges.

### 2.1. Why ReLU and not the older alternatives

The history is short and consequential. The sigmoid, $\sigma(z) = 1/(1 + e^{-z})$, and tanh were the standard hidden-layer activations for decades. Both *saturate*: for inputs far from zero the curve flattens and its slope falls towards zero. That matters because a network learns by sending a correction signal, the gradient of posts 10 to 21, backwards through its layers, and at every activation the signal is multiplied by the slope of that activation. The script `snippets/relu.py` prints the three slopes side by side:

```text
largest sigmoid slope, at z = 0: 0.25
largest tanh slope, at z = 0:    1.0
    z   ReLU slope   sigmoid slope   tanh slope
  0.5            1        2.35e-01     7.86e-01
  2.5            1        7.01e-02     2.66e-02
  5.0            1        6.65e-03     1.82e-04
 10.0            1        4.54e-05     8.24e-09
0.25 ** 10 = 9.5367431640625e-07
```

The sigmoid's slope is never more than 0.25, so a signal that crosses ten sigmoid layers is scaled by at most $0.25^{10} \approx 9.5 \times 10^{-7}$ by the activations alone, before the weights are counted, and by far less wherever a neuron sits on a flat part of the curve. This is the **vanishing-gradient problem**: in a deep network the signal fades before it reaches the early layers, and they stop learning (Hochreiter, 1991). ReLU's slope is exactly 1 for every positive input, however large, so on that side nothing shrinks.

Nair and Hinton (2010) showed that rectified linear units improve restricted Boltzmann machines, and Glorot, Bordes, and Bengio (2011) showed that deep networks of rectifier units match or beat tanh networks and can be trained with plain supervised learning, without the unsupervised pre-training that deep networks had commonly relied on until then. Krizhevsky, Sutskever, and Hinton's AlexNet (2012) used ReLU in all of its hidden layers. Their paper reports that a small convolutional network with ReLUs reached 25 percent training error on CIFAR-10 six times faster than the same network with tanh units, and AlexNet won the 2012 ImageNet competition with a top-5 test error of 15.3 percent against 26.2 percent for the runner-up, a margin that turned computer vision towards learned features. ReLU has been the usual choice for hidden layers since; Goodfellow, Bengio, and Courville (2016, chapter 6) give it as the default recommendation. Replacing a curve by a function that costs one comparison per number proved to be one of the more consequential changes in the field.

### 2.2. What ReLU is *not*

A boundary section, because the function has well-known limitations that later posts will refine.

- **ReLU is not smooth.** Its slope jumps from 0 to 1 at $z = 0$, so its derivative is not defined there. Implementations pick a value and move on; the backward pass of post 17 uses 0. It is often said that an input is never exactly zero in practice. In this series it is: the first sample of every spiral arm is the origin (post 04), so with zero biases all of its weighted sums are exactly 0, and in the forward pass of section 5 that makes 9 of the 900 inputs to the ReLU exact zeros. The value chosen at the kink is therefore used on real data, not only in theory.
- **ReLU does not centre activations around zero.** All outputs are non-negative, so their average is above zero. `snippets/relu.py` draws 100,000 standard-normal inputs whose mean is 0.002 and whose mean after ReLU is 0.398; the exact value for a standard normal is $1/\sqrt{2\pi} \approx 0.399$. The inputs of the next layer are therefore shifted to one side. Batch normalisation (`cnn-014`, Convolutional Neural Networks from Scratch) subtracts a mean from the values it is applied to, which is one way of removing a shift of this kind.
- **ReLU does not save dead neurons.** A **dead neuron** is a ReLU neuron whose weighted sum is negative for every sample. It outputs zero everywhere, the correction signal that would repair its weights is multiplied by a slope of zero, and so its weights stop updating. Section 7 builds one on the spiral data. This "dying ReLU" problem motivates variants such as Leaky ReLU, which gives negative inputs a small slope in place of zero. GELU (Hendrycks and Gimpel, 2016), a smooth relative of ReLU, does not set negative inputs to exactly zero either. Post 17 names them; this series does not need them.
- **ReLU is not a probability function.** Its outputs can be any non-negative number, including 0, 5, or 10,000. Classification needs something else for the output layer.

### 2.3. Implementation

Wrapped in the same class pattern as `Layer_Dense`:

```python
class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
```

One line. `np.maximum(0, inputs)` is element-wise: it compares every entry with 0 and returns an array of the same shape with the negatives replaced by zero. This is different from `np.max(inputs)`, which collapses the whole array to a single scalar.

```python
inputs = np.array([1, -2, 3, -0.5, 0])
print(np.maximum(0, inputs))   # [1. 0. 3. 0. 0.]
```

For the same array `np.max(inputs)` is `3.0`: one number where five were needed.

The class stores the result on `self.output` for the next layer to consume, matching the `Layer_Dense` convention, and like `Layer_Dense.forward` it returns nothing. An activation owns no weights and no biases, so the class has nothing to allocate and needs no `__init__`, and the array it stores always has the shape of the array it was given. `snippets/relu.py` passes it a batch of 2 samples and 3 neurons, `[[1.5, -0.3, 0.0], [-2.0, 4.0, -0.1]]`, and prints the shapes and the output:

```text
batch shape: (2, 3) -> output shape: (2, 3)
[[1.5 0.  0. ]
 [0.  4.  0. ]]
```

---

## 3. Why ReLU is not enough for the output layer

ReLU produces non-negative numbers of any size. Classification needs three properties ReLU does not provide:

1. **Outputs in `[0, 1]`.** A class probability cannot be 5.
2. **Outputs that sum to 1.** Across the possible classes for one sample, the probabilities must form a valid distribution.
3. **A "confidence" interpretation.** "This sample is class 2 with 80 percent probability" is what a downstream loss function and a downstream human consumer both expect.

Softmax provides all three with one well-defined operation.

---

## 4. Softmax, formally

For one sample, the last dense layer produces a row of $K$ raw scores $z_1, z_2, \dots, z_K$, one per class. These raw scores are called **logits**. **Softmax** turns them into the predicted probabilities $\hat{y}_1, \dots, \hat{y}_K$:

$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.$$

The numerator exponentiates one score. The denominator is the sum of the exponentials of every score in the same sample's row. Three guarantees fall out of the definition:

| Property | Why it holds |
|---|---|
| All outputs `> 0` | the exponential is always positive |
| All outputs `< 1` | the numerator is one term of the denominator sum, and with two or more classes the other terms are positive |
| Outputs sum to 1 | $\sum_k \frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{\sum_k e^{z_k}}{\sum_j e^{z_j}} = 1$ |

One row by hand, as `snippets/softmax.py` prints it for the logits $[1, 2, 3]$:

```text
1. logits           [1. 2. 3.]
   exp of each      [ 2.71828  7.38906 20.08554]
   their sum        30.19287
   each over sum    [0.09003  0.24473  0.66524]
   class output     [0.09003  0.24473  0.66524]
   sum of the row   1.0
```

The dense layer that feeds softmax does not produce probabilities. It produces logits, which can be negative, positive, large, or small. Softmax rescales them into a probability distribution without adding information. Bigger logits become bigger probabilities; smaller logits become smaller probabilities; the row totals to 1. The order never changes, so the class with the largest logit is the class with the largest probability.

Two consequences of the exponential are worth reading off the example. The ratio of two probabilities depends only on the gap between their logits, $\hat{y}_a / \hat{y}_b = e^{z_a - z_b}$: a gap of 1 is a factor of $e \approx 2.718$, which is why 0.24473 is 2.718 times 0.09003. And when all the logits of a row are equal, every exponential is the same and every probability is exactly $1/K$, whatever the common value: the script returns three probabilities of 0.33333333 for the row $[100, 100, 100]$ and for the row $[0, 0, 0]$ alike.

### 4.1. Where softmax comes from

The function is older than its name: it is the Boltzmann distribution of statistical physics and the multinomial logistic model of statistics. Bridle's 1990 paper *"Probabilistic Interpretation of Feedforward Classification Network Outputs"* named it softmax and proposed it, as the generalisation of the sigmoid to more than two classes, for the output layer of classification networks, so that the outputs can be read as probabilities. Minimising the cross-entropy loss of post 08 on softmax outputs is maximum-likelihood estimation of the network's parameters under a categorical distribution, and that pairing is the standard choice for multi-class classification today.

### 4.2. The numerical-stability trick

The naive softmax can overflow. `np.exp` returns `inf` as soon as its result is too large for the array's type: in `float64` that happens for inputs above 709.78, and in `float32`, the type that `nnfs.init()` gives the arrays of this series, for inputs above 88.72. A row of logits such as $[1000, 1001, 999]$ therefore exponentiates to three `inf` values, and the ratio `inf / inf` is `nan`. NumPy raises no error. It prints two `RuntimeWarning` lines, and the `nan` values flow into everything computed from them.

The fix is to subtract the per-row maximum from every logit before exponentiating:

$$\hat{y}_k = \frac{e^{z_k - \max_j z_j}}{\sum_{j} e^{z_j - \max_j z_j}}.$$

This is mathematically identical to the original definition. Subtracting the same constant $c$ from every logit multiplies both the numerator and denominator by $e^{-c}$, which cancels:

$$\frac{e^{z_k - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c} e^{z_k}}{e^{-c} \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}.$$

Softmax therefore sees only the differences between the logits of a row, never their overall level. The script confirms it: the rows $[1, 2, 3]$, $[1001, 1002, 1003]$, and $[-999, -998, -997]$ all give $[0.09003, 0.24473, 0.66524]$.

![Two worked columns for the logits 1000, 1001, 999. Naive, exp of the logits: every exponential is inf and every softmax entry is nan. Stable, exp of the logits minus the max of 1001: the shifted logits are -1, 0, -2, their exponentials 0.368, 1.000, 0.135, and the softmax 0.245, 0.665, 0.090. A band below writes out the identity in which the factor e to the power minus c cancels between numerator and denominator.](diagrams/02-softmax-stability.svg)

*Same probabilities, far safer intermediates: after the subtraction every exponent is at most 0 and every exponential lies in the interval from 0 to 1. The figure writes a logit as $o_i$, and its "above ~700" is the `float64` limit; in `float32` the limit is 88.72.*

After the subtraction the largest exponent is exactly zero, so the largest exponential is exactly one. No overflow is possible, and the denominator is at least 1, so the division is always safe. The script runs the figure's row both ways:

```text
3. np.exp of [1000. 1001.  999.] = [inf inf inf]
   naive softmax    [nan nan nan]
     RuntimeWarning: overflow encountered in exp
     RuntimeWarning: invalid value encountered in divide
   shifted logits   [-1.  0. -2.]
   exp of those     [0.368 1.    0.135]
   stable softmax   [0.245 0.665 0.09 ]
```

The naive formula has a second failure that the same shift removes. For very negative logits such as $[-1000, -999, -1001]$ every exponential underflows to 0, the denominator is 0, and `0 / 0` is `nan` again; the shifted version returns $[0.245, 0.665, 0.09]$. The cost of the fix is two extra array operations per batch, one `np.max` and one broadcast subtraction; the benefit is that the numerics simply work.

### 4.3. Implementation

```python
class Activation_Softmax:

    def forward(self, inputs):
        # Subtract the per-row max for stability.
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        # Exponentiate and normalise per row.
        exp_values    = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities
```

The `axis=1, keepdims=True` argument pair is exactly what [post 05](../05-array-summation-keepdims-and-broadcasting/index.md) was for. `axis=1` reduces along the class axis, giving one result per sample; `keepdims=True` returns a column of shape `(N, 1)` that broadcasts correctly when subtracted from the original `(N, K)` matrix. Without `keepdims=True` the maxima come back with shape `(N,)` and line up with the columns, the bug of post 05, section 3.1. On the `(300, 3)` batch of this series that stops the program with a `ValueError`, which is the lucky outcome. On a square batch, with as many samples as classes, it runs, and section 7 shows the result: rows that still sum to 1 and hold the wrong probabilities.

The two classes of this post differ in one structural way. ReLU treats every entry on its own. Softmax couples the entries of a row: each output depends on every logit in that row, and raising one logit lowers the probability of every other class. The difference returns in the backward pass, where ReLU needs only a mask (post 17) and softmax needs a post of its own (post 19).

### 4.4. Temperature

One extension is common enough to name. Dividing every logit by a positive constant $T$, the *temperature*, before the softmax controls how sharp the distribution is:

$$\hat{y}_k = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}.$$

For the logits $[1, 2, 3]$ the script prints:

```text
8. T =  0.1: [2.06e-09  4.54e-05  0.99995]
8. T =  1.0: [0.09003  0.24473  0.66524]
8. T = 10.0: [0.30061  0.33222  0.36717]
```

A low temperature stretches the gaps between the logits and pushes nearly all of the probability onto one class; a high temperature shrinks the gaps and pushes it towards uniform; $T = 1$ is the plain softmax. The name comes from the Boltzmann distribution of section 4.1. The classes of this series always use $T = 1$, and the three rows are here for the contrast with section 4.2: adding a constant to every logit changes nothing, while multiplying every logit by a constant changes every probability.

---

## 5. The forward pass, end to end

The pieces from posts 04 and 05 plus the two activations from this post now form a complete (untrained) classifier. The script `snippets/forward_pass.py` defines `Layer_Dense` exactly as [post 04](../04-dense-layer-class-and-spiral-data/index.md) wrote it, then the two `Activation_` classes above, and runs them in order:

```python
nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1      = Layer_Dense(2, 3)            # 2 inputs, 3 hidden neurons
activation1 = Activation_ReLU()

dense2      = Layer_Dense(3, 3)            # 3 inputs, 3 output neurons (1 per class)
activation2 = Activation_Softmax()

dense1.forward(X)                          # linear
activation1.forward(dense1.output)         # ReLU
dense2.forward(activation1.output)         # linear: the logits
activation2.forward(dense2.output)         # softmax: logits to probabilities

print(activation2.output[:5])
```

**Output (first five samples):**

```text
[[0.33333334 0.33333334 0.33333334]
 [0.3333332  0.3333332  0.33333364]
 [0.3333329  0.33333293 0.3333342 ]
 [0.3333326  0.33333263 0.33333477]
 [0.33333233 0.3333324  0.33333528]]
```

Every entry of these five rows is within 0.000002 of one third, because the network is untrained and, more to the point, its weights are small. With the 0.01 scale of post 04 the largest logit over all 300 rows is 0.00023 in absolute value, and logits that are nearly equal give probabilities that are nearly equal (section 4). The first row is exactly uniform: that sample is the origin, all its weighted sums are 0, and three equal logits give exactly one third each. The row furthest from uniform is row 99, `[0.33329675 0.33330163 0.33340165]`, still within 0.00007 of one third. Once training starts in post 22 the weights move, the logits spread apart, and rows sharpen towards confident predictions of the form `[0.95, 0.03, 0.02]`, an illustration of the pattern and not the output of a script.

Each row sums to 1 up to `float32` rounding: the smallest of the 300 row sums is 0.9999999 and the largest 1.0000001. The hidden layer shows what ReLU does on real data. Of the 900 weighted sums that `dense1` produces, 447 are negative, 9 are exactly zero, and 444 are positive, so ReLU hands 456 zeros to `dense2`, a little over half of its outputs.

![A left-to-right pipeline. X of shape (N, 2), the spiral input, enters dense1, Layer_Dense(2, 3), which computes X times W1 plus b1. Next, act1, Activation_ReLU, applies max(0, .). Then dense2, Layer_Dense(3, 3), computes A1 times W2 plus b2 and produces the logits, and act2, Activation_Softmax, turns them into the probabilities y-hat. Every array after the input carries the shape badge (N, 3).](diagrams/03-forward-pass-pipeline.svg)

*Four objects in a row. Each one stores its output on `self`, and the next object reads the previous one's `self.output`; no external state, no glue code. The figure's `act1` and `act2` are the script's `activation1` and `activation2`.*

The script also prints the shape and type of every array on the way through:

```text
X                  (300, 2)  float32
dense1.output      (300, 3)  float32
activation1.output (300, 3)  float32
dense2.output      (300, 3)  float32
activation2.output (300, 3)  float32
```

Only the dense layers change a shape, and only its second number. Neither activation does.

### 5.1. The default pattern

For every classification network in this series, and for most of those in use elsewhere, the activations follow this pattern:

| Layer position | Activation | Reason |
|---|---|---|
| every hidden layer | **ReLU** | cheap non-linearity, gradient-friendly, the usual default since the early 2010s |
| output layer (classification) | **Softmax** | probability distribution over classes |
| output layer (regression) | none (identity) | the prediction is a real number, not a probability |
| output layer (binary classification) | **sigmoid** | a single probability for "is this class 1?"; covered in post 34 |

A network that follows this default has its activations in the right places without further thought. Departures from the pattern have specific reasons; without one, follow the default. Four placement rules follow from what each function does.

- **An activation goes after a dense layer, between it and the next one.** That is the only position in which it prevents the collapse of section 1: two dense layers with nothing between them merge into one, wherever else in the network an activation has been placed.
- **The input data gets no activation.** The input is not the output of a layer; it is the original feature vector. Applying ReLU to it would destroy every negative feature for no benefit, and 305 of the 600 spiral coordinates are negative.
- **The logits get softmax, never ReLU and then softmax.** A ReLU in front of the softmax throws away every negative logit before it can be exponentiated; section 7 shows a row of three different negative logits coming out as exactly uniform.
- **A regression output gets no activation.** The prediction has to be free to take any real value, positive or negative, and passing it through ReLU or softmax would constrain it in a way the task does not want.

During training, softmax is still applied to the logits, exactly as here. What changes in post 19 is the backward pass: differentiating softmax and the cross-entropy loss together gives a formula so short that the full softmax derivative never has to be built. For now, applying the two separately is correct.

---

## 6. Make it run: five short scripts

Every number and every printed block in this post comes from a script in `snippets/`. Each runs from the series root in under a second on a CPU, for example `python posts/06-activation-functions-relu-and-softmax/snippets/softmax.py`, and prints the same output on every run. Three need only NumPy; `forward_pass.py` and `pitfalls.py` also need the `nnfs` package (`pip install nnfs`) for the spiral data.

| Script | Section | What it prints |
|---|---|---|
| `why_nonlinearity.py` | 1.1 | the table of section 1.1: two layers without an activation equal to the line $x - 2$, and the zigzag with ReLU between them |
| `relu.py` | 2 | `[1. 0. 3. 0. 0.]`, the `(2, 3)` batch of section 2.3, the slope table of section 2.1, and the means 0.002 and 0.398 of section 2.2 |
| `softmax.py` | 4 | one row by hand, the shift, the overflow, the two overflow limits, equal logits, a batch of three rows, and three temperatures |
| `forward_pass.py` | 5 | the five rows of section 5, the shape of every array, the counts 447, 9, and 444, and the row sums |
| `pitfalls.py` | 7 | every error message and wrong result quoted in section 7 |

`softmax.py` prints eight numbered items. Item 1 is the row worked by hand in section 4, item 3 the overflow of section 4.2, and item 8 the temperatures of section 4.4. Item 2 is the shift check of section 4.2, three rows that differ by a constant and print the same probabilities. Items 4 to 7 are:

```text
4. largest float64 = exp(709.78)   largest float32 = exp(88.72)
   float64: exp(709) = 8.218e+307   exp(710) = inf
   float32: exp(88)  = 1.652e+38   exp(89)  = inf
5. naive softmax of [-1000.  -999. -1001.] = [nan nan nan]
     RuntimeWarning: invalid value encountered in divide
   stable softmax   [0.245 0.665 0.09 ]
6. equal logits [100, 100, 100] and [0, 0, 0]:
[[0.33333333 0.33333333 0.33333333]
 [0.33333333 0.33333333 0.33333333]]
7. a batch of three rows:
[[0.09  0.245 0.665]
 [0.333 0.333 0.333]
 [0.007 0.987 0.007]]
   row sums: [1. 1. 1.]
```

Item 4 is where the two limits of section 4.2 come from: the natural logarithm of the largest `float64` is 709.78 and that of the largest `float32` is 88.72, so `np.exp(709)` is still a number in `float64` and `np.exp(710)` is not, and the same holds for 88 and 89 in `float32`. Item 7 passes three rows at once. Each is normalised on its own: the row $[2, 2, 2]$ comes out uniform, and the row $[0, 5, 0]$ puts 0.987 on its middle class.

The naive formula that items 3 and 5 run is the definition of section 4 typed in directly, with no shift:

```python
def naive_softmax(logits):
    """The definition with no shift: it breaks when a logit is far from zero."""
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
```

---

## 7. What can go wrong?

Each failure below is triggered on purpose by `snippets/pitfalls.py`, which prints the messages and arrays quoted here.

- **`np.max` is written where `np.maximum` was meant.** `np.max(inputs)` runs and returns `3.0` for the five-element array of section 2.3, a single number in place of an array. With the zero in front, `np.max(0, inputs)` raises `TypeError: only integer scalar arrays can be converted to a scalar index`, because `np.max` reads its second argument as an axis.
- **`keepdims=True` is left off the softmax.** On a batch of 4 rows and 3 columns NumPy stops with `ValueError: operands could not be broadcast together with shapes (4,3) (4,)`. On a square batch of 3 rows and 3 columns nothing is raised. With `keepdims` missing from the max alone, the rows $[1, 2, 3]$, $[3, 2, 1]$, and $[0, 0, 5]$ come out as `[0.212 0.576 0.212]`, `[0.721 0.265 0.013]`, and `[0.045 0.045 0.909]`: every row sums to 1 and every row is wrong, the first one being `[0.09 0.245 0.665]` when computed correctly. With `keepdims` missing from the sum as well, the row sums are 0.6, 1.848, and 1.023 and one "probability" is 1.566, which at least looks wrong.
- **`axis=1` is left off as well.** `np.max(inputs)` and `np.sum(exp_values)` then run over the whole batch, which is normalised as if it were one long row. All 12 entries of the 4-row batch sum to 1, and the four row sums are 0.13, 0.13, 0.646, and 0.095. No error is raised.
- **The max subtraction is skipped.** The naive formula works on small logits and fails on the first large one. In `float32` the row $[90, 91, 89]$ already gives `[nan nan nan]`, where the shifted version gives `[0.245 0.665 0.09]`. The only sign is a pair of `RuntimeWarning` lines, and every number computed from a `nan` is a `nan`.
- **A ReLU is placed between the last dense layer and the softmax.** The logits $[-2, -1, -3]$ should give `[0.245 0.665 0.09]`. After a ReLU they are $[0, 0, 0]$ and the output is `[0.333 0.333 0.333]`: the network's preference for the second class is gone. For $[2, -1, -3]$ the correct `[0.946 0.047 0.006]` becomes `[0.787 0.107 0.107]`.
- **A probability underflows to exactly zero.** The table of section 4 says every output is positive, which holds for real numbers and not for floats: the softmax of $[0, -800]$ is `[1. 0.]`, with a second entry equal to 0.0. The loss of post 08 takes the logarithm of a probability, and $\log 0$ is $-\infty$, which is why that post clips its inputs first.
- **A ReLU neuron dies.** A neuron with weights $(1, 1)$ and bias $-3$ has a negative weighted sum on every one of the 300 spiral samples, the largest being $-1.653$, so 0 of its 300 outputs are non-zero. Nothing downstream can tell it apart from a missing neuron, and section 2.2 gives the reason it stays that way. A large negative bias produces one directly, as here; during training, an update that pushes a neuron's weights too far can do the same.
- **A single sample is passed as a 1-D array.** The class reduces along `axis=1`, and a 1-D array has no axis 1: `AxisError: axis 1 is out of bounds for array of dimension 1`. One sample is a batch of one row, shape `(1, 3)`, written with two pairs of brackets.

---

## 8. Summary

| Concept | Takeaway |
|---|---|
| Without activations | Any depth of layers collapses to one linear map |
| ReLU | $\max(0, z)$; cheap, gradient-friendly; the usual hidden activation |
| One kink per neuron | Three ReLU neurons turn the line $x - 2$ into a zigzag with bends at 0, 1, and 2 |
| Softmax | $e^{z_k} / \sum_j e^{z_j}$ per row; outputs are probabilities that sum to 1 |
| Stability trick | Subtract the per-row max before exponentiating; `np.exp` overflows above 88.72 in `float32` and 709.78 in `float64` |
| `axis=1, keepdims=True` | Required so the per-row max and sum broadcast back as columns |
| Untrained output | Rows within 0.00007 of `[1/3, 1/3, 1/3]`, because 0.01-scaled weights give logits below 0.00024 |
| Default pattern | ReLU on every hidden layer; softmax on the output for classification |

---

## Common pitfalls

1. **Using `np.max` instead of `np.maximum`.** `np.max` collapses the array to a scalar; `np.maximum` operates element-wise. ReLU needs the latter.
2. **Omitting `axis=1, keepdims=True` in softmax.** Without `keepdims` the code raises a `ValueError` on most batches and returns wrong probabilities on a square one; without `axis` it normalises the whole batch as one row and raises nothing.
3. **Skipping the max-subtraction trick.** Code that works on small toy logits produces `nan` values the first time a logit passes 88.72 in `float32`. Always subtract the max.
4. **Applying ReLU to the output layer of a classifier.** The output goes to softmax. Inserting ReLU first throws away every negative logit before it can be exponentiated, and the resulting probabilities are skewed towards uniform.
5. **Applying softmax to the input data.** Softmax is for raw logits, not for raw inputs. It keeps only the differences inside each row, so distinct inputs become identical and everything downstream works from damaged data.
6. **Using sigmoid for hidden layers in a deep network.** The sigmoid's slope is at most 0.25 and near zero where it saturates, which stalls training in deep networks. ReLU is the right choice for hidden layers unless there is a specific reason not to use it.

---

## Further reading

- Bridle, J. S., *"Probabilistic Interpretation of Feedforward Classification Network Outputs"* (Neurocomputing, NATO ASI Series, 1990).
- Cybenko, G., *"Approximation by Superpositions of a Sigmoidal Function"* (Mathematics of Control, Signals and Systems, 1989).
- Glorot, X., Bordes, A., and Bengio, Y., *"Deep Sparse Rectifier Neural Networks"* (AISTATS, 2011).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 6, "Deep Feedforward Networks" (MIT Press, 2016).
- Hendrycks, D. and Gimpel, K., *"Gaussian Error Linear Units (GELUs)"* (arXiv:1606.08415, 2016).
- Hochreiter, S., *"Untersuchungen zu dynamischen neuronalen Netzen"* (Diploma thesis, TU München, 1991).
- Hornik, K., *"Approximation Capabilities of Multilayer Feedforward Networks"* (Neural Networks, 1991).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 4 (2020).
- Krizhevsky, A., Sutskever, I., and Hinton, G., *"ImageNet Classification with Deep Convolutional Neural Networks"* (NeurIPS, 2012).
- Leshno, M., Lin, V. Ya., Pinkus, A., and Schocken, S., *"Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function"* (Neural Networks, 1993).
- Nair, V. and Hinton, G. E., *"Rectified Linear Units Improve Restricted Boltzmann Machines"* (ICML, 2010).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 07 - Coding the complete forward pass](../07-coding-the-complete-forward-pass/index.md):** the four objects of section 5 assembled into one script, with the shape of every intermediate array audited against the architecture.
- **[Post 19 - Softmax derivatives and the combined backward pass](../19-softmax-derivatives-and-the-combined-backward-pass/index.md):** the backward pass of the softmax written here, and the cancellation that makes it short when it is paired with the cross-entropy loss of post 08.
