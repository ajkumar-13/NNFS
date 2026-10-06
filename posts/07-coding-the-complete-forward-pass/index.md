# 07 - Coding the complete forward pass

> **TL;DR.** The three classes built so far (`Layer_Dense`, `Activation_ReLU`, `Activation_Softmax`) wire into a working two-layer classifier with four objects and four `forward` calls. On the 300 spiral points every intermediate array has 300 rows, only a dense layer changes the number of columns, and every entry of the untrained output lies within 0.00007 of $1/3$, because two layers of 0.01-scaled weights shrink the logits to at most 0.00023 in size. That output is the baseline training has to beat: rows that sum to 1, and an accuracy of 0.34 that is no better than answering one class for every point.
>
> **Prerequisites:** [Post 04](../04-dense-layer-class-and-spiral-data/index.md), [Post 06](../06-activation-functions-relu-and-softmax/index.md).
> **Safe to skip?** Skip it if the reader can already wire `Layer_Dense`, `Activation_ReLU`, and `Activation_Softmax` into a forward pass from memory, give the shape of every array in it, and say why its untrained output is close to $1/3$ in every entry.
>
> **After reading, you will be able to:**
>
> - Assemble a complete forward-pass script from the Layer_Dense, Activation_ReLU, and Activation_Softmax classes.
> - Audit the shape of every intermediate array of a forward pass against the network's architecture.
> - Explain why an untrained network gives every class a probability close to one over the number of classes.
> - Check an untrained forward pass against its baseline: rows that sum to 1 and accuracy near one in three.
> - Extend the script to a deeper network by adding Layer_Dense and Activation_ReLU pairs whose shapes chain.

![The forward pass as a chain of five arrays drawn as bands, one column per feature: X with 2 columns, then dense1.output, activation1.output, dense2.output and activation2.output with 3 each, named Z1, A1, Z2 for the logits and y-hat for the probabilities. Between them the objects dense1, computing X W1 plus b1, activation1, taking max of 0 and each entry, dense2, computing A1 W2 plus b2, and activation2, the softmax of each row. Below, a card with the four lines that build the objects and the four forward calls, and the first five rows the script prints: row 0, the origin, exactly 1/3 in every entry, rows 1 to 4 within 0.000002 of 1/3, and every entry of all 300 rows between 0.333297 and 0.333402.](diagrams/01-pipeline.svg)

*One script, four objects, end to end. The loss of post 08 is computed from its output, and the backward pass of posts 12 to 21 walks it in reverse.*

---

## 1. The question: how do the pieces fit together, and what should come out?

The last four posts each added one piece:

- **[Post 03](../03-stacking-layers-and-the-forward-pass/index.md)** chained two `np.dot` calls into a forward pass.
- **[Post 04](../04-dense-layer-class-and-spiral-data/index.md)** wrapped the call in a reusable `Layer_Dense` class and introduced the spiral dataset.
- **[Post 05](../05-array-summation-keepdims-and-broadcasting/index.md)** settled the `axis`, `keepdims`, and broadcasting rules that softmax needs.
- **[Post 06](../06-activation-functions-relu-and-softmax/index.md)** added the `Activation_ReLU` and `Activation_Softmax` classes.

This post puts all of them into one self-contained script and asks two questions of it. How do the three classes fit together, and what shape does every array have along the way? And what should an untrained network print, so that a correct pipeline can be told from a broken one before any training exists?

The second question matters because the numbers the script prints are uninformative on purpose: nothing has been trained yet. What is solid after this post is the forward pass itself, ready to be paired with a loss function in post 08 and with a learning algorithm in the posts after it. The output is not a trained classifier; it is an object with the shape of a classifier, whose outputs are well formed and say nothing yet about the classes. Training will later push that same output into something useful.

In production terms, this is the **inference path** of a classification network: data in, probabilities out, no weight changed. Posts 12 to 27 build the **training path** that adjusts the weights inside `dense1` and `dense2` so that the same forward pass produces useful predictions.

---

## 2. The architecture

The network is small on purpose: one hidden layer of three neurons, then an output layer with one neuron per class.

| Layer | Object | Input shape | Output shape | Role |
|---|---|---|---|---|
| 1 | `dense1` (`Layer_Dense(2, 3)`) | `(N, 2)` | `(N, 3)` | combine the two input features into three hidden units |
| 1 | `activation1` (`Activation_ReLU()`) | `(N, 3)` | `(N, 3)` | inject the non-linearity |
| 2 | `dense2` (`Layer_Dense(3, 3)`) | `(N, 3)` | `(N, 3)` | combine the hidden units into three class logits |
| 2 | `activation2` (`Activation_Softmax()`) | `(N, 3)` | `(N, 3)` | turn logits into a per-row probability distribution |

In the symbols of the notation guide the four steps are $\mathbf{Z}_1 = \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1$, $\mathbf{A}_1 = \text{ReLU}(\mathbf{Z}_1)$, $\mathbf{Z}_2 = \mathbf{A}_1\mathbf{W}_2 + \mathbf{b}_2$, and $\hat{\mathbf{y}} = \text{softmax}(\mathbf{Z}_2)$. Written as one line:

$$\hat{\mathbf{y}} = \text{softmax}\bigl( \text{ReLU}(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\,\mathbf{W}_2 + \mathbf{b}_2 \bigr)$$

ReLU acts on every entry and softmax on every row. $\mathbf{Z}_2$, the output of the last dense layer, holds the **logits**: the raw scores, one per class, that softmax turns into probabilities.

Three hidden neurons are enough to make the function non-linear, and few enough that every array can be printed and checked by eye. They are not enough to separate the spiral classes: three ReLU neurons cut the plane with three straight lines into at most seven pieces, and on each piece the logits are a linear function of the input plus a constant, which is too coarse for three arms that wrap around one another. Post 22 widens the hidden layer to 64 neurons before it trains. The activations hold no parameters, so the network has the 21 numbers that post 04 counted for its two dense layers: 9 in `dense1` and 12 in `dense2`.

Every component is something the previous posts already built. This post calls them in order.

---

## 3. The complete script

The listing is `snippets/forward_pass.py` without its docstring.

```python
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# ============================ Classes (built in posts 04 and 06) ============================

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        shifted       = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values    = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities


# ============================ Data ============================

X, y = spiral_data(samples=100, classes=3)

# ============================ Build the network ============================

dense1      = Layer_Dense(2, 3)         # 2 input features, 3 hidden neurons
activation1 = Activation_ReLU()

dense2      = Layer_Dense(3, 3)         # 3 inputs (hidden), 3 outputs (one per class)
activation2 = Activation_Softmax()

# ============================ Forward pass ============================

dense1.forward(X)                       # step 1: linear
activation1.forward(dense1.output)      # step 2: ReLU
dense2.forward(activation1.output)      # step 3: linear, gives the logits
activation2.forward(dense2.output)      # step 4: softmax, gives the probabilities

# ============================ Inspect ============================

print(activation2.output[:5])
```

**Output:**

```text
[[0.33333334 0.33333334 0.33333334]
 [0.3333332  0.3333332  0.33333364]
 [0.3333329  0.33333293 0.3333342 ]
 [0.3333326  0.33333263 0.33333477]
 [0.33333233 0.3333324  0.33333528]]
```

Slicing with `[:5]` keeps the output readable; the full array has 300 rows, one per spiral point. Every row shown is $[1/3, 1/3, 1/3]$ to four decimal places. The network is producing well-formed probability distributions; it has no opinion yet about which distribution is right.

The first row is exactly uniform. The first spiral point is the origin, the biases are all zero, and an all-zero input then gives all-zero logits whatever the weights are; the softmax of three equal logits is exactly $1/3$ each, which `float32` stores as 0.33333334. The other four rows belong to the next four points along the first arm, and none of their entries is more than two millionths from $1/3$. Section 5 explains why the drift is that small.

Apart from the imports, the `nnfs.init()` call, and the class definitions, the script is ten lines: one for the data, four that build the objects, four `forward` calls, and one `print`. Each object stores its result in its own `output` attribute, and the next call reads it from there. The wiring never calls `np.dot` or `np.exp` directly; the classes hide them. The figure at the top of the post draws this script: the arrays it hands from object to object, its eight lines, and the five rows it prints.

---

## 4. Tracing the shapes

For the standard spiral input (100 samples per class times 3 classes, so 300 rows of 2 features), the figure draws every array of the pass twice, once for all 300 rows and once for the first 7:

![The five arrays of the forward pass drawn as bands, one column per feature. For all 300 spiral points X is (300, 2) and dense1.output, activation1.output, dense2.output and activation2.output are (300, 3) each; between them dense1 with weights (2, 3), activation1 with no weights, dense2 with weights (3, 3) and activation2 with no weights. A bracket marks the four arrays that share the shape (300, 3), so a mix-up between them raises no error. Below, the same pass on the first 7 rows, drawn cell by cell: (7, 2), then (7, 3) four times.](diagrams/02-shape-audit.svg)

*The batch sets the row count at every step. Only a dense layer changes the column count, to the number of its neurons.*

The diary in table form:

| Step | Array | Operation | In shape | Out shape | Notes |
|---|---|---|---|---|---|
| 0 | `X` | input data | none | `(300, 2)` | `spiral_data(samples=100, classes=3)` |
| 1 | `dense1.output` | $\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1$ | `(300, 2)` | `(300, 3)` | $\mathbf{W}_1$ is `(2, 3)`, $\mathbf{b}_1$ is `(1, 3)` |
| 2 | `activation1.output` | $\max(0, \cdot)$ | `(300, 3)` | `(300, 3)` | element-wise; shape unchanged |
| 3 | `dense2.output` | $\mathbf{A}_1\mathbf{W}_2 + \mathbf{b}_2$ | `(300, 3)` | `(300, 3)` | $\mathbf{W}_2$ is `(3, 3)`, $\mathbf{b}_2$ is `(1, 3)` |
| 4 | `activation2.output` | softmax along `axis=1` | `(300, 3)` | `(300, 3)` | per-row normalisation; shape unchanged |

Here $\mathbf{A}_1$ is the ReLU output from step 2 (`activation1.output`), the array fed forward into `dense2`.

`snippets/shape_audit.py` runs the script of section 3 and prints the same diary from the live arrays:

```python
# The audit: one line per array, in the order the forward pass creates them.
steps = [
    ("X",                  X,                  "the input batch"),
    ("dense1.output",      dense1.output,      "X . W1 + b1"),
    ("activation1.output", activation1.output, "max(0, .) on every entry"),
    ("dense2.output",      dense2.output,      "A1 . W2 + b2, the logits"),
    ("activation2.output", activation2.output, "softmax along axis 1"),
]
for number, (name, array, operation) in enumerate(steps):
    print(f"step {number}  {name:<19} {str(array.shape):<9} {operation}")
```

```text
step 0  X                   (300, 2)  the input batch
step 1  dense1.output       (300, 3)  X . W1 + b1
step 2  activation1.output  (300, 3)  max(0, .) on every entry
step 3  dense2.output       (300, 3)  A1 . W2 + b2, the logits
step 4  activation2.output  (300, 3)  softmax along axis 1
parameters
  dense1: weights (2, 3), biases (1, 3), 9 numbers
  dense2: weights (3, 3), biases (1, 3), 12 numbers
  total: 21
rows at every step: [300, 300, 300, 300, 300]
labels y: (300,) one class index per row
a batch of 7 rows: (7, 2) -> (7, 3) -> (7, 3) -> (7, 3) -> (7, 3)
```

Two observations matter for the next several posts.

First, **the batch dimension 300 never changes**. Every step keeps one row per sample, from the input to the probabilities. An array whose first dimension is not the batch size has been transposed, or reduced along the wrong axis, somewhere before it. The last line of the output repeats the pass on the first 7 rows of `X`: only the first number of each shape moves, because the batch decides the row count and the layers decide the column count.

Second, **activation functions never change the shape**. ReLU acts on each entry and softmax normalises each row. The shape changes only at a dense layer, where the new feature count is the layer's neuron count.

The audit has a blind spot: a mistake that leaves every shape intact passes it. In this network four of the five arrays share the shape `(300, 3)`, because the hidden layer happens to be as wide as the number of classes, so a call that is handed the wrong one of them raises no error. A softmax that normalises along the wrong axis keeps its shape as well. Section 9 runs both mistakes. Catching them takes a check on the values, and the next section builds the two that every forward pass should pass.

---

## 5. What the uniform output means

A trained classifier on this dataset would output rows such as `[0.95, 0.03, 0.02]` for a confident class-0 sample and `[0.10, 0.85, 0.05]` for a class-1 sample; those two rows are illustrations, not measurements. The output here is `[0.333, 0.333, 0.333]` for every sample, to three decimal places. Two facts explain it, and `snippets/uniform_baseline.py` measures both.

**The weights are random and small.** `0.01 * np.random.randn(n_inputs, n_neurons)` draws values with a standard deviation of 0.01. The spiral points lie within distance 1 of the origin, so the first layer multiplies inputs no larger than 1 by weights near 0.01, and the largest entry of `dense1.output` is 0.017 in absolute value. After the ReLU, 456 of the 900 hidden values are exactly zero, and the rest have passed through unchanged. The second layer multiplies by weights near 0.01 once more, so the logits are about 75 times smaller again: the largest is 0.00023 in absolute value, and their mean absolute value is 0.000046. A logit of this network is measured in ten-thousandths, not in hundredths; the hundredths belong to the hidden layer.

**Softmax of near-equal logits is near-uniform.** With $K$ classes, $K$ equal logits give exactly $1/K$ each, as post 06 showed, and logits that differ by a few ten-thousandths give almost that. The definition of softmax says by how much. For two classes $a$ and $b$ of the same row, the shared denominator cancels:

$$\frac{\hat{y}_a}{\hat{y}_b} = \frac{e^{z_a}}{e^{z_b}} = e^{z_a - z_b}$$

Only the difference between two logits matters, and a difference near zero gives a ratio near 1. The least uniform of the 300 rows is row 99, the outermost point of the first arm, with logits $-0.0002333$, $-0.0002187$, and $0.0000813$. Its largest logit gap is $0.0003146$, and $e^{0.0003146} = 1.000315$, so its largest probability is only 0.03 percent bigger than its smallest: 0.333402 against 0.333297. Those two are also the extremes of the whole output, in which no probability is further than 0.00007 from $1/3$. The figure draws both facts from the script's arrays.

![Left, a dot plot on a log scale of the size of the entries at each stage, the mean absolute entry as a grey circle and the largest as a green diamond: X, largest 0.97939; dense1.output, 0.01746; activation1.output, 0.01731; dense2.output, the logits, 0.00023. Right, row 99, the least uniform row: logits minus 0.0002333, minus 0.0002187 and 0.0000813, a largest logit gap of 0.0003146, and e to that gap, 1.000315, the largest probability over the smallest; then its probabilities 0.333297, 0.333302 and 0.333402 as three bars on a scale from 0 to 1 that look equal.](diagrams/03-why-uniform.svg)

*Two layers of 0.01-scaled weights shrink the entries to logits of ten-thousandths, and even the least uniform row is 1/3 to three decimal places.*

Both facts turn into checks that the pipeline must pass before any training. The script computes them in three lines:

```python
row_sums    = np.sum(probabilities, axis=1)         # (300,): one sum per row
predictions = np.argmax(probabilities, axis=1)      # (300,): one class index per row
accuracy    = np.mean(predictions == y)             # fraction of rows predicted correctly
```

- **The output is well formed.** Every row must be a probability distribution: positive entries that sum to 1. Here 261 of the 300 row sums are exactly 1, and the other 39 miss by at most $1.2 \times 10^{-7}$, which is `float32` rounding and is normal. With three classes, a row sum that is off by $10^{-5}$ or more is not rounding; it means the softmax is broken. `np.allclose(row_sums, 1)` is the one-line test, and its default tolerance sits at that threshold.
- **The output beats chance by nothing.** The network can already make predictions: the predicted class of a row is the index of its largest probability, which `np.argmax(probabilities, axis=1)` returns for all 300 rows at once. Those predictions are right for 102 of the 300 points, so the accuracy, the fraction of correct predictions, is 0.34. The three classes hold 100 points each, so answering any one class for every point scores exactly 100 of 300. Training must beat this baseline to do anything useful.

The untrained network reaches its 0.34 by doing almost exactly that, and not by guessing at random: 292 of its 300 predictions are class 2 and the other 8 are class 0, as the figure below counts them by true class. The reason is in the nine numbers of `dense2.weights`. In each of its three rows the largest weight sits in column 2, and after ReLU no hidden value is negative, so the class-2 logit is the largest of the three whenever at least one hidden value is positive. The 8 exceptions are the rows whose three hidden values are all zero, the origins of the three arms among them. Their logits are exactly 0, 0, 0, the three probabilities tie, and `np.argmax` returns the first index of a tie, which is 0. Another seed draws other weights and collapses onto another class, or splits its predictions between two classes or among all three, and lands in the same place: over seeds 0 to 199 the script measures a mean accuracy of 0.333, with single runs between 0.24 and 0.42.

![Three stacked horizontal bars, one per true class of 100 points, split by the predicted class: true class 0, 3 predicted as class 0 and 97 as class 2; true class 1, 4 and 96; true class 2, 1 and 99, each split written after its bar as 3 + 97, 4 + 96 and 1 + 99. No point is predicted as class 1. A column on the right counts the correct answers, 3, 0 and 99, 102 of 300, an accuracy of 0.34, and a note says that the 8 points predicted as class 0 have three logits of exactly 0, a tie np.argmax gives to class 0.](diagrams/04-baseline.svg)

*Almost every point gets the same answer, class 2, so the accuracy is what one fixed answer scores: about a third.*

The training loop of post 22 will repeatedly compute this forward pass, measure the loss against the true labels, and adjust the weights to reduce that loss. The arithmetic of the forward pass does not change; the weights do.

---

## 6. What this script is *not*

A boundary section, because the integration is the headline and the limits are easy to overlook.

- **It is not training.** The weights are random and stay random. No gradients are computed. No optimiser runs.
- **It is not a measurement of model quality.** The script prints probabilities and nothing else. Section 5 scores them once by hand; the loss, the number that training will push down, arrives in post 08 together with accuracy as a reported metric.
- **It is not the only architecture this code supports.** The same three classes wire up a feed-forward classifier of any depth. Adding a second hidden layer is one more `Layer_Dense` and `Activation_ReLU` pair in front of the output layer.
- **It is not a final implementation.** Post 16 revisits `Layer_Dense.forward` and adds `self.inputs = inputs` so that the backward pass can use it. The current version is the smallest forward-pass-only form.

---

## 7. Extending the depth

Adding layers requires no new code patterns. `snippets/deeper_network.py` keeps the classes and the data of section 3 and builds two hidden layers of 64 neurons:

```python
dense1      = Layer_Dense(2, 64)
activation1 = Activation_ReLU()

dense2      = Layer_Dense(64, 64)
activation2 = Activation_ReLU()

dense3      = Layer_Dense(64, 3)
activation3 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

print(activation3.output[:5])
```

```text
[[0.33333334 0.33333334 0.33333334]
 [0.33333325 0.33333337 0.33333337]
 [0.33333316 0.33333355 0.33333334]
 [0.33333313 0.3333336  0.33333325]
 [0.33333308 0.3333337  0.33333325]]
shapes
  X                   (300, 2)   largest |entry| 9.8e-01
  dense1.output       (300, 64)  largest |entry| 3.0e-02
  activation1.output  (300, 64)  largest |entry| 3.0e-02
  dense2.output       (300, 64)  largest |entry| 2.1e-03
  activation2.output  (300, 64)  largest |entry| 1.4e-03
  dense3.output       (300, 3)   largest |entry| 4.4e-05
  activation3.output  (300, 3)   largest |entry| 3.3e-01
parameters
  dense1: weights (2, 64), biases (1, 64), 192 numbers
  dense2: weights (64, 64), biases (1, 64), 4,160 numbers
  dense3: weights (64, 3), biases (1, 3), 195 numbers
  total: 4,547
farthest any probability is from 1/3: 1.4e-05
largest |row sum - 1|: 1.2e-07
predictions per class: [  3 191 106]
correct: 84 of 300, accuracy 0.28
```

The same two calls per layer, a dense `forward` and an activation `forward`, repeated. The only constraint is the shape-continuity rule from [post 03](../03-stacking-layers-and-the-forward-pass/index.md): each layer's `n_inputs` must equal the previous layer's `n_neurons`, here 2, then 64, then 64. The audit of section 4 reads the same way as before: 300 rows at every step, and a column count that follows the dense layers from 2 to 64 to 64 to 3.

The untrained output is uniform again, and more closely than before. A third layer of small weights shrinks the logits a third time, to at most 0.000044 in size, and no probability is further than 0.000014 from $1/3$. The 4,547 parameters of this network buy nothing over the 21 of section 3 until they are trained: 84 of the 300 predictions are right, an accuracy of 0.28, inside the range that section 5 measured for the small network across seeds.

Production code typically wraps these calls in a model class that stores a list of layers and walks through them in order. The posts of this series do not build that wrapper; they keep every layer as a named variable so that each call stays visible.

---

## 8. Make it run: five short scripts

Every number and every printed block in this post comes from a script in `snippets/`. Each one needs NumPy and the `nnfs` package (`pip install nnfs`), runs from the series root in about a second on a CPU, and prints the same output on every run, for example `python posts/07-coding-the-complete-forward-pass/snippets/forward_pass.py`.

| Script | Section | What it prints |
|---|---|---|
| `forward_pass.py` | 3 | the first five rows of the output |
| `shape_audit.py` | 4 | the shape of every array, the 21 parameters, and the same pass on a batch of 7 rows |
| `uniform_baseline.py` | 5 | the block below |
| `deeper_network.py` | 7 | the block of section 7: five rows, shapes, 4,547 parameters, 84 of 300 correct |
| `pitfalls.py` | 9 | every error message and wrong result quoted in section 9 |

`uniform_baseline.py` prints:

```text
1. the numbers shrink at every layer
   X                       largest |entry| 0.97939   mean |entry| 0.318836
   dense1.output           largest |entry| 0.01746   mean |entry| 0.004043
   activation1.output      largest |entry| 0.01731   mean |entry| 0.002029
   dense2.output (logits)  largest |entry| 0.00023   mean |entry| 0.000046
   ReLU output: 456 of 900 entries are exactly 0
2. softmax of logits this small
   smallest probability 0.333297, largest 0.333402
   farthest any probability is from 1/3: 6.8e-05
   row 99, the least uniform: logits [-2.3326118e-04 -2.1870324e-04  8.1336591e-05]
   row 99 probabilities: [0.33329675 0.33330163 0.33340165]
   row 99: largest logit gap 0.0003146, exp of that gap 1.000315
   row 99: largest probability over smallest 1.000315
3. every row is a probability distribution
   row sums run from 0.9999999 to 1.0000001
   rows that sum to exactly 1: 261 of 300
   largest |row sum - 1|: 1.2e-07
   np.allclose(row_sums, 1): True
   every probability positive: True
4. predictions and accuracy
   predictions per class: [  8   0 292]
   correct: 102 of 300, accuracy 0.34
   always answering class 0: 100 of 300 correct
   always answering class 1: 100 of 300 correct
   always answering class 2: 100 of 300 correct
5. why class 2 collects the predictions
   dense2.weights, one row per hidden neuron, one column per class:
     [-0.01334258 -0.01346717  0.00693773]
     [-0.00159573 -0.00133702  0.01077744]
     [-0.01126826 -0.00730678 -0.0038488 ]
   column of the largest weight in each row: [2 2 2]
   rows whose three logits are all exactly 0: 8, at indices [  0  62  66 100 116 118 190 200]
   predictions on those rows: [0 0 0 0 0 0 0 0] | true classes: [0 0 0 1 1 1 1 2]
   predictions on the other 292 rows: [2]
6. the same network under seeds 0 to 199
   accuracy: mean 0.333, lowest 0.24, highest 0.42
   rows given to the most-predicted class: mean 192 of 300, fewest 108
```

Block 1 is the first fact of section 5, block 2 the second, and blocks 3 and 4 are the two baseline checks. Block 5 accounts for the split of the predictions: the 292 rows with at least one positive hidden value go to class 2, and the 8 rows of exact ties go to class 0, where 3 of them belong. With the 99 class-2 points that are predicted as class 2, that makes the 102 correct answers. Block 6 repeats the whole experiment under 200 seeds; on average the most-predicted class takes 192 of the 300 rows, so collapsing most of the predictions onto one class is the usual state of this untrained network and not a quirk of seed 0.

---

## 9. What can go wrong?

`snippets/pitfalls.py` makes each mistake below on purpose, on the network of section 3, and prints what NumPy does with it:

```text
1. dense2 fed dense1.output: the ReLU is skipped and nothing complains
   output shape: (300, 3)
   largest difference from the single layer X . (W1 . W2): 1.5e-11
   largest difference from the correct logits: 2.2e-04
   its probabilities, farthest from 1/3: 2.4e-05 | rows sum to 1: True
2. dense2 sized from the wrong neighbour
   ValueError: shapes (300,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)
3. reading the return value of forward
   forward returned: None
   AttributeError: 'NoneType' object has no attribute 'shape'
4. softmax along the wrong axis
   output shape: (300, 3)
   first three row sums:    [0.01000017 0.01000017 0.01000018]
   the three column sums:   [1.0000005 0.9999992 1.0000005]
   np.allclose(row sums, 1): False
5. softmax used as the hidden activation
   hidden rows sum to 1: True
   hidden values run from 0.328 to 0.339
6. np.argmax without axis=1
   np.argmax(activation2.output): 299
   np.unravel_index(299, (300, 3)): (99, 2)
   np.argmax(activation2.output, axis=1): shape (300,) first five [0 2 2 2 2]
   np.argmax of a three-way tie [1/3, 1/3, 1/3]: 0
7. the same seed without nnfs.init()
   without it: float64 data, float64 probabilities, second row [0.33333317 0.33333318 0.33333364]
   with it:    float32 data, float32 probabilities, second row [0.3333332  0.3333332  0.33333364]
   largest difference between the two outputs: 4.6e-08
8. the layers created before the data
   first weight row of dense1, as in section 3: [-0.01306527  0.01658131 -0.00118164]
   first weight row of dense1, created first:   [0.01764052 0.00400157 0.00978738]
   predictions per class, as in section 3: [  8   0 292] | created first: [256   0  44]
   accuracy, created first: 0.34
```

The two differences printed in cases 1 and 7, $1.5 \times 10^{-11}$ and $4.6 \times 10^{-8}$, are rounding residues. Their exact size can differ slightly from one machine to another; their order of magnitude does not.

- **Skipping the activation raises nothing.** `dense2.forward(dense1.output)` in place of `dense2.forward(activation1.output)` runs, because both arrays have shape `(300, 3)`. Without the ReLU the two dense layers collapse into one linear layer, as post 03 showed: the biases are zero here, and the output equals $\mathbf{X}(\mathbf{W}_1\mathbf{W}_2)$ to within $1.5 \times 10^{-11}$, which is rounding. It differs from the correct logits by up to 0.00022, as much as the logits themselves. Neither the shape audit nor the baseline checks see this mistake, since a linear network with small weights also prints a near-uniform output: its rows sum to 1 and none of its probabilities is further than 0.000024 from $1/3$. Comparing the logits with that single-layer product, as case 1 does, is the direct test.
- **A layer sized from the wrong neighbour fails at once.** `Layer_Dense(2, 3)` used as the second layer expects 2 inputs and receives the 3 hidden values, and `np.dot` raises `ValueError: shapes (300,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)`. The message names both shapes. The fix is `Layer_Dense(3, 3)`, whose `n_inputs` is the `n_neurons` of the layer before it.
- **`forward` returns nothing.** Every `forward` in this series stores its result in `self.output` and returns `None`, so `result = activation2.forward(dense2.output)` followed by `result.shape` raises `AttributeError: 'NoneType' object has no attribute 'shape'`. The probabilities are read from `activation2.output`.
- **A softmax along the wrong axis passes the shape audit.** With `axis=0` in place of `axis=1` the output still has shape `(300, 3)`, but now each column sums to 1 and each row sums to about 0.01: a total of 3 shared among 300 nearly identical rows. The row-sum check of section 5 catches it, and `np.allclose(row_sums, 1)` returns `False`.
- **Softmax in a hidden position ties the hidden values together.** Softmax belongs at the very end of a classification network. Used in place of the ReLU it forces every hidden row to sum to 1, so each hidden value depends on the other two and the three of them carry only two independent numbers. Here it also squeezes all of them into the range 0.328 to 0.339.
- **`np.argmax` without `axis=1` returns one number.** `np.argmax(activation2.output)` flattens the array and returns 299, the position of its single largest entry, which is row 99, column 2. The per-row prediction needs `axis=1`, which returns 300 class indices. On an exact tie `np.argmax` returns the first index, which is why the first row, the origin, is predicted as class 0.
- **The numbers differ from the ones printed here.** `nnfs.init()` seeds NumPy's random generator with 0, so the script prints the same numbers on every run; the call belongs directly after the imports. With `np.random.seed(0)` in its place the arrays are `float64` and the probabilities agree with the `float32` ones to within $4.6 \times 10^{-8}$. With no seed at all, the data and the weights change on every run. The order of creation matters as well, as post 04 showed: when both layers are created before the data, `dense1` receives the first six numbers of the random stream, and the predictions split 256, 0, 44 where section 5 found 8, 0, 292. The accuracy is 0.34 in both cases: each run gives most of its rows to one class, and that scores about a third whichever class it is.

---

## 10. Summary

| Concept | Takeaway |
|---|---|
| End-to-end pipeline | `dense1`, ReLU, `dense2`, softmax: four objects and four `forward` calls |
| Shape invariants | the batch dimension never changes; activations keep the shape; a dense layer sets the feature dimension to its neuron count |
| Untrained output | within 0.00007 of $1/3$ in every entry, because the logits are at most 0.00023 in size |
| Baseline checks | every row sums to 1 to within $1.2 \times 10^{-7}$; accuracy 0.34, against 100 of 300 for one fixed answer |
| Encapsulation | the wiring never calls `np.dot` or `np.exp` directly; the classes hide them |
| Extending depth | add `Layer_Dense` and `Activation_ReLU` pairs; no new code patterns, and an untrained output that is still uniform |

---

## Common pitfalls

1. **Feeding `dense2.forward(dense1.output)` instead of `dense2.forward(activation1.output)`.** Skipping the activation removes the non-linearity and collapses the two dense layers back into one. The code still runs; the network is now linear and can do no better on the spiral than a single layer.
2. **Mismatched sizes between layers.** `dense2 = Layer_Dense(2, 3)` after `dense1 = Layer_Dense(2, 3)` raises a `ValueError` at `dense2.forward(activation1.output)`, because the input has shape `(N, 3)`, not `(N, 2)`. The fix is `dense2 = Layer_Dense(3, 3)`.
3. **Forgetting `nnfs.init()`.** Without it, and without another seed, the results vary from run to run, and NumPy computes in `float64` where the printed outputs of this series are `float32`. The forward pass still works; reproducing the printed numbers does not.
4. **Reading the output along the wrong axis.** The output has shape `(N, K)`, where $K$ is the number of classes: each row is one sample's probability distribution, and each column holds one class's probability for all $N$ samples. Predictions come from `np.argmax(..., axis=1)` and row sums from `np.sum(..., axis=1)`; without the axis, or with `axis=0`, both answer a different question.
5. **Putting softmax in the middle of the network.** Softmax belongs at the very end of a classification network. Inserted earlier it makes every hidden row sum to 1, which ties the hidden values to one another and is rarely what is wanted.
6. **Assuming the network is doing something useful.** It is not. An accuracy of 0.34 on three balanced classes is what one fixed answer scores. Until the training loop of post 22, the output is structurally correct and says nothing about the classes.

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 6, "Deep Feedforward Networks" (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 4 (2020). Activation functions, and the Dense, ReLU, Dense, Softmax forward pass on the spiral data.
- NumPy documentation, `numpy.argmax` (latest). The reduction that turns a row of probabilities into a predicted class, and its first-index rule for ties.
- Stanford CS231n, *Convolutional Neural Networks for Visual Recognition*, course notes, ["Putting it together: Minimal Neural Network Case Study"](https://cs231n.github.io/neural-networks-case-study/). A two-layer network on the same kind of spiral data, with the remark that small initial random weights give every class a probability of about one third.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 08 - Loss: categorical cross-entropy](../08-loss-categorical-cross-entropy/index.md):** the loss function that turns this output into one number to minimise, which for a uniform output over three classes is $\ln 3 \approx 1.0986$.
- **[Post 20 - Assembling full backpropagation](../20-assembling-full-backpropagation/index.md):** the same Dense, ReLU, Dense, Softmax pipeline walked in reverse, once every component has a `backward` method.
