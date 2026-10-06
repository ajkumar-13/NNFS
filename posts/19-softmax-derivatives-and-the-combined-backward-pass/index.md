# 19 - Softmax derivatives and the combined backward pass

> **TL;DR.** Every softmax output depends on every logit, so the softmax Jacobian of one sample is a full $K \times K$ matrix with $\hat{y}_k(1 - \hat{y}_k)$ on the diagonal and $-\hat{y}_k \hat{y}_j$ off it. Multiplied by the cross-entropy gradient $-\mathbf{y}/\hat{\mathbf{y}}$ of post 18, the matrix collapses: each division by $\hat{y}_k$ meets a factor $\hat{y}_k$, and what is left is $\partial L / \partial \mathbf{Z} = (\hat{\mathbf{y}} - \mathbf{y})/N$ for one-hot labels and a batch-mean loss. `Activation_Softmax_Loss_CategoricalCrossentropy` computes that in three lines, agrees with the Jacobian route to $10^{-16}$ and with a central difference to $10^{-10}$, and stays finite where the separate route returns `nan`.
>
> **Prerequisites:** [Post 17](../17-backpropagation-through-activation-functions/index.md), [Post 18](../18-backpropagation-through-the-loss-function/index.md).
> **Safe to skip?** Skip it if the reader can already derive the softmax Jacobian, show that it multiplies the cross-entropy gradient into $(\hat{\mathbf{y}} - \mathbf{y})/N$, and code that result for integer and one-hot labels.
>
> **After reading, you will be able to:**
>
> - Explain why softmax's Jacobian is full rather than diagonal.
> - Apply the combined formula (y-hat - y) / N and recognise the cancellation that produces it.
> - Implement Activation_Softmax_Loss_CategoricalCrossentropy with a three-line backward method.

![Two routes for one sample with softmax output 0.7, 0.2, 0.1 and true class 0. Top, the Jacobian route: the row minus y over y-hat, -1.43, 0, 0, times the 3 by 3 softmax Jacobian gives -0.30, 0.20, 0.10; the non-zero entry and the first Jacobian row are outlined, and a worked line shows that row is 0.7 times 0.3, -0.2, -0.1, so the 0.7 cancels the 1/0.7. Bottom, the combined route: y-hat minus the one-hot y, 1, 0, 0, gives the same -0.30, 0.20, 0.10.](diagrams/01-two-routes.svg)

*Both routes give the same row. The combined route never builds the Jacobian and never divides by a prediction.*

---

## 1. The question: does the softmax backward need a matrix?

One `backward` method is still missing after post 18, the softmax's. The loss now produces the first gradient, $\partial L / \partial \hat{\mathbf{y}} = -\mathbf{y} / (N \hat{\mathbf{y}})$, and the softmax has to turn it into $\partial L / \partial \mathbf{Z}$, the gradient with respect to the logits, which the last dense layer receives as its `dvalues`.

Post 17 showed why this step cannot be the one-line multiply of ReLU. For one sample with logits $z_1, \dots, z_K$, softmax is

$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}},$$

and the denominator contains every logit. Post 17 wrote the output as $a_k$; at the output layer it is the prediction $\hat{y}_k$ of post 08. Raising $z_2$ raises the denominator, which changes $\hat{y}_1$, $\hat{y}_3$, and every other output, so the table of local derivatives $\partial \hat{y}_k / \partial z_j$, the Jacobian, has no zero entries. Post 17 quoted those entries and promised a derivation.

This post gives it, and then answers the practical question. Building a $K \times K$ matrix for every sample of the batch works and is wasteful. Softmax is almost always followed by categorical cross-entropy, and differentiating the two together cancels nearly all of the matrix.

---

## 2. The softmax Jacobian

### 2.1. One new derivative: the exponential

Post 10 left the exponential for this post. Its difference quotient is

$$\frac{e^{x + h} - e^{x}}{h} = e^{x} \cdot \frac{e^{h} - 1}{h},$$

and $(e^h - 1)/h$ tends to 1 as $h$ shrinks, which is the property that singles out the base $e$ (the mirror image of the limit post 18 used for the logarithm). So

$$\frac{d}{dx} e^{x} = e^{x}:$$

the slope of the exponential is its own value. A central difference agrees at every point tried: at $x = 1$ it measures 2.718282, which is $e$ (`snippets/softmax_jacobian.py`).

### 2.2. The entries

Taking the logarithm of softmax turns the quotient into a difference:

$$\log \hat{y}_k = z_k - \log \sum_{j} e^{z_j}.$$

Both sides are now differentiated with respect to one logit $z_j$, the others held fixed.

- **Left side.** By the chain rule of post 11 and the derivative of the logarithm from post 18, it is $\dfrac{1}{\hat{y}_k} \dfrac{\partial \hat{y}_k}{\partial z_j}$.
- **First term on the right.** $z_k$ has slope 1 with respect to $z_j$ when $k = j$ and slope 0 otherwise.
- **Second term on the right.** The chain rule again: the outer logarithm gives one over the sum, and inside the sum only $e^{z_j}$ depends on $z_j$, with slope $e^{z_j}$. The product is $e^{z_j} / \sum_{m} e^{z_m}$, which is $\hat{y}_j$.

Multiplying through by $\hat{y}_k$ gives the entries post 17 quoted:

$$\frac{\partial \hat{y}_k}{\partial z_j} = \begin{cases} \hat{y}_k (1 - \hat{y}_k) & k = j \\ -\hat{y}_k \hat{y}_j & k \ne j. \end{cases}$$

The convention is that of post 17: row $k$ is an output, column $j$ is a logit. Every probability lies strictly between 0 and 1, so no entry is zero. The diagonal is positive, since raising a logit raises its own probability, and everything off the diagonal is negative, since that probability is taken from the other classes. This is the whole reason the Jacobian is **full**: an element-wise activation has $\partial a_k / \partial z_j = 0$ for $k \ne j$, and softmax has $-\hat{y}_k \hat{y}_j$ there.

### 2.3. The Jacobian of one sample

For the softmax output $[0.7, 0.2, 0.1]$ the diagonal is $0.7 \times 0.3$, $0.2 \times 0.8$, $0.1 \times 0.9$, and the off-diagonal entries are the negated pairwise products. In code the matrix is `np.diagflat(y_hat) - np.outer(y_hat, y_hat)`, and `snippets/softmax_jacobian.py` prints

```text
[[ 0.210000 -0.140000 -0.070000]
 [-0.140000  0.160000 -0.020000]
 [-0.070000 -0.020000  0.090000]]
largest gap to a central difference: 7.6e-12
column sums: [0.000000 0.000000 0.000000]  symmetric: True
```

All nine entries are non-zero and match slopes measured by nudging one logit at a time. The matrix is symmetric, and each column sums to zero because the three outputs always sum to 1: what one output gains, the others lose.

---

## 3. The cancellation

The backward step of an activation is the incoming gradient times the Jacobian (post 17). For one sample the incoming gradient is that of post 18, $\partial L_i / \partial \hat{y}_k = -y_k / \hat{y}_k$, so

$$\frac{\partial L_i}{\partial z_j} = \sum_{k} \left(-\frac{y_k}{\hat{y}_k}\right) \frac{\partial \hat{y}_k}{\partial z_j}.$$

Every Jacobian entry in row $k$ carries a factor $\hat{y}_k$, and it cancels the division by $\hat{y}_k$. The term with $k = j$ becomes $-y_j (1 - \hat{y}_j)$ and each other term becomes $+y_k \hat{y}_j$:

$$\frac{\partial L_i}{\partial z_j} = -y_j + \hat{y}_j \sum_{k} y_k = \hat{y}_j - y_j,$$

because the entries of a label row sum to 1. Predicted minus true. The exponentials, the logarithm and the division are gone, and so is the sum over $k$: the gradient at logit $j$ needs only the prediction and the label of class $j$.

The script runs the product for $[0.7, 0.2, 0.1]$ with true class 0:

```text
dvalues           = [-1.428571  0.000000  0.000000]
dvalues @ J       = [-0.300000  0.200000  0.100000]
y_hat - y         = [-0.300000  0.200000  0.100000]
```

The single non-zero entry of post 18's gradient, $-1/0.7$, picks out the first row of the Jacobian and rescales it into $\hat{\mathbf{y}} - \mathbf{y}$. The figure at the top of the post draws this product and the subtraction side by side.

For a batch, $L$ is the mean of the $L_i$ and logit $z_{i,k}$ belongs to sample $i$ alone, so the factor $1/N$ of post 18 carries over unchanged:

$$\frac{\partial L}{\partial \mathbf{Z}} = \frac{\hat{\mathbf{y}} - \mathbf{y}}{N}.$$

The division by $N$ happens once. In the Jacobian route it is the one inside `Loss_CategoricalCrossentropy.backward`; the combined route skips that method and has to divide for itself.

### 3.1. Why this is not magic

A second derivation reaches the same place without a Jacobian. With a one-hot label whose true class is $c$, the loss of the sample is $L_i = -\log \hat{y}_c$, and the identity of section 2.2 writes it in the logits directly:

$$L_i = -z_c + \log \sum_{j} e^{z_j}.$$

For $[0.7, 0.2, 0.1]$ both forms give 0.356675. Differentiating with respect to $z_k$ takes two steps:

- The term $-z_c$ contributes $-1$ when $k = c$ and 0 otherwise, which is exactly $-y_k$.
- The term $\log \sum_j e^{z_j}$ contributes $\hat{y}_k$, as in section 2.2.

The sum is $\hat{y}_k - y_k$ again. The Jacobian has not disappeared; its effect is folded into the rewritten loss. The logarithm in the loss undoes the exponential in softmax, and that is why this particular pair cancels and an arbitrary activation and loss do not.

The result reads easily. The entry at the true class is $\hat{y}_c - 1$, negative, and every other entry is $\hat{y}_k$, positive. Gradient descent subtracts the gradient, so the true-class logit is pushed up and the others down, each by an amount proportional to its own error, and a row sums to zero. A prediction that already equals its label has a zero gradient.

### 3.2. Why frameworks ship a combined version

![Two panels. Left, a table of the numbers each route computes for one sample: 9 against 3 at 3 classes, 1,000,000 against 1,000 at 1,000 classes, and 2,500,000,000 against 50,000 at 50,000 classes. Right, the logits 0, -800, 0 with true class 1: the softmax output 0.5, 0, 0.5, the Jacobian route's minus y over y-hat with minus infinity in the middle and the nan row it ends in, and the combined route's finite 0.5, -1, 0.5.](diagrams/02-why-combined.svg)

*The gap between the two routes is a factor of $K$, the number of classes, and only the Jacobian route divides by a probability.*

Two reasons, and both are counted or measured in this post, not timed; the figure above shows one of each.

**Work.** The Jacobian route builds $K^2$ numbers for every sample and multiplies a row into them; the combined route computes $K$. For the spiral's three classes that is 9 against 3. At 1,000 classes it is 1,000,000 against 1,000, and at 50,000 classes, the size of a language-model vocabulary, 2,500,000,000 against 50,000 for each sample. The ratio is $K$ itself.

**No division.** The separate route divides by $\hat{y}$ and then multiplies by it. On paper the two cancel; in floating point they do not when $\hat{y}$ has underflowed to exactly 0, and section 8 shows the separate route returning `nan` where $\hat{\mathbf{y}} - \mathbf{y}$ is finite. Every entry of $\hat{\mathbf{y}} - \mathbf{y}$ lies between $-1$ and 1.

Framework losses go one step further. PyTorch's `CrossEntropyLoss`, for instance, takes the logits and computes the logarithm of the softmax in one step. The form $-z_c + \log \sum_j e^{z_j}$ is what makes that possible: with the max-subtraction of post 06 it needs no probability before the logarithm, and so no clip. The class of this series joins the two functions for the backward pass only: its forward pass still runs softmax and then the clipped loss of post 08. The same cancellation appears once more in the series, for sigmoid with binary cross-entropy (post 34).

---

## 4. Worked example

The batch of post 08, three samples and three classes:

| Sample | Softmax output $\hat{\mathbf{y}}$ | True class | $\mathbf{y}$ (one-hot) | $\hat{\mathbf{y}} - \mathbf{y}$ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | $[0.7, 0.1, 0.2]$ | 0 | $[1, 0, 0]$ | $[-0.3, 0.1, 0.2]$ |
| 2 | $[0.1, 0.5, 0.4]$ | 1 | $[0, 1, 0]$ | $[0.1, -0.5, 0.4]$ |
| 3 | $[0.02, 0.9, 0.08]$ | 1 | $[0, 1, 0]$ | $[0.02, -0.1, 0.08]$ |

Dividing by $N = 3$:

$$\frac{\partial L}{\partial \mathbf{Z}} = \frac{1}{3}\begin{bmatrix} -0.3 & 0.1 & 0.2 \\ 0.1 & -0.5 & 0.4 \\ 0.02 & -0.1 & 0.08 \end{bmatrix} = \begin{bmatrix} -0.100 & 0.033 & 0.067 \\ 0.033 & -0.167 & 0.133 \\ 0.007 & -0.033 & 0.027 \end{bmatrix}.$$

In code (`snippets/worked_example.py`):

```python
import numpy as np

softmax_output = np.array([[0.7,  0.1, 0.2 ],
                           [0.1,  0.5, 0.4 ],
                           [0.02, 0.9, 0.08]])

y_true = np.array([0, 1, 1])   # integer class indices

# Combined backward.
dinputs = softmax_output.copy()
dinputs[range(len(y_true)), y_true] -= 1      # subtract 1 at the true class
dinputs /= len(y_true)                         # divide by batch size

print(dinputs)
```

**Output:**

```text
[[-0.1         0.03333333  0.06666667]
 [ 0.03333333 -0.16666667  0.13333333]
 [ 0.00666667 -0.03333333  0.02666667]]
```

![The three lines of the combined backward, as in the snippet and as in the class of section 5, above the batch they act on: the labels 0, 1, 1; the copy of the softmax output; the copy with 1 subtracted at each true class, the three changed cells outlined; and dinputs after the division by 3, rows -0.100, 0.033, 0.067; 0.033, -0.167, 0.133; 0.007, -0.033, 0.027.](diagrams/03-three-lines.svg)

*Line 2 changes one entry per row, at the true class; line 3 divides every entry by the batch size.*

The figure follows the batch through the three lines. Post 18's gradient had one non-zero entry in each row. This one has none that is zero: the Jacobian has spread each row's single entry over all the logits. Sample 3, the most confident and correct, has the smallest gradient, and sample 2, with only 0.5 on its true class, the largest.

The subtraction uses the advanced indexing of post 08: row $i$, column `y_true[i]`. With integer labels the one-hot matrix is never built, because subtracting $\mathbf{y}$ changes only one entry per row.

---

## 5. The combined class

For completeness, and for the check of section 7, the Jacobian route is a `backward` method on the softmax class of post 06:

```python
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        # One Jacobian and one product for every sample of the batch.
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            jacobian = np.diagflat(single_output) - np.outer(single_output, single_output)
            self.dinputs[index] = single_dvalues @ jacobian
```

It needs a Python loop over the samples, because each row has its own matrix. The series does not use it for training. It uses this class:

```python
class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot, convert to indices.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Three lines: copy, subtract 1 at the true class, normalise.
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples
```

![The combined class between the last dense layer and the loss. Its forward card takes the logits, shape N by K, stores self.output, tagged cached, and returns the loss, one number. A purple arrow carries softmax_loss.output, N by K, down into the backward card as dvalues, and a purple arrow carries dinputs, N by K, left to the dense layer, which reads it as its dvalues. The dense layer appears twice, a forward card and a backward card joined by a dashed edge labelled the same object. A note says neither standalone backward method is called, and a key gives grey for forward, purple for backward and the cached tag.](diagrams/04-combined-class.svg)

*The caller passes the cached `output` back as `dvalues`, and `dinputs` goes straight to the last dense layer.*

Three things to flag, each visible in the figure.

**`forward` is a thin wrapper.** It runs the softmax of post 06 on the logits and the loss of post 08 on the result, stores the probabilities as `self.output`, and returns the batch loss. The reported loss is the number the two separate classes would give.

**`dvalues` is the softmax output.** As in post 18, nothing comes after this class, so its upstream gradient is 1 and the argument carries the predictions: the caller passes `softmax_loss.output`. The copy keeps the subtraction from changing that array.

**`dinputs` is the gradient with respect to the logits.** It has shape $(N, K)$, one row per sample and one column per class, the shape of the logits, and the last dense layer reads it as its `dvalues`. All $K$ columns are needed because column $k$ feeds the weights of output neuron $k$. `Loss_CategoricalCrossentropy.backward` and `Activation_Softmax.backward` are not called at all; calling either as well applies part of the chain twice (section 8).

---

## 6. Label formats

The class works on integer labels, because the subtraction is an indexing operation. One-hot labels are converted first: `np.argmax(y_true, axis=1)` returns the column of the single 1 in each row. This is the reverse of post 18, whose `backward` turned integer labels into one-hot rows for its division. After the conversion `y_true` is a 1-D array of class indices, and the three lines do not branch on the format.

Two inputs pass the check on the number of dimensions and are still wrong. Integer labels stored as an $(N, 1)$ column are taken for one-hot rows, as in posts 08 and 18. Soft labels such as $[0.9, 0.05, 0.05]$ are two-dimensional as well, and `argmax` reduces each to its largest entry. The derivation of section 3 does hold for them, since it needed only a label row that sums to 1, but the code for it is `(dvalues - y_true) / samples`, not an index, and `forward` would have to change with it: the multiply-and-sum path of post 08 reports the cross-entropy for one-hot rows only. The series uses hard labels only; section 8 measures both failures.

---

## 7. Make it run: both routes against a central difference

`snippets/combined_class.py` holds the two `backward` methods of section 5 with the softmax class of post 06 and the loss classes of posts 08 and 18 around them. It needs NumPy only and runs in under a second from the series root with `python posts/19-softmax-derivatives-and-the-combined-backward-pass/snippets/combined_class.py`.

The script draws five rows of four logits and five labels in float64 from a generator with seed 4, chosen so that the five labels cover all four classes. It computes $\partial L / \partial \mathbf{Z}$ three ways: with the combined class, with post 18's loss `backward` followed by the softmax `backward`, and with the central difference of post 10 ($h = 10^{-5}$), which nudges each of the 20 logits in turn and divides the change in the loss returned by `forward` by $2h$.

```text
labels: [1 2 1 3 0]  loss: 1.891199
combined dinputs, shape (5, 4)
[[ 0.012157 -0.180410  0.123154  0.045099]
 [ 0.013429  0.068968 -0.162834  0.080437]
 [ 0.005289 -0.166367  0.033419  0.127659]
 [ 0.021484  0.026081  0.003517 -0.051082]
 [-0.193136  0.140296  0.033517  0.019323]]
row sums: [0.000000 0.000000 0.000000 0.000000 0.000000]
largest |combined - Jacobian route|     = 2.78e-17
largest |combined - central difference| = 2.27e-11
largest |one-hot labels - central difference| = 2.27e-11
section 4 batch, largest |combined - Jacobian route| = 5.55e-17
numbers built: Jacobian route 80  combined 20
```

The two analytic routes differ by rounding in the last bit, on this batch and on the batch of section 4. That is the statement that the shortcut is exact and not an approximation. Both agree with the measured slopes to about $2 \times 10^{-11}$, for integer and for one-hot labels, and since the measurement differentiates the batch mean, it confirms the single division by $N$ as well. The Jacobian route built $5 \times 4^2 = 80$ numbers to get there and the combined route 20.

Three more scripts in the same directory print the other numbers of this post, each in under a second: `softmax_jacobian.py` (sections 2 and 3), `worked_example.py` (section 4), and `what_can_go_wrong.py` (section 8).

---

## 8. What can go wrong?

`snippets/what_can_go_wrong.py` runs each case on the seeded batch of section 7 unless another input is named.

- **The division by $N$ is left out, or done twice.** Without `self.dinputs /= samples` every entry is exactly 5 times the measured slope at $N = 5$. A second division, in a layer or an optimiser, leaves 0.2 times the slope. Neither raises an error or changes the direction of the step.

- **The standalone `backward` methods are called as well.** Passing the combined `dinputs` through `Activation_Softmax.backward` multiplies a finished gradient by the Jacobian a second time. The first row changes from `[0.0122 -0.1804 0.1232 0.0451]` to `[-0.0035 -0.0244 0.0333 -0.0054]`, with two signs flipped, and the largest gap to the central difference is 0.183. The shape is right, so nothing complains.

- **A soft label is passed.** For the label $[0.9, 0.05, 0.05]$ and the prediction $[0.7, 0.1, 0.2]$:

```text
   combined class:       [[-0.3  0.1  0.2]]
   Jacobian route:       [[-0.2   0.05  0.15]]
   central difference:   [[-0.2   0.05  0.15]]
```

The class returns the gradient for the hard label $[1, 0, 0]$. The measured slopes of the cross-entropy $-\sum_k y_k \log \hat{y}_k$ are $\hat{\mathbf{y}} - \mathbf{y}$ with the soft label, which the Jacobian route reproduces because post 18's `backward` keeps the general division.

- **Integer labels arrive as a column.** With the labels `[1 2 1 3 0]` reshaped to $(5, 1)$, `argmax` over a row of one number returns 0 every time, so the 1 is subtracted in column 0 of every row. The largest gap to the central difference is 0.200, which is $1/N$, and no error is raised. `y.ravel()` restores the right answer.

- **A probability is exactly zero.** The logits $[0, -800, 0]$ with true class 1 give the softmax output `[0.5 0. 0.5]`:

```text
   Jacobian route: [[nan nan nan]]
   raised:         RuntimeWarning: divide by zero encountered in divide
   raised:         RuntimeWarning: invalid value encountered in matmul
   combined class: [[ 0.5 -1.   0.5]]
```

The separate route computes $-1/0 = -\infty$ and then multiplies it by Jacobian entries that are 0, which is `nan`. The combined class returns $\hat{\mathbf{y}} - \mathbf{y}$, the correct gradient, and the sample that most needs a correction gets the largest one.

- **The gradient is expected to follow the clipped loss.** For the logits $[-20, 0, 0]$ with true class 0, the true-class probability is $1.031 \times 10^{-9}$, below the lower clip bound of post 08. `forward` reports 16.1181 where the unclipped loss is 20.6931, and the reported number is flat there: its measured slopes are `[0. 0. 0.]`. `backward` returns `[-1. 0.5 0.5]`, the gradient of the unclipped loss. The clip changes the number that is logged and leaves the gradient alone, which is the useful behaviour; a gradient check on such a sample fails for a reason that is not a bug.

- **The formula is used with another loss.** The cancellation belongs to softmax with cross-entropy. For softmax followed by a squared error against the one-hot labels, the measured slopes of the first row are `[-0.0069 -0.0489 0.0666 -0.0108]` and $(\hat{\mathbf{y}} - \mathbf{y})/N$ is `[0.0122 -0.1804 0.1232 0.0451]`: different sizes and, in two places, different signs.

---

## 9. Summary

| Concept | Takeaway |
|---|---|
| Derivative of the exponential | $\frac{d}{dx} e^x = e^x$ |
| Softmax Jacobian | $\hat{y}_k(1 - \hat{y}_k)$ on the diagonal, $-\hat{y}_k \hat{y}_j$ off it; full, symmetric, columns sum to zero |
| Jacobian route | `-y / y_hat`, then one $K \times K$ matrix and one product per sample; $N K^2$ numbers |
| The cancellation | each $1/\hat{y}_k$ meets a factor $\hat{y}_k$, and the label row sums to 1 |
| Combined gradient | $\partial L / \partial \mathbf{Z} = (\hat{\mathbf{y}} - \mathbf{y})/N$; $N K$ numbers, no division by a prediction |
| Second derivation | $L_i = -z_c + \log \sum_j e^{z_j}$, differentiated term by term |
| The class | copy the softmax output, subtract 1 at the true class, divide by $N$ once |
| Exactness | equal to the Jacobian route to rounding; $2 \times 10^{-11}$ from a central difference |

---

## Common pitfalls

1. **Forgetting to divide by `samples`.** The combined class replaces the loss `backward`, which is where the $1/N$ lived. Without it the gradient is $N$ times too large.
2. **Calling the standalone `backward` methods too.** The combined `dinputs` goes straight to the last dense layer. Passing it through the softmax or the loss `backward` gives a wrong gradient of the right shape.
3. **Using $\hat{\mathbf{y}} - \mathbf{y}$ with another activation or loss.** The cancellation is a property of softmax with cross-entropy (and of sigmoid with binary cross-entropy), not of output layers in general.
4. **Passing soft labels or an $(N, 1)$ column.** Both are two-dimensional, both go through `argmax`, and both produce a wrong gradient without an error.
5. **Reading `dvalues` as a gradient.** In this class it is the softmax output; the upstream gradient is 1.
6. **Treating the shortcut as an approximation.** It is the same derivative as the Jacobian route, equal to rounding, and better behaved when a probability underflows.

---

## Further reading

- Bishop, C. M., *Pattern Recognition and Machine Learning*, section 4.3.4 (Springer, 2006).
- Bridle, J. S., *"Probabilistic Interpretation of Feedforward Classification Network Outputs"* (Neurocomputing, NATO ASI Series, 1990).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.2.2.3 (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (2020).
- PyTorch documentation, `torch.nn.CrossEntropyLoss` (latest).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 20 - Assembling full backpropagation](../20-assembling-full-backpropagation/index.md):** every `backward` method of posts 16 to 19 chained into one pass, with this class at its start.
- **[Post 34 - Sigmoid and binary cross-entropy](../34-sigmoid-and-binary-cross-entropy/index.md):** the same cancellation for a single yes-or-no output.
