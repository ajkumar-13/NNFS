# 18 - Backpropagation through the loss function

> **TL;DR.** The backward pass starts at the loss: the gradient of the batch loss $L$ with respect to the predictions $\hat{\mathbf{y}}$ is the first upstream gradient, and every other `backward` consumes something derived from it. For categorical cross-entropy it is the element-wise division $-\mathbf{y} / \hat{\mathbf{y}}$, divided once by the batch size $N$ because $L$ is a mean, so with one-hot labels each row holds a single non-zero entry, $-1/(N \hat{y}_{i,c_i})$. `Loss_CategoricalCrossentropy.backward` computes it in two lines for integer or one-hot labels and agrees with a central difference to better than $10^{-9}$. The division returns `nan` or `-inf` on an exact zero in the predictions, and the clip inside `forward` does not prevent that.
>
> **Prerequisites:** [Post 08](../08-loss-categorical-cross-entropy/index.md), [Post 16](../16-coding-backpropagation/index.md).
> **Safe to skip?** Skip it if the reader can already derive $\partial L / \partial \hat{y}_{i,k} = -y_{i,k} / (N \hat{y}_{i,k})$ from the definition of the loss, code it for both label formats, and say where the $1/N$ comes from.
>
> **After reading, you will be able to:**
>
> - Derive the cross-entropy gradient -y / y-hat from the definition of the loss.
> - Implement Loss_CategoricalCrossentropy.backward for integer and one-hot labels.
> - Explain why the gradient is divided by the batch size.

![Four 3 by 3 grids for the batch of section 3.1. The one-hot labels divided element by element by the predictions 0.7, 0.2, 0.1; 0.1, 0.6, 0.3; 0.2, 0.3, 0.5, and negated, give each sample's gradient: -1.429, -1.667 and -2.000 on the true classes and 0 elsewhere. Divided by N = 3 they give -0.476, -0.556 and -0.667. Below, the two lines of backward that do the two steps, and the third sample worked out: -1 / (3 times 0.5) = -0.667.](diagrams/01-cross-entropy-backward.svg)

*The backward pass starts at the loss. With one-hot labels a single entry per row survives the division; the rest are zero.*

---

## 1. The question: where does the first gradient come from?

The classification pipeline of post 07, with the loss of post 08 on the end, is

$$\text{inputs} \to \text{Dense} \to \text{ReLU} \to \text{Dense} \to \text{Softmax} \to \hat{\mathbf{y}} \to \text{cross-entropy} \to L.$$

Backpropagation walks this pipeline from right to left. Post 16 gave `Layer_Dense` and `Activation_ReLU` a `backward(dvalues)` method, and each of them needs to be handed `dvalues`, the gradient of the loss with respect to its own output. Something has to produce the first such gradient without being handed one. That is the loss: its local derivative, $\partial L / \partial \hat{\mathbf{y}}$, is the upstream gradient of the softmax, and everything further left receives something derived from it. If it is wrong, every weight update in the network is wrong.

This post derives $\partial L / \partial \hat{\mathbf{y}}$ for categorical cross-entropy and adds it to the loss class of post 08 as a `backward` method.

---

## 2. The gradient, from the definition

Post 08 defined the loss of one sample $i$, with one-hot label row $\mathbf{y}_i$ and predicted probabilities $\hat{\mathbf{y}}_i$ over $K$ classes, as

$$L_i = -\sum_{k} y_{i,k} \log \hat{y}_{i,k}.$$

The partial derivative with respect to one prediction $\hat{y}_{i,k}$ holds the other predictions fixed (post 10). Only one term of the sum contains $\hat{y}_{i,k}$, and its label $y_{i,k}$ is a constant factor, so the whole calculation is the derivative of a logarithm.

### 2.1. The derivative of the logarithm

For $f(x) = \log x$, the difference quotient of post 10 is

$$\frac{\log(x + h) - \log x}{h} = \frac{\log(1 + h/x)}{h} = \frac{1}{x} \cdot \frac{\log(1 + u)}{u}, \qquad u = \frac{h}{x}.$$

The first step is the rule of post 08, that the logarithm of a product is the sum of the logarithms, read backwards: a difference of two logarithms is the logarithm of the quotient, here $(x + h)/x = 1 + h/x$. The second writes $h$ as $u x$. As $h$ shrinks, so does $u$, and $\log(1 + u) / u$ tends to 1: near 1 the natural logarithm rises at exactly the rate of its argument, which is the property that singles out the base $e$. The limit is therefore

$$\frac{d}{dx} \log x = \frac{1}{x}, \qquad \frac{d}{dx} \bigl[-\log x\bigr] = -\frac{1}{x}.$$

A central difference agrees: at $x = 0.3$ the measured slope of $-\log x$ is $-3.33333$, which is $-1/0.3$ (`snippets/what_can_go_wrong.py`, part 2). This is the slope of the curve in post 08's hero figure: nearly flat close to 1 and steep close to 0.

### 2.2. The gradient of one sample

Applying the rule to the one term that contains $\hat{y}_{i,k}$:

$$\frac{\partial L_i}{\partial \hat{y}_{i,k}} = -\frac{y_{i,k}}{\hat{y}_{i,k}}.$$

In array form the row of partial derivatives is the element-wise division $-\mathbf{y}_i / \hat{\mathbf{y}}_i$. Two facts follow from the label being one-hot.

**Only the true-class entry is non-zero.** $y_{i,k}$ is 1 at the true class $c_i$ and 0 everywhere else, so every other entry of the row is $-0 / \hat{y}_{i,k} = 0$.

**The non-zero entry is $-1/\hat{y}_{i,c_i}$.** It is negative, because raising the probability on the true class lowers the loss, and its size grows without bound as that probability falls: $-1.43$ at 0.7, $-10$ at 0.1, $-100$ at 0.01. A confident wrong prediction is a small $\hat{y}_{i,c_i}$, so it sends the largest signal.

For the label $[1, 0, 0]$ and the prediction $[0.7, 0.2, 0.1]$:

$$\frac{\partial L_i}{\partial \hat{\mathbf{y}}_i} = -\frac{[1, 0, 0]}{[0.7, 0.2, 0.1]} = [-1.429,\ 0,\ 0].$$

The row has the shape of the prediction and one meaningful number. The zeros do not say that the wrong-class probabilities are free to take any value. The derivation treated the $K$ predictions as independent variables, and they are not: a softmax row sums to 1. That coupling belongs to the softmax, and its backward pass (post 19) is where the single entry is spread over all $K$ logits.

---

## 3. Batch behaviour

Post 08 defined the loss of a batch of $N$ samples as the mean of the per-sample losses,

$$L = \frac{1}{N} \sum_{i} L_i.$$

The prediction $\hat{y}_{i,k}$ appears in $L_i$ and in no other sample's loss, so differentiating the mean leaves one term and its factor $1/N$:

$$\frac{\partial L}{\partial \hat{y}_{i,k}} = \frac{1}{N} \frac{\partial L_i}{\partial \hat{y}_{i,k}} = -\frac{y_{i,k}}{N \, \hat{y}_{i,k}}.$$

The gradient is an $(N, K)$ array, the shape of $\hat{\mathbf{y}}$. Each row depends on its own label and its own prediction only, and every row carries the same factor $1/N$.

### 3.1. Worked batch example

```python
y_true = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

y_pred = np.array([[0.7, 0.2, 0.1],
                   [0.1, 0.6, 0.3],
                   [0.2, 0.3, 0.5]])

N = len(y_pred)
dinputs = (-y_true / y_pred) / N
print(dinputs)
```

**Output:**

```text
[[-0.47619048  0.          0.        ]
 [ 0.         -0.55555556  0.        ]
 [ 0.          0.         -0.66666667]]
```

Each row has one non-zero entry, $-1/(N \hat{y}_{i,c_i})$: $-1/(3 \times 0.7)$, $-1/(3 \times 0.6)$, and $-1/(3 \times 0.5)$. The third sample has the smallest probability on its true class and the largest gradient. These are the numbers of the figure at the top of the post.

---

## 4. Integer labels

The formula is written for one-hot labels, and `spiral_data` returns integer class indices (post 08, section 5). The conversion is one line:

```python
y_true_indices = np.array([0, 1, 2])

n_labels = y_pred.shape[1]                  # number of classes
y_true_onehot = np.eye(n_labels)[y_true_indices]
```

`np.eye(n)` is the $n \times n$ identity matrix. Indexing it with an array of integers picks one row per label, and row $c$ of the identity is the one-hot vector of class $c$, so the result is the $(N, K)$ one-hot matrix; here it is the `y_true` of section 3.1. The method below checks the number of dimensions of `y_true`, as `forward` does: one dimension is converted, two are taken as one-hot.

---

## 5. The complete `backward` method

The class of post 08 keeps its `forward` and gains a `backward`:

```python
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels  = len(dvalues[0])

        # Integer labels become one-hot rows.
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # The gradient of each sample's loss, then the 1/N of the batch mean.
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
```

The two assignments are sections 2 and 3: the division $-\mathbf{y} / \hat{\mathbf{y}}$, then the factor $1/N$. The result is stored as `self.dinputs`, the gradient with respect to the input of the loss, which is the softmax output; the softmax's `backward` receives it as its `dvalues`. Nothing is cached in `forward`, because both things the derivative needs arrive as arguments. The figure below puts the class into the pipeline of section 1, with what reaches it from each side.

![Top, the classifier of section 1 as a row of cards, Dense, ReLU, Dense, Softmax and Loss, with forward arrows left to right and backward arrows right to left; the loss sends dinputs to the softmax, and to its right a dashed box holds nothing. Below, the loss class enlarged: the predictions y-hat of shape (N, K) and y_true arrive from the left at forward and at backward, where y-hat fills the dvalues slot, self.dinputs of shape (N, K) goes back to the left as the softmax's dvalues, the per-sample losses leave to the right, and where an upstream gradient would arrive there is nothing, because dL/dL = 1.](diagrams/02-where-backprop-starts.svg)

*The loss is the one class with nothing after it, which is why its `dvalues` is not a gradient.*

**`dvalues` here is the prediction array.** Everywhere else in the series `dvalues` is the gradient arriving from the next component. The loss has no next component. Its upstream gradient is $\partial L / \partial L = 1$, so the chain-rule multiplication by the upstream is a multiplication by 1 and is left out, and the argument slot carries the predictions that the local derivative needs. The name is kept so that every `backward` in the series has the same first argument.

**`backward` does not clip.** The `forward` of post 08 clips a copy of the predictions and leaves the caller's array as it was, so the array handed to `backward` is the unclipped softmax output and the division sees whatever zeros it contains. Section 8 measures what happens then.

---

## 6. Why divide by the batch size

The division by `samples` is the derivative of the `np.mean` in `Loss.calculate`. `forward` returns $N$ per-sample losses, `calculate` averages them into the one number that is reported and minimised, and the backward pass has to differentiate that number, so the $1/N$ of the mean appears in the gradient.

The division happens in this method and nowhere else. The dense layer of post 16 turns its `dvalues` into parameter gradients by summing over the rows (`np.dot(self.inputs.T, dvalues)` and `np.sum(dvalues, axis=0, keepdims=True)`), and a sum over $N$ rows that each carry $1/N$ is an average over the batch. `snippets/gradient_check.py` repeats the batch of section 3.1 ten and a hundred times, which changes $N$ and nothing else:

```text
N =   3  loss 0.5202  entry [0, 0]: -0.476190  summed over rows with 1/N: [-0.4762 -0.5556 -0.6667]  without: [-1.43 -1.67 -2.  ]
N =  30  loss 0.5202  entry [0, 0]: -0.047619  summed over rows with 1/N: [-0.4762 -0.5556 -0.6667]  without: [-14.29 -16.67 -20.  ]
N = 300  loss 0.5202  entry [0, 0]: -0.004762  summed over rows with 1/N: [-0.4762 -0.5556 -0.6667]  without: [-142.86 -166.67 -200.  ]
```

The loss is 0.5202 at every size, as a mean should be. A single entry of the gradient shrinks in proportion to $1/N$, and the sum over the rows, which is what reaches the parameters, does not move. Without the division that sum grows in proportion to $N$: 100 times larger at 300 samples than at 3. The figure below draws the class-0 column of the three lines on log axes.

![A chart on log axes of the class-0 column of the gradient against the batch size N = 3, 30 and 300, sizes drawn since every value is negative. Summed over the rows with the 1/N it is 0.476 at every N, a flat line; a single entry falls as 0.476, 0.0476, 0.00476; summed without the 1/N it rises as 1.43, 14.29, 142.86.](diagrams/03-batch-size.svg)

*With the 1/N the sum that reaches the parameters is flat; without it the sum grows in step with the batch.*

The summed version is the gradient of the summed loss, so it points in the same direction, and a learning rate $N$ times smaller would produce the same step. The cost is that the learning rate then means something different at every batch size. This is the dependence that post 08 said a summed loss would bring.

---

## 7. Make it run: the backward method against a central difference

`snippets/loss_backward.py` holds the code blocks of sections 3 to 5 in order, with the two classes of post 08 around the new method. It needs NumPy only and runs in under a second from the series root with `python posts/18-backpropagation-through-the-loss-function/snippets/loss_backward.py`. Its last lines call the class on the batch of section 3.1 with integer labels and then with one-hot labels:

```text
loss: 0.5202159160882228
[[-0.47619048 -0.         -0.        ]
 [-0.         -0.55555556 -0.        ]
 [-0.         -0.         -0.66666667]]
same array from one-hot labels: True
shape: (3, 3)  non-zero entries per row: [1 1 1]
```

The array is that of section 3.1 for both label formats. The `-0.` entries are the floating-point negative zero, the result of negating the float zeros that `np.eye` produces; it compares equal to 0.

A `backward` method is a claim about a derivative, and post 10 gave the test: a central difference with $h = 10^{-5}$ in float64. `snippets/gradient_check.py` draws five rows of four probabilities from a seeded generator, nudges each of the 20 predictions up and down by $h$ in turn, with the other 19 held fixed, and divides the change in `loss_fn.calculate` by $2h$:

```text
labels: [2 0 2 2 3]  loss: 1.362287
integer labels: largest |backward - central difference| = 8.07e-10
one-hot labels: largest |backward - central difference| = 8.07e-10
sample 0, true class 2: prediction 0.202537, backward -0.987476, central difference -0.987476
1/N left out: backward / central difference = [5. 5. 5. 5. 5.]
```

The largest disagreement over the 20 entries is $8 \times 10^{-10}$ for either label format. For sample 0 the value is $-1/(5 \times 0.202537) = -0.987476$. The check differentiates `calculate`, the mean, so it also tests the $1/N$: with the division left out, every true-class entry is $N = 5$ times the measured slope, to the four decimals printed. The script runs in under a second, as does `snippets/what_can_go_wrong.py`, which prints the numbers of the next section.

---

## 8. What can go wrong?

`snippets/what_can_go_wrong.py` calls the method on four inputs it was not written for. Its first three parts print:

```text
1. exact zeros in the predictions
   forward, per sample: [3.56674944e-01 1.00000005e-07 1.61180957e+01]
   backward: [[-0.47619048 -0.         -0.        ]
   backward:  [        nan         nan -0.33333333]
   backward:  [       -inf -0.         -0.        ]]
   raised:   RuntimeWarning: divide by zero encountered in divide
   raised:   RuntimeWarning: invalid value encountered in divide
   clipped:  [[      -0.4762        0.            0.    ]
   clipped:   [       0.            0.           -0.3333]
   clipped:   [-3333333.3333        0.            0.    ]]
2. what the clip does to the gradient: one sample, true class 0
   prediction 0.3: loss 1.2040, its slope by central difference -3.33333, backward -3.33333, clipped backward -3.33333
   prediction 1e-09: loss 16.1181, its slope by central difference 0, backward -1e+09, clipped backward -1e+07
3. a soft label
   backward:                       [[-1.2857 -0.5    -0.25  ]]
   slopes of the cross-entropy:    [[-1.2857 -0.5    -0.25  ]]  value 0.5166
   slopes of what forward returns: [[-1.3953 -0.0775 -0.0775]]  value 0.4385
```

- **A prediction is exactly zero.** Post 08 showed that a softmax output can underflow to exactly 0. In part 1 the second row has zeros on its two wrong classes and the third has a zero on its true class. `forward` returns three finite losses, because it clips. `backward` divides by the raw array: a zero on a wrong class is $-0/0$, which is `nan`, and a zero on the true class is $-1/0$, which is `-inf`. NumPy raises warnings, not errors, and a `nan` contaminates every sum it enters, so it spreads into the parameter gradients. Clipping in `forward` does not protect `backward`.
- **The clip is added to `backward`.** Dividing by `np.clip(dvalues, 1e-7, 1 - 1e-7)`, the bounds of post 08, turns the same two rows into finite numbers, the `clipped` lines of part 1. Part 2 puts three quantities side by side for one sample. An ordinary prediction is untouched. Below the lower bound the three quantities part. The clipped loss is flat at 16.118 there, so its slope is 0; the unclipped formula returns $-10^{9}$; the clipped division returns $-10^{7}$, which is $-1/10^{-7}$. The clip therefore caps the size of the gradient at $10^{7}/N$ per entry and keeps its sign. It is a guard, not the derivative of the clipped loss. The series does not rely on it: the combined class of post 19 removes the division altogether.
- **A soft label is passed.** Part 3 takes the label $[0.9, 0.05, 0.05]$ and the prediction $[0.7, 0.1, 0.2]$ of post 08. `backward` uses the general form $-\mathbf{y} / \hat{\mathbf{y}}$, so all three entries are non-zero and they are the measured slopes of the cross-entropy $-\sum_k y_k \log \hat{y}_k$. They are not the slopes of the number `forward` reports, because post 08's multiply-and-sum path computes $-\log \sum_k y_k \hat{y}_k$, which is guaranteed to equal the cross-entropy only for one-hot rows. With soft labels this class would report one function and descend another. The series uses hard labels only.
- **Integer labels arrive as a column.** Labels of shape $(N, 1)$ have two dimensions, are taken as one-hot, and broadcast. For the labels 0, 1, 2 on the batch of section 3.1 the method returns a first row of zeros, a second row of $-3.333, -0.556, -1.111$ and a third of $-3.333, -2.222, -1.333$, with no error (part 4 of the script); `y.ravel()` restores the array of section 3.1.
- **The division by $N$ is missing, or done twice.** Leaving it out makes every entry $N$ times too large (section 7). Adding a second division in a layer or an optimiser, on the grounds that the gradient should be averaged, makes every parameter gradient $N$ times too small: the first entry of section 3.1 would be $-0.1587$ in place of $-0.4762$. Neither mistake raises an error or changes the direction of the step, so each shows up only as a learning rate that seems to need retuning whenever the batch size changes.

---

## 9. Summary

| Concept | Takeaway |
|---|---|
| Where the backward pass starts | $\partial L / \partial \hat{\mathbf{y}}$ is the first gradient; the loss computes it without receiving one |
| Derivative of the logarithm | $\frac{d}{dx}[-\log x] = -1/x$ |
| Cross-entropy gradient | $\partial L_i / \partial \hat{y}_{i,k} = -y_{i,k} / \hat{y}_{i,k}$, an element-wise division |
| One-hot labels | One non-zero entry per row, $-1/\hat{y}_{i,c_i}$ at the true class |
| Batch mean | $L$ is a mean, so the gradient is divided by $N$ once, in the loss |
| Two label formats | Integer labels become one-hot rows with `np.eye(labels)[y_true]` |
| Exact zeros | `forward` clips a copy; `backward` divides by the raw predictions and returns `nan` or `-inf` |

---

## Common pitfalls

1. **Forgetting the division by $N$.** The gradient is then $N$ times too large and the usable learning rate depends on the batch size.
2. **Dividing by $N$ a second time.** The loss already averages; the layers that receive its gradient only sum over the rows.
3. **Assuming the clip in `forward` protects `backward`.** It clips a local copy. An exact zero in the predictions still gives `nan` or `-inf` in the gradient.
4. **Reading `dvalues` as a gradient in the loss class.** It holds the predictions; the upstream gradient of the loss is 1.
5. **Passing integer labels as an $(N, 1)$ column.** The method treats them as one-hot and returns a wrong array without an error. Integer labels must have shape $(N,)$.
6. **Treating the zeros in each row as a bug.** One non-zero entry per row is correct for one-hot labels; the softmax backward of post 19 spreads it over all the logits.

---

## Further reading

- Bishop, C. M., *Pattern Recognition and Machine Learning*, section 4.3.4 (Springer, 2006).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, sections 6.2.2 and 6.5 (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (2020).
- Murphy, K. P., *Probabilistic Machine Learning: An Introduction*, chapter 10 (MIT Press, 2022).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 19 - Softmax derivatives and the combined backward pass](../19-softmax-derivatives-and-the-combined-backward-pass/index.md):** the softmax Jacobian, and the cancellation that turns this division into $(\hat{\mathbf{y}} - \mathbf{y})/N$.
- **[Post 34 - Sigmoid and binary cross-entropy](../34-sigmoid-and-binary-cross-entropy/index.md):** the same derivation for the two-term loss of a yes-or-no output.
