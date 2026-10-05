# 08 - Loss: categorical cross-entropy

> **TL;DR.** Categorical cross-entropy scores one sample as the negative natural logarithm of the probability the network gave to the true class, and a batch as the mean of those scores. The score is 0 only when that probability is exactly 1, grows without bound as it approaches 0, and equals $\ln K$ for a uniform guess over $K$ classes, which is why the untrained spiral network of post 07 scores 1.0986. Predictions are clipped to $[10^{-7}, 1 - 10^{-7}]$ before the logarithm so that an exact zero cannot make the loss infinite. Two classes, `Loss` and `Loss_CategoricalCrossentropy`, carry the calculation for integer and one-hot labels, and accuracy is reported beside the loss because it ignores confidence and cannot be differentiated.
>
> **Prerequisites:** [Post 06](../06-activation-functions-relu-and-softmax/index.md), [Post 07](../07-coding-the-complete-forward-pass/index.md).
> **Safe to skip?** Skip it if the reader can already compute the categorical cross-entropy of a batch from integer or one-hot labels, say why an untrained three-class network scores about 1.0986, and say what the clip before the logarithm protects against.
>
> **After reading, you will be able to:**
>
> - Compute the categorical cross-entropy of a batch of softmax outputs against integer or one-hot labels.
> - Explain in one sentence why the negative logarithm is the right function for turning a probability into a loss.
> - State why predictions are clipped to [1e-7, 1 - 1e-7] before the logarithm is taken.
> - Interpret a loss value using the uniform-guess baseline ln K and the accuracy reported beside it.

![A chart of the loss, minus log p, against the probability p given to the correct class. The curve falls from about 4.6 at p = 0.01 to 0 at p = 1, with marked points 2.30 at p = 0.1, 1.20 at 0.3, 0.69 at 0.5, 0.36 at 0.7 and 0.11 at 0.9. A side panel states that the loss is never negative, that it is zero only at p = 1, that p = 0.01 costs about 6.6 times as much as p = 0.5, and that the per-sample loss is averaged over the batch.](diagrams/01-cross-entropy-curve.svg)

*Categorical cross-entropy is the negative log of the probability given to the correct class. The curve is nearly flat close to 1 and steep close to 0, and that shape does the work.*

---

## 1. The question: how wrong is a batch of predictions, as one number?

The forward pass of post 07 turns each of the 300 spiral points into a row of three probabilities. With untrained weights, every row is approximately `[1/3, 1/3, 1/3]`. The forward pass cannot, by itself, tell whether those probabilities are good or bad. Something else has to turn the prediction and the true label into one summary number.

That number is the **loss**. A loss function takes a model's output and the true target and returns one number, lower being better: small when the prediction is good and large when it is bad. Training is the process of adjusting the weights to reduce this number.

Two roles fall out of this definition:

- **The scoreboard.** A loss number is a single comparable measurement: 1.10 is worse than 0.40, and the relative magnitude is meaningful, not just the ordering.
- **The optimisation target.** Training works by computing the gradient of the loss with respect to every weight (post 09 introduces the idea and post 10 the calculus), so the loss has to be differentiable. The number is what gets minimised; the gradient says *how* to minimise it.

For multi-class classification with softmax outputs, the standard loss is **categorical cross-entropy**. This post asks how that loss turns a batch of softmax outputs and its labels into one number, and why the negative logarithm is the function that does it.

---

## 2. Categorical cross-entropy, formally

Take one sample $i$ of a problem with $K$ classes. Its label, written as a one-hot vector $\mathbf{y}_i$, has a 1 at the true class $c_i$ and a 0 everywhere else. Its prediction $\hat{\mathbf{y}}_i$ is one row of the softmax output: $K$ probabilities that sum to 1. The loss of that sample is

$$L_i = -\sum_{k} y_{i,k} \log \hat{y}_{i,k},$$

where the sum runs over the $K$ classes. Throughout the series $\log$ is the natural logarithm, base $e$, which is what `np.log` computes; a quoted value is written with $\ln$, so that $-\log(1/3) = \ln 3$.

Because $\mathbf{y}_i$ is one-hot, every term of the sum where $y_{i,k} = 0$ vanishes. Only the term for the true class survives:

$$L_i = -\log \hat{y}_{i,c_i}.$$

This is the entire formula. The loss of a sample is the negative logarithm of the probability the network assigned to the true class. The probabilities assigned to the wrong classes do not enter the formula directly. They enter through softmax: a row sums to 1, so probability placed on a wrong class is probability taken from the true one, and a confident wrong prediction is the same thing as a small $\hat{y}_{i,c_i}$ and a large loss.

The loss of a batch of $N$ samples is the mean of the per-sample losses,

$$L = \frac{1}{N} \sum_{i} L_i = -\frac{1}{N} \sum_{i} \log \hat{y}_{i,c_i},$$

and this $L$ is the number that is reported and minimised.

### 2.1. Where the formula comes from

The shape is not arbitrary. Categorical cross-entropy is the **negative log-likelihood** of the labels under the model. If a row of the output is read as a probability distribution over the classes, the probability the model gives to the label that was actually observed is $\hat{y}_{i,c_i}$; that number is the likelihood of the sample. For a batch of independent samples the likelihood is the product of the per-sample values, and the logarithm turns that product into a sum:

$$\log \prod_{i} \hat{y}_{i,c_i} = \sum_{i} \log \hat{y}_{i,c_i} = -N L.$$

Maximising the likelihood of the training labels is therefore the same as minimising $L$; the sign is flipped only because an optimiser minimises. The sum is also the form a computer can hold. For 300 samples that each have probability $1/3$ on the true class, the product is $7.3 \times 10^{-144}$, which float32, the number type of this series' scripts, stores as exactly 0, while the sum of the 300 logarithms is $-329.58$ and its negated mean is 1.0986 (`snippets/neg_log_curve.py`).

The name is from information theory. Shannon's entropy of a distribution $p$ (1948) is $-\sum_k p_k \log p_k$; the cross-entropy of a second distribution $q$ measured against $p$ keeps the weights and changes the term inside the logarithm, $-\sum_k p_k \log q_k$. With the one-hot label as $p$ and the prediction as $q$, that is the formula of this section, and "categorical" says that the label is one of $K$ categories.

The pairing with softmax is not a coincidence either. Bridle (1990), the paper that named softmax (post 06), set out the pair: a softmax output trained with a probability score in place of squared error. Minimising cross-entropy over a softmax output is maximum-likelihood estimation under a categorical distribution, and Goodfellow, Bengio, and Courville (*Deep Learning*, sections 5.5 and 6.2.2) give the derivation in full. For this series the practical takeaway is that **softmax on the output layer and cross-entropy as the loss are the default pair for multi-class classification**.

A second property of the pair is worth flagging: their combined derivative is unusually short (post 19). The two functions are not just compatible; differentiated together, they need less work in the backward pass than either one alone.

### 2.2. What this loss is *not*

A boundary section, because the function gets misapplied often.

- **It is not for regression.** Cross-entropy expects probabilities and class labels, not arbitrary real numbers. Regression uses mean squared error, which this series trains with only in its California housing regression project (post 12 borrows the squared error of a single neuron as an example). Nothing in the code stops the mistake: the clip of section 6 forces whatever it is given into the valid range, and the result is a number without meaning. Handed the logits of section 8's network in place of its probabilities, the class of section 7 returns 14.3504 where the loss is 1.0986 (`snippets/pitfalls.py`).
- **It is not the usual choice for binary classification.** With two classes the usual choice is *binary* cross-entropy (post 34), the single-output equivalent with a simpler formula. Categorical cross-entropy still works for two classes, with two softmax outputs; binary cross-entropy is the more economical pairing.
- **It does not produce a gradient on its own.** The class of this post only measures. Post 18 adds the `backward` method that differentiates the loss, and the optimisers from post 22 onward use that derivative to update the weights.
- **It is not the same as accuracy.** Two networks can have the same accuracy (the same count of correct top predictions) and very different losses, because the loss also depends on confidence. Section 8.2 returns to this.

---

## 3. The intuition of the `-log` curve

Tabulating $-\log(p)$ for $p \in (0, 1]$ shows why this particular function is the right loss for a probability:

| Probability $p$ on the true class | $-\log(p)$ | Interpretation |
|:---:|:---:|:---|
| 1.00 | 0.000 | perfect prediction; zero loss |
| 0.90 | 0.105 | confident and right; small loss |
| 0.70 | 0.357 | fairly confident; moderate loss |
| 0.50 | 0.693 | half the probability on the true class; $\ln 2$ |
| 0.10 | 2.303 | a tenth of the probability on the true class; large loss |
| 0.01 | 4.605 | almost nothing on the true class; very large loss |

Two properties hold for every point on the curve:

- **Loss is non-negative.** The probability lives in $(0, 1]$, so $\log(p) \le 0$, so $-\log(p) \ge 0$.
- **Loss is zero only at perfect confidence.** $\log(1) = 0$, so the only way to score zero loss on a sample is to give the true class a probability of exactly 1.

The curve steepens as $p$ approaches zero. A probability of 0.01 on the true class costs 4.605, which is 6.64 times the 0.693 of a probability of 0.5. Raising a probability from 0.01 to 0.10 removes 2.303 of loss; the step of the same size from 0.90 to 0.99 removes 0.095. A loss that fell in a straight line, such as $1 - p$, would pay 0.09 for either step. This uneven charge is the source of the loss's training signal: the samples the network gets most wrong dominate the mean, so the largest corrections go to them. Post 19 computes the gradient that carries those corrections.

In one sentence: $-\log(p)$ is zero when the true class gets probability 1, positive otherwise, unbounded as that probability goes to 0, and it adds over samples in the same way that their probabilities multiply. The table and the figures of this section are printed by `snippets/neg_log_curve.py`.

---

## 4. Worked example

Three samples through a network produce three rows of softmax output. The true labels are class 0, class 1, class 1. For each sample, the loss is the negative logarithm of the probability assigned to that sample's true class:

| Sample | Softmax output | True class | Probability on the true class | Loss $-\log(p)$ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | `[0.7, 0.1, 0.2]` | 0 | 0.70 | 0.357 |
| 2 | `[0.1, 0.5, 0.4]` | 1 | 0.50 | 0.693 |
| 3 | `[0.02, 0.9, 0.08]` | 1 | 0.90 | 0.105 |

The **batch loss** is the mean of the per-sample losses:

$$L = \frac{0.357 + 0.693 + 0.105}{3} = 0.385.$$

Sample 3 has the lowest loss (0.90 on the right class). Sample 2 has the highest loss (only 0.50, even though its top prediction is correct). The wrong-class probabilities in each row never enter the calculation.

---

## 5. Two label formats, two ways to index

Datasets ship class labels in one of two formats. Both formats encode the same information; the implementation has to handle each one differently.

| Format | The labels of section 4 | Shape | Where it comes from |
|---|---|:---:|---|
| **Integer index** | `[0, 1, 1]` | $(N,)$ | one integer per sample; what `spiral_data` returns |
| **One-hot vector** | `[[1,0,0], [0,1,0], [0,1,0]]` | $(N, K)$ | an encoding step, such as `np.eye(3)[class_targets]` |

![Two panels on the worked batch of softmax outputs. Left, integer labels 0, 1, 1: indexing with the row numbers and the labels picks row 0 column 0, 0.70, row 1 column 1, 0.50, and row 2 column 1, 0.90. Right, one-hot labels: the element-wise product turns every wrong-class entry into 0.00 and leaves 0.70, 0.50 and 0.90, which the row sums recover. Both panels end in the same array of 0.7, 0.5 and 0.9.](diagrams/02-indexing-methods.svg)

*Same arithmetic, two paths. Either one ends in the same vector of probabilities on the true class.*

### 5.1. Integer labels: advanced indexing

```python
import numpy as np

softmax_outputs = np.array([[0.7,  0.1, 0.2 ],
                            [0.1,  0.5, 0.4 ],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

correct_confidences = softmax_outputs[
    range(len(softmax_outputs)),
    class_targets
]
print(correct_confidences)
```

`range(len(softmax_outputs))` produces the row indices 0, 1, 2. NumPy's advanced indexing pairs them element-wise with the column indices in `class_targets`:

- row 0, column 0 gives 0.7
- row 1, column 1 gives 0.5
- row 2, column 1 gives 0.9

The result is a 1-D array with one entry per sample, the probability on the true class, which the code calls the correct confidence.

### 5.2. One-hot labels: element-wise multiply, then sum

```python
class_targets_onehot = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]])

correct_confidences = np.sum(
    class_targets_onehot * softmax_outputs,
    axis=1
)
print(correct_confidences)
```

The element-wise product `class_targets_onehot * softmax_outputs` zeros out every wrong-class column. Summing along `axis=1` collapses each row to the single surviving value: the probability on the true class. No `keepdims` is needed here, because the result is not broadcast back against the batch (post 05).

The two paths produce identical results, and the losses of section 4 take two more lines:

```python
sample_losses = -np.log(correct_confidences)
print(sample_losses)
print(np.mean(sample_losses))
```

The three blocks are consecutive lines of `snippets/worked_example.py`, whose first four lines of output are:

```text
[0.7 0.5 0.9]
[0.7 0.5 0.9]
[0.35667494 0.69314718 0.10536052]
0.38506088005216804
```

The implementation in section 7 checks the number of dimensions of the label array and picks the matching path.

One-hot labels cost more memory and one more operation, and they exist because they generalise. A row such as `[0.9, 0.05, 0.05]`, a *soft label*, can say that a sample is mostly class 0; label smoothing and mixup are built on such rows. This series uses hard labels only, and section 9 shows that the multiply-and-sum path is not the right formula for soft ones.

---

## 6. Clipping: never take `log(0)`

A softmax output can be exactly zero in floating point. Post 06 showed the softmax of $[0, -800]$ coming out as `[1. 0.]`. In float32, the type that `nnfs.init()` sets, the exponential underflows to exactly 0 once a logit sits about 104 below the largest logit of its row (about 745 in float64). If that zero is the probability of the true class, the loss is $-\log(0)$: NumPy returns `inf`, and one infinite sample makes the mean of the whole batch infinite.

The standard fix is to **clip** predictions into a safe range before taking the logarithm:

```python
y_pred = np.array([[0.0, 1.0, 0.0],      # probability exactly 0 on the true class
                   [1.0, 0.0, 0.0],      # probability exactly 1 on the true class
                   [0.7, 0.1, 0.2]])     # an ordinary prediction
y_true = np.array([0, 0, 0])

y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
```

This keeps every value in $[10^{-7}, 1 - 10^{-7}]$. `snippets/clipping.py` takes the loss of those three rows without the clip and with it:

```text
1. three predictions for true class 0, without and with the clip
   without: [       inf 0.         0.35667494]  mean: inf
   raised:  RuntimeWarning: divide by zero encountered in log
   with:    [1.61180957e+01 1.00000005e-07 3.56674944e-01]  mean: 5.491590231632352
   the largest loss the clip allows, -log(1e-7): 16.118
   the smallest, -log(1 - 1e-7): 1.000e-07
```

Three things are visible. The lower bound is the one that matters: it replaces an infinite loss by $-\log(10^{-7}) = 16.118$, large enough to mark the sample as badly wrong and finite, so the mean survives. The upper bound turns a loss of exactly 0 into about $10^{-7}$ and protects nothing in this class, because $\log(1) = 0$ is harmless. It is there because the reference implementation (Kinsley and Kukieła) clips both sides by the same amount, so that the clip does not pull the mean towards either end, and it becomes necessary in binary cross-entropy (post 34), where $\log(1 - \hat{y})$ is taken as well and a prediction of exactly 1 is the one that produces $\log(0)$. And the ordinary prediction is untouched, 0.35667494 both times; the worked batch of section 4 has the same mean loss, 0.38506088005216804, with and without the clip.

The value $10^{-7}$ is a convention with one reason behind it. Any small positive lower bound would cap the loss. The upper bound $1 - \epsilon$, though, has to stay below 1 in float32, and $1 - 10^{-8}$ does not: it rounds to exactly 1.0, while $1 - 10^{-7}$ is stored as 0.9999999. So $10^{-7}$ is the smallest power of ten that works at both ends.

The clip belongs to implementations that take the logarithm of a probability. Framework losses such as PyTorch's `CrossEntropyLoss` take the logits instead and compute the logarithm of the softmax in one step; post 19 joins the same two functions for the backward pass.

---

## 7. The two classes: `Loss` and `Loss_CategoricalCrossentropy`

The implementation splits into a base class that handles the batch average and a subclass that computes the per-sample losses.

```python
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss     = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples        = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Integer labels:
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # One-hot labels:
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
```

Three design choices are worth naming.

- **`forward` returns per-sample losses.** One number per sample, shape $(N,)$. This is the array to look at when the mean is bad, because it shows which samples are responsible.
- **`calculate` returns the batch mean.** A single scalar is what the optimiser will minimise, and what gets logged each epoch. The method sits in the base class because every loss needs it, and the name `data_loss` looks ahead to post 30, which adds a second term to the same base class, the regularisation penalty.
- **The label format is detected at run time.** Checking `len(y_true.shape)` lets the caller pass either format without converting beforehand: one dimension selects the indexing path of section 5.1, two dimensions the multiply-and-sum path of section 5.2. The check looks at nothing but the number of dimensions, and section 9 lists what that lets through.

---

## 8. Make it run: the forward pass with a loss and an accuracy

`snippets/forward_pass_with_loss.py` is the script of post 07 with a measurement at the end. The file opens with the three imports of post 07 (`numpy`, `nnfs`, and `spiral_data`), the classes `Layer_Dense`, `Activation_ReLU`, and `Activation_Softmax` as posts 04 and 06 wrote them, and the two loss classes of section 7. The rest of the file is below. It needs NumPy and the `nnfs` package, and it runs in under a second from the series root with `python posts/08-loss-categorical-cross-entropy/snippets/forward_pass_with_loss.py`.

```python
nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1      = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2      = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_fn     = Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_fn.calculate(activation2.output, y)
print(f"Loss: {loss:.4f}")

predictions = np.argmax(activation2.output, axis=1)
accuracy    = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.3f}")

# What stands behind the two numbers.
sample_losses = loss_fn.forward(activation2.output, y)
print("per-sample losses:", sample_losses.shape, "from", sample_losses.min(), "to", sample_losses.max())
print("log(3):", np.log(3))
print("loss with one-hot labels:", loss_fn.calculate(activation2.output, np.eye(3)[y]))
print("correct:", np.sum(predictions == y), "of", len(y))
print("predictions per class:", np.bincount(predictions, minlength=3))
```

**Output:**

```text
Loss: 1.0986
Accuracy: 0.340
per-sample losses: (300,) from 1.0984129 to 1.098722
log(3): 1.0986122886681098
loss with one-hot labels: 1.0986104
correct: 102 of 300
predictions per class: [  8   0 292]
```

The loss of 1.0986 is not a random number. With three classes and an output of $1/3$ per class, the loss is $-\log(1/3) = \ln 3 = 1.0986122886681098$, and the script's float32 value, 1.0986104, agrees with it to five decimals. The 300 per-sample losses that `forward` returns run only from 1.0984 to 1.0987, and the one-hot path gives the same mean as the integer path. The match confirms two things: the softmax is producing the expected near-uniform distribution, and the cross-entropy is computing the right loss from it.

Five more scripts in the same directory produce the numbers of the other sections: `neg_log_curve.py` (sections 2.1, 3, and 8.1), `worked_example.py` (sections 4 and 5), `clipping.py` (section 6), `loss_vs_accuracy.py` (section 8.2), and `pitfalls.py` (sections 2.2 and 9). Each runs the same way in under a second; all need NumPy only, except `pitfalls.py`, which also loads the spiral data from `nnfs`.

### 8.1. Reference numbers: the uniform-guess baseline

| Number of classes $K$ | Loss of a uniform guess, $\ln K$ |
|:---:|:---:|
| 2 | 0.693 |
| 3 | 1.099 |
| 10 | 2.303 |
| 100 | 4.605 |
| 1,000 | 6.908 |

Before training, a network with small random weights should score close to the baseline for its class count; a first loss far from it points at a bug in the forward pass or in the loss, not at an unlucky initialisation.

The baseline is also what makes a later loss readable. A mean loss $L$ corresponds to a typical probability on the true class of $e^{-L}$, the geometric mean of the batch's true-class probabilities: $e^{-1.0986} = 1/3$ for the untrained network, and $e^{-0.385} = 0.680$ for the worked batch of section 4, whose three values were 0.7, 0.5, and 0.9. A loss below $\ln K$ therefore means the network puts, in that sense, more probability on the true class than a uniform guess does, and training pushes the number down from there.

The converse needs care: a loss above the baseline does not show that the top predictions are at chance. Nine predictions of 0.9 on the true class and one of $10^{-6}$ have an accuracy of 0.9 and a loss of 1.476, above $\ln 3 = 1.099$, because the single confident mistake costs 13.8 by itself (`snippets/pitfalls.py`).

### 8.2. Accuracy: the complementary metric

Accuracy counts how often the top prediction is correct, ignoring how confident the network was. The script computes it in two lines. `np.argmax(activation2.output, axis=1)` returns, for every row, the column index of the largest value: the class the network is betting on. `np.mean(predictions == y)` is the fraction of rows where that index equals the true label.

The script prints 0.340: 102 of the 300 predictions are right. That is the score of one fixed answer and not of 300 informed guesses, and post 07 traced the reason: 292 of the predictions are class 2, the other 8 are class 0, and each class holds 100 points. A uniform guess over $K$ balanced classes scores about $1/K$ in accuracy, as it scores $\ln K$ in loss.

| Metric | What it measures | Differentiable? | Used for |
|---|---|:---:|---|
| Categorical cross-entropy | how much probability the network put on the true class | yes | optimisation (training) |
| Accuracy | whether the top prediction matches the truth | no | human-readable reporting |

![Two batches of three softmax rows with true classes 0, 1 and 1. The left batch, the worked batch of section 4, has 0.7, 0.5 and 0.9 on the true classes, per-sample losses 0.357, 0.693 and 0.105, and a mean loss of 0.385. The right batch has 0.95, 0.96 and 0.97 on them, per-sample losses 0.051, 0.041 and 0.030, and a mean loss of 0.041. Both batches show an accuracy of 3 out of 3, 100 percent.](diagrams/03-loss-vs-accuracy.svg)

*Both batches are right on every sample, so accuracy reports 100 percent for each. The whole difference between them lands in the loss.*

Accuracy is coarser. In the figure both batches are right on all three samples, so both have an accuracy of 1.0, while their mean losses are 0.385 and 0.041, a factor of 9.4 (`snippets/loss_vs_accuracy.py`). For a single two-class sample of class 0, the predictions `[0.51, 0.49]` and `[0.99, 0.01]` are both correct by accuracy, and their losses are 0.673 and 0.010. Networks are optimised on loss and reported on accuracy because the two metrics answer two different questions.

Accuracy cannot stand in for the loss during training. A tiny change in the weights either leaves every argmax where it was, and the accuracy does not move at all, or flips one, and the accuracy jumps by $1/N$. A quantity that is flat almost everywhere gives gradient descent nothing to follow; cross-entropy changes smoothly with every probability and is the smooth stand-in that training minimises. In the other direction, accuracy is the number to report: "loss 0.385" means little to a reader outside the field, while "3 right out of 3" needs no explanation. When a loss has to be quoted, it should come with its baseline $\ln K$.

---

## 9. What can go wrong?

- **A zero reaches the logarithm.** Without the clip, a probability of exactly 0 on the true class gives a per-sample loss of `inf` and a batch mean of `inf` (section 6). A version that codes the full sum, `-np.sum(y_true * np.log(y_pred), axis=1)`, fails more easily: a zero on a *wrong* class produces $0 \times (-\infty)$, which is `nan`, on the row `[0.6, 0.0, 0.4]` with true class 0, where the class of section 7 returns 0.5108.
- **The labels have the wrong number of dimensions.** The class chooses between its two paths from `len(y_true.shape)` alone. Integer labels stored as a column of shape $(N, 1)$ count as two-dimensional, take the one-hot path, and broadcast. For the labels 0, 1, 2 on the batch of section 4 the class then returns per-sample losses of `inf`, 0, and $-0.693$, a negative loss, where the correct values are 0.357, 0.693, and 2.526. No error is raised. Flattening the labels with `y.ravel()` restores the right answer.
- **A hand-written path meets the other format.** Outside the class, each path of section 5 fails in its own way when it is given the other kind of label. On the 300 spiral samples, integer labels in the multiply-and-sum path raise `ValueError: operands could not be broadcast together with shapes (300,3) (300,)`, and one-hot labels in the indexing path raise `IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (300,) (300,3)`. With as many samples as classes there is no error at all: on the three samples of section 4 the first mistake returns `[0.3 0.9 0.98]` in place of `[0.7 0.5 0.9]`, and the second returns an array of shape $(3, 3)$. This is the square-array trap of post 05.
- **Soft labels go through the one-hot path.** The multiply-and-sum path computes $-\log \sum_k y_k \hat{y}_k$, which is guaranteed to equal the cross-entropy $-\sum_k y_k \log \hat{y}_k$ only when the label row is exactly one-hot. For the soft label `[0.9, 0.05, 0.05]` against the prediction `[0.7, 0.1, 0.2]`, the class returns 0.4385 and the cross-entropy is 0.5166. Soft labels need the full sum, taken on clipped predictions.
- **The per-sample losses are summed instead of averaged.** The batch of section 4 has a sum of 1.155 and a mean of 0.385; the same batch repeated ten times has a sum of 11.552 and a mean of 0.385. A summed loss grows with the batch size at the same prediction quality, so losses from batches of different sizes stop being comparable, and post 18 shows that the gradient, and with it the usable learning rate, would depend on the batch size as well.
- **The loss is read without its baseline.** A loss of 1.5 is worse than a uniform guess for 3 classes ($\ln 3 = 1.099$) and far better than one for 100 classes ($\ln 100 = 4.605$). A first loss of 1.099 on a three-class problem is not a decent start; it is exactly the uniform guess, and nothing has been learned yet.

`snippets/pitfalls.py` runs every one of these cases and prints the values and error messages quoted here.

---

## 10. Summary

| Concept | Takeaway |
|---|---|
| Loss function | One number that says how wrong the network's predictions are, lower being better; the quantity training minimises |
| Categorical cross-entropy | $L_i = -\log \hat{y}_{i,c_i}$ per sample, because a one-hot label leaves one term of the sum; the batch loss $L$ is the mean |
| The `-log` curve | Zero at a probability of 1, never negative, unbounded as the probability goes to 0 |
| Two label formats | Integer indices use advanced indexing; one-hot rows use element-wise multiply and sum; identical result |
| Clipping | `np.clip(y_pred, 1e-7, 1 - 1e-7)` before `log` caps the per-sample loss at 16.118 in place of `inf` |
| Baseline | A uniform guess over $K$ classes has loss $\ln K$; 1.0986 for the untrained spiral network |
| Loss and accuracy | Train on the loss (differentiable); report accuracy (readable); 1.0986 and 0.340 before training |

---

## Common pitfalls

1. **Taking the logarithm without clipping.** A zero on the true class gives a loss of `inf`, and one infinite sample makes the batch mean `inf`. Clip to `[1e-7, 1 - 1e-7]` first.
2. **Passing integer labels in the wrong shape.** The class tells integer labels from one-hot labels by the number of dimensions alone. A column of integer labels, shape $(N, 1)$, is treated as one-hot and gives wrong losses, some infinite or negative, without an error. Integer labels must have shape $(N,)$.
3. **Summing the per-sample losses instead of averaging.** A batch of 30 should not report ten times the loss of a batch of 3 with the same predictions. Average.
4. **Reading a loss without its baseline.** A loss of 1.5 is worse than a uniform guess for 3 classes and far better than one for 100, and 1.099 on a three-class problem means nothing has been learned. Compare every loss with $\ln K$.
5. **Using the one-hot path for soft labels.** `np.sum(y_pred * y_true, axis=1)` followed by a logarithm is guaranteed to be the cross-entropy only for rows that are exactly one-hot. Soft labels need $-\sum_k y_k \log \hat{y}_k$.
6. **Handing the loss something that is not a probability distribution.** Regression outputs, or the logits in place of the softmax output, raise no error: the clip forces every value into range and the class returns a number with no meaning. Each task has its own loss, and this one takes the softmax output.

---

## Further reading

- Bishop, C. M., *Pattern Recognition and Machine Learning*, section 4.3.4 (Springer, 2006).
- Bridle, J. S., *"Probabilistic Interpretation of Feedforward Classification Network Outputs"* (Neurocomputing, NATO ASI Series, 1990).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, sections 5.5 and 6.2.2 (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 5 (2020).
- PyTorch documentation, `torch.nn.CrossEntropyLoss` (latest).
- Shannon, C. E., *"A Mathematical Theory of Communication"* (Bell System Technical Journal, 1948).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 09 - Introduction to optimisation](../09-introduction-to-optimisation/index.md):** the search for weights that lower this loss, starting from why random search fails.
- **[Post 18 - Backpropagation through the loss function](../18-backpropagation-through-the-loss-function/index.md):** the derivative of categorical cross-entropy with respect to the predictions, where the backward pass starts.
