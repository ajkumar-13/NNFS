# 32 - Mini-batching

> **TL;DR.** A mini-batch loop shuffles the rows once per epoch, cuts them into batches of $B$ rows, and runs one forward pass, one backward pass and one update per batch, so an epoch holds $\lceil N/B \rceil$ updates instead of one. The classes of the earlier posts do not change; the training script gains an inner loop, and `Optimizer_Adam` now counts batches, so a decay chosen per epoch has to be chosen again. On the spiral, over ten seeds, batches of 32 beat the full batch in 9 runs of 10 after the same number of epochs and show no reliable order against it after the same number of updates, which they reach with a tenth of the arithmetic.
>
> **Prerequisites:** [Post 22](../22-gradient-descent-optimiser/index.md), [Post 28](../28-generalization-and-testing/index.md).
> **Safe to skip?** Skip it if the reader can already write a shuffled two-loop training script, and can say how many updates an epoch holds and what the optimiser's step counter then counts.
>
> **After reading, you will be able to:**
>
> - Distinguish full-batch gradient descent, true SGD and mini-batch SGD by what each step consumes and the gradient noise that results.
> - Extend the single-loop training pattern into the two-loop epoch-by-batch structure with shuffling.
> - Choose a batch size for a given dataset.

![Three paths cross the same elliptical loss contours from one start to the minimum: a smooth full-batch path of 4 steps, a mildly noisy mini-batch path of 12 steps and a jagged pure SGD path of 30 steps. A table beside them gives steps per epoch for 60,000 samples: 1 at batch size 60,000, 469 at 128 and 60,000 at 1.](diagrams/01-batch-size-trajectories.svg)

*Three regimes on one loss surface. The paths are a schematic, not a run; the counts in the table are closed forms, and the measured noise is in section 2.*

---

## 1. The question: what did the full-batch loop leave out?

Every training loop of posts 22 to 31 has the same shape: one forward pass over all of `X`, one backward pass, one call of the optimiser, repeated 10,001 times. `X` held the 300 points of the spiral, so every row took part in every gradient, and an epoch, one pass over the training set, was one update. This is **full-batch gradient descent**. The question of this post is: **what limits that loop on a larger dataset, and what replaces it?**

`snippets/counts.py` prints the sizes. Two limits appear, and they are not equally binding.

**Memory, on large data.** A full-batch forward pass holds one row of activations per sample. For the 60,000 training images of MNIST that is modest: the inputs take 188.2 MB in float32 and a first hidden layer of 128 neurons another 30.7 MB. For 1,281,167 images of $224 \times 224 \times 3$ values, the size of a common image benchmark, the inputs alone take 771 GB. Whether memory binds depends on the dataset; for MNIST it does not.

**One update per pass, always.** Twenty full-batch epochs are 20 updates, however large the dataset. The released project `nn-p01` trains on MNIST for 20 epochs in batches of 128 rows, which is 469 updates per epoch and 9,380 in all, and its published result is 98.00 percent test accuracy (9,800 of 10,000 images). Each of those updates uses a gradient computed from 128 rows instead of 60,000. Section 2 measures how good such a gradient is, and section 6 measures what the trade does to training.

---

## 2. The three regimes

One number sets the regime: the **batch size** $B$, the number of rows that contribute to each gradient. $N$ is the size of the training set.

| Regime | Batch size $B$ | Updates per epoch | Each update consumes | Gradient |
|---|:---:|:---:|---|---|
| **Full-batch gradient descent** | $N$ | 1 | every row | exact for the training set |
| **Mini-batch SGD** | $1 < B < N$ | $\lceil N / B \rceil$ | $B$ rows | an estimate with sampling noise |
| **True SGD** | 1 | $N$ | one row | the noisiest estimate |

SGD is stochastic gradient descent (post 22). The name first meant $B = 1$; today it is used for any batch drawn at random, and the $B = 1$ case is called true, pure or online SGD. A **mini-batch** is a batch with $1 < B < N$.

![Three cards, each with a bar of the same width for one epoch of 60,000 samples. The full-batch card cuts the bar into 1 batch, the mini-batch card into 469 batches of 128, and the pure SGD card into 60,000 batches of 1. Each card states its weight updates per epoch and a relative gradient noise: 0.004, 0.088 and 1.000.](diagrams/02-same-cost-more-steps.svg)

*Every regime passes over the same rows in an epoch; the number of updates differs. The noise figures on the cards are $1/\sqrt{B}$, which overstates the full-batch case: its sampling noise is zero, as the measurement below shows.*

**What a batch gradient is.** The loss is a mean over rows, so the full-batch gradient $\mathbf{g}$ is the mean of the $N$ single-row gradients. A batch gradient $\mathbf{g}_B$ is the mean of $B$ of them. If the batch is drawn at random, its expectation is $\mathbf{g}$: the estimate is unbiased. Its error has a closed form. With $\sigma^2$ the summed variance of the single-row gradients over the training set, and the $B$ rows drawn without replacement,

$$\mathbb{E} \, \lVert \mathbf{g}_B - \mathbf{g} \rVert^2 = \frac{\sigma^2}{B} \cdot \frac{N - B}{N - 1}$$

The first factor is the familiar one: the variance falls as $1/B$, so the noise, its square root, falls as $1/\sqrt{B}$. The second factor is exact for a finite training set: it is 1 at $B = 1$ and 0 at $B = N$, where the batch is the whole set and nothing is left to chance.

`snippets/gradient_noise.py` checks all of this on the spiral network with fixed weights, in float64. It computes the 300 single-row gradients, then draws 2,000 random batches of each size:

```text
length of the full-batch gradient: 0.4570
largest difference from the full-batch gradient, weighted by batch size: 5.6e-17
spread of the 300 single-row gradients, sigma: 3.0631
   B  measured  sigma/sqrt(B)  with the factor sqrt((N-B)/(N-1))  measured / length of full gradient
   1    3.0115         3.0631                               3.0631        6.59
   8    1.0647         1.0830                               1.0702        2.33
  32    0.5124         0.5415                               0.5126        1.12
 100    0.2489         0.3063                               0.2505        0.54
 128    0.2069         0.2707                               0.2053        0.45
 300    0.0000         0.1768                               0.0000        0.00
```

The second line is the unbiasedness in its exact form: over one epoch of batches of 32, the batch gradients weighted by their sizes average to the full-batch gradient to rounding. The measured noise follows the closed form within 2 percent at every size, and the plain $\sigma / \sqrt{B}$ overstates it by 23 percent at $B = 100$, a third of this small dataset. On these weights a single-row gradient misses the full gradient by 6.59 times the full gradient's own length, and a batch of 32 by 1.12 times: a batch of 32 is still a rough estimate, and going from 32 to 128 rows costs four times the arithmetic for a noise that is 2.5 times smaller. That is the diminishing return behind small batches.

**The cost of an epoch is not the same in practice.** Every regime touches each row once per epoch, so the arithmetic on the rows is the same. The work per update is not: the optimiser touches every parameter once per update, and each update pays Python's overhead. `snippets/timing.py` times one epoch of a $784 \to 128 \to 10$ network on 6,400 random rows. The times vary by machine and by run. In the runs behind this post an epoch at $B = 1$ took on the order of a hundred times as long as a full-batch epoch, and an epoch at $B = 128$ stayed within the order of magnitude of a full-batch one. True SGD is rare in NumPy code for this reason alone.

---

## 3. The two-loop structure

The loop of post 22 gains one inner loop. This is `train()` of `snippets/network.py`:

```python
    n_samples = len(X)
    losses, accuracies = [], []

    for epoch in range(epochs):
        # 1. A new order of the rows for every epoch.
        idx = order(n_samples)
        X_shuf = X[idx]
        y_shuf = y[idx]

        epoch_loss = 0.0
        n_correct = 0
        n_batches = 0

        # 2. One pass over the data, batch_size rows at a time; the last batch may be shorter.
        for start in range(0, n_samples, batch_size):
            X_batch = X_shuf[start:start + batch_size]
            y_batch = y_shuf[start:start + batch_size]

            # 3. Forward, on this batch only.
            dense1.forward(X_batch)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            loss = loss_activation.forward(dense2.output, y_batch)

            predictions = np.argmax(loss_activation.output, axis=1)
            n_correct += int(np.sum(predictions == y_batch))

            # 4. Backward: the gradient of this batch's mean loss.
            loss_activation.backward(loss_activation.output, y_batch)
            dense2.backward(loss_activation.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            # 5. One update per batch.
            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.post_update_params()

            epoch_loss += float(loss)
            n_batches += 1

        losses.append(epoch_loss / n_batches)       # mean of the batch losses, every batch counted once
        accuracies.append(n_correct / n_samples)    # share of the rows predicted correctly when they were seen
```

Steps 3, 4 and 5 are the loop body of post 22 with `X_batch` and `y_batch` in place of `X` and `y`. `Optimizer_Adam` is the class of post 27, unchanged: it reads `layer.dweights` and `layer.dbiases`, which each layer computes from whatever batch it saw last. The division by the number of samples in the combined class (post 19) is a division by the rows of the batch, so every batch gradient is the gradient of that batch's mean loss, whatever its size.

**One shuffle per epoch, of rows and labels together.** `idx` is a random order of the row numbers, and the same `idx` indexes `X` and `y`, so each row keeps its label. Cutting that order into consecutive slices gives batches that are disjoint and cover every row exactly once per epoch. Section 10 runs the loop without the shuffle, and with the labels left behind.

**The inner loop runs $\lceil N / B \rceil$ times.** For $N = 60{,}000$ and $B = 128$ that is 469 updates per epoch, and 20 epochs are 9,380 updates. For the 300 spiral points and $B = 32$ it is 10.

**The last batch is usually shorter.** $60{,}000 / 128 = 468.75$, so the 469th batch holds 96 rows; with 300 rows and $B = 32$ the tenth batch holds 12. The slice `X_shuf[start:start + batch_size]` stops at the end of the array without an error, and every class of the series takes its row count from the array it is given, so the short batch needs no special case. Its gradient is noisier than the others, by the formula of section 2, and its update counts as one update like any other.

**The loop of `nn-p01` is this loop.** The project indexes `X[order[start:start + batch_size]]` instead of making shuffled copies, which gives the same batches without a second copy of the data. It averages its epoch figures in the same way, over a loss that includes its L2 penalty and with dropout active.

---

## 4. The complete loop with Adam: a documented run

The setup is that of [post 28](../28-generalization-and-testing/index.md), section 3, and `snippets/network.py` prints it: `nnfs.init()` once (seed 0, float32, and its own `np.dot`); then `np.random.seed(0)`, `spiral_data(samples=100, classes=3)`, `Layer_Dense(2, 64)`, ReLU, `Layer_Dense(64, 3)`, the combined softmax and loss, weights of `0.01 * randn`, a test set from a second call of `spiral_data`, and `Optimizer_Adam(learning_rate=0.02, decay=1e-5)`. The shuffles draw from the same global stream, after the test set. The run uses `BATCH_SIZE = 32` for 1,000 epochs:

```text
seed 0, 300 rows, BATCH_SIZE = 32, EPOCHS = 1000: 10 batches an epoch, the last of 12 rows
epoch   0 | loss 1.1028 | acc 0.3467 | updates    10 | lr 0.019998
epoch   9 | loss 1.0655 | acc 0.4100 | updates   100 | lr 0.019980
epoch  99 | loss 0.7145 | acc 0.6600 | updates  1000 | lr 0.019802
epoch 499 | loss 0.4364 | acc 0.8233 | updates  5000 | lr 0.019048
epoch 998 | loss 0.3210 | acc 0.8700 | updates  9990 | lr 0.018184
epoch 999 | loss 0.3275 | acc 0.8733 | updates 10000 | lr 0.018182
forward-only after the last update: training loss 0.3025, accuracy 0.8733; test loss 0.5408, accuracy 0.7700
epoch loss over the last 100 epochs: 0.2965 to 0.3972; epoch accuracy: 0.8400 to 0.8833
```

**The epoch figures describe a moving network.** Each batch loss is measured before that batch's update, so the ten losses of an epoch belong to ten different sets of weights. The epoch loss of the last epoch is 0.3275; the loss of the finished network on all 300 rows, measured forward-only as in post 28, is 0.3025. The epoch figures are a cheap running report. A figure that is quoted or compared comes from a forward-only pass.

**The epoch figures jump.** Over the last 100 epochs the epoch loss moves between 0.2965 and 0.3972 and the epoch accuracy between 84.00 and 88.33 percent, although the learning rate barely changes. One run read at one epoch is a draw from that band.

**The epoch loss is the plain mean of the batch losses.** `epoch_loss / n_batches` counts every batch once, so the 12 rows of the short batch weigh as much as 32 rows of a full one. The mean over rows is the mean weighted by batch size. On the trained network, with the rows cut in their stored order and no update in between, the script prints:

```text
batch sizes: [32, 32, 32, 32, 32, 32, 32, 32, 32, 12]
mean of the batch losses 0.2912; weighted by batch size 0.3025; loss of all 300 rows at once 0.3025
```

The stored order makes the short batch 12 rows of one class, which shows the difference at its plainest; in a shuffled epoch it is a random error and not a constant one. The epoch accuracy, `n_correct / n_samples`, is weighted correctly, because it counts rows. When $B$ divides $N$ the two means agree.

**With `batch_size = len(X)` the two loops are the old loop.** `snippets/full_batch.py` trains seed 0 for 10,001 epochs with one batch of 300 rows, once with the rows in stored order (`order=np.arange`) and once shuffled:

```text
the weights of dense1 are equal for 1 updates; after update 2 the largest difference is 1.3e-07 (float32)
rows in          updates  train loss  train acc  test loss  test acc
stored order       10001      0.0806     0.9633     1.1168    0.8233
shuffled order     10001      0.0873     0.9567     0.9908    0.8333
```

In stored order the run is the documented run of post 28 to the digit: 96.33 percent on the training set and 82.33 percent on the test set. Shuffled, the same 300 rows are summed in another order, the rounding differs in the last bit from the second update on, and the remaining updates amplify that into a different network.

---

## 5. The counter counts updates, so the decay is chosen again

`Optimizer_Adam` raises `iterations` once per call of `post_update_params`, which is once per batch. Two things read that counter. The bias correction of post 27 uses it as $t$, and counting updates is what the correction needs: the counter is never reset between epochs. The decay of post 23 uses it too, and that changes the meaning of a decay value:

$$\alpha_t = \frac{\alpha_0}{1 + \text{decay} \cdot t}$$

with $t$ the number of updates so far, in the code `iterations`. The rate has halved after $1/\text{decay}$ updates. Under full-batch training that was $1/\text{decay}$ epochs; with $\lceil N/B \rceil$ updates per epoch it is that many times sooner. `snippets/counts.py` prints the consequence for the settings of post 23, a learning rate of 1.0 and a decay of $10^{-3}$ over 10,001 epochs:

| Batch size | Updates | Decay | Rate in the last update |
|:---:|:---:|:---:|:---:|
| 300 | 10,001 | $10^{-3}$ | 0.0909 |
| 32 | 100,010 | $10^{-3}$ | 0.0099 |
| 32 | 100,010 | $10^{-4}$ | 0.0909 |

A decay that is to end a run at the same rate is divided by the number of updates per epoch, here 10. This is the reason `nn-p01` uses $10^{-4}$: its rate falls from 0.001 to 0.000516 over its 9,380 updates, and with $10^{-3}$ it would end at 0.000096.

`snippets/decay.py` measures what an unchanged decay does, on the setup of section 4 with 1,000 epochs and a deliberately strong decay of $10^{-2}$, which takes the rate to about a tenth after 1,000 full-batch updates. Ranges are over seeds 0 to 4, forward-only after the last update:

| Batch size | Decay | Updates | Last rate | Training loss (mean) | Training accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 300 | $10^{-2}$ | 1,000 | 0.001820 | 0.3778 to 0.7427 (0.5591) | 65.67 to 86.67 |
| 32 | $10^{-2}$ | 10,000 | 0.000198 | 0.8844 to 0.9541 (0.9146) | 50.33 to 63.00 |
| 32 | $10^{-3}$ | 10,000 | 0.001818 | 0.3950 to 0.5213 (0.4790) | 75.33 to 86.00 |

**The unchanged decay made mini-batching worse than the full batch.** With ten times as many updates, the runs with batches of 32 ended at a higher training loss than the full-batch runs on all five seeds, and on 9 of 10 when the script is given the seeds `0 1 2 3 4 5 6 7 8 9`, for which it prints the counts of this paragraph and the next. The rate had fallen to about a tenth after 100 epochs and to about a hundredth by the end.

**The decay chosen again restored the advantage.** With $10^{-3}$ the training loss is lower than with $10^{-2}$ on all ten seeds, and lower than the full-batch run on 9 of 10. The table does not say that $10^{-3}$ is a good decay for this problem: the runs of section 6 use $10^{-5}$ and end lower still, on all five seeds. It says that a decay value is a statement about a number of updates.

---

## 6. What the trade does to training, measured

All runs use the setup of section 4; only the batch size and the number of epochs change. Each row is five seeds, 0 to 4, measured forward-only after the last update. The loss is the mean over seeds; accuracies are ranges in percent with the mean in brackets.

| Batch size | Epochs | Updates | Script | Training loss | Training accuracy | Test accuracy |
|:---:|:---:|:---:|---|:---:|:---:|:---:|
| 300 | 1,000 | 1,000 | `head_to_head.py` | 0.4294 | 73.67 to 91.33 (82.47) | 66.67 to 79.67 (73.33) |
| 100 | 1,000 | 3,000 | `batch_100.py` | 0.4193 | 79.00 to 87.33 (82.47) | 75.67 to 79.67 (76.87) |
| 32 | 1,000 | 10,000 | `head_to_head.py` | 0.2809 | 86.67 to 91.67 (89.07) | 77.00 to 82.00 (79.60) |
| 300 | 10,000 | 10,000 | `head_to_head.py` | 0.1992 | 87.67 to 95.67 (91.27) | 76.00 to 84.00 (80.67) |
| 100 | 3,334 | 10,002 | `batch_100.py` | 0.2358 | 87.33 to 93.67 (90.93) | 78.33 to 83.33 (81.13) |
| 8 | 264 | 10,032 | `batch_8.py` | 0.6766 | 61.67 to 65.67 (63.87) | 54.00 to 63.33 (59.07) |
| 32 | 100 | 1,000 | `head_to_head.py` | 0.7399 | 56.33 to 69.00 (65.60) | 52.33 to 67.33 (60.87) |

Equal epochs is not equal updates, so the table is read three ways.

**The same number of epochs: the smaller batch won.** The first three rows pass over the data 1,000 times each. Batches of 32 make ten times as many updates as the full batch and end at a lower training loss on four of the five seeds and a higher test accuracy on four of the five. `head_to_head.py` trains both for each seed and prints the counts: 9 of 10 for both on the seeds `0` to `9`, and 8 of 10 for both on the seeds `10` to `19`. The same arithmetic on the rows bought a better network.

**The same number of updates, about 10,000: no reliable order between 32 and 300.** The full batch needs 10,000 epochs for them and batches of 32 need 1,000. Seed by seed, the batches of 32 have the lower training loss on 5 of the seeds `0` to `9` and the higher test accuracy on 6; on the seeds `10` to `19` the counts are 2 and 6. With the same number of updates the full batch tends to fit the training set a little better and tests no better, and each of its updates costs about ten times the arithmetic. That, and not the noise, is the measured case for mini-batches on this problem.

**The same number of updates early in training: the exact gradient was worth more.** After 1,000 updates, the first and the last row, the full batch has the lower training loss on 9 of the seeds `0` to `9` and on all of the seeds `10` to `19`. The two findings are consistent: an update on 32 rows is a cheaper, rougher step, and whether ten of them beat one exact step depends on where the run is.

**Batches of 8 failed at this learning rate.** At about 10,000 updates their best training accuracy, 65.67 percent, is below the worst of the batches of 32, 86.67. The last column of the scripts' tables shows the damage: 44 to 52 of the 64 hidden neurons are dead, zero on every training row (post 17, counted as in post 22), against 27 to 38 with batches of 32 and 10 to 26 with the full batch. The length of Adam's step is set by the learning rate and not by the size of the gradient (post 27), so a noisier gradient gives steps of the same length in worse directions, and at a rate of 0.02 they push ReLU neurons to zero on every row, where no gradient reaches them again. With half the learning rate (`batch_8.py`, second table) fewer neurons die on every seed, 32 to 44, and the training accuracy is higher on all five seeds and on 9 of the ten seeds `0` to `9`, at 67.00 to 79.00 percent, still below the batches of 32. A batch size and a learning rate are chosen together.

**Does the noise help generalisation?** Keskar et al. (2017) report, on six networks for image and speech data, that large-batch training reached sharper minima and generalised worse than small-batch training; the finding is theirs and the explanation is debated. The spiral runs neither confirm nor contradict it. At equal updates the test accuracy of batches of 32 is higher on 6 of 10 seeds, which is no order, and at equal epochs it is higher on 9 of 10 together with a lower training loss, so it cannot be told apart from better optimisation. No run in this post isolates an effect of the noise on generalisation.

**The optimiser already averages the noise.** Adam's first moment is a moving average of batch gradients with $\beta_1 = 0.9$. For batches drawn independently at fixed weights its noise is $\sqrt{(1 - \beta_1)/(1 + \beta_1)} = 0.2294$ of one batch's, and `snippets/gradient_noise.py` measures 0.2284 over 20,000 such batches of 32. The batches of a shuffled epoch are not independent: they are disjoint, so their deviations cancel over the epoch, and for them the script measures 0.1388. Either way the step follows a much quieter gradient than one batch gives, bought with memory of old gradients instead of arithmetic.

---

## 7. Choosing a batch size

The batch size is a hyperparameter of the training schedule (post 29), and the measurements above give its two bounds on this problem.

| Constraint | What it says |
|---|---|
| Memory | an upper bound: the activations of one batch have to fit, with the gradients of the backward pass |
| Time per epoch | very small batches pay the per-update overhead $N/B$ times (section 2) |
| Noise | the noise grows as $B$ shrinks, and the usable learning rate shrinks with it (section 6) |
| Updates | the number of updates per epoch falls as $B$ grows; with a fixed epoch budget a large batch takes few steps |

Between the bounds the choice is wide. On the spiral, 32 and 100 were hard to separate at equal updates, and 8 was clearly too small for a rate of 0.02. A reasonable procedure is to start from a customary value, such as the 128 that `nn-p01` uses for 60,000 images, and to compare a factor of two to four on either side on validation data over several seeds, with the learning rate tuned for each.

**Powers of two are a convention.** 32, 64, 128 and 256 are the customary values. Goodfellow, Bengio and Courville (2016, section 8.1.3) note that some hardware, GPUs in particular, runs faster with particular array sizes. No such effect appears in the NumPy code of this series: in the runs of `snippets/timing.py` the time per row at $B = 100$ and at $B = 128$ differed by less than the variation between runs. Nothing in the mathematics prefers 128 to 100, and with $B = 100$ the 60,000 MNIST images give 600 batches with no short one.

**Batch size and learning rate move together.** Goyal et al. (2017) report that scaling the learning rate in proportion to the batch size, with a warm-up, kept the accuracy of an image classifier up to batches of about 8,000, and Masters and Luschi (2018) report their best test results with batches of 32 or fewer. Both are findings on those models; the evidence of this post is that halving the rate helped the batches of 8.

---

## 8. Variants and naming

- **Iteration, step, update.** Three words for one forward pass, one backward pass and one parameter update on one batch.
- **Dropping the last batch.** Some loaders discard a short last batch so that every batch has the same size. The loop of this post keeps it.
- **Sampling with replacement.** Each batch is drawn independently from the whole set, the scheme most of the theory of SGD assumes (Bottou, Curtis and Nocedal, 2018). A row can then appear twice in an epoch or not at all.
- **Gradient accumulation.** The gradients of several small batches are summed before one update, to imitate a batch that memory cannot hold. The layers of this series assign `dweights` in every `backward` call, so the sum would be kept outside them.
- **Evaluation in batches.** A forward-only pass needs no shuffle. A test set that does not fit in memory is passed in slices, and its loss is the mean weighted by slice size (section 4).

---

## 9. Make it run: the scripts

Every code block and every number of this post comes from a script in `snippets/`, run from the series root, for example:

```text
python posts/32-mini-batching/snippets/head_to_head.py
```

| Script | Contents | Time |
|---|---|---|
| `network.py` | the classes, `train()`, the setup; runs the documented run of section 4 and prints the last-batch losses of section 10 | 3 s |
| `counts.py` | the closed forms of sections 1, 3 and 5 | under 1 s |
| `gradient_noise.py` | the batch gradient against the full gradient (sections 2 and 6) | 3 s |
| `timing.py` | the time of one epoch against the batch size (sections 2 and 7) | 20 s |
| `full_batch.py` | one batch of 300 rows, stored order and shuffled (section 4) | 15 s |
| `decay.py` | the decay kept and chosen again, five seeds (section 5) | 35 s |
| `head_to_head.py` | the full batch and batches of 32, five seeds, two epoch counts each, and the seed-by-seed counts (section 6) | 50 s |
| `batch_100.py`, `batch_8.py` | five seeds each, two settings each (section 6) | 20 to 25 s each |
| `no_shuffle.py`, `what_can_go_wrong.py` | the mistakes of section 10 | 15 to 20 s each |

The training scripts take other seeds as arguments, as in `batch_100.py 5 6 7 8 9`; `head_to_head.py`, `decay.py` and `batch_8.py` print their counts for the seeds they are given, ten seeds in about twice the time. All need NumPy; all but `counts.py` and `timing.py` need the `nnfs` package for `spiral_data`. The training scripts call `nnfs.init()` once and reseed with `np.random.seed` for each run. `counts.py`, `gradient_noise.py` and `timing.py` do not call it, and the second of these runs in float64.

---

## 10. What can go wrong?

**No shuffle, on data stored class by class.** `spiral_data` returns the 100 rows of class 0, then class 1, then class 2. `snippets/no_shuffle.py` cuts the batches from that order in every epoch:

```text
batch size 100: [(100, 0, 0), (0, 100, 0), (0, 0, 100)]
```

Each batch holds one class, so each update pulls every output towards one label, and the next update pulls it towards another. No batch gradient resembles the full gradient, and the estimate is no longer unbiased. Over five seeds and 1,000 epochs the training accuracy ends at 44.00 to 60.33 percent with batches of 100 and at 38.00 to 61.67 percent with batches of 32, against 79.00 to 87.33 and 86.67 to 91.67 percent with the shuffle (section 6). The one-class updates leave 48 to 63 of the 64 hidden neurons dead, and nothing is raised. Real datasets are often stored in some order.

**The rows shuffled and the labels not.** `snippets/what_can_go_wrong.py` indexes `X` with `idx` and leaves `y` alone, for 200 epochs:

```text
ln 3 = 1.0986
seed  first epoch loss  last epoch loss  train loss  train acc
   0            1.1962           1.1005      1.1025     0.3333
```

Every row now carries the label of another row. The epoch loss starts above $\ln 3$, the loss of a uniform guess (post 08), is still above it after 200 epochs, and the accuracy is exactly one third, on all five seeds. A loss that stays at $\ln K$ for $K$ classes from the first epoch is the sign of labels that do not belong to their inputs.

**The optimiser created inside the batch loop.** Every batch then meets an `Optimizer_Adam` whose `iterations` is 0:

```text
seed  counter at the end  last rate  train loss  train acc
   0                   1   0.020000      0.4920     0.7700
   1                   1   0.020000      0.2845     0.8933
the corrections with the counter stuck at 0: first moment / 0.1, second moment / 0.001, step x 0.316
```

The correct loop ends at 0.3025 on seed 0 and 0.2417 on seed 1, and the table of `head_to_head.py` for batch size 32 and 1,000 epochs has the lower training loss on each of the five seeds. The run with the mistake still learns, so it is easy to miss. In this series the momentum and cache arrays are stored on the layers (post 27), so they survive the new optimiser. What is lost is the counter: the decay never starts, and the bias correction is applied with $t = 1$ to averages that are long past their start, which scales every step by $10 / \sqrt{1{,}000} = 0.316$. Post 27, section 12, shows the same stuck counter from a missing `post_update_params`. The optimiser is created once, before both loops.

**The last batch's loss read as the epoch's.** A script that prints `loss` after the inner loop prints the loss of the short last batch. For the run of section 4, `snippets/network.py` prints both:

```text
epoch loss over the last 100 epochs: 0.2965 to 0.3972; epoch accuracy: 0.8400 to 0.8833
loss of the last batch (12 rows) over the same 100 epochs: 0.0282 to 0.8937
```

Twelve rows give a figure that swings by a factor of 30 between epochs of a run that is barely changing.

---

## 11. Summary

| Concept | Takeaway |
|---|---|
| Full batch | one update per epoch, exact gradient; memory and the update count limit it |
| Mini-batch | $\lceil N/B \rceil$ updates per epoch, each from $B$ rows; the last batch may be short |
| True SGD | $B = 1$; the noisiest gradient and, in NumPy, the slowest epoch |
| Batch gradient | unbiased if the batch is random; noise $\sigma / \sqrt{B}$ times $\sqrt{(N - B)/(N - 1)}$ |
| Two loops | epochs outside, batches inside; one shuffle per epoch, of rows and labels together |
| What changes | the training script only; layers, loss and optimiser classes are reused |
| Counter | `iterations` counts updates; a decay is divided by the updates per epoch |
| Reported figures | epoch loss and accuracy are running reports; compare forward-only figures |
| Measured, ten seeds | batches of 32 beat the full batch at equal epochs on 9; no reliable order at equal updates; at a rate of 0.02 batches of 8 kill most hidden neurons |

---

## Common pitfalls

1. **Leaving out the shuffle.** On data stored by class every batch holds one class and training stalls without an error.
2. **Shuffling `X` and `y` separately.** One index array orders both. A loss stuck at $\ln K$ is the symptom.
3. **Keeping a decay chosen for full-batch epochs.** The rate now falls $\lceil N/B \rceil$ times as fast per epoch. The decay is divided by the updates per epoch, or tuned again.
4. **Creating the optimiser inside a loop.** Its counter restarts, so the decay never acts and the bias correction is wrong at every step.
5. **Comparing batch sizes at equal epochs only.** Equal epochs is not equal updates; both counts are reported, over several seeds.
6. **Quoting an epoch figure as a result.** It averages over weights that were still moving, and with a short last batch it is not the mean over rows.

---

## Further reading

- Bottou, L., Curtis, F. E., and Nocedal, J., *"Optimization Methods for Large-Scale Machine Learning"* (SIAM Review, 2018).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 8.1.3 (MIT Press, 2016).
- Goyal, P., et al., *"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"* (arXiv:1706.02677, 2017).
- Keskar, N. S., et al., *"On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"* (ICLR, 2017).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 19 (2020).
- Masters, D. and Luschi, C., *"Revisiting Small Batch Training for Deep Neural Networks"* (arXiv:1804.07612, 2018).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 33 - Weight initialisation](../33-weight-initialisation/index.md):** the scale `0.01 * randn` that every layer has used so far, and what replaces it at depth.
- **[Post 23 - Learning-rate decay](../23-learning-rate-decay/index.md):** the schedule whose counter now counts batches, and how its decay was chosen for full-batch epochs.
