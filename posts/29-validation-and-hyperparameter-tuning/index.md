# 29 - Validation and hyperparameter tuning

> **TL;DR.** A hyperparameter is chosen by a score on data the weights never saw, and a number that helped to choose can no longer serve as the report: the three-way split gives the choosing to a validation set and keeps the test set for one final measurement. On the spiral, choosing the best of eight learning rates by test accuracy inflated the reported accuracy by an average of 0.05 to 1.8 percentage points on test sets of 300 and of 2.9 to 5.0 points on test sets of 60, over five seeds, while a choice made on a separate set left the average test number within 0.21 points of the truth. With only 300 points, 5-fold cross-validation lets every point validate once, and its mean moved 2.6 times less than a single split under reshuffling. That was enough to rule out learning rates of 0.5 and 1.0 and a hidden width of 16 in every seed, and not enough to separate 0.05 from 0.1.
>
> **Prerequisites:** [Post 28](../28-generalization-and-testing/index.md).
> **Safe to skip?** Skip it if the reader can already write a k-fold split that validates every sample exactly once, say what a test accuracy is worth after it has been used to pick a learning rate, and place a preprocessing step correctly inside a cross-validation loop.
>
> **After reading, you will be able to:**
>
> - State why the test set is touched exactly once and what reusing it for tuning does to the reported number.
> - Implement k_fold_split in NumPy so that every sample is validated exactly once.
> - Compare hyperparameter candidates by their k-fold scores and read the result against its spread.
> - Recognise the common forms of data leakage and state the preprocessing rules that prevent them.

![Two panels. On the left one bar is cut into 60 percent training, 20 percent validation and 20 percent test data, with a card for the role of each. On the right five rows show five equal folds, a different one marked as the validation slot in each row, with an accuracy per row and their mean. A timeline below: train, tune, open the test set once.](diagrams/01-three-way-split-and-kfold.svg)

*The three-way split on the left, the 5-fold rotation on the right. The fold accuracies drawn in the figure are placeholders; the measured ones are in section 5.*

---

## 1. The question: how can a search leave a test number worth trusting?

A neural network has two kinds of numbers: **parameters**, which gradient descent learns (every weight and bias of every layer), and **hyperparameters**, which someone must choose before training starts. Hyperparameters are the dials that backpropagation cannot turn.

| Category | Examples |
|---|---|
| Architecture | number of hidden layers, neurons per layer, choice of activation |
| Optimiser | algorithm (Part VI), learning rate $\alpha$, momentum, decay |
| Regularisation | strength of an L1 or L2 penalty (post 30), dropout rate (post 31) |
| Training schedule | number of epochs, batch size (post 32) |
| Initialisation | scale of the random draw (post 33) |

Each choice is a hypothesis: this combination will generalise well on this data. [Post 28](../28-generalization-and-testing/index.md) showed that such a hypothesis can only be tested on data the model never saw during training. A search tests many of them, and that raises the question of this post: **when a held-out score has been used to pick the winner of a search, what is left to report as the winner's accuracy on unseen data?**

Not that same score. Every held-out score is the model's true accuracy plus the luck of the particular points it was measured on, and picking the highest of several scores picks, in part, the luckiest. The set used for choosing must therefore be a different set from the one used for the final report. That is the entire reason validation data exists.

---

## 2. The three-way split

When data is plentiful, the discipline is simple: three disjoint slices.

| Slice | Typical share | Role | When it is touched |
|---|:---:|---|---|
| **Training** | 60 to 80 percent | the weights and biases are learned from it | every forward and backward pass |
| **Validation** | 10 to 20 percent | hyperparameters are chosen by comparing validation scores | between experiments, freely |
| **Test** | 10 to 20 percent | one final report of generalisation | exactly once, after every design decision is frozen |

The workflow has four steps:

1. Train every candidate (a setting of the hyperparameters) on the training set.
2. Evaluate each candidate on the validation set, forward only, and choose the one with the highest validation accuracy or the lowest validation loss. The selection logic is the same for either score, though the two need not pick the same candidate: in post 28 the lowest held-out loss and the highest held-out accuracy fell at different epochs.
3. Optionally retrain the winning configuration on the union of training and validation data.
4. Evaluate that final model on the test set and report the number. Do not go back and tune again.

Step 4 is the rule most often broken. The moment a disappointing test number leads to a change of the model, the test set has been used for tuning, and it has become a second validation set.

`snippets/test_set_tuning.py` measures what that costs. For each of five seeds it trains eight candidates, learning rates from 0.01 to 0.1, on the same 300 spiral points (the training setup is stated in section 5), and draws 200 test sets of 300 points each. On every test set it does what the rule forbids: it picks the candidate with the highest test accuracy and reports that accuracy. A further 30,000 points, which no choice ever looks at, say how good the chosen model really is.

```text
seed 3  accuracy on the 30,000 further points, per candidate: [0.732 0.75  0.767 0.754 0.681 0.742 0.762 0.759]
        200 test sets of 300: chosen by test accuracy, mean (test - further) +0.0182, positive on 169, negative on 31
        chosen on one set, reported on the next: mean (test - further) +0.0011
chosen by test accuracy, 300 points      over the five seeds: +0.0005 to +0.0182, mean +0.0085
chosen by test accuracy, 60 points       over the five seeds: +0.0286 to +0.0502, mean +0.0392
chosen on one set, reported on the next  over the five seeds: -0.0021 to +0.0015, mean -0.0002
```

The reported number is too high on average in every one of the five seeds: by 0.05 to 1.8 percentage points when the test set has 300 points, and by 2.9 to 5.0 points when it has 60. The third line of the summary is the three-way split: the candidate is chosen on one set, which plays the validation set, and reported on the next, which plays the test set. Averaged over the 200 sets, that report is off by 0.21 points at most, in either direction.

The sizes follow from the noise. A model with a true accuracy of 0.8 scores $0.8 \pm \sqrt{0.8 \cdot 0.2 / 300} \approx 0.8 \pm 0.023$ on 300 points and $0.8 \pm 0.052$ on 60. In seed 3 five of the eight candidates lie between 0.750 and 0.767, closer together than that noise, so the choice among them is made almost entirely by luck, and the inflation is the largest of the five seeds. In seed 1 one candidate is clearly ahead of the rest (0.839 against at most 0.797 on the further points), and the inflation almost vanishes. Tuning on the test set costs most when the candidates are close and the test set is small, and, because the largest of more noisy numbers is larger, when the candidates are many. At 300 points the bias is real and small: its mean of 0.85 points over the five seeds lies well inside the 2.3 points by which a single test accuracy scatters, and in every seed the report fell below the truth on 31 to 96 of the 200 test sets.

---

## 3. k-fold cross-validation

The 300 points of the spiral leave a validation slice of 20 percent with only 60 points, and the previous section put the noise of a 60-point accuracy at about 0.05. Two candidates that differ by less cannot be told apart. The standard remedy is **k-fold cross-validation**: the validation slot rotates through the data.

The recipe with $k = 5$:

1. Shuffle the data once and cut it into 5 parts of equal size, the folds $A, B, C, D, E$.
2. For each fold $i$ from 1 to 5: train a new model from scratch on the other four folds, and record its accuracy $E_i$ on fold $i$.
3. The mean $\bar{E} = \frac{1}{5} \sum_{i=1}^{5} E_i$ is the score of the candidate.

| Fold | Validation part | Training parts | Validation accuracy |
|:---:|:---:|:---:|:---:|
| 1 | A | B, C, D, E | $E_1$ |
| 2 | B | A, C, D, E | $E_2$ |
| 3 | C | A, B, D, E | $E_3$ |
| 4 | D | A, B, C, E | $E_4$ |
| 5 | E | A, B, C, D | $E_5$ |

Two properties make this work.

**Every sample takes one turn in the validation slot.** Across the five runs every sample is validated exactly once, by a model that did not train on it, so $\bar{E}$ is an accuracy over all 300 points and not over 60.

**The mean moves less than a single fold.** If the five $E_i$ were independent, averaging them would shrink the standard deviation by $\sqrt{5} \approx 2.24$. They are not independent, since any two of the five models share three of their four training folds, so the factor has to be measured. `snippets/steadiness.py` keeps the data, the learning rate (0.1) and the initial weights fixed and changes only the shuffle, eight times:

```text
one split of 60 (40 values):  0.600 to 0.883, standard deviation 0.064
5-fold mean (8 values):       0.723 to 0.800, standard deviation 0.025
ratio of the standard deviations 2.59   sqrt(5) = 2.24
```

One and the same setting scores anything from 0.600 to 0.883 on a single split of 60, depending on which 60 points are held out. The 5-fold mean stays between 0.723 and 0.800. The ratio of 2.59 comes from eight means and is itself rough, and it counts only the noise of the shuffle: all eight means are scored on the same 300 points, and no reshuffling removes the luck of having drawn those 300.

The price is $k$ trainings per candidate instead of one. The common choices are $k = 5$ and $k = 10$; Kohavi (1995) compared cross-validation and the bootstrap on real datasets and recommended stratified 10-fold cross-validation for model selection.

Cross-validation takes the place of the validation set only. The test set stays outside the rotation, untouched.

---

## 4. A minimal k-fold implementation

The whole logic is a dozen lines of NumPy:

```python
def k_fold_indices(n, k=5, shuffle=True, seed=0):
    """Yield (train_idx, val_idx) for each of the k folds of n samples."""
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)    # a generator of its own, shuffled once

    folds = np.array_split(idx, k)                  # sizes differ by at most one
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def k_fold_split(X, y, k=5, shuffle=True, seed=0):
    """Yield (train_X, train_y, val_X, val_y) for each fold."""
    for train_idx, val_idx in k_fold_indices(len(X), k, shuffle, seed):
        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]
```

Four design choices are worth noting.

**The shuffle happens once, before the cut.** The folds are slices of one shuffled index array, so they cannot overlap. A new shuffle for every fold would let one sample be validated twice and another never.

**The shuffle has its own generator and a fixed seed.** `spiral_data` and the initial weights of `Layer_Dense` draw from NumPy's global random stream. `np.random.default_rng(seed)` is separate from it, so calling the split changes no later draw, and a second call with the same seed returns the same folds. Every candidate can then be scored on identical folds; without that, one candidate might win because its folds were easier.

**`np.array_split` handles a remainder.** When $k$ does not divide $N$, the first $N \bmod k$ folds get one sample more. A fold size of `len(X) // k` loses the remainder (section 10).

**The functions use `yield`.** The $k$ splits are produced one at a time, never all held in memory at once.

`snippets/kfold.py` checks the properties that section 3 relies on:

```text
n=300 k=5  validation sizes [60, 60, 60, 60, 60]  every sample validated once: True  largest train/validation overlap: 0  train + validation = n: True
n=300 k=7  validation sizes [43, 43, 43, 43, 43, 43, 42]  every sample validated once: True  largest train/validation overlap: 0  train + validation = n: True
same seed, same folds: True   other seed, same folds: False
global random stream untouched by the split: True
classes per validation fold, shuffled: [[18, 24, 18], [22, 19, 19], [17, 16, 27], [21, 21, 18], [22, 20, 18]]
```

---

## 5. Comparing learning rates with k-fold

One function scores one candidate. It builds a new network for every fold, with `train` (1,000 full-batch epochs of the Adam optimiser of [post 27](../27-adam-optimiser/index.md)) and `accuracy` (the forward-only pass of post 28), both defined in `snippets/kfold.py`:

```python
def k_fold_accuracies(X, y, learning_rate, n_neurons=64, k=5, seed=0):
    """The k validation accuracies of one candidate, on folds and initial weights fixed by seed."""
    fold_accs = []
    for tr_X, tr_y, va_X, va_y in k_fold_split(X, y, k=k, seed=seed):
        model = train(tr_X, tr_y, learning_rate, n_neurons, seed=100 + seed)   # a new network every fold
        fold_accs.append(accuracy(model, va_X, va_y))
    return np.array(fold_accs)
```

The search is a loop over the candidates around it (`snippets/search.py`, with `CANDIDATES = [0.01, 0.05, 0.1, 0.5, 1.0]`):

```python
    for lr in CANDIDATES:
        fold_accs = k_fold_accuracies(X, y, lr, seed=seed)
        results[lr] = fold_accs
        if verbose:
            print(f'lr={lr:<6} mean_acc={np.mean(fold_accs):.3f} '
                  f'std={np.std(fold_accs):.3f}  folds {np.round(fold_accs, 3)}')
```

The setup, stated once for every training run of this post: `nnfs.init()` is called (float32 arrays and a float32 `np.dot`); for seed $s$ the data are `spiral_data(samples=100, classes=3)` after `np.random.seed(s)`, and a second call of `spiral_data` on the same stream draws the test set; the network is `Layer_Dense(2, 64)`, ReLU, `Layer_Dense(64, 3)` and the combined softmax and loss, with the classes of posts 16 and 19 unchanged; the weights are drawn after `np.random.seed(100 + s)`, so every fold, and every candidate of the same width, starts from the same weights; the optimiser is `Optimizer_Adam(learning_rate=lr)` with no decay; and the folds come from `np.random.default_rng(s)`. This is a shorter and plainer run than the 10,001 epochs with decay of post 27, so the accuracies are not comparable with that post's, nor with those of post 28, which reuses that run. Every standard deviation printed in this post is `np.std` with its default, the population form that divides by the number of values.

```text
lr=0.01   mean_acc=0.720 std=0.053  folds [0.7   0.767 0.717 0.783 0.633]
lr=0.05   mean_acc=0.770 std=0.057  folds [0.733 0.8   0.783 0.85  0.683]
lr=0.1    mean_acc=0.773 std=0.040  folds [0.8   0.817 0.717 0.8   0.733]
lr=0.5    mean_acc=0.423 std=0.083  folds [0.5   0.45  0.267 0.483 0.417]
lr=1.0    mean_acc=0.347 std=0.058  folds [0.4   0.433 0.3   0.3   0.3  ]

seed 0  0.01: 0.720  0.05: 0.770  0.1: 0.773  0.5: 0.423  1.0: 0.347  | best 0.1, ahead of 0.05 by 0.003
best learning rate on each single fold: [[0.1], [0.1], [0.05], [0.05], [0.1]]
```

Two readings, each checked on four more seeds by `snippets/seeds_1_2.py` and `snippets/seeds_3_4.py`, which draw new data, new folds and new initial weights:

```text
learning rates   seed 1  0.01: 0.660  0.05: 0.727  0.1: 0.737  0.5: 0.350  1.0: 0.343  | best 0.1, ahead of 0.05 by 0.010
learning rates   seed 2  0.01: 0.640  0.05: 0.737  0.1: 0.713  0.5: 0.387  1.0: 0.360  | best 0.05, ahead of 0.1 by 0.023
learning rates   seed 3  0.01: 0.673  0.05: 0.603  0.1: 0.713  0.5: 0.453  1.0: 0.403  | best 0.1, ahead of 0.01 by 0.040
learning rates   seed 4  0.01: 0.730  0.05: 0.760  0.1: 0.797  0.5: 0.453  1.0: 0.403  | best 0.1, ahead of 0.05 by 0.037
```

**The large learning rates lose in every seed.** With $\alpha = 0.5$ the mean lies between 0.350 and 0.453, and with $\alpha = 1.0$ between 0.343 and 0.403, little above the 0.333 of guessing among three classes. The three smaller rates score at least 0.603 in every seed. A gap of that size is far beyond the spread across folds, and cross-validation settles it.

**The two best learning rates are not separated.** In seed 0 the lead of 0.1 over 0.05 is 0.003, which is one validation sample out of 300, while the accuracies of one candidate differ between folds with a standard deviation of 0.04 to 0.06. Over the five seeds 0.1 has the highest mean four times and 0.05 once, and the difference between the two, 0.1 minus 0.05, is +0.003, +0.010, -0.023, +0.110 and +0.037: a single seed could report anything from a small loss to a clear win. These runs favour 0.1 and do not establish it; an honest report says that 0.05 and 0.1 are not distinguished by this data. The rate 0.01 is another matter: it lies behind 0.1 in all five seeds, by 0.040 to 0.077.

The last printed line shows what a single validation split would have done on seed 0: three of the five folds, taken alone, choose 0.1 and two choose 0.05.

---

## 6. Data leakage: the silent failure mode

A clean split only helps if nothing about the held-out data reaches the model. **Data leakage** is any path by which information from the validation or test data influences training or model choice. Tuning on the test set (section 2) is one such path. Three more are common.

**Preprocessing fitted on all the data.** Any step that estimates something from the data (the mean and standard deviation of a standardiser, a principal-component basis, a vocabulary, a choice of features) must be fitted on the training part and then applied, unchanged, to the validation and test parts.

**Look-ahead features in sequential data.** A feature that depends on a value from after the moment of prediction, such as a moving average that includes tomorrow's price in a forecast of tomorrow, cannot exist when the model is used. Sequential data is the exception to shuffling: it is split by time, training on the past and testing on the future.

**Features that stand in for the label.** A feature can be so tightly coupled to the label that it acts as one. In a model that predicts whether a customer will buy, a feature such as "called to cancel the order" exists only after the outcome is known. It predicts perfectly in the collected data and is absent for a new customer.

![Two pipelines over the same three steps. The left one, labelled split first, shuffles and splits, fits the scaler on the training part and applies it to the validation and test parts, and is marked clean. The right one, labelled preprocess first, fits the scaler on everything, then splits and trains, and is marked leaked. Three cards below name global preprocessing, look-ahead features and label-correlated metadata.](diagrams/02-data-leakage.svg)

*The same three operations in two orders. In the right-hand pipeline one step happens a position too early.*

How much a preprocessing leak costs depends on the step. `snippets/leakage.py` takes a case in which the leak is the whole result, a mistake described by Hastie, Tibshirani and Friedman in their chapter on model assessment. The data are pure noise: 100 rows, 2,000 random features, and labels that are a random half 0 and half 1, so no classifier can do better than 0.5 on new rows on average. The preprocessing step keeps the 20 features whose class means differ most, and a nearest-class-mean classifier is scored by 5-fold cross-validation.

```python
def leaky(X, y, seed):
    columns = select_features(X, y)                   # fitted on all rows, validation rows included
    return np.mean([centroid_accuracy(tr_X[:, columns], tr_y, va_X[:, columns], va_y)
                    for tr_X, tr_y, va_X, va_y in k_fold_split(X, y, k=5, seed=seed)])


def clean(X, y, seed):
    fold_accs = []
    for tr_X, tr_y, va_X, va_y in k_fold_split(X, y, k=5, seed=seed):
        columns = select_features(tr_X, tr_y)         # fitted on the training rows of this fold
        fold_accs.append(centroid_accuracy(tr_X[:, columns], tr_y, va_X[:, columns], va_y))
    return np.mean(fold_accs)
```

```text
selected before the split   [0.92 0.89 0.87 0.82 0.89 0.9  0.92 0.85 0.92 0.89]  mean 0.887
selected inside each fold   [0.58 0.48 0.55 0.38 0.58 0.62 0.46 0.49 0.57 0.52]  mean 0.523
leaky pipeline on new rows  [0.52 0.5  0.51 0.49 0.51 0.5  0.51 0.51 0.5  0.49]  mean 0.503
```

With the selection made before the split, cross-validation reports 0.82 to 0.92 over ten seeds, on data that contain nothing to learn. Among 2,000 noise features some agree with 100 random labels by chance, and the selection has already seen the labels of the validation rows. With the selection inside the loop the score is 0.523 on average, and the leaky pipeline's own model scores 0.503 on 2,000 new rows. The split was correct in both versions; one line was in the wrong place.

A standardiser has far less to leak, because it uses no labels and two numbers per feature barely depend on 60 rows out of 300: on the spiral, whose features have a standard deviation of about 0.41, the full-data means differ from those of a training fold by at most 0.029 and the standard deviations by at most 0.017. The rule is the same in both cases, because the size of a leak is not known in advance:

- Split before any step that is fitted to data.
- Fit every preprocessor on the training part only; in k-fold, inside the loop, on each fold's training part.
- Treat a feature that predicts suspiciously well as a possible leak until its origin is known.

---

## 7. Picking $k$ and other practical defaults

The table gives conventions, not measurements; the sizes are orders of magnitude.

| Dataset size | Usual choice | Why |
|---|---|---|
| Very large ($10^6$ and more) | one 80/10/10 split | a validation set of 100,000 points has negligible noise, and $k$ trainings are expensive |
| Large ($10^4$ to $10^6$) | one split, or 5-fold | a trade of compute against precision |
| Medium ($10^2$ to $10^4$) | 5-fold or 10-fold | the noise of one validation slice matters, as in section 3 |
| Small (under $10^2$) | leave-one-out, which is k-fold with $k = N$ | every sample is needed for training |

Leave-one-out needs $N$ trainings, and its estimate can have a high variance, because the $N$ training sets are nearly identical (Hastie, Tibshirani and Friedman, chapter 7), so it is kept for very small data. When a model is too slow for five trainings per candidate, $k = 3$ or a single validation split are the fallbacks.

**Stratification.** The shuffled folds of section 4 hold between 16 and 27 samples of a class where 20 would be even. **Stratified k-fold** shuffles and cuts each class separately, so that every fold has the class proportions of the whole dataset. It matters most when a class is rare: an unstratified fold can then contain no sample of it at all.

**The k-fold score describes models trained on four fifths of the data.** The final model is trained on all of it and is usually a little better, so the k-fold mean tends to understate it. On the spiral this is visible: trained on all 300 points with $\alpha = 0.1$, the network scores 0.818, 0.839, 0.806, 0.759 and 0.827 on the 30,000 further points of section 2 for seeds 0 to 4, against 5-fold means of 0.773, 0.737, 0.713, 0.713 and 0.797. The final model is ahead in all five seeds, by 0.03 to 0.10. The smaller training sets are one reason and need not be the only one: the two numbers are also measured on different points, 300 held-out ones against 30,000 new ones, and these runs do not separate the two effects.

**The winner's k-fold score is itself a selected number.** It is the highest of several noisy means, which is the situation of section 2 with the validation data in the place of the test set. The two biases pull in opposite directions and neither is known, so the k-fold score of the winner is a tool for choosing and is not the report. The most careful protocol, **nested cross-validation**, wraps the whole search in an outer k-fold loop and costs the product of the two fold counts in trainings per candidate. With a held-out test set, one final evaluation does the same job.

---

## 8. The grid-search loop

k-fold is the measuring instrument; a search strategy decides which candidates to measure. **Grid search** takes a short list of values for each hyperparameter and scores every combination. `snippets/grid.py` does it for two learning rates and two hidden widths, then finishes the protocol of section 2:

```python
    for lr in [0.05, 0.1]:
        for n_neurons in [16, 64]:
            if (lr, n_neurons) not in scores:
                scores[(lr, n_neurons)] = k_fold_accuracies(X, y, lr, n_neurons, seed=seed)
            fold_accs = scores[(lr, n_neurons)]
            if verbose:
                print(f'lr={lr:<5} n_neurons={n_neurons:<3} mean_acc={np.mean(fold_accs):.3f} '
                      f'std={np.std(fold_accs):.3f}')

    grid_scores = {c: scores[c] for c in scores if c[0] in (0.05, 0.1) and c[1] in (16, 64)}
    best_lr, best_n = max(grid_scores, key=lambda c: np.mean(grid_scores[c]))

    # Every decision is made. Retrain the winner on all 300 points, then open the test set, once.
    final_model = train(X, y, best_lr, best_n, seed=100 + seed)
```

```text
lr=0.05  n_neurons=16  mean_acc=0.537 std=0.066
lr=0.05  n_neurons=64  mean_acc=0.770 std=0.057
lr=0.1   n_neurons=16  mean_acc=0.423 std=0.060
lr=0.1   n_neurons=64  mean_acc=0.773 std=0.040

winner (learning rate, width): (0.1, 64)
k-fold mean of the winner 0.773
retrained on all 300 points: training accuracy 0.927, test accuracy 0.793 (238 of 300)
```

The width is settled and the learning rate is not, as in section 5. In all five seeds both settings with 64 neurons (0.603 to 0.797) lie above both settings with 16 (0.423 to 0.537). The winner is $(0.1, 64)$ in four seeds and $(0.05, 64)$ in seed 2.

The number to report for seed 0 is the test accuracy, **0.793**. It is neither the k-fold mean of 0.773 nor the training accuracy of 0.927, which is 0.13 higher than the test accuracy: the gap of post 28. The other four seeds, each with its own data and its own test set, print test accuracies of 0.857, 0.713, 0.770 and 0.863 beside k-fold means of 0.737, 0.737, 0.713 and 0.797. Whatever the test number is, the search is over once it has been read.

A grid grows quickly: three values for each of four hyperparameters are 81 candidates, and 405 trainings with 5-fold validation. **Random search** draws each candidate at random instead, the learning rate usually from a log-uniform distribution. Bergstra and Bengio (2012) showed that it finds good settings with fewer trials than a grid when only a few of the hyperparameters matter: a grid spends its trials on repeated values of the important ones, while random draws give every trial a new value of each. **Bayesian optimisation** (Snoek, Larochelle and Adams, 2012) goes a step further and fits a probabilistic model to the scores seen so far, which proposes the next candidate. The skeleton is the same for all three: each candidate is scored on validation data, and the test set is opened once at the end.

---

## 9. Make it run: the splits, the searches and their spread

Eight scripts hold every code block and every printed number of this post. All are seeded and need NumPy and the `nnfs` package.

```text
python posts/29-validation-and-hyperparameter-tuning/snippets/kfold.py
python posts/29-validation-and-hyperparameter-tuning/snippets/search.py
python posts/29-validation-and-hyperparameter-tuning/snippets/seeds_1_2.py
python posts/29-validation-and-hyperparameter-tuning/snippets/seeds_3_4.py
python posts/29-validation-and-hyperparameter-tuning/snippets/steadiness.py
python posts/29-validation-and-hyperparameter-tuning/snippets/test_set_tuning.py
python posts/29-validation-and-hyperparameter-tuning/snippets/leakage.py
python posts/29-validation-and-hyperparameter-tuning/snippets/grid.py
```

`kfold.py` defines the classes, the split, `train` and `accuracy`, and the other scripts import them from it. On the machine that produced these numbers one training of 1,000 epochs takes under a second, `kfold.py` and `leakage.py` finish in a few seconds, and each of the others in 10 to 45 seconds. The five seeds of sections 5 and 8 are spread over `search.py`, `grid.py` and the two `seeds` scripts to keep every script short; they run the same functions.

---

## 10. What can go wrong?

None of the four mistakes below raises an error.

**A fold size of `len(X) // k`.** A split that computes one fold size by integer division and slices $k$ folds of that size works when $k$ divides $N$ and silently drops the remainder otherwise:

```text
n=300 k=5  fold size 60  samples validated 300  never validated 0
n=300 k=7  fold size 42  samples validated 294  never validated 6
```

With $k = 7$ six samples are never validated; they sit in the training part of every fold.

**No shuffle on sorted data.** `spiral_data` returns its samples sorted by class, 100 of class 0, then 100 of class 1, then 100 of class 2.

```text
shuffle=False  classes per validation fold [[60, 0, 0], [40, 20, 0], [0, 60, 0], [0, 20, 40], [0, 0, 60]]
               fold accuracies [0.05  0.167 0.217 0.133 0.033]  mean 0.120
shuffle=True   classes per validation fold [[18, 24, 18], [22, 19, 19], [17, 16, 27], [21, 21, 18], [22, 20, 18]]
               fold accuracies [0.733 0.8   0.783 0.85  0.683]  mean 0.770
```

Without the shuffle each validation fold is a stretch of one or two spiral arms that is missing from the training data, and the same setting that scores 0.770 on shuffled folds scores 0.120, well below guessing.

**A ranking read from one fold or one seed.** On seed 0 two of the five single folds prefer 0.05 and three prefer 0.1, and the five seeds do not agree either (section 5). A difference smaller than the spread across folds is not a finding, however many decimals are printed.

**Candidates scored on different folds.** In section 3 the shuffle alone moved the 5-fold mean of one fixed setting between 0.723 and 0.800. If each candidate gets its own shuffle, that movement is added to every comparison. One fixed split seed, used for all candidates, removes it.

---

## 11. Summary

| Concept | Takeaway |
|---|---|
| Three-way split | training data for the weights, validation data for the hyperparameters, test data for the final report only |
| Test-set discipline | the best of several test scores is biased upward: by up to 1.8 points on 300 test points and up to 5.0 on 60 in the runs of section 2 |
| k-fold cross-validation | every sample validates once; the mean of the $k$ accuracies is the candidate's score |
| Steadiness | under reshuffling a 5-fold mean varied 2.6 times less than a single split of 60 |
| Reading a search | large gaps (learning rates of 0.5 and 1.0, width 16) hold in every seed; 0.05 against 0.1 stays open |
| Data leakage | fit every preprocessing step on the training part, inside the fold loop |
| Search strategy | grid for a few values, random search when only some hyperparameters matter, Bayesian optimisation for costly trainings |

---

## Common pitfalls

1. **Reporting the winner's validation or k-fold score as its accuracy.** That score took part in the choice. The report is the one test accuracy measured afterwards.
2. **Tuning after seeing the test number.** From then on the test set is a second validation set, and a new, untouched test set is needed for an honest report.
3. **Fitting a preprocessing step before the split.** Standardisers, feature selection and vocabularies are fitted on the training part of each fold.
4. **Treating early stopping as free.** Choosing the epoch with the best validation score is hyperparameter tuning on the validation set, so the reported number still comes from the test set.
5. **Splitting sorted data without a shuffle, or shuffling sequential data.** Class-sorted data must be shuffled or stratified; data ordered in time must be split by time.
6. **Calling a winner inside the noise.** A lead smaller than the spread across folds and seeds is reported as a tie.

---

## Further reading

- Bergstra, J. and Bengio, Y., *"Random Search for Hyper-Parameter Optimization"* (Journal of Machine Learning Research, 2012).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 5 (MIT Press, 2016).
- Hastie, T., Tibshirani, R., and Friedman, J., *The Elements of Statistical Learning*, chapter 7, section 7.10 (Springer, 2009).
- Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O., *"Leakage in Data Mining: Formulation, Detection, and Avoidance"* (ACM Transactions on Knowledge Discovery from Data, 2012). A formal treatment of the leaks of section 6.
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 11 (2020).
- Kohavi, R., *"A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection"* (IJCAI, 1995).
- Snoek, J., Larochelle, H., and Adams, R. P., *"Practical Bayesian Optimization of Machine Learning Algorithms"* (NeurIPS, 2012).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 30 - L1 and L2 regularisation](../30-l1-and-l2-regularisation/index.md):** a penalty on weight magnitude whose strength $\lambda$ is the next hyperparameter to choose on validation data.
- **[Post 31 - Dropout](../31-dropout/index.md):** a second regulariser, with a rate that is tuned by the same protocol.
