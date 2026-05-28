---
slug: 29-validation-and-hyperparameter-tuning
title: "Part 29 · Validation and hyperparameter tuning"
date: 2026-05-29
tags: [neural-networks, from-scratch, validation, hyperparameters, cross-validation, generalisation]
hero: diagrams/01-three-way-split-and-kfold.svg
reading_time: 11
part: "Part VII — Generalisation and regularisation"
---

# Part 29 · Validation and hyperparameter tuning

> **TL;DR.** Part 28 measured the generalisation gap; this lecture introduces the discipline required to *close* it without cheating. Two ideas do the work. First, the **three-way split**: training data trains the weights, **validation** data picks the hyperparameters, and **test** data is touched exactly once at the end. Mixing those roles is the most common source of inflated benchmarks in published ML. Second, **k-fold cross-validation**: when data is scarce, every example takes a turn in the validation slot, so the same dataset can serve both roles without throwing away signal. Together the two patterns let a practitioner search a large hyperparameter space and still report a number that is honest about the model's behaviour on truly unseen data.
>
> **Reading time:** ~11 minutes.
>
> **After reading this you will be able to:**
> - State why the test set must be touched exactly once, and explain the failure mode when it is reused for tuning.
> - Implement a clean `k_fold_split` in NumPy and use it to compare hyperparameter candidates.
> - Recognise the most common forms of data leakage and the preprocessing rules that prevent them.

![Three-way split and k-fold cross-validation: the training set teaches the weights, the validation set chooses the knobs, and the test set is opened once at the very end. K-fold reuses the training set as a rotating validation slot when data is scarce.](diagrams/01-three-way-split-and-kfold.svg)
*Two splitting patterns side by side. The three-way split on the left is the default when data is plentiful; the 5-fold rotation on the right is the default when it is not.*

---

## 1. The hyperparameter problem

A neural network has two kinds of numbers: **parameters** that gradient descent learns (every weight and bias in every layer), and **hyperparameters** that someone must choose before training starts. Hyperparameters are the dials backprop cannot turn.

A non-exhaustive list of choices made before the first forward pass:

| Category | Examples |
|---|---|
| Architecture | Number of hidden layers, neurons per layer, choice of activation |
| Optimiser | Algorithm (SGD vs Adam), learning rate $\alpha$, momentum $\beta_1$, decay |
| Regularisation | L1 or L2 weight decay (Part 30), dropout rate (Part 31), batch size |
| Training schedule | Number of epochs, early-stopping patience, learning-rate warmup |
| Initialisation | He, Xavier, scale of the random draw |

Each choice is a hypothesis: "this combination will generalise well on this data". The only honest way to test a hypothesis is to evaluate it on data the model never saw during training. The set used for *that* evaluation must be different from the set used for the *final* evaluation, or the practitioner has tuned the design against the test set and the reported number is no longer trustworthy.

That is the entire reason validation data exists. Without a separate validation set, every hyperparameter choice secretly turns into a tiny gradient step that the human takes against the test set.

---

## 2. The three-way split

When data is plentiful, the discipline is simple: carve three disjoint slices.

| Slice | Typical share | Role | When it is touched |
|---|:---:|---|---|
| **Training** | 60–80% | Weights and biases are learned from it | Every forward/backward pass |
| **Validation** | 10–20% | Hyperparameters are chosen by comparing their validation losses | Between experiments, freely |
| **Test** | 10–20% | Final, single-number report of generalisation | Exactly once, after all design decisions are frozen |

The workflow is:

1. Train many candidate models (different hyperparameters) on the training set.
2. Evaluate each candidate on the validation set. Choose the one with the lowest validation loss (or highest validation accuracy).
3. Retrain the winning configuration on the union of training and validation if desired.
4. Evaluate that final model on the test set. Report that number. Do **not** go back and re-tune.

Step 4's "do not go back" is the rule the field most frequently breaks. Every time a researcher reads a disappointing test number and tweaks the model, the test set has been used for tuning and the reported "test accuracy" is overstated.

---

## 3. K-fold cross-validation

When the dataset is small (say, a few hundred examples), a 20% validation slice might be only 60 examples. The validation loss measured on so few examples is too noisy to distinguish similar candidates. The standard fix is **k-fold cross-validation**: rotate the validation slice through the data.

The recipe with $k = 5$:

1. Shuffle the training data. Split it into 5 equal parts: $A, B, C, D, E$.
2. For each fold $i \in \{1, \dots, 5\}$:
   - Use the $i$-th part as the validation slot.
   - Use the other four parts as the training slot.
   - Train a *fresh* model from scratch, then record its validation accuracy $E_i$.
3. The **mean validation accuracy** $\bar{E} = \frac{1}{5} \sum_i E_i$ is the score of the hyperparameter candidate.

| Fold | Validation part | Training parts | Validation acc. |
|:---:|:---:|:---:|:---:|
| 1 | A | B, C, D, E | $E_1$ |
| 2 | B | A, C, D, E | $E_2$ |
| 3 | C | A, B, D, E | $E_3$ |
| 4 | D | A, B, C, E | $E_4$ |
| 5 | E | A, B, C, D | $E_5$ |

Two properties make this work.

**Every example takes a turn in the validation slot.** Across the five runs, every training example is validated against exactly once. No data is wasted; nothing is reserved permanently.

**The variance of $\bar{E}$ is much smaller than the variance of a single $E_i$.** Averaging five validation losses on disjoint subsets gives a more reliable estimate of the model's true validation behaviour than any one slice could. For two hyperparameter candidates whose single-fold accuracies differ by less than a per-fold standard deviation, the mean over 5 folds usually resolves the comparison.

The most common choices are $k = 5$ and $k = 10$. Higher $k$ means more reliable estimates but proportionally more training runs; the original $k = 10$ recommendation comes from a 1995 study by Kohavi showing diminishing returns past that point.

---

## 4. A minimal k-fold implementation

The `sklearn.model_selection.KFold` API is the standard, but the underlying logic is twelve lines of NumPy:

```python
import numpy as np

def k_fold_split(X, y, k=5, shuffle=True, seed=0):
    """
    Yield (train_X, train_y, val_X, val_y) for each fold.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)

    fold_size = len(X) // k
    for i in range(k):
        val_start = i * fold_size
        val_end   = val_start + fold_size

        val_idx   = idx[val_start:val_end]
        train_idx = np.concatenate([idx[:val_start], idx[val_end:]])

        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]
```

Two design choices worth noting.

**`seed` defaults to 0**, so the splits are reproducible across runs. Without a fixed seed, comparing two hyperparameter candidates is confounded by the shuffle: one configuration might win simply because it got an easier fold.

**The function uses `yield`, not `return`.** That avoids materialising all `k` splits in memory at once and lets the caller stop early (useful when one fold's validation accuracy already invalidates the candidate).

### 4.1. Using k-fold for hyperparameter search

The outer loop searches the hyperparameter space; the inner loop runs k-fold for each candidate:

```python
candidates = [0.01, 0.05, 0.1, 0.5, 1.0]   # learning rates to compare

for lr in candidates:
    fold_accs = []
    for tr_X, tr_y, va_X, va_y in k_fold_split(X, y, k=5):
        model     = build_fresh_model()                # new weights every fold
        optimizer = Optimizer_Adam(learning_rate=lr)

        for epoch in range(1000):
            train_one_epoch(model, optimizer, tr_X, tr_y)

        fold_accs.append(evaluate_accuracy(model, va_X, va_y))

    print(f'lr={lr:<6} mean_acc={np.mean(fold_accs):.3f} '
          f'std={np.std(fold_accs):.3f}')
```

A representative run on the spiral data:

```
lr=0.01   mean_acc=0.783  std=0.014
lr=0.05   mean_acc=0.893  std=0.018
lr=0.1    mean_acc=0.877  std=0.018
lr=0.5    mean_acc=0.657  std=0.018
lr=1.0    mean_acc=0.337  std=0.007
```

Two observations.

**$\alpha = 0.05$ wins.** It has the highest mean accuracy and a low standard deviation, so the win is consistent across folds.

**The losers expose themselves clearly.** $\alpha = 1.0$ diverges to random guessing on every fold; $\alpha = 0.5$ converges to a poor local minimum. Both are obvious from the per-fold pattern. K-fold makes good and bad hyperparameters look obviously different even on small datasets.

---

## 5. Data leakage: the silent failure mode

A clean three-way split or a perfect k-fold rotation only helps if the model never gets a peek at the held-out data. **Data leakage** is any pathway through which information from the test (or validation) set reaches the model during training. It is the single most common source of inflated benchmarks and embarrassed retractions.

Three common leak patterns:

**Preprocessing across the full dataset.** Standardising features with the *full*-dataset mean and standard deviation, then splitting, leaks the test distribution into the training set. The standardiser must be fitted on the training fold and *then applied* to the validation/test folds. Same rule for PCA, target encoding, vocabulary construction, and anything else that involves estimated statistics.

**Look-ahead features in time-series data.** Predicting tomorrow's stock price using a feature that secretly depends on tomorrow's closing price is the classic example. For sequential data the split must respect time: training is the past, test is the future, and no feature computed at test time can use information that would not have been available then.

**Label-correlated metadata.** Sometimes a feature is technically not the label but is so tightly coupled that it acts as one. The textbook example: predicting whether a customer will buy a product using a feature like "did the customer call to cancel?". Cancel calls only happen after a non-purchase; the feature encodes the label. On truly new customers the feature is meaningless and the model fails.

The defence against leakage is mechanical rather than clever:

- Shuffle and split *before* any global preprocessing.
- Fit every preprocessor (scaler, encoder, vocabulary, …) on the training fold only.
- Be suspicious of any feature whose marginal predictive power feels "too good".
- For k-fold, refit every preprocessor inside the loop, on each fold's training portion.

The mantra: **the test set is invisible until the final report**. Every line of code that touches it before then is a candidate leak.

---

## 6. Picking $k$ and other practical defaults

A short table of working defaults:

| Dataset size | Recommended split | Why |
|---|---|---|
| Very large ($\ge 10^6$) | Single 80/10/10 split | A 100 000-sample validation set has negligible variance; k-fold is wasted compute |
| Large ($10^4$ to $10^6$) | Single 80/10/10 split or 5-fold | Either works; the choice is a compute/precision trade-off |
| Medium ($10^2$ to $10^4$) | 5-fold or 10-fold CV | Validation variance starts to matter |
| Small ($< 100$) | Leave-one-out CV (extreme of k-fold with $k = n$) | Every example used as its own one-sample validation set |

Stratification matters too. For classification, **stratified k-fold** ensures each fold has approximately the same class distribution as the full dataset; without it, a fold might happen to contain none of a rare class and the validation accuracy becomes meaningless. `sklearn.model_selection.StratifiedKFold` is the standard tool; a from-scratch version applies the same shuffle-then-split logic class by class.

---

## 7. The grid-search loop

K-fold is the *measurement* tool; the *search* loop wraps it. The two most common search strategies:

**Grid search.** Enumerate every combination from a small list of values per hyperparameter. Simple, exhaustive, expensive when the grid grows large.

```python
for lr in [1e-4, 1e-3, 1e-2]:
    for hidden in [32, 64, 128]:
        for batch in [16, 32, 64]:
            mean_acc = run_kfold_cv(lr=lr, hidden=hidden, batch=batch)
            log_result(lr, hidden, batch, mean_acc)
```

**Random search.** Sample hyperparameter values from a distribution (uniform, log-uniform). Bergstra and Bengio (2012) showed empirically that random search outperforms grid search when only a few hyperparameters are decisive: the random sampler spends fewer evaluations on the dimensions that do not matter.

For very large spaces, the modern default is **Bayesian optimisation** (libraries: Optuna, Ax, hyperopt) which fits a probabilistic model over the loss landscape and proposes the next candidate where the expected improvement is highest. The skeleton stays the same: each proposal is scored by k-fold CV on the training set, and the test set is opened once at the end.

---

## 8. Anticipated questions

- **Should I shuffle before splitting?** Almost always yes. The only exception is genuinely sequential data (time series, ordered text), where shuffling destroys the time structure and creates lookahead leakage.
- **Can I tune the number of epochs using the test set?** No. Use the validation set for early stopping; the test set's role is the final report only.
- **What if my model is too slow to run k-fold?** Drop $k$ to 3, or use a held-out validation set instead. K-fold is a precision tool; if a single split gives a sharp enough signal, it is unnecessary.
- **Does k-fold give an honest estimate of the *final* generalisation error?** Almost. The mean validation error from k-fold is unbiased for *a model with this hyperparameter choice*. It is slightly biased for the final model (which is typically trained on more data — the full training fold rather than $k-1$ slices) and so usually underestimates the final test performance by a small amount.
- **What about nested cross-validation?** It is the most rigorous version: an outer k-fold for the test estimate, an inner k-fold inside each outer fold for hyperparameter selection. The compute cost is $k_\text{outer} \times k_\text{inner}$ trainings per candidate, so it is reserved for small datasets where every drop of precision matters.

---

## 9. Summary

| Concept | Takeaway |
|---|---|
| Three-way split | Training (weights), validation (hyperparameters), test (final report only) |
| Test-set discipline | Touch it exactly once; every re-tune secretly trains on it |
| K-fold CV | Rotate the validation slot through the training set; mean accuracy is the score |
| Default $k$ | 5 (or 10) for medium-sized datasets; leave-one-out for very small data |
| Stratified k-fold | Required for classification when classes are imbalanced |
| Data leakage | Any pathway from held-out to training; the silent killer of benchmarks |
| Search strategy | Grid (small spaces), random (medium), Bayesian (large) |

---

## Common pitfalls

- **Reporting the best validation accuracy as the model's accuracy.** Validation accuracy is an internal selection metric; report the test accuracy. The two often differ by a few percentage points because the winning candidate was selected by ranking against the validation set (a small but real form of optimism bias).
- **Tuning after seeing the test number.** The instant the test set has influenced a design decision it has become a second validation set. Reset the test split before reporting.
- **Standardising features before splitting.** Mean and standard deviation computed on the full dataset leak the test distribution into the training fold. Fit the scaler inside each fold.
- **Forgetting to set a random seed.** Without `np.random.default_rng(seed)`, two runs of the same comparison will pick different splits and produce different rankings.
- **Comparing candidates on different folds.** Every candidate must be evaluated on the *same* splits, or the comparison is confounded by which examples ended up where. Cache the fold indices and reuse them.
- **Treating early stopping as free.** Choosing the epoch with the best validation accuracy *is* hyperparameter tuning on the validation set. The final reported number must be measured on a held-out test set, not on the validation curve.
- **Letting one class dominate a fold.** If the spiral data was sorted by class before splitting, the first fold's validation slice might contain only class 0 and the validation accuracy would be useless. Always shuffle (or stratify).

---

## Further reading

- Bergstra, J. and Bengio, Y., *"Random Search for Hyper-Parameter Optimization"* (Journal of Machine Learning Research, 2012) — why random search is often better than grid.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning* — chapter 5 (Machine Learning Basics) (MIT Press, 2016).
- Hastie, T., Tibshirani, R., and Friedman, J., *The Elements of Statistical Learning* — chapter 7 (Model Assessment and Selection) (Springer, 2009).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python* — chapter 29 (2020).
- Kohavi, R., *"A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection"* (IJCAI, 1995) — the 10-fold recommendation.
- Snoek, J., Larochelle, H., and Adams, R. P., *"Practical Bayesian Optimization of Machine Learning Algorithms"* (NeurIPS, 2012) — modern hyperparameter search.

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

Tuning hyperparameters tells you *which* model to ship; the next two lectures show *how* to make that model less prone to overfit in the first place.

- **[Part 30 — L1 and L2 regularisation](../30-l1-and-l2-regularisation/index.md)** — add a weight-magnitude penalty to the loss; force the optimiser to prefer simpler hypotheses.
- **[Part 31 — Dropout](../31-dropout/index.md)** — randomly mask neurons during training so the network cannot rely on any single one.

---

> **Try it yourself:** Hands-on exercises and quizzes for this lecture live in [Exercises](../../exercises.md) and [Quizzes](../../quizzes.md).
