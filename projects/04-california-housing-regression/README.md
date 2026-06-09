# Project 04 · California housing regression

> **TL;DR.** The first three projects in this series were all *classification* — predict a discrete label from a probability distribution. This project is *regression*: predict a continuous target (the median house value of a California census block group) from 8 numerical features. The architectural and code changes from project 01 are small but precise: **no softmax**, **no sigmoid**, the last Dense layer's raw scalar output is the prediction; **mean squared error** replaces cross-entropy as the loss; and the target gets **standardised to zero mean and unit variance** before training (then de-standardised at evaluation so the reported error is in dollars). A 64-64 hidden network trained with Adam for 200 epochs hits an R² of ~0.82 on the test set with RMSE around $49 000 — competitive with a tree-ensemble baseline on this dataset.

---

## What this project demonstrates

- **Regression vs classification.** Same forward pipeline of Dense + ReLU layers; only the **last activation and the loss** change. No softmax, no probabilities, no cross-entropy.
- **Standardisation done correctly.** Training-fold statistics are used to fit the scaler, then applied to the test fold without re-fitting. This is the no-leakage rule from [post 29 §5](../../posts/29-validation-and-hyperparameter-tuning/) made explicit.
- **The dataset's quirks show in the scatter.** California housing is top-coded at $500 000 — any block group with a higher actual value was clipped to the cap. The trained model has never seen un-capped truth above $500k and predicts ~$200k-$280k for those samples, producing the distinctive vertical cluster at the right edge of the predicted-vs-actual scatter.

## File layout

```
04-california-housing-regression/
├── README.md
├── requirements.txt    ← numpy + (optional) sklearn
├── nn.py               ← Dense, ReLU, Loss_MSE, Adam, regularization_loss
├── data.py             ← cal-housing loader, train/test split, standardise/destandardise
├── train.py            ← MSE training loop with mini-batches
└── evaluate.py         ← train + test RMSE/MAE/R², predicted-vs-actual dump
```

## Quick start

```bash
cd projects/04-california-housing-regression
pip install -r requirements.txt
python train.py              # ~30 sec on a modern laptop CPU
python evaluate.py
```

The first call to `train.py` downloads the dataset via `sklearn.datasets.fetch_california_housing`. The manual backend (`--backend manual`) pulls `cal_housing.tgz` from the StatLib mirror and parses it with stdlib only.

---

## 1. The dataset

The California housing dataset comes from the 1990 US Census via Pace and Barry (1997). Each row is a *block group* — the smallest geographical unit the Census Bureau publishes data for, typically 600 to 3 000 people. The eight features are the kind of aggregate statistics any real-estate model wants:

| Index | Feature | Meaning |
|:---:|---|---|
| 0 | `MedInc` | Median income in the block group |
| 1 | `HouseAge` | Median house age |
| 2 | `AveRooms` | Average number of rooms per household |
| 3 | `AveBedrms` | Average number of bedrooms per household |
| 4 | `Population` | Total population |
| 5 | `AveOccup` | Average household occupancy |
| 6 | `Latitude` | Block group latitude |
| 7 | `Longitude` | Block group longitude |

The target is **median house value in the block group**, in units of $100 000. So a target value of 2.5 means $250 000. The dataset is **top-coded at $500 000**: any block group with a true median above the cap was clipped. About 5% of the data sits exactly at the cap, which becomes the vertical cluster in the hero diagram.

20 640 samples in total. The `train_test_split` in `data.py` splits 80/20 by default, leaving 16 512 training and 4 128 test samples.

## 2. Why standardise?

Raw feature values span very different ranges:

| Feature | Range |
|---|:---:|
| `MedInc` | 0.5 – 15 |
| `HouseAge` | 1 – 52 |
| `Population` | 3 – 35 000 |
| `Latitude` | 32 – 42 |

A weight that fits `MedInc` well (multiplier in the 0.0–0.5 range) is the wrong magnitude for `Population` (where the same data-space step requires a 0.00005 multiplier). The optimiser cannot balance these per-feature scales fast — and Adam's per-parameter rescaling helps but does not fully fix the problem.

Standardisation gives every feature zero mean and unit variance:

$$x^{\text{std}}_i = \frac{x_i - \bar{x}}{\sigma_x}$$

After standardisation, every column has the same scale and the initial random weights operate on a level playing field. Training converges in a fraction of the epochs.

Same idea applies to the target: with standardised `y`, the MSE loss values are on the same scale as the lecture examples (in the 0–1 range early in training, dropping to ~0.2 at convergence). Without standardisation, the raw loss is in the 10⁹ range and Adam's default learning rates don't fit.

The de-standardisation step lives in `evaluate.py`: predictions in standardised units multiply by `y_std`, add `y_mean`, multiply by $100 000, and become dollars. The reported RMSE/MAE numbers are honest dollar errors.

## 3. What changes from project 01

The full list of differences from `projects/01-mnist-from-scratch/`:

| Piece | Project 01 (MNIST classifier) | Project 04 (regression) |
|---|---|---|
| Last layer activation | Softmax | None (raw linear output) |
| Loss class | `Activation_Softmax_Loss_CategoricalCrossentropy` | `Loss_MSE` |
| Output shape | (N, 10) probabilities | (N, 1) scalar |
| Target shape | (N,) integer labels | (N,) float values |
| Metrics | Accuracy, confusion matrix | RMSE, MAE, R² |
| Architecture | 784 → 128 → 128 → 10 | 8 → 64 → 64 → 1 |
| Loss range | 0 - ~2.3 (cross-entropy) | 0 - ~5 in std units |
| Backward call | `loss_act.backward(loss_act.output, y_batch)` | `loss_fn.backward(dense3.output)` |

The forward and backward pass logic is otherwise identical: linear → ReLU → linear → ReLU → linear → loss. The chain rule does not care whether the loss is cross-entropy or MSE.

## 4. The MSE loss class

```python
class Loss_MSE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = np.asarray(y_true).reshape(y_pred.shape)
        return float(np.mean((y_pred - self.y_true) ** 2))

    def backward(self, y_pred, y_true=None):
        y_true = self.y_true if y_true is None else \
            np.asarray(y_true).reshape(y_pred.shape)
        samples = len(y_pred)
        outputs = y_pred.shape[1] if y_pred.ndim > 1 else 1
        self.dinputs = 2 * (y_pred - y_true) / (samples * outputs)
```

Two points worth noting:

**No combined-class trick.** The classification projects used `Activation_Softmax_Loss_CategoricalCrossentropy` to absorb the softmax forward + cross-entropy backward into a single class with the simplified `(y_pred - y_true) / N` derivative. For MSE there is no comparable simplification: the gradient is already `2 * (y_pred - y_true) / N` and the output has no activation to absorb.

**Mean over both batch and output dims.** For a (N, K) output, the loss is the mean over all NK elements; the gradient divides by `samples * outputs` accordingly. For this project K = 1, so it's just division by N, but the formula handles vector regression too.

## 5. Architecture

```
input (8) → Dense(8, 64) → ReLU → Dense(64, 64) → ReLU → Dense(64, 1)
```

Parameter count:

| Layer | Weights | Biases |
|---|:---:|:---:|
| Dense(8, 64)  | 512   | 64 |
| Dense(64, 64) | 4 096 | 64 |
| Dense(64, 1)  | 64    | 1  |
| **Total** |  | **4 801** |

About 4 800 parameters — small. The dataset has 16 512 training samples, so we have roughly 3.4 training samples per parameter, well into the "no overfitting expected" regime. L2 weight decay (`lambda=1e-4`) is included for hygiene but contributes a tiny fraction of the loss.

## 6. Result

Representative final metrics from a typical 200-epoch run:

| Metric | Train | Test |
|---|:---:|:---:|
| MSE (standardised units) | ~0.15 | ~0.18 |
| RMSE ($) | ~$45 000 | ~$49 000 |
| MAE ($) | ~$31 000 | ~$34 000 |
| R² | ~0.85 | ~0.82 |

The ~2-point train/test R² gap is mild — barely overfit. The RMSE of $49k on a target with mean ~$207k is about a 24% relative error, which sounds high but is typical for "all of California" regression: the dataset bundles together expensive coastal block groups and cheap inland ones, and 8 numerical features cannot fully separate them.

### 6.1 Comparing to baselines

| Model | Test R² | Notes |
|---|:---:|---|
| Predict the training mean | 0.000 | The zero-knowledge floor |
| Ordinary least squares (linear regression) | ~0.64 | A linear model on the 8 features |
| **2-layer MLP (this project)** | **~0.82** | What we built |
| sklearn `GradientBoostingRegressor` (defaults) | ~0.81 | A strong tree-ensemble baseline; the from-scratch MLP edges it here |
| xgboost with tuning | ~0.85 | The competitive ML ceiling on this dataset |

The MLP comfortably beats linear regression and gets within striking distance of well-tuned gradient boosting. For a single hidden-layer change, that's the right shape.

## 7. The cap-effect cluster

The most visually distinctive feature of the predicted-vs-actual scatter is the vertical column of red dots at the right edge — actual value = $500 000, predicted values scattered from ~$200k to ~$320k.

This is **not a model failure**; it's the data telling the truth. The Census Bureau top-coded any block group above $500k, so:

- The model never saw an uncapped truth value above $500k.
- For block groups that *would have been* worth $750k or $1.2M, the loss penalised any prediction differently from $500k.
- The model learned that the features associated with high-end housing predict values around $300k-$500k, not higher.
- At test time, those same high-end block groups still appear at the cap and the model under-predicts.

The fix is **not** a different model architecture — it's a different *target*. If you cared about top-end predictions, you would either find an uncapped dataset, model the cap as a censored regression problem, or train on a transformed target (e.g. log(price)) that compresses the high end.

## 8. Stretch goals

| Goal | Difficulty | Hint |
|---|---|---|
| Hit R² > 0.80 | medium | Wider hidden layers (128, 128) or a 3rd hidden layer |
| Compare to ordinary least squares | easy | One `np.linalg.lstsq` call as a baseline; report both R² |
| Plot residuals vs actual | easy | `residuals = y_pred - y_true; plt.scatter(y_true, residuals)` |
| Per-region accuracy | medium | Slice the test set by Latitude buckets; the model is more accurate in the SF Bay area than in inland regions |
| Predict log(price) instead | medium | Standardise `log(y)` instead of `y`; exponentiate at evaluation; reduces cap-effect underprediction |
| Add a validation set + early stopping | medium | Split the training fold further; halt when val RMSE plateaus |

## 9. Related lectures

| Lecture | Used here for |
|---|---|
| [Part 4 — Dense layer class](../../posts/04-dense-layer-class-and-spiral-data/) | `Layer_Dense` |
| [Part 6 — Activations](../../posts/06-activation-functions-relu-and-softmax/) | `Activation_ReLU` |
| [Part 21 — Coding full backpropagation](../../posts/21-coding-the-full-backpropagation/) | Forward + backward composition |
| [Part 27 — Adam](../../posts/27-adam-optimiser/) | `Optimizer_Adam` |
| [Part 29 — Validation](../../posts/29-validation-and-hyperparameter-tuning/) | Train/test split discipline, no-leakage standardisation |
| [Part 30 — L2 regularisation](../../posts/30-l1-and-l2-regularisation/) | `weight_regularizer_l2` |

## 10. Common pitfalls

- **Forgetting to standardise the features.** Raw values span 4-5 orders of magnitude. Without standardisation, training takes hundreds of epochs to even start moving.
- **Fitting the standardiser on the whole dataset, not the training fold.** Leakage; the test set's distribution informs the training set's preprocessing. `data.py` enforces the right pattern with `standardise_fit` (training-only) and `standardise_apply` (any fold).
- **Reporting MSE in standardised units as if they were dollars.** Always de-standardise before reporting dollar metrics.
- **Including a softmax or sigmoid at the output.** Regression outputs are unbounded; an activation at the end *restricts* the range and biases predictions. Last Dense layer's raw output IS the prediction.
- **Using accuracy as the metric.** Accuracy is a classification concept. Use RMSE, MAE, or R² for regression.
- **Confusing R² with correlation r².** They are related but not the same. R² is `1 - SS_res / SS_tot`; it can be negative if the model is worse than predicting the mean. Correlation r² is always in [0, 1].
- **Comparing models that predicted the raw target vs the log-target without re-aligning units.** Always evaluate in the same units; log-space MSE looks much smaller than raw-space MSE but is not better, just different.

---

> *Project 04 of N. See [projects/README.md](../README.md) for the project index. The from-scratch series lives in [posts/](../../posts/).*
