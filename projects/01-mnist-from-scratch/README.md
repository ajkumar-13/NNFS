# Project 01 · MNIST from scratch

> **TL;DR.** Train the from-scratch classes built across [the series](../../posts/) on real handwritten digits. A two-hidden-layer network (784 → 128 → 128 → 10) with Adam, L2 weight decay, and dropout reaches roughly **97% test accuracy** on the standard MNIST split using only NumPy. No PyTorch, no TensorFlow, no SciPy beyond `np.random`. The project is split into four runnable files (`data.py`, `nn.py`, `train.py`, `evaluate.py`) plus this writeup, so the training loop, the network definition, and the evaluation are each in their own place rather than wedged into one notebook.

---

## What this project demonstrates

- The from-scratch stack composes into a single working model: Dense layers + ReLU + Softmax + cross-entropy + Adam + L2 + Dropout, all imported from `nn.py`.
- Mini-batch training (introduced in [post 32](../../posts/32-mini-batching/)) is applied here to a real dataset: a compact outer loop in `train.py` shuffles, slices, and iterates the dataset in batches of 128.
- The train/test discipline from [post 28](../../posts/28-generalization-and-testing/) and the train-vs-test dropout switch from [post 31](../../posts/31-dropout/) carry over directly to a real dataset.

## File layout

```
01-mnist-from-scratch/
├── README.md          ← this file
├── requirements.txt   ← numpy + (optional) scikit-learn
├── nn.py              ← every class lifted from the series (posts 1–32)
├── data.py            ← MNIST loading (sklearn or manual download)
├── train.py           ← training loop + checkpoint
└── evaluate.py        ← test loss, accuracy, confusion matrix
```

## Quick start

```bash
cd projects/01-mnist-from-scratch
pip install -r requirements.txt
python train.py              # ~20 epochs, ~2-5 min on a modern laptop CPU
python evaluate.py
```

The first call to `train.py` (with the default sklearn backend) downloads the dataset via `fetch_openml` and caches it in `~/scikit_learn_data/`. The manual backend (`--backend manual`) downloads the four idx-ubyte.gz files into a local `mnist_cache/` folder and parses them with stdlib only.

---

## 1. Why MNIST?

| Property | Value |
|---|:---:|
| Input | 784 features (28 × 28 grayscale pixels, flattened) |
| Classes | 10 (digits 0–9) |
| Training samples | 60 000 |
| Test samples | 10 000 |
| Modern SOTA | > 99.9% (with convolutions, augmentation, ensembles) |
| Plain-MLP ceiling | ~98% (with the kind of tricks taught in this series) |
| What this project hits | ~97% (a tuned subset of those tricks) |

MNIST is the natural next step after the spiral dataset from the lectures: a real image dataset, ten classes instead of three, two orders of magnitude more samples, and well-studied baselines so the result can be sanity-checked. It is small enough to train on a CPU in minutes and large enough that overfitting becomes a real concern, which is why dropout earns its keep here in a way it could not on the spiral.

## 2. The architecture

```
input (784) → Dense(784, 128) → ReLU → Dropout(0.1)
            → Dense(128, 128) → ReLU → Dropout(0.1)
            → Dense(128, 10)  → Softmax + categorical cross-entropy
```

Three design choices worth pinning down.

**Two hidden layers, 128 units each.** One hidden layer can fit MNIST but generalises worse; three or four hidden layers help marginally and cost more compute. 128 is a sensible width — enough capacity for ten classes without exploding the parameter count past ~120k.

**Dropout 0.1, not 0.5.** The original dropout paper used 0.5 for hidden layers on much wider networks. With 128-unit hidden layers, dropping half kills too much capacity. 0.1 gives a measurable regularisation benefit without crushing training accuracy. See [post 31, §3](../../posts/31-dropout/) for the rate-vs-capacity discussion.

**L2 weight decay of 5e-4.** The small value comes from the rule of thumb in [post 30, §6](../../posts/30-l1-and-l2-regularisation/): low enough that training accuracy is not damaged, high enough that the train/test gap closes.

Total parameter count:

| Layer | Weights | Biases | Total |
|---|:---:|:---:|:---:|
| Dense(784, 128) | 100 352 | 128 | 100 480 |
| Dense(128, 128) | 16 384  | 128 | 16 512 |
| Dense(128, 10)  | 1 280   | 10  | 1 290 |
| **Sum** |  |  | **118 282** |

## 3. The training loop

`train.py` runs 20 epochs of mini-batch Adam with the following structure (simplified):

```python
for epoch in range(epochs):
    idx = np.random.permutation(len(X_train))
    X_shuf, y_shuf = X_train[idx], y_train[idx]

    for start in range(0, len(X_train), batch_size):
        X_batch = X_shuf[start:start + batch_size]
        y_batch = y_shuf[start:start + batch_size]

        # Forward (training=True so dropout is active)
        # Backward (chain rule through every layer)
        # Update (optimizer.pre_update_params → update_params per layer → post_update_params)
```

The full implementation in `train.py` adds:

- L2 regularisation loss computed per layer and added to the data loss
- A running tally of correct predictions for the epoch
- Per-epoch logging of loss, training accuracy, and current learning rate
- A pickle checkpoint at the end so `evaluate.py` does not have to retrain

The whole loop is about 70 lines including the model construction and CLI parsing.

## 4. The evaluation script

`evaluate.py` loads the checkpoint, runs a single forward pass on the full test set (with **`training=False`** so dropout is off), and reports:

- Test loss and accuracy
- Per-class accuracy (which digits the network finds easy vs hard)
- A printed 10 × 10 confusion matrix
- Indices of the first 20 misclassified samples (so you can pull them up in a notebook to inspect)

The most common output looks like:

```
  loss     0.0892
  accuracy 0.9714  (9714/10000)

Per-class accuracy:
  digit 0: 0.9888  (980 samples)
  digit 1: 0.9885  (1135 samples)
  digit 2: 0.9670  (1032 samples)
  digit 3: 0.9663  (1010 samples)
  ...
```

Per-class accuracy almost always reveals that 8 and 5 are the hardest digits (they share strokes with several others), and 1 is the easiest.

## 5. Stretch goals

| Goal | Difficulty | Hint |
|---|---|---|
| Reach 98% test accuracy | medium | Wider hidden layers (256 instead of 128) or add a 3rd hidden layer |
| Add early stopping | easy | Track validation loss in `train.py`, halt when it plateaus for N epochs |
| Switch optimisers and compare | medium | Swap `Optimizer_Adam` for SGD+momentum or RMSProp; compare convergence curves |
| Plot the loss curve | easy | Append `(epoch, train_loss, train_acc)` to a list each epoch and `matplotlib.pyplot.plot` them |
| Data augmentation (random shifts) | hard | Shift each image by ±2 pixels in x/y at random; expect ~+0.5% test accuracy |
| Train on Fashion-MNIST | easy | Same code, swap the OpenML name; a harder dataset that exposes overfitting better |

## 6. What this project does *not* do

Things deliberately out of scope so the project stays a faithful demonstration of what the series covers, rather than a from-scratch reproduction of all of deep learning:

- **No convolutional layers.** Convolutions are not in the series; using them here would obscure which improvement comes from the model and which from the architecture family.
- **No batch normalisation.** Not in the series. Adam + dropout + L2 is enough to hit 97% on a small MLP.
- **No GPU code.** NumPy on CPU. Training takes a few minutes; that is the point.
- **No external optimiser libraries.** Adam in `nn.py` is the same class built in [post 27](../../posts/27-adam-optimiser/), to the line.

## 7. Related lectures

| Lecture | Used here for |
|---|---|
| [Part 4 — Dense layer class](../../posts/04-dense-layer-class-and-spiral-data/) | `Layer_Dense` |
| [Part 6 — Activation functions](../../posts/06-activation-functions-relu-and-softmax/) | `Activation_ReLU` |
| [Part 19 — Softmax + cross-entropy combined](../../posts/19-softmax-derivatives-and-the-combined-backward-pass/) | `Activation_Softmax_Loss_CategoricalCrossentropy` |
| [Part 21 — Coding full backpropagation](../../posts/21-coding-the-full-backpropagation/) | The full forward/backward stack |
| [Part 27 — Adam](../../posts/27-adam-optimiser/) | `Optimizer_Adam` |
| [Part 28 — Generalisation and testing](../../posts/28-generalization-and-testing/) | Train/test split discipline |
| [Part 30 — L1 / L2 regularisation](../../posts/30-l1-and-l2-regularisation/) | `weight_regularizer_l2`, `regularization_loss` |
| [Part 31 — Dropout](../../posts/31-dropout/) | `Layer_Dropout`, train-vs-test switch |
| [Part 32 — Mini-batching](../../posts/32-mini-batching/) | The epoch × batch training loop in `train.py` |

## 8. Common pitfalls

- **Pixel values not normalised to [0, 1].** Adam diverges on raw 0–255 pixel ranges. `data.py` already divides by 255; if you re-implement loading, do the same.
- **Forgetting `training=False` at evaluation.** Test accuracy becomes the accuracy of a random subnetwork. `evaluate.py` sets this correctly; if you copy the loop manually, be careful.
- **One huge batch instead of mini-batches.** Adam needs the stochastic noise to escape local minima; full-batch updates on 60k samples both slow training and reduce final accuracy.
- **Re-initialising the optimiser inside the epoch loop.** The momentum and cache buffers must persist across batches. Construct `Optimizer_Adam(...)` once, outside the loop.
- **Using `np.random.seed` only at the top of the script.** That seeds the global RNG but not the per-layer dropout masks if you forget to use the same RNG. `train.py` calls `np.random.seed(seed)` once at the start, which is enough as long as no other code paths reseed.
- **Cheating with the test set.** If a hyperparameter choice was informed by test-set behaviour, that test number is no longer honest. Hold out a separate validation slice from the training set if you want to tune.

---

> *Built on the foundation of the series. Source: [INDEX.md](../../INDEX.md).*
