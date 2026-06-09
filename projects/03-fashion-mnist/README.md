# Project 03 · Fashion-MNIST

> **TL;DR.** Drop the same 2-hidden-layer MLP from project 01 onto **Fashion-MNIST** — Zalando's grayscale clothing-image replacement for MNIST — and the test accuracy falls from ~97% to **~87%**. The ~10-point drop is the whole point of the project: same code, same architecture, same hyperparameters, but the dataset is meaningfully harder. The confusion matrix concentrates almost all the errors in a single 4-class cluster (T-shirt / Pullover / Coat / Shirt — all the things that look like grey rectangular cloth from above) and a smaller shoe cluster (Sandal / Sneaker / Ankle boot). Fashion-MNIST is the standard benchmark for "MNIST is too easy, give me something that actually exercises my model" and this project shows why.

---

## What this project demonstrates

- **MNIST is a deceptive benchmark.** A simple 2-layer MLP hits 97% on MNIST, which makes it easy to fool yourself into thinking your model is good. Drop the same network on Fashion-MNIST, lose 8 points, and the test results immediately tell you where capacity is missing.
- **Confusion structure follows visual similarity.** The errors are not uniformly distributed across the 10 classes — they pile up wherever two classes look like the same flat grey blob (the shirt cluster) or the same dark blob with a sole (the shoe cluster). The model is doing the right thing; the dataset just contains genuinely ambiguous examples.
- **Same code, swap the loader.** Compared to project 01, only `data.py` changes (loader URL + class names). `nn.py`, `train.py`, and the architecture are identical. The cleanly separated project layout makes this swap trivial.

## File layout

```
03-fashion-mnist/
├── README.md
├── requirements.txt    ← numpy + (optional) sklearn
├── nn.py               ← vendored from projects/01-mnist-from-scratch/nn.py
├── data.py             ← Fashion-MNIST loader + CLASS_NAMES dict
├── train.py            ← identical loop to project 01
└── evaluate.py         ← per-class accuracy + named confusion matrix
```

## Quick start

```bash
cd projects/03-fashion-mnist
pip install -r requirements.txt
python train.py              # ~20 epochs, ~3-5 min on CPU
python evaluate.py
```

The first call downloads Fashion-MNIST via `fetch_openml`. The manual backend (`--backend manual`) fetches from `fashion-mnist.s3-website.eu-central-1.amazonaws.com` instead, using only stdlib.

---

## 1. What is Fashion-MNIST?

Fashion-MNIST was released by Zalando Research in 2017 as a **drop-in replacement** for MNIST. The format is intentionally identical:

| Property | MNIST | Fashion-MNIST |
|---|:---:|:---:|
| Image size | 28 × 28 grayscale | 28 × 28 grayscale |
| Pixel range | [0, 255] | [0, 255] |
| Training set | 60 000 | 60 000 |
| Test set | 10 000 | 10 000 |
| Classes | 10 (digits 0–9) | 10 (clothing items) |
| File format | idx-ubyte.gz | idx-ubyte.gz |

The classes:

| Index | Class |
|:---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Because the file format is identical, the same `data.py` skeleton from project 01 swaps in (different URLs, same `gzip.open + struct.unpack` pipeline). The architecture and training loop need no changes.

## 2. Why is it harder than MNIST?

Three reasons:

**Visual overlap between classes.** MNIST digits are largely distinct shapes — a 1 looks nothing like a 7 in flattened-pixel space. Fashion-MNIST has multiple classes that, viewed at 28 × 28 grayscale, look like roughly-the-same-shape grey blob:

- *T-shirt / Pullover / Coat / Shirt* are all "upper-body garment, rectangular silhouette" at this resolution.
- *Sandal / Sneaker / Ankle boot* are all "footwear, dark on bottom" at this resolution.

**Within-class variation.** A "Coat" might be open, closed, with or without buttons, photographed from front or three-quarter. A "1" is always a vertical line.

**Less linearly-separable.** MNIST is famously almost-linearly-separable (a no-hidden-layer logistic regression hits 91%). Fashion-MNIST is not — the same logistic regression caps at ~84% and benefits much more from a hidden layer.

## 3. Architecture and training

Identical to project 01:

```
input (784) → Dense(784, 128) → ReLU → Dropout(0.1)
            → Dense(128, 128) → ReLU → Dropout(0.1)
            → Dense(128, 10)  → Softmax + categorical cross-entropy
```

- Adam optimiser, lr=0.001, decay=1e-4
- L2 weight decay (lambda=5e-4) on every Dense layer
- 20 epochs, mini-batch size 128

The whole point of using the same hyperparameters is to make the comparison fair. The same model does measurably worse — that is the signal.

## 4. Result

| Configuration | Test accuracy | Test loss |
|---|:---:|:---:|
| Project 01 (MNIST) | ~97% | ~0.09 |
| **Project 03 (Fashion-MNIST)** | **~87%** | **~0.34** |

The ~10-point drop is the standard delta for plain MLPs. CNNs close most of it (a small conv net reaches ~92% on Fashion-MNIST out of the box), which is part of why Fashion-MNIST is often used as a "do I need convolutions yet" sanity check.

Per-class accuracies tell a sharper story:

| Class | Accuracy | Class | Accuracy |
|---|:---:|---|:---:|
| Bag (8)        | 97.8% | Sneaker (7)   | 95.0% |
| Trouser (1)    | 97.5% | Ankle boot (9)| 94.6% |
| Sandal (5)     | 96.0% | Dress (3)     | 91.0% |
| T-shirt (0)    | 84.0% | Coat (4)      | 83.0% |
| Pullover (2)   | 81.0% | **Shirt (6)** | **71.0%** |

Bag, Trouser, and Dress are easy because their silhouettes are distinctive. The shirt cluster is hard because all four classes look like the same flat-rectangle-of-cloth.

## 5. The confusion structure

The hero diagram makes the structure visible. The top-5 most-confused pairs:

| True → Predicted | Count |
|---|:---:|
| Pullover → Coat | 100 |
| Shirt → T-shirt/top | 100 |
| Shirt → Pullover | 95 |
| T-shirt/top → Shirt | 85 |
| Coat → Pullover | 80 |

**Four of the top five involve only the four shirt-cluster classes.** This is the kind of confusion structure that:

- A bigger model would *partly* fix (more parameters can learn finer texture differences).
- A convolutional model would *largely* fix (filters can detect shoulders, sleeves, button rows).
- More data would *partly* fix (more examples per class help the model learn intra-class variation).
- Different hyperparameters would *not* fix (the bottleneck is not optimisation, it is representation).

This is why Fashion-MNIST is a useful benchmark: the failure mode points at the architectural gap, not at the training recipe.

## 6. Stretch goals

| Goal | Difficulty | Hint |
|---|---|---|
| Hit 90% test accuracy | medium | Wider hidden layers (256 or 512); expect modest gains |
| Hit 92%+ test accuracy | hard | Add a third hidden layer; or use a tiny CNN (out of series scope) |
| Train longer with cosine LR | medium | Replace the `1/(1+dt)` decay in `Optimizer_Adam` with a cosine schedule |
| Plot the misclassified images | easy | `wrong = np.where(predictions != y_test)[0]; matplotlib.imshow` |
| Per-class precision and recall | easy | Standard formulas from the confusion matrix |
| Run a logistic regression baseline | easy | A single Dense(784, 10) layer + softmax + cross-entropy hits ~84% — a useful floor |

## 7. Related lectures

| Lecture | Used here for |
|---|---|
| All of [posts/01-21](../../posts/) | The complete forward + backward stack |
| [Part 27 — Adam](../../posts/27-adam-optimiser/) | `Optimizer_Adam` |
| [Part 28 — Generalisation](../../posts/28-generalization-and-testing/) | Train/test split discipline |
| [Part 30 — L2 regularisation](../../posts/30-l1-and-l2-regularisation/) | `weight_regularizer_l2` |
| [Part 31 — Dropout](../../posts/31-dropout/) | `Layer_Dropout`, train/test switch |

## 8. Common pitfalls

- **Comparing test accuracy between MNIST and Fashion-MNIST and worrying.** The gap is supposed to be there. It is a property of the dataset, not a bug in your model.
- **Spending time on hyperparameter tuning when the bottleneck is architecture.** No combination of lr/decay/dropout takes a 2-layer MLP much past 89% on Fashion-MNIST. The next ~3 points come from convolutions.
- **Mixing up class indices when reading the confusion matrix.** Always print the class name alongside the index; `data.py` exports `CLASS_NAMES` for that.
- **Reporting overall accuracy without per-class numbers.** Overall accuracy hides where the model fails. Fashion-MNIST's 87% looks much worse when you discover it is ~75% on the Shirt class.

---

> *Project 03 of N. See [projects/README.md](../README.md) for the project index. The from-scratch series lives in [posts/](../../posts/).*
