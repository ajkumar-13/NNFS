# Evaluation

Measured: 2026-10-05. Every figure on this page was printed by the commands shown, run from this directory in a freshly created environment, or is a closed form written out beside it.

## The claim

Trained from seed 0 with the documented command, the 118,282-parameter network classifies 9,800 of the 10,000 MNIST test images correctly, a test accuracy of 98.00 percent.

The claim is false if the commands below, run in the locked environment, print a lower accuracy. The last command makes that a check: it exits with status 1 when the accuracy is under 0.98.

## How to reproduce it

```bash
uv sync --frozen
uv run python scripts/download_mnist.py
uv run python -m mnist_from_scratch.train
uv run python -m mnist_from_scratch.evaluate --min-accuracy 0.98
```

| Item | Value |
|---|---|
| Seed | 0 (the default of `--seed`) |
| Data | the four MNIST files with the SHA-256 checksums recorded in `src/mnist_from_scratch/data.py`; 60,000 training and 10,000 test images, the standard split |
| Training | 20 epochs, mini-batches of 128, 469 steps an epoch, Adam with learning rate 0.001 and decay 0.0001, L2 0.0005 on the hidden weights, dropout 0.1 |
| Software | Python 3.13.5, NumPy 2.3.5, installed from `uv.lock` by uv 0.11.7 |
| Machine | Intel Core i7-9750H (2.60 GHz), 7.8 GB of memory, Windows 10, CPU only |

## Results

| Figure | Value |
|---|---|
| Test accuracy | 0.9800 (9,800 of 10,000) |
| Test loss, mean cross-entropy without the penalty | 0.0628 |
| Misclassified test images | 200 |
| Parameters | 118,282 |
| Training accuracy in epoch 20, with dropout active | 0.9847 |
| Training loss in epoch 20, data loss plus L2 penalty | 0.1104 |
| Training time, 20 epochs | 85.3 s |
| Fingerprint of the weights | `a38647f981b44e6b77d52938a2664af031836569e852cb1d0471a6a7a7dc47e5` |

The parameter count is also a closed form: $(784 \times 128 + 128) + (128 \times 128 + 128) + (128 \times 10 + 10) = 100{,}480 + 16{,}512 + 1{,}290 = 118{,}282$.

### Per digit

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Test images | 980 | 1,135 | 1,032 | 1,010 | 982 | 892 | 958 | 1,028 | 974 | 1,009 |
| Correct | 969 | 1,125 | 1,018 | 992 | 971 | 871 | 940 | 1,005 | 949 | 960 |
| Accuracy | 0.9888 | 0.9912 | 0.9864 | 0.9822 | 0.9888 | 0.9765 | 0.9812 | 0.9776 | 0.9743 | 0.9514 |

In this run 1 is the easiest digit and 9 the hardest. Of the 49 nines that are misread, 18 are read as 4, the largest entry off the diagonal of the confusion matrix; the next largest are 8, for threes read as 2 and for nines read as 3.

### The training log

```text
loading MNIST from mnist_cache
  train: (60000, 784)    test: (10000, 784)
  parameters 118,282
training for 20 epochs, batch_size=128 (469 steps/epoch), seed=0

epoch   1 | loss 0.6208 | train_acc 0.8244 | lr 0.000955
epoch   2 | loss 0.3106 | train_acc 0.9309 | lr 0.000914
epoch   3 | loss 0.2490 | train_acc 0.9500 | lr 0.000877
epoch   4 | loss 0.2217 | train_acc 0.9582 | lr 0.000842
epoch   5 | loss 0.1995 | train_acc 0.9648 | lr 0.000810
epoch   6 | loss 0.1870 | train_acc 0.9672 | lr 0.000780
epoch   7 | loss 0.1748 | train_acc 0.9708 | lr 0.000753
epoch   8 | loss 0.1681 | train_acc 0.9720 | lr 0.000727
epoch   9 | loss 0.1595 | train_acc 0.9736 | lr 0.000703
epoch  10 | loss 0.1535 | train_acc 0.9757 | lr 0.000681
epoch  11 | loss 0.1465 | train_acc 0.9768 | lr 0.000660
epoch  12 | loss 0.1385 | train_acc 0.9784 | lr 0.000640
epoch  13 | loss 0.1347 | train_acc 0.9792 | lr 0.000621
epoch  14 | loss 0.1303 | train_acc 0.9803 | lr 0.000604
epoch  15 | loss 0.1268 | train_acc 0.9811 | lr 0.000587
epoch  16 | loss 0.1232 | train_acc 0.9817 | lr 0.000571
epoch  17 | loss 0.1177 | train_acc 0.9829 | lr 0.000556
epoch  18 | loss 0.1164 | train_acc 0.9825 | lr 0.000542
epoch  19 | loss 0.1117 | train_acc 0.9840 | lr 0.000529
epoch  20 | loss 0.1104 | train_acc 0.9847 | lr 0.000516

trained in 85.3 s
wrote weights to mnist_weights.npz
weights sha256 a38647f981b44e6b77d52938a2664af031836569e852cb1d0471a6a7a7dc47e5
```

The loss column is the mean over the epoch's batches of the cross-entropy plus the L2 penalty, and `train_acc` counts the predictions made while training, with dropout active. Neither is comparable with the test figures below, which use no dropout and no penalty.

### The evaluation report

```text
restoring weights from mnist_weights.npz
loading MNIST from mnist_cache
  parameters 118,282
  weights    sha256 a38647f981b44e6b77d52938a2664af031836569e852cb1d0471a6a7a7dc47e5

  loss     0.0628
  accuracy 0.9800  (9800/10000)

Per-class accuracy:
  digit 0: 0.9888  (980 samples)
  digit 1: 0.9912  (1135 samples)
  digit 2: 0.9864  (1032 samples)
  digit 3: 0.9822  (1010 samples)
  digit 4: 0.9888  (982 samples)
  digit 5: 0.9765  (892 samples)
  digit 6: 0.9812  (958 samples)
  digit 7: 0.9776  (1028 samples)
  digit 8: 0.9743  (974 samples)
  digit 9: 0.9514  (1009 samples)

Confusion matrix (rows = true class, cols = predicted class):
             0      1      2      3      4      5      6      7      8      9
  true 0:   969      0      3      2      0      0      4      1      1      0
  true 1:     0   1125      3      1      0      0      1      1      4      0
  true 2:     1      1   1018      0      1      0      2      6      3      0
  true 3:     0      0      8    992      0      3      0      4      3      0
  true 4:     0      0      3      0    971      0      3      1      0      4
  true 5:     2      1      0      7      1    871      6      1      2      1
  true 6:     4      3      3      1      4      2    940      0      1      0
  true 7:     1      6      6      3      0      0      0   1005      1      6
  true 8:     1      0      6      4      3      2      2      5    949      2
  true 9:     1      4      0      8     18      2      2      7      7    960

200 misclassified samples. First 20 indices: [8, 61, 151, 247, 259, 274, 321, 340, 359, 381, 445, 448, 449, 495, 582, 619, 646, 659, 684, 691]
```

## Repeatability

The run was made twice on 2026-10-05 in two separately created environments, and both printed the fingerprint above: on this machine, with the locked packages, the result repeats bit for bit. The same weights also came out of the project's scripts as they stood before version 1.0.0, so restructuring the code into a package changed nothing in the model (record [0003](adr/0003-pin-numpy-for-bit-reproducibility.md)).

## Other seeds

Two more runs, identical except for `--seed`, show how much of the headline belongs to the seed:

```bash
uv run python -m mnist_from_scratch.train --seed 1 --weights seed1.npz
uv run python -m mnist_from_scratch.evaluate --weights seed1.npz
```

| Seed | Test accuracy | Correct | Test loss | Easiest digit | Hardest digit |
|---|---|---|---|---|---|
| 0 | 0.9800 | 9,800 | 0.0628 | 1, at 0.9912 | 9, at 0.9514 |
| 1 | 0.9791 | 9,791 | 0.0653 | 1, at 0.9938 | 2, at 0.9564 |
| 2 | 0.9801 | 9,801 | 0.0640 | 1, at 0.9921 | 9, at 0.9653 |

The three runs span 10 test images, from 9,791 to 9,801. The digit 1 is the easiest every time; the hardest digit is not the same every time.

## Limits of the claim

- **It is one run.** The headline is seed 0. The three seeds measured lie within 0.10 percentage points of each other, and one of them is below 98 percent. Three runs are not a distribution, and no confidence interval over seeds is claimed.
- **The test set is finite.** For an accuracy of 0.98 on 10,000 images the binomial standard error is $\sqrt{0.98 \times 0.02 / 10{,}000} = 0.0014$, or 14 images. Differences of a tenth of a percentage point between two models on this test set are inside that noise.
- **Per-digit figures are weaker than the headline.** Each rests on about 1,000 images, and the comparison of seeds shows that the ranking of the hard digits moves from run to run. Quote a per-digit figure with its seed.
- **Bit-identity is local.** It was shown for one processor and the locked NumPy 2.3.5. Under NumPy 2.5.3 the weights after one epoch were already not bit-identical on the same machine, which is why the version is pinned; the 20-epoch accuracy under any other version, and under any other processor, was not measured.
- **The training time is a wall-clock reading** on a laptop that was running other work at the same time. The four 20-epoch runs made that day (seed 0 twice, seeds 1 and 2) printed 85.3, 114.0, 116.9 and 126.6 s, so read it as "one to two minutes", not more.
- **No validation set was held out.** The settings were fixed before this evaluation and were not tuned as part of it; the run trains on all 60,000 training images, and the test set is used only to report. The figure is not evidence that these settings are the best ones.
- **MNIST flatters dense networks.** The digits are centred and size-normalised, so a network that ignores the layout of the pixels still does well. The result does not transfer to harder image data.

## Use

- 2026-10-05: the sibling series Convolutional Neural Networks from Scratch uses this run as its dense baseline. `cnn-011` quotes the 98.00 percent after 20 epochs and the 118,282 parameters, and `cnn-012` compares its convolutional network against the 200 test errors and the parameter count. Those are the figures on this page.
