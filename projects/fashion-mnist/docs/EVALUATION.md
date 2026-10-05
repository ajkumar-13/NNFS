# Evaluation

Measured: 2026-10-05. Every figure on this page was printed by the commands shown, run from this directory in a freshly created environment, or is a closed form written out beside it.

## The claim

Trained from seed 0 with the documented command, the 118,282-parameter network classifies 8,712 of the 10,000 Fashion-MNIST test images correctly, a test accuracy of 87.12 percent. That is 1,088 images fewer than the same network, trained the same way, classifies on MNIST (`nn-p01`: 9,800 of 10,000).

The claim is false if the commands below, run in the locked environment, print a lower accuracy. The last command makes that a check: it exits with status 1 when the accuracy is under 0.8712.

## How to reproduce it

```bash
uv sync --frozen
uv run python scripts/download_fashion_mnist.py
uv run python -m fashion_mnist.train
uv run python -m fashion_mnist.evaluate --min-accuracy 0.8712
```

| Item | Value |
|---|---|
| Seed | 0 (the default of `--seed`) |
| Data | the four Fashion-MNIST files with the SHA-256 checksums recorded in `src/fashion_mnist/data.py`; 60,000 training and 10,000 test images, the standard split |
| Training | 20 epochs, mini-batches of 128, 469 steps an epoch, Adam with learning rate 0.001 and decay 0.0001, L2 0.0005 on the hidden weights, dropout 0.1 |
| Software | Python 3.13.5, NumPy 2.3.5, installed from `uv.lock` by uv 0.11.7 |
| Machine | Intel Core i7-9750H (2.60 GHz), 7.8 GB of memory, Windows 10, CPU only |

The network, the training settings and the seed are those of the series' MNIST project. Only the dataset differs.

## Results

| Figure | Value |
|---|---|
| Test accuracy | 0.8712 (8,712 of 10,000) |
| Test loss, mean cross-entropy without the penalty | 0.3419 |
| Misclassified test images | 1,288 |
| Parameters | 118,282 |
| Training accuracy in epoch 20, with dropout active | 0.8928 |
| Training loss in epoch 20, data loss plus L2 penalty | 0.3537 |
| Training time, 20 epochs | 116.1 s |
| Fingerprint of the weights | `bb6c47c027bc09f00a1e0438f682933710e74d08db368365a47316774a19cc5b` |

The parameter count is also a closed form: $(784 \times 128 + 128) + (128 \times 128 + 128) + (128 \times 10 + 10) = 100{,}480 + 16{,}512 + 1{,}290 = 118{,}282$.

### Per class

Every class has 1,000 test images, so a count of correct images is also an accuracy in tenths of a percent.

| Label | Class | Correct of 1,000 | Accuracy | Times predicted | Correct among those predictions |
|---|---|---|---|---|---|
| 0 | T-shirt/top | 754 | 0.7540 | 857 | $754 / 857 = 0.8798$ |
| 1 | Trouser | 964 | 0.9640 | 971 | $964 / 971 = 0.9928$ |
| 2 | Pullover | 726 | 0.7260 | 870 | $726 / 870 = 0.8345$ |
| 3 | Dress | 906 | 0.9060 | 1,058 | $906 / 1{,}058 = 0.8563$ |
| 4 | Coat | 801 | 0.8010 | 1,028 | $801 / 1{,}028 = 0.7792$ |
| 5 | Sandal | 929 | 0.9290 | 950 | $929 / 950 = 0.9779$ |
| 6 | Shirt | 752 | 0.7520 | 1,210 | $752 / 1{,}210 = 0.6215$ |
| 7 | Sneaker | 941 | 0.9410 | 1,009 | $941 / 1{,}009 = 0.9326$ |
| 8 | Bag | 969 | 0.9690 | 1,000 | $969 / 1{,}000 = 0.9690$ |
| 9 | Ankle boot | 970 | 0.9700 | 1,047 | $970 / 1{,}047 = 0.9265$ |

The first two numeric columns are printed by the evaluation. The last two are the column sums of the confusion matrix below and the diagonal divided by them.

In this run Ankle boot is the easiest class and Pullover the hardest, at 726 correct; Shirt, at 752, and T-shirt/top, at 754, follow. Three readings of the matrix:

- **Shirt is over-predicted.** The network answers Shirt 1,210 times for 1,000 shirts, and 458 of those answers are wrong: 185 T-shirts/tops, 129 pullovers, 97 coats, 33 dresses, 12 bags, 1 trouser and 1 ankle boot. Of the 1,000 real shirts it misses 248. The class is confused in both directions, and more often as a wrong answer than as a missed one.
- **The errors stay inside two groups.** Among the four upper-body garments (T-shirt/top, Pullover, Coat, Shirt) there are 816 confusions, the sum of the twelve cells off the diagonal between those four classes, which is 63.4 percent of the 1,288 errors. Among the footwear (Sandal, Sneaker, Ankle boot) there are 157, another 12.2 percent. Together they are 973 errors, 75.5 percent.
- **The five largest confusions** are T-shirt/top read as Shirt (185), Pullover read as Shirt (129), Pullover read as Coat (121), Coat read as Shirt (97) and Shirt read as T-shirt/top (80). All five are inside the upper-body group.

### Against the same network on MNIST

| | MNIST (`nn-p01`) | Fashion-MNIST (this project) |
|---|---|---|
| Test accuracy | 0.9800 | 0.8712 |
| Misclassified test images | 200 | 1,288 |
| Test loss | 0.0628 | 0.3419 |
| Lowest accuracy of a class | 0.9514 | 0.7260 |
| Training accuracy in epoch 20, dropout active | 0.9847 | 0.8928 |

The MNIST column is that project's documented seed 0 run. The difference is $0.9800 - 0.8712 = 0.1088$, or 10.88 percentage points, and $1{,}288 / 200 = 6.44$ times as many errors. The comparison is one seed against one seed; the section on other seeds below says how far the Fashion-MNIST figure moves.

### The training log

```text
loading Fashion-MNIST from fashion_mnist_cache
  train: (60000, 784)    test: (10000, 784)
  parameters 118,282
training for 20 epochs, batch_size=128 (469 steps/epoch), seed=0

epoch   1 | loss 0.8341 | train_acc 0.7036 | lr 0.000955
epoch   2 | loss 0.5597 | train_acc 0.8248 | lr 0.000914
epoch   3 | loss 0.5079 | train_acc 0.8426 | lr 0.000877
epoch   4 | loss 0.4767 | train_acc 0.8538 | lr 0.000842
epoch   5 | loss 0.4565 | train_acc 0.8618 | lr 0.000810
epoch   6 | loss 0.4406 | train_acc 0.8678 | lr 0.000780
epoch   7 | loss 0.4310 | train_acc 0.8685 | lr 0.000753
epoch   8 | loss 0.4200 | train_acc 0.8730 | lr 0.000727
epoch   9 | loss 0.4105 | train_acc 0.8763 | lr 0.000703
epoch  10 | loss 0.4045 | train_acc 0.8778 | lr 0.000681
epoch  11 | loss 0.3968 | train_acc 0.8795 | lr 0.000660
epoch  12 | loss 0.3887 | train_acc 0.8826 | lr 0.000640
epoch  13 | loss 0.3834 | train_acc 0.8838 | lr 0.000621
epoch  14 | loss 0.3790 | train_acc 0.8850 | lr 0.000604
epoch  15 | loss 0.3727 | train_acc 0.8877 | lr 0.000587
epoch  16 | loss 0.3687 | train_acc 0.8882 | lr 0.000571
epoch  17 | loss 0.3642 | train_acc 0.8907 | lr 0.000556
epoch  18 | loss 0.3602 | train_acc 0.8910 | lr 0.000542
epoch  19 | loss 0.3572 | train_acc 0.8923 | lr 0.000529
epoch  20 | loss 0.3537 | train_acc 0.8928 | lr 0.000516

trained in 116.1 s
wrote weights to fashion_mnist_weights.npz
weights sha256 bb6c47c027bc09f00a1e0438f682933710e74d08db368365a47316774a19cc5b
```

The loss column is the mean over the epoch's batches of the cross-entropy plus the L2 penalty, and `train_acc` counts the predictions made while training, with dropout active. Neither is comparable with the test figures below, which use no dropout and no penalty.

### The evaluation report

```text
restoring weights from fashion_mnist_weights.npz
loading Fashion-MNIST from fashion_mnist_cache
  parameters 118,282
  weights    sha256 bb6c47c027bc09f00a1e0438f682933710e74d08db368365a47316774a19cc5b

  loss     0.3419
  accuracy 0.8712  (8712/10000)

Per-class accuracy:
  T-shirt/top (0): 0.7540  (1000 samples)
      Trouser (1): 0.9640  (1000 samples)
     Pullover (2): 0.7260  (1000 samples)
        Dress (3): 0.9060  (1000 samples)
         Coat (4): 0.8010  (1000 samples)
       Sandal (5): 0.9290  (1000 samples)
        Shirt (6): 0.7520  (1000 samples)
      Sneaker (7): 0.9410  (1000 samples)
          Bag (8): 0.9690  (1000 samples)
   Ankle boot (9): 0.9700  (1000 samples)

Confusion matrix (rows = true class, cols = predicted class):
                       0      1      2      3      4      5      6      7      8      9
  T-shirt/top (0):   754      1     12     34      4      1    185      0      9      0
      Trouser (1):     2    964      2     22      7      0      1      0      2      0
     Pullover (2):     8      0    726     15    121      0    129      0      1      0
        Dress (3):    12      5      7    906     34      0     33      0      3      0
         Coat (4):     0      0     65     35    801      0     97      0      2      0
       Sandal (5):     0      0      0      1      0    929      0     38      1     31
        Shirt (6):    80      1     57     39     58      0    752      0     13      0
      Sneaker (7):     0      0      0      0      0     13      0    941      0     46
          Bag (8):     1      0      1      6      3      3     12      5    969      0
   Ankle boot (9):     0      0      0      0      0      4      1     25      0    970

Top 5 most-confused class pairs (true -> predicted):
  T-shirt/top (0) ->       Shirt (6):   185 times
     Pullover (2) ->       Shirt (6):   129 times
     Pullover (2) ->        Coat (4):   121 times
         Coat (4) ->       Shirt (6):    97 times
        Shirt (6) -> T-shirt/top (0):    80 times

1288 misclassified samples. First 20 indices: [12, 17, 23, 25, 42, 49, 51, 66, 67, 68, 103, 127, 135, 136, 141, 147, 150, 153, 155, 170]
```

## Repeatability

The documented run was made twice on 2026-10-05 in two separately created environments, the second after deleting the environment, installing it again from the lockfile, and downloading the four files afresh into an empty directory. Both printed the fingerprint above and the same report: on this machine, with the locked packages, the result repeats bit for bit. The second run's training time was 131.2 s.

The same weights also came out of the project's scripts as they stood before version 1.0.0, run that day with their documented command, so restructuring the code into a package changed nothing in the model (record [0003](adr/0003-pin-numpy-for-bit-reproducibility.md)). Those scripts read the dataset from OpenML where the package reads the four original files, so the two sources held the same images in the same order on that day (record [0004](adr/0004-idx-files-with-checksums.md)). The file those scripts wrote was byte for byte the checkpoint the author had kept from a run of 2026-06-09, which is the checkpoint the sibling series took its figures from.

## Other seeds

Two more runs, identical except for `--seed`, show how much of the headline belongs to the seed:

```bash
uv run python -m fashion_mnist.train --seed 1 --weights seed1.npz
uv run python -m fashion_mnist.evaluate --weights seed1.npz
```

| Seed | Test accuracy | Correct | Test loss | Easiest class | Hardest class | Shirt | Largest confusion |
|---|---|---|---|---|---|---|---|
| 0 | 0.8712 | 8,712 | 0.3419 | Ankle boot, at 0.9700 | Pullover, at 0.7260 | 0.7520 | T-shirt/top read as Shirt, 185 |
| 1 | 0.8831 | 8,831 | 0.3261 | Bag, at 0.9820 | Shirt, at 0.6980 | 0.6980 | Shirt read as T-shirt/top, 120 |
| 2 | 0.8764 | 8,764 | 0.3404 | Sandal, at 0.9730 | Shirt, at 0.6770 | 0.6770 | Coat read as Pullover, 145 |

The three runs span 119 test images, from 8,712 to 8,831, which is 1.19 percentage points. On MNIST the same three seeds span 10 images. The documented seed 0 run is the lowest of the three here. The easiest class, the hardest class and the largest confusion are different in every run; what all three share is that the five largest confusions are all among T-shirt/top, Pullover, Coat and Shirt.

## Limits of the claim

- **It is one run, and the seed matters here.** The headline is seed 0. The three seeds measured lie within 1.19 percentage points of each other, and the other two are above the headline. Three runs are not a distribution, and no mean or confidence interval over seeds is claimed. A comparison against this baseline that rests on a difference of a point or less is a comparison of seeds.
- **The test set is finite.** For an accuracy of 0.8712 on 10,000 images the binomial standard error is $\sqrt{0.8712 \times 0.1288 / 10{,}000} = 0.0033$, or 33 images. The spread between seeds, 119 images, is larger than that, so the seed is the bigger uncertainty.
- **Per-class figures are weaker than the headline.** Each rests on 1,000 images, and the comparison of seeds shows them moving by several points: Shirt is at 0.7520, 0.6980 and 0.6770 in the three runs, and Pullover at 0.7260, 0.7980 and 0.8580. Which class is hardest and which confusion is largest depend on the seed. Quote a per-class figure or a confusion count with its seed.
- **Bit-identity is local.** It was shown for one processor and the locked NumPy 2.3.5. Whether another NumPy version or another processor gives the same weights was not measured for this project; the MNIST project, which shares the network and the loop, found that NumPy 2.5.3 did not, which is why the version is pinned.
- **The training time is a wall-clock reading** on a laptop that was running other work at the same time. The four 20-epoch runs made that day (seed 0 twice, seeds 1 and 2) printed 116.1, 131.2, 132.1 and 162.5 s, so read it as "two to three minutes", not more.
- **No validation set was held out, and nothing was tuned.** The settings are the MNIST project's, fixed before this evaluation; the run trains on all 60,000 training images, and the test set is used only to report. The figure is not evidence that these settings are good ones for Fashion-MNIST, and it is not the best a dense network can do there.
- **The explanation of the errors is a reading, not a measurement.** The matrix shows which classes are confused. That the cause is a dense network's blindness to where the pixels are is the natural account, and this project does not test it: it trains no other architecture.

## Use

- 2026-10-05: the sibling series Convolutional Neural Networks from Scratch uses this run as its dense baseline on Fashion-MNIST, quoting the 87.12 percent, the per-class accuracies and the confusions with Shirt. Those are the seed 0 figures on this page.
