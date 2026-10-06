"""Post 33, section 8: the three values of init on the series' spiral network, compared seed by seed.

Run from the series root:
    python posts/33-weight-initialisation/snippets/spiral_wins.py
    python posts/33-weight-initialisation/snippets/spiral_wins.py 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

Every run is the documented run of post 28 with init="small", "he" or "xavier" on both layers; the
setup is printed by the script. With init="small" it is post 28's run itself, and seed 0 reproduces
its 96.33 and 82.33 percent. Accuracies are measured forward-only after the last update; the test
set is a second spiral_data call made after training. For every seed the script trains all three,
then prints the range and mean per value of init and counts on how many seeds one value has the
higher accuracy than another.

Without arguments it runs seed 0 alone, the documented seed of Part VI. The ranges and counts of
section 8 are printed by the second command, seeds 0 to 19, which takes about seven minutes.

Needs NumPy and the nnfs package. Takes 20 to 25 seconds for one seed.
"""
import sys

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from network import Optimizer_Adam, accuracy_of, train_stack

nnfs.init()
INITS = ("small", "he", "xavier")
seeds = tuple(int(argument) for argument in sys.argv[1:]) or (0,)

print("setup: nnfs.init() once (seed 0, float32, float32 np.dot), then per run np.random.seed(s), "
      "spiral_data(samples=100, classes=3), Layer_Dense(2, 64, init=i), ReLU, Layer_Dense(64, 3, init=i), "
      "softmax and cross-entropy, Optimizer_Adam(learning_rate=0.02, decay=1e-5), 10,001 full-batch epochs; "
      "test set: a second spiral_data call after training; i = 'small', 'he', 'xavier'")
print("seed  train acc: small     he  xavier   test acc: small     he  xavier")
train = {init: [] for init in INITS}
test = {init: [] for init in INITS}
for seed in seeds:
    for init in INITS:
        r = train_stack(seed, 1, init, Optimizer_Adam(learning_rate=0.02, decay=1e-5), 10001)
        train[init].append(100 * accuracy_of(r["X"], r["y"], r["dense"], r["activations"]))
        X_test, y_test = spiral_data(samples=100, classes=3)
        test[init].append(100 * accuracy_of(X_test, y_test, r["dense"], r["activations"]))
    print(f"{seed:4d}             " + "  ".join(f"{train[init][-1]:5.2f}" for init in INITS) +
          "             " + "  ".join(f"{test[init][-1]:5.2f}" for init in INITS), flush=True)

print(f"range and mean over the {len(seeds)} seeds, in percent")
print("init     training accuracy         test accuracy")
for init in INITS:
    print(f"{init:<7}  {min(train[init]):5.2f} to {max(train[init]):5.2f} ({np.mean(train[init]):5.2f})    "
          f"{min(test[init]):5.2f} to {max(test[init]):5.2f} ({np.mean(test[init]):5.2f})")

print(f"seeds on which the first has the higher accuracy than the second, of {len(seeds)}; in brackets, seeds on which the two are equal")
print("pair               training   test")
for first, second in (("he", "small"), ("xavier", "small"), ("xavier", "he")):
    cells = []
    for figures in (train, test):
        higher = sum(a > b for a, b in zip(figures[first], figures[second]))
        equal = sum(a == b for a, b in zip(figures[first], figures[second]))
        cells.append(f"{higher} ({equal})")
    print(f"{first + ' over ' + second:<17}  {cells[0]:<9}  {cells[1]}")
