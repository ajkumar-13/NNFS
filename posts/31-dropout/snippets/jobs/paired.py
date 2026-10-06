"""Post 31, section 8: every seed-by-seed count of the section, and a fitted network handed to dropout.

Run from the series root:
    python posts/31-dropout/snippets/jobs/paired.py

Part 1: seeds 0 to 9 without a dropout layer and with Layer_Dropout at 0.1, 0.2 and 0.5, the
setup of network.py, and after each rate the comparison with the run of the same seed without
the layer. The rows of seeds 0 to 4 are those of the four seeds_*.py scripts.
Part 2: seeds 0 to 4 trained without dropout for 10,001 epochs, then for 10,001 more with
Layer_Dropout(0.1) and the same optimiser object. Other seeds can be given as arguments.

Needs NumPy and the nnfs package. Forty-five runs, about 6 to 12 minutes, which is why it is under jobs/.
"""
import sys
from pathlib import Path

import nnfs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from network import SETUP, Layer_Dropout, build, evaluate, paired, spread, train

nnfs.init()
print(SETUP)
seeds = tuple(int(argument) for argument in sys.argv[1:]) or tuple(range(10))

print()
base = spread(rate=None, seeds=seeds)
for rate in (0.1, 0.2, 0.5):
    print()
    rows = spread(rate=rate, seeds=seeds)
    paired(base, rows, f"Layer_Dropout({rate:g})")

print()
print("fitted without dropout for 10,001 epochs, then trained on for 10,001 epochs with Layer_Dropout(0.1)")
print("seed  before: train, test   after, mask off: train, test")
falls = []
for seed in seeds[:5]:
    X, y, X_test, y_test, dense1, activation1, dropout1, dense2, loss_activation, optimizer = build(seed, None)
    train(X, y, dense1, activation1, dropout1, dense2, loss_activation, optimizer)
    network = (dense1, activation1, Layer_Dropout(0.1), dense2, loss_activation)
    before = evaluate(X, y, *network)[1], evaluate(X_test, y_test, *network)[1]
    train(X, y, *network, optimizer)
    after = evaluate(X, y, *network)[1], evaluate(X_test, y_test, *network)[1]
    falls.append((100 * (before[0] - after[0]), 100 * (before[1] - after[1])))
    print(f"{seed:4d}  {before[0]:13.4f}, {before[1]:.4f}  {after[0]:23.4f}, {after[1]:.4f}", flush=True)
print(f"training accuracy, mask off, fell on {sum(f[0] > 0 for f in falls)} of {len(falls)} seeds, by "
      f"{min(f[0] for f in falls):.2f} to {max(f[0] for f in falls):.2f} points; test accuracy fell on "
      f"{sum(f[1] > 0 for f in falls)} of {len(falls)}, by {min(f[1] for f in falls):.2f} to {max(f[1] for f in falls):.2f} points")
