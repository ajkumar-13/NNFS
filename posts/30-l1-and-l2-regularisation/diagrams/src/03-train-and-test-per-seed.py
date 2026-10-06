"""Post 30, section 7: training and test accuracy of every run, no penalty against L2 and L1 at 5e-4, seeds 0 to 4.

Run from anywhere:  python posts/30-l1-and-l2-regularisation/diagrams/src/03-train-and-test-per-seed.py
Writes posts/30-l1-and-l2-regularisation/diagrams/03-train-and-test-per-seed.svg. Takes about a minute.

The three scripts of section 7, snippets/seeds_none.py, seeds_l2.py and seeds_l1.py, are run here as they are, side
by side (_runs.py), and their printed rows give every value drawn. The ranges of section 7's first table are
recomputed from the rows and asserted against index.md, and so are the paired statements of the text: the rise in
test accuracy and the narrowing of the gap on each seed, and the seed 0 figures.

Layout: one row per run, grouped by seed; each run a line from its test accuracy (solid mark) to its training
accuracy (hollow mark), so the line's length is the gap; test accuracy and gap printed in two columns.
"""
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figkit import Figure, rich, sup, num, text_width  # noqa: E402
from _runs import INDEX, runs, rng  # noqa: E402

NAMES = ("seeds_none", "seeds_l2", "seeds_l1")
R = runs(NAMES)
SEEDS = range(5)


def col(n, k):
    return [float(r[k]) if k in ("train_pct", "test_pct") else float(r[k]) for r in R[n]["rows"]]


# -- section 7's first table, row by row
for n, label in (("seeds_none", "none"), ("seeds_l2", "L2, $5 \\times 10^{-4}$"), ("seeds_l1", "L1, $5 \\times 10^{-4}$")):
    cells = (f"| {label} | `{n}.py` | {rng(col(n, 'train_pct'))} | {rng(col(n, 'test_pct'))} | "
             f"{rng(col(n, 'gap'))} | {re.search(r'test data loss (\d\.\d+ to \d\.\d+)$', R[n]['out'], re.M).group(1)} |")
    assert cells in INDEX, cells

TEST = {n: col(n, "test_pct") for n in NAMES}
TRAIN = {n: col(n, "train_pct") for n in NAMES}
GAP = {n: [a - b for a, b in zip(TRAIN[n], TEST[n])] for n in NAMES}     # exact; _runs asserts the printed gap
for n, rise, narrow in (("seeds_l2", "3.33 to 12.67", "3.00 to 5.67"), ("seeds_l1", "1.00 to 13.67", "0.67 to 3.33")):
    up = [a - b for a, b in zip(TEST[n], TEST["seeds_none"])]
    down = [b - a for a, b in zip(GAP[n], GAP["seeds_none"])]
    assert min(up) > 0 and min(down) > 0, n                           # on each of the five seeds
    assert rng(up) == rise and rng(down) == narrow, (n, rng(up), rng(down))
assert "test accuracy rose by 3.33 to 12.67 points, the gap narrowed by 3.00 to 5.67 points" in INDEX
assert "Test accuracy rose by 1.00 to 13.67 points and the gap narrowed by 0.67 to 3.33 points" in INDEX
assert (f"{TRAIN['seeds_none'][0]:.2f}", f"{TRAIN['seeds_l2'][0]:.2f}", f"{TEST['seeds_none'][0]:.2f}",
        f"{TEST['seeds_l2'][0]:.2f}") == ("95.33", "95.33", "78.67", "84.00")
assert "On seed 0 the figures are 95.33 percent in training with and without the penalty, and 78.67 against 84.00" \
       in INDEX
assert all(t1 > t2 for t1, t2 in zip(TRAIN["seeds_l1"], TRAIN["seeds_l2"]))  # L1 kept the higher training accuracy
assert "because L1 kept the higher training accuracy on all five seeds" in INDEX

STRENGTH = rich("5 × ", sup("10", num(-4), italic=False))
SET = [("seeds_none", "none", "ink-muted", "circle"), ("seeds_l2", "L2", "blue", "diamond"),
       ("seeds_l1", "L1", "ink", "square")]


def listing(n):
    return "; ".join(f"seed {s} {TEST[n][s]:.2f} to {TRAIN[n][s]:.2f}" for s in SEEDS)


fig = Figure(
    "03-train-and-test-per-seed", "Each penalty raised test accuracy on all five seeds",
    "One row per run for seeds 0 to 4, three runs per seed: no penalty, L2 at 5 times 10 to the minus 4 and L1 at "
    "the same strength on the first layer, on an accuracy axis from 70 to 100 percent. Each run is a line from its "
    "test accuracy, a solid mark, to its training accuracy, a hollow mark; the length is the gap. No penalty, test "
    f"to training accuracy in percent: {listing('seeds_none')}. L2: {listing('seeds_l2')}. L1: "
    f"{listing('seeds_l1')}. Two columns print the test accuracy and the gap in points of each run. On every seed "
    "both penalties have the higher test accuracy and the shorter line.",
    subtitle=rich("Seeds 0 to 4, 10,001 epochs, L1 or L2 at ", STRENGTH,
                  " on the first layer. Each line: test to training accuracy."),
    height=720, data_w=True)

LO, HI, TICKS = 70, 100, [70, 75, 80, 85, 90, 95, 100]
PX0, PX1 = 176, 712
TEST_X, GAP_X = 816, 920
sx = lambda v: PX0 + (PX1 - PX0) * (v - LO) / (HI - LO)  # noqa: E731
TOP, PITCH, GROUP = 160, 24, 88
cy = lambda s, k: TOP + 16 + GROUP * s + PITCH * k  # noqa: E731
AXIS = cy(4, 2) + 24
assert AXIS <= 600, AXIS

# -- column headings and the key on one line
HY = 140
fig.text(40, HY, "seed", "note")
fig.text(88, HY, "penalty", "note")
fig.text(TEST_X, HY, "test accuracy", "note", anchor="end")
fig.text(GAP_X, HY, "gap, points", "note", anchor="end")
with fig.data():
    for t in TICKS:
        fig.edge((sx(t), TOP), (sx(t), AXIS), color="grid", width=0.75)
    fig.edge((PX0, AXIS), (PX1, AXIS), color="ink-muted", width=1)
    for t in TICKS:
        fig.text(sx(t), AXIS + 20, f"{t}", "tick", anchor="middle", snap=False)
    for s in SEEDS:
        fig.text(40, cy(s, 0) + 5, str(s), "label", snap=False)
        for k, (n, label, color, shape) in enumerate(SET):
            y, a, b = cy(s, k), TEST[n][s], TRAIN[n][s]
            fig.edge((sx(a), y), (sx(b), y), color=color, width=1.5)
            fig.marker(sx(b), y, shape, color, size=10, hollow=True)
            fig.marker(sx(a), y, shape, color, size=10)
            fig.text(88, y + 5, label, "label", color=color if color != "ink-muted" else None, snap=False)
            fig.text(TEST_X, y + 5, f"{a:.2f}", "value", anchor="end", color=color, snap=False)
            fig.text(GAP_X, y + 5, f"{GAP[n][s]:.2f}", "value", anchor="end", color=color, snap=False)
fig.text((PX0 + PX1) / 2, AXIS + 44, "accuracy, percent", "note", anchor="middle")

# -- the key, on the headings' line over the plot: solid and hollow marks in the neutral
KY = HY
x = PX0
with fig.data():
    for hollow, text in ((False, "test accuracy"), (True, "training accuracy")):
        fig.marker(x + 8, KY - 4, "circle", "ink-muted", size=10, hollow=hollow)
        fig.text(x + 24, KY, text, "label", snap=False)
        x += 24 + math.ceil(text_width(text, 14, weight=400)) + 40
fig.caption("L2 raised test accuracy by 3.33 to 12.67 points, L1 by 1.00 to 13.67, and both shortened every gap.")
fig.write()
