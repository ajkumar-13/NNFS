"""Post 28 hero (section 3): the same weights on the training points and on 300 new points, seeds 0 to 4.

Run from anywhere:  python posts/28-generalization-and-testing/diagrams/src/01-gap-over-seeds.py
Writes posts/28-generalization-and-testing/diagrams/01-gap-over-seeds.svg.
Source: snippets/seed_spread.py takes about 45 seconds, so its section 3 table is parsed from the listing in
index.md, which the series lint (--run) keeps equal to the snippet's output. Each printed accuracy is a count out of
300; each row's gap is recomputed from the two counts with the snippet's format, and the ranges the post states
are asserted on the rows. Seed 0's counts
289 and 247 of 300 are asserted against the test_pass.py listing.

Layout: one dot plot, a row per seed, the training accuracy and the test accuracy joined by a line, and three
value columns (train, test, gap).
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
INDEX = (POST / "index.md").read_text(encoding="utf-8")

HEAD = "seed  train loss  train acc  test loss  test acc  gap in points"
block = INDEX[INDEX.index(HEAD):].split("```")[0].splitlines()[1:6]
ROWS = []
for line in block:
    m = re.fullmatch(r" +(\d)  +(\d\.\d{4})  +(\d\.\d{4})  +(\d\.\d{4})  +(\d\.\d{4})  +(\d+\.\d{2})", line)
    assert m, line
    seed, tr, te, gap = int(m.group(1)), float(m.group(3)), float(m.group(5)), m.group(6)
    n_tr, n_te = round(300 * tr), round(300 * te)       # the printed accuracies are counts out of 300
    assert f"{n_tr / 300:.4f}" == m.group(3) and f"{n_te / 300:.4f}" == m.group(5)
    assert f"{(n_tr - n_te) / 3:.2f}" == gap, (seed, gap)
    ROWS.append((seed, n_tr / 300, n_te / 300, gap))
assert [r[0] for r in ROWS] == [0, 1, 2, 3, 4]
TRAIN = [100 * r[1] for r in ROWS]
TEST = [100 * r[2] for r in ROWS]
GAPS = [float(r[3]) for r in ROWS]
assert (f"{min(GAPS):.2f}", f"{max(GAPS):.2f}") == ("10.67", "14.00") and all(g > 0 for g in GAPS)
assert (f"{min(TRAIN):.2f}", f"{max(TRAIN):.2f}") == ("78.00", "96.33")
assert (f"{min(TEST):.2f}", f"{max(TEST):.2f}") == ("67.33", "82.67")
assert "The gap is the steadier quantity: 10.67 to 14.00 points, and positive in every one of the five runs." in INDEX
assert "training data  loss 0.0806  acc 0.9633  (289 of 300)" in INDEX
assert "test data      loss 1.1168  acc 0.8233  (247 of 300)" in INDEX
assert round(TRAIN[0] * 3) == 289 and round(TEST[0] * 3) == 247

GREEN, BLUE = "output", "blue"        # accuracy in its role colour; the test pass, the post's method, in blue
fmt = lambda v: f"{v:.2f}"             # noqa: E731
TEST_SIZE = 13                         # a hollow diamond that clears a size-10 circle inside it

fig = Figure(
    "01-gap-over-seeds", "The same weights, 10.67 to 14.00 points lower on new points",
    "A dot plot with one row per seed, 0 to 4, on an accuracy axis from 60 to 100 percent, for the 64-neuron "
    "network of post 27 after 10,001 epochs. A green circle marks the accuracy on its 300 training points and a "
    "hollow blue diamond the accuracy on 300 new points from the same generator, read forward-only with the same "
    "weights; a line joins the two. Training and test: "
    + "; ".join(f"seed {r[0]} {fmt(a)} and {fmt(b)} percent, gap {r[3]} points" for r, a, b in zip(ROWS, TRAIN, TEST))
    + ". Seed 0 is the documented run, 289 and 247 of 300 points. The test figure is below the training figure "
    "on every seed.",
    subtitle="Network of post 27, 10,001 epochs: accuracy on its 300 training points and on 300 new points.",
    data_w=True)

LABEL_W, PAD_R = 96, 280
rows = [(f"seed {r[0]}", []) for r in ROWS]
ax = fig.dot_plot(Box(40, 136, 880, 296), rows, 60, 100, [60, 70, 80, 90, 100], fmt=lambda v: num(v), unit="%",
                  label_w=LABEL_W, pad_right=PAD_R, axis_label="accuracy")
with fig.data():                               # the dumbbells by hand: the test mark hollow, drawn last, so a
    for k, (a, b) in enumerate(zip(TRAIN, TEST)):   # training circle under it stays visible
        fig.edge((ax.sx(b), ax.sy(k)), (ax.sx(a), ax.sy(k)), color="rule", width=1.5)
        fig.marker(ax.sx(a), ax.sy(k), "circle", GREEN, size=10)
        fig.marker(ax.sx(b), ax.sy(k), "diamond", BLUE, size=TEST_SIZE, hollow=True)
XT, XE, XG = 728, 816, 920                     # right ends of the value columns
with fig.data():
    for k, (r, a, b) in enumerate(zip(ROWS, TRAIN, TEST)):
        y = ax.sy(k) + 5
        fig.text(XT, y, fmt(a), "value", anchor="end", color=GREEN, snap=False)
        fig.text(XE, y, fmt(b), "value", anchor="end", color=BLUE, snap=False)
        fig.text(XG, y, r[3], "label", anchor="end", snap=False, bold=(k == 0))
    fig.text(ax.sx(TEST[0]) - 12, ax.sy(0) + 5, "documented run, 289 and 247 of 300", "note", anchor="end",
             snap=False)
fig.text(XT, 120, "train", "note", anchor="end")
fig.text(XE, 120, "test", "note", anchor="end")
fig.text(XG, 120, "gap, points", "note", anchor="end")

fig.legend(240, 468, [dict(color=GREEN, label="training points, the 300 it was fitted to", mark="circle"),
                      dict(color=BLUE, label="300 new points, forward-only", mark="diamond", hollow=True)],
           direction="row", gap=40)
fig.caption("The training figure runs from 78.00 to 96.33 percent; the gap stays between 10.67 and 14.00 points.")
fig.write()
