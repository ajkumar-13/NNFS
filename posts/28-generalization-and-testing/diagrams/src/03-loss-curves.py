"""Post 28, section 5.1: seed 0's loss and accuracy on the training points and on the test points, every 100 epochs.

Run from anywhere:  python posts/28-generalization-and-testing/diagrams/src/03-loss-curves.py   (about 15 to 30 s)
Writes posts/28-generalization-and-testing/diagrams/03-loss-curves.svg.
snippets/seed_spread.py's run(0) is called here, as its main block calls it for seed 0: it trains the documented
network and records both sets forward-only at the start of every 100th epoch, 101 checks. Every row the snippet
prints from that record (epochs 0, 100, 300, 700, 1,000, 2,000, 5,000 and 10,000) is formatted as the snippet
formats it and asserted to be a line of the section 5.1 listing in index.md; the check of lowest test loss (epoch
700) is asserted against the seed 0 row of the second listing. The curves draw all 101 checks; the dots mark the
printed epochs.

Layout: two charts on one epoch axis, the loss above and the accuracy below.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
INDEX = (POST / "index.md").read_text(encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()):
    import nnfs  # noqa: E402
    from seed_spread import run  # noqa: E402
nnfs.init()                                  # as the snippet's main block does before its runs
final, lowest = run(0)
CURVE = lowest["curve"]
assert len(CURVE) == 101 and [c[0] for c in CURVE] == list(range(0, 10001, 100))
PRINTED = (0, 100, 300, 700, 1000, 2000, 5000, 10000)
AT = {c[0]: c for c in CURVE}
for e in PRINTED:
    _, trl, tra, tel, tea = AT[e]
    line = f"{e:5d}  {trl:10.4f}  {tra:9.4f}  {tel:9.4f}  {tea:8.4f}"
    assert line in INDEX, line
assert lowest["epoch"] == 700 and min(CURVE, key=lambda c: c[3])[0] == 700
assert (f"{lowest['test_loss']:.4f}", f"{final['test_loss']:.4f}") == ("0.4900", "1.1168")
assert "   0    700  0.4900 -> 1.1168     0.8133 -> 0.8233" in INDEX
assert f"{np.log(3):.4f}" == "1.0986"

EP = [c[0] for c in CURVE]
RED, GREEN, BLUE = "error", "output", "blue"
f4 = lambda v: f"{v:.4f}"                    # noqa: E731
pc = lambda v: f"{100 * v:.2f}"              # noqa: E731
v = {e: AT[e] for e in PRINTED}

fig = Figure(
    "03-loss-curves", "The test loss bottoms out at epoch 700 and then climbs",
    "Two charts of seed 0's documented run against epochs 0 to 10,000, read forward-only every 100 epochs on the "
    "300 training points and on the 300 test points; circles (training) and hollow diamonds (test) mark the epochs the script prints. Top, the loss: both "
    "start at 1.0986, ln 3. The training loss keeps falling, to 0.0806 at epoch 10,000. The test loss falls "
    "to its lowest, 0.4900, at epoch 700, stays near 0.5 up to epoch 2,000 (0.5347), climbs to 0.8000 at epoch "
    "5,000 and to 1.1165 at epoch 10,000. Bottom, the accuracy: the training accuracy rises from 36.00 to 96.33 "
    "percent; the test accuracy is 81.33 percent at epoch 700, 84.67 at epoch 5,000 and 82.33 at epoch 10,000. A "
    "dotted vertical line marks epoch 700 in both charts.",
    subtitle="Seed 0, the documented run: both sets read forward-only at the start of every 100th epoch.",
    height=720, data_w=True)

X_AX = (0, 10000, [0, 2000, 4000, 6000, 8000, 10000])
LABEL_W = 152
REF700 = [dict(x=700, label="lowest test loss, epoch 700")]

# -- the loss
fig.text(40, 120, "Loss", "head")
ax = fig.line_chart(
    Box(40, 128, 880, 256),
    [dict(xs=EP, ys=[c[1] for c in CURVE], color=RED, points=False, label=None),
     dict(xs=EP, ys=[c[3] for c in CURVE], color=BLUE, points=False, label=None)],
    x=X_AX, y=(0, 1.4, [0, 0.4, 0.8, 1.2]), fmt_x=lambda t: num(t), fmt_y=lambda t: f"{t:.1f}",
    labels=False, label_w=LABEL_W,
    ref_lines=REF700 + [dict(y=float(np.log(3)), label="ln 3 = 1.0986", at="middle")])
for e in PRINTED:
    ax.point(e, v[e][1], "circle", RED, size=8)
    ax.point(e, v[e][3], "diamond", BLUE, size=13, hollow=True)
ax.text(10000, v[10000][3], "test, 1.1165", "value", color=BLUE, dx=12, dy=5)
ax.text(10000, v[10000][1], "training, 0.0806", "value", color=RED, dx=12, dy=5)
ax.text(760, 0.37, "lowest, 0.4900", "value", color=BLUE, dy=5)
ax.text(5000, v[5000][3], "0.8000", "value", color=BLUE, anchor="middle", dy=-12)

# -- the accuracy
fig.text(40, 424, "Accuracy", "head")
ax = fig.line_chart(
    Box(40, 432, 880, 224),
    [dict(xs=EP, ys=[100 * c[2] for c in CURVE], color=GREEN, points=False, label=None),
     dict(xs=EP, ys=[100 * c[4] for c in CURVE], color=BLUE, points=False, label=None)],
    x=X_AX, y=(30, 100, [40, 60, 80, 100]), fmt_x=lambda t: num(t), fmt_y=lambda t: f"{t:.0f}%",
    labels=False, label_w=LABEL_W, x_label="epoch", ref_lines=[dict(x=700)])
for e in PRINTED:
    ax.point(e, 100 * v[e][2], "circle", GREEN, size=8)
    ax.point(e, 100 * v[e][4], "diamond", BLUE, size=13, hollow=True)
ax.text(10000, 100 * v[10000][2], "training, 96.33%", "value", color=GREEN, dx=12, dy=-2)
ax.text(10000, 100 * v[10000][4], "test, 82.33%", "value", color=BLUE, dx=12, dy=12)
ax.text(700, 100 * v[700][4], "81.33%", "value", color=BLUE, dx=8, dy=24)
ax.text(5000, 100 * v[5000][4], "84.67%", "value", color=BLUE, anchor="middle", dy=24)

fig.caption("Circles: the training points. Hollow diamonds: the test points. Marks sit at the epochs the script prints.")
fig.write()
