"""Post 20, section 5: the fall in loss of one update, for ten seeds, on the post's batch and on a balanced batch.

Run from anywhere:  python posts/20-assembling-full-backpropagation/diagrams/src/03-fall-per-seed.py
Writes posts/20-assembling-full-backpropagation/diagrams/03-fall-per-seed.svg.
snippets/assemble.py is run here (runpy) and its one_step function gives every fall drawn: seeds 0 to 9 on the
batch of four and on the balanced batch of six. The extremes are asserted against the two lines the snippet prints.
The bias gradient of the last layer on both batches is computed here with the snippet's classes, for the note.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, sup, num  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "assemble.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
one_step = s["one_step"]
X, y, XB, YB = s["X"], s["y"], s["X_balanced"], s["y_balanced"]

POST = [a - b for a, b in (one_step(seed) for seed in range(10))]
BAL = [a - b for a, b in (one_step(seed, X=XB, y=YB) for seed in range(10))]
assert f"fall in loss: smallest {min(POST):.3e}, largest {max(POST):.3e}" in OUT
assert f"balanced batch of six: smallest {min(BAL):.3e}, largest {max(BAL):.3e}" in OUT
assert (f"{min(POST):.3e}", f"{max(POST):.3e}") == ("4.169e-04", "4.191e-04")
assert (f"{min(BAL):.3e}", f"{max(BAL):.3e}") == ("2.723e-07", "1.409e-06")
assert min(POST) / max(BAL) > 100                         # the caption: more than two powers of ten apart
assert sorted(np.bincount(YB)) == [2, 2, 2] and sorted(np.bincount(y)) == [1, 1, 2]


def db2(Xb, yb, seed=0):
    """dense2.dbiases after one backward pass of the snippet's network, built from a seed."""
    np.random.seed(seed)
    d1, a1 = s["Layer_Dense"](2, 3), s["Activation_ReLU"]()
    d2, la = s["Layer_Dense"](3, 3), s["Activation_Softmax_Loss_CategoricalCrossentropy"]()
    d1.forward(Xb); a1.forward(d1.output); d2.forward(a1.output); la.forward(d2.output, yb)
    la.backward(la.output, yb); d2.backward(la.dinputs)
    return d2.dbiases


assert np.abs(db2(XB, YB)).max() < 0.01 * np.abs(db2(X, y)).max()             # nearly cancels when balanced


def sci(v):
    m, e = f"{v:.3e}".split("e")
    return rich(m, " × ", sup("10", num(int(e)), italic=False))


fig = Figure(
    "03-fall-per-seed", "One step's fall depends on the batch, not the seed",
    "Two ranges on a logarithmic axis from 10 to the minus 7 to 10 to the minus 3: the fall in loss after one "
    "update with learning rate 0.01, one tick per seed, seeds 0 to 9. On the post's batch of four, labels 0, 1, 2, 1, "
    "the ten falls lie between 4.169 times 10 to the minus 4 and 4.191 times 10 to the minus 4, all in one place at "
    "this scale. On a balanced batch of six, labels 0, 1, 2, 1, 0, 2, they range from 2.723 times 10 to the minus 7 "
    "to 1.409 times 10 to the minus 6. A note says that on the balanced batch the bias gradient of dense2 nearly "
    "cancels, and what is left is small because the weights of the 0.01 initialisation are.",
    subtitle="One update with learning rate 0.01, from each of the seeds 0 to 9. Logarithmic axis.",
    data_w=True)

LO, HI = 1e-7, 1e-3
TICKS = [10.0 ** e for e in range(-7, -2)]
assert LO < min(BAL) and max(POST) < HI
ax = fig.dot_plot(Box(40, 128, 880, 216), [("post's batch of four", []), ("balanced batch of six", [])],
                  LO, HI, TICKS, x_log=True, label_w=200, pad_right=24, axis_label="fall in loss")
with fig.data():
    for i, (vals, lab) in enumerate(((POST, y), (BAL, YB))):
        fig.text(40, ax.sy(i) + 25, "labels " + ", ".join(str(v) for v in lab), "note")
        fig.range_mark(ax.sx(min(vals)), ax.sx(max(vals)), ax.sy(i), ticks=[ax.sx(v) for v in vals],
                       color="gradient", dot=ax.sx(float(np.mean(vals))) if i == 0 else None)
# each range's min and max, labelled once: the post's to the left of its band, the balanced one's to the right
ax.text(min(POST), 0, rich(sci(min(POST)), "  to ", sci(max(POST))), "value", anchor="end", color="gradient",
        dx=-16, dy=5)
ax.text(float(np.mean(POST)), 0, "all ten in one place", "note", anchor="middle", dy=36)
ax.text(max(BAL), 1, rich(sci(min(BAL)), "  to ", sci(max(BAL))), "value", color="gradient", dx=16, dy=5)
fig.legend(240, 392, [dict(color="gradient", label="range over seeds 0 to 9, one tick per seed", mark="band")],
           direction="row")
fig.text(240, 428, "On the balanced batch the bias gradient of dense2 nearly cancels; what is left", "note")
fig.text(240, 448, "is small because the weights of the 0.01 initialisation are.", "note")

fig.caption("The seed barely moves the fall; on the balanced batch one step lowers the loss by a millionth or less.")
fig.write()
