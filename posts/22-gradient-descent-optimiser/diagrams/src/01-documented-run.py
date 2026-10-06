"""Post 22, sections 3 and 4: the documented run, every epoch of it, against the eleven rows its log prints.

Run from anywhere:  python posts/22-gradient-descent-optimiser/diagrams/src/01-documented-run.py
Writes posts/22-gradient-descent-optimiser/diagrams/01-documented-run.svg.
snippets/train_sgd.py is run here (runpy, about 7 s): its own losses and accuracies lists give every value drawn,
and the lines it prints (the log every 1,000 epochs and the summary of what happened between them) are asserted
against the numbers the figure labels. The thin curves are the min and max of every 10 epochs in time order
(about 0.7 viewBox units per bin), so every peak and trough of the run is kept while the file stays small.
"""
import contextlib
import io
import math
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, num, MINUS  # noqa: E402

SNIPPETS = Path(__file__).resolve().parents[2] / "snippets"
sys.path.insert(0, str(SNIPPETS))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPETS / "train_sgd.py"), run_name="snippet")
OUT = buf.getvalue()
L = np.asarray(s["losses"], dtype=float)
A = np.asarray(s["accuracies"], dtype=float)
assert len(L) == len(A) == 10001

# -- what the log prints, and what the figure labels
LOGGED = list(range(0, 10001, 1000))
for e in LOGGED:
    assert f"epoch {e:5d}  loss {L[e]:.4f}  acc {A[e]:.4f}" in OUT
assert f"{L[0]:.4f}" == "1.0986" and f"{L[10000]:.4f}" == "0.8737" and f"{A[10000]:.4f}" == "0.6467"
steps = np.diff(L)
RISES = int(np.sum(steps > 0))
JUMP_AT = int(steps.argmax()) + 1                 # the epoch whose loss is the largest rise above the one before
LOW_AT = int(L.argmin())
assert "the loss rose on 4,628 of 10,000 updates; largest single rise 0.0553, at epoch 1,604" in OUT
assert (RISES, JUMP_AT, f"{steps.max():.4f}") == (4628, 1604, "0.0553")
assert "lowest loss 0.8343, at epoch 7,874" in OUT and (LOW_AT, f"{L.min():.4f}") == (7874, "0.8343")
LAST = slice(9001, 10001)
assert "epochs 9,001 to 10,000: loss 0.8350 to 0.9330, accuracy 0.5633 to 0.6533" in OUT
A_LO, A_HI = A[LAST].min(), A[LAST].max()
assert (f"{A_LO:.4f}", f"{A_HI:.4f}") == ("0.5633", "0.6533")
LN3 = math.log(3)
assert f"{LN3:.4f}" == "1.0986"


def envelope(v, k=10):
    """The min and the max of every k epochs, in the order they occur, as (epoch, value) points."""
    pts = []
    for a in range(0, len(v), k):
        w = v[a:a + k]
        i, j = a + int(w.argmin()), a + int(w.argmax())
        for t in sorted({i, j}):
            pts.append((t, float(v[t])))
    return pts


THETA, ALPHA = var("θ"), var("α")
fig = Figure(
    "01-documented-run", "The log reads as a descent; the loss rose on 4,628 updates",
    "Two charts against the epoch, 0 to 10,000, of the documented run: seed 0, the update theta minus alpha times "
    "g with alpha = 1, all 300 spiral samples in every update. A thin line draws every epoch and an ink line "
    "with dots the 11 rows the log prints, one every 1,000 epochs. Top, the loss: it starts at ln 3 = 1.0986, "
    "marked by a dotted line, and the printed rows fall to 0.8737 at epoch 10,000, but the thin line bounces "
    "and rose on 4,628 of the 10,000 updates; the largest single rise, 0.0553, ends at epoch 1,604, and the "
    "lowest loss is 0.8343, at epoch 7,874. Bottom, the accuracy: the printed rows go from 0.3600 to 0.6467, "
    "and over epochs 9,001 to 10,000, shaded, the accuracy moved between 0.5633 and 0.6533.",
    subtitle=rich("The documented run: seed 0, ", THETA, " ← ", THETA, " ", MINUS, " ", ALPHA, " ", var("g"),
                  " with ", ALPHA, " = 1, all 300 samples in every update."),
    height=720, data_w=True)

X_AX = (0, 10000, [0, 2000, 4000, 6000, 8000, 10000])
FMT_X = lambda v: num(int(v))  # noqa: E731
LABEL_W = 120

# -- top: the loss
top = fig.panel(Box(40, 104, 880, 296), rich("Loss: it rose on ", num(RISES), " of the 10,000 updates"))
axl = fig.line_chart(
    top, [dict(xs=LOGGED, ys=[float(L[e]) for e in LOGGED], color="ink", label=f"{L[10000]:.4f}")],
    x=X_AX, y=(0.75, 1.2, [0.8, 0.9, 1.0, 1.1, 1.2]), fmt_x=FMT_X, fmt_y=lambda v: f"{v:.1f}",
    label_w=LABEL_W, points=False, ref_lines=[dict(y=LN3, label=rich("ln 3 = ", f"{LN3:.4f}"), at="right")])
axl.polyline(envelope(L), color="error", width=1)
axl.polyline([(e, float(L[e])) for e in LOGGED], color="ink", width=1.5)
for e in LOGGED:
    axl.point(e, float(L[e]), "circle", "ink", size=8)
axl.point(JUMP_AT, float(L[JUMP_AT]), "circle", "error", size=10, hollow=True)
axl.text(JUMP_AT, float(L[JUMP_AT]), rich("largest rise, ", f"{steps.max():.4f}", ", epoch ", num(JUMP_AT)),
         "note", dx=12, dy=-4)
axl.point(LOW_AT, float(L[LOW_AT]), "circle", "error", size=10, hollow=True)
axl.text(LOW_AT, float(L[LOW_AT]), rich("lowest, ", f"{L.min():.4f}", ", epoch ", num(LOW_AT)), "note",
         anchor="middle", dy=24)

# -- bottom: the accuracy
bot = fig.panel(Box(40, 424, 880, 232), rich("Accuracy: ", f"{A_LO:.4f}", " to ", f"{A_HI:.4f}",
                                                  " in the shaded last 1,000 epochs"))
axa = fig.line_chart(
    bot, [dict(xs=LOGGED, ys=[float(A[e]) for e in LOGGED], color="ink", label=f"{A[10000]:.4f}")],
    x=X_AX, y=(0.25, 0.75, [0.3, 0.4, 0.5, 0.6, 0.7]), fmt_x=FMT_X, fmt_y=lambda v: f"{v:.1f}", x_label="epoch",
    label_w=LABEL_W, points=False, bands=[(9001, 10000, "output")])
axa.polyline(envelope(A), color="output", width=1)
axa.polyline([(e, float(A[e])) for e in LOGGED], color="ink", width=1.5)
for e in LOGGED:
    axa.point(e, float(A[e]), "circle", "ink", size=8)
axa.text(0, float(A[0]), f"{A[0]:.4f}", "note", dx=8, dy=20)

# -- the key, on the loss heading's row
KY = 120
fig.legend(560, KY, [dict(color="error", label="every epoch", mark="line"),
                     dict(color="ink", label="the 11 printed rows", mark="circle")], direction="row")

fig.caption("Eleven printed rows read as a steady descent; the loss rose on nearly half of the updates.")
fig.write()
