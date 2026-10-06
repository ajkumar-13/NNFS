"""Post 08 hero (section 3): the loss -log(p) against the probability p on the true class.

Run from anywhere:  python posts/08-loss-categorical-cross-entropy/diagrams/src/01-cross-entropy-curve.py
Writes posts/08-loss-categorical-cross-entropy/diagrams/01-cross-entropy-curve.svg.
The curve is -np.log evaluated here; every value the figure prints is checked against what
snippets/neg_log_curve.py and snippets/clipping.py print.

Layout: one line chart. The curve -log(p) from p = 0.01 to 1, the straight-line loss 1 - p dashed for
comparison, and two shaded steps of the same width in p (0.01 to 0.10 and 0.90 to 0.99) with the loss
each one removes. The uniform guess over three classes, p = 1/3, is a blue diamond at ln 3.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Box, Figure, MINUS, rich, sup, var, num  # noqa: E402

SNIPPETS = Path(__file__).resolve().parents[2] / "snippets"


def run(name):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(SNIPPETS / f"{name}.py"), run_name="snippet")
    return buf.getvalue()


curve_out, clip_out = run("neg_log_curve"), run("clipping")

L = lambda p: float(-np.log(p))  # noqa: E731   the loss of one sample, natural log as in the post
# the table of section 3, as the snippet prints it
for p in [1.00, 0.90, 0.70, 0.50, 0.10, 0.01]:
    assert f"p = {p:.2f}   -log(p) = {abs(L(p)):.3f}" in curve_out, p
MARKS = [0.01, 0.10, 0.50, 1.00]                         # points labelled with their loss
LN3 = L(1 / 3)
assert f"K = 3      log(K) = {LN3:.3f}" in curve_out and f"{LN3:.3f}" == "1.099"
assert f"{L(0.01):.3f}" == "4.605" and f"{L(0.10):.3f}" == "2.303" and f"{L(0.50):.3f}" == "0.693"
STEPS = [(0.01, 0.10), (0.90, 0.99)]
DROP = [L(a) - L(b) for a, b in STEPS]
assert f"raising p from 0.01 to 0.10 removes {DROP[0]:.3f} of loss" in curve_out
assert f"raising p from 0.90 to 0.99 removes {DROP[1]:.3f} of loss" in curve_out
LINEAR = [(1 - a) - (1 - b) for a, b in STEPS]
assert f"a straight-line loss 1 - p would remove {LINEAR[0]:.2f} and {LINEAR[1]:.2f}" in curve_out
CLIP = L(1e-7)
assert f"the largest loss the clip allows, -log(1e-7): {CLIP:.3f}" in clip_out and f"{CLIP:.3f}" == "16.118"
assert f"the points marked in the first figure: " + ", ".join(
    f"{p} -> {abs(L(p)):.3f}" for p in MARKS) in curve_out

NEGLOG = rich(MINUS, "log(", var("p"), ")")
fig = Figure(
    "01-cross-entropy-curve", "The loss is the negative log of the true-class probability",
    "A line chart of the loss minus log p against the probability p on the true class, p from 0.01 to 1. "
    f"The curve falls from {L(0.01):.3f} at p = 0.01 through {L(0.10):.3f} at 0.10 and {L(0.50):.3f} at 0.50 "
    f"to 0 at p = 1. A blue diamond marks p = 1/3, the uniform guess over three classes, at ln 3 = {LN3:.3f}. "
    f"Two shaded steps of the same width in p: from 0.01 to 0.10 the loss falls by {DROP[0]:.3f}, from 0.90 to "
    f"0.99 by {DROP[1]:.3f}. A dashed straight line, 1 minus p, falls by {LINEAR[0]:.2f} on each step. The "
    f"caption adds that at the clip bound p = 10 to the minus 7 the loss is {CLIP:.3f}.",
    subtitle=rich("Zero at ", var("p"), " = 1, never negative, and steeper the closer ", var("p"),
                  " comes to 0; natural log."))

BOX = Box(40, 104, 880, 372)
LABEL_W = 120
X_LO, X_HI, Y_LO, Y_HI = 0, 1, 0, 5
# the plot area line_chart will use (same arithmetic as the kit: 56 for y ticks, 24 above, 48 below)
PX, PY = BOX.x + 56, BOX.y + 24
PW, PH = BOX.w - 56 - LABEL_W, BOX.h - 24 - 48
sx = lambda v: PX + PW * (v - X_LO) / (X_HI - X_LO)       # noqa: E731
sy = lambda v: PY + PH - PH * (v - Y_LO) / (Y_HI - Y_LO)  # noqa: E731

# the two steps of equal width, shaded behind the chart
for a, b in STEPS:
    fig.fill(Box(sx(a), PY, sx(b) - sx(a), PH), "input-soft", fit=False)

# the straight-line loss 1 - p, for comparison, under the curve
with fig.data():
    fig.edge((sx(0.01), sy(0.99)), (sx(1.0), sy(0.0)), color="ink-muted", width=1.5, dash="ref")

ps = np.linspace(0.01, 1.0, 199)
series = [
    dict(xs=ps.tolist(), ys=[abs(L(p)) for p in ps], color="error", points=False, label=NEGLOG),
]
plot = fig.line_chart(BOX, series, x=(X_LO, X_HI, [0, 0.2, 0.4, 0.6, 0.8, 1.0]), y=(Y_LO, Y_HI, [0, 1, 2, 3, 4, 5]),
                      x_label=rich("probability ", var("p"), " on the true class"),
                      y_label=rich("loss ", NEGLOG), label_w=LABEL_W,
                      fmt_x=lambda v: f"{v:.1f}", fmt_y=lambda v: str(v))
assert (plot.x, plot.y, plot.w, plot.h) == (PX, PY, PW, PH)

with fig.data():
    for p in MARKS:
        fig.marker(sx(p), sy(abs(L(p))), "circle", color="error")
    fig.marker(sx(1 / 3), sy(LN3), "diamond", color="input")
    # value labels, each to the right of its point and clear of the curve
    fig.text(sx(0.01) + 12, sy(L(0.01)) + 5, f"{L(0.01):.3f}", "value", color="error", snap=False)
    fig.text(sx(0.10) + 12, sy(L(0.10)) + 5, f"{L(0.10):.3f}", "value", color="error", snap=False)
    fig.text(sx(0.50) + 10, sy(L(0.50)) - 12, f"{L(0.50):.3f}", "value", color="error", snap=False)
    fig.text(sx(1 / 3) + 14, sy(LN3) - 10, rich("ln 3 = ", f"{LN3:.3f}", ", a uniform guess over 3 classes"),
             "label", color="input", snap=False)
    # the two steps: each label stacked on three lines just right of its own band, in the same way
    for (a, b), drop, top in zip(STEPS, DROP, (3.9, 1.6)):
        x = sx(b) + 12
        fig.text(x, sy(top), rich(var("p"), f" from {a:.2f}"), "label", snap=False)
        fig.text(x, sy(top) + 20, f"to {b:.2f}:", "label", snap=False)
        fig.text(x, sy(top) + 40, rich(MINUS, f"{drop:.3f}"), "label", snap=False)
    fig.text(sx(0.10) + 12, sy(0.25), rich("dashed, 1 ", MINUS, " ", var("p"), ": ", MINUS,
             f"{LINEAR[0]:.2f}", " on each step"), "note", snap=False)
fig.caption(rich("Off the chart to the left, the clip bound ", var("p"), " = ", sup("10", num(-7), italic=False),
                 " costs ", f"{CLIP:.3f}", "; without the clip the loss has no upper bound."))
fig.write()
