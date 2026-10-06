"""Post 06 sections 2 and 2.1: the ReLU graph, and the slopes of ReLU, sigmoid and tanh.

Run from anywhere:  python posts/06-activation-functions-relu-and-softmax/diagrams/src/02-relu.py
Writes posts/06-activation-functions-relu-and-softmax/diagrams/02-relu.svg.
snippets/relu.py is run (runpy). The left curve is the snippet's Activation_ReLU applied here to a fine grid of
inputs; the dots are the five inputs of section 2.3 and the outputs the snippet prints for them. The right curves
are the snippet's own sigmoid_slope and tanh_slope evaluated on a fine grid; the ReLU slope is 0 below zero and 1
above it, with 0 at z = 0, the value the post says the backward pass uses. The peak slopes and the slope table
are asserted against the snippet's printout.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, num  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "relu.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    S = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

# -- the function, from the post's class
relu = S["Activation_ReLU"]()
INPUTS = S["inputs"]
relu.forward(INPUTS)
OUTPUTS = relu.output
assert INPUTS.tolist() == [1, -2, 3, -0.5, 0] and OUTPUTS.tolist() == [1, 0, 3, 0, 0]
assert OUT.startswith(f"{OUTPUTS}\n") and str(OUTPUTS) == "[1. 0. 3. 0. 0.]"
Z = np.linspace(-3.0, 3.0, 601)
relu.forward(Z)
F = relu.output
assert np.array_equal(F, np.where(Z > 0, Z, 0.0))

# -- the slopes, from the post's functions
sig_slope, tanh_slope = S["sigmoid_slope"], S["tanh_slope"]
ZS = np.linspace(-5.0, 5.0, 401)
SIG, TANH = sig_slope(ZS), tanh_slope(ZS)
assert ZS[200] == 0.0 and SIG.max() == SIG[200] == 0.25 and TANH.max() == TANH[200] == 1.0
assert "largest sigmoid slope, at z = 0: 0.25" in OUT and "largest tanh slope, at z = 0:    1.0" in OUT
for z in [0.5, 2.5, 5.0, 10.0]:
    assert f"{z:5.1f}   {1:10.0f}   {sig_slope(z):13.2e}   {tanh_slope(z):10.2e}" in OUT

fig = Figure(
    "02-relu", "ReLU zeroes negatives and keeps a slope of 1 above zero",
    "Two charts. Left, ReLU of z = max(0, z) for z from minus 3 to 3, computed with the post's Activation_ReLU: "
    "a flat line at 0 for negative inputs, labelled negative in, 0 out, a kink at z = 0, and a diagonal for positive "
    "inputs, labelled positive in, unchanged. Dots mark the five inputs of section 2.3, 1, minus 2, 3, minus 0.5 and "
    "0, at their outputs 1, 0, 3, 0 and 0. Right, the slopes for z from minus 5 to 5: ReLU's slope is 0 below zero "
    "and 1 above it, with a solid dot at 0 and an open dot at 1 for z = 0; the sigmoid's slope peaks at 0.25 at "
    "z = 0 and tanh's at 1, and both fall towards 0 on either side.",
    subtitle="Left: what the post's Activation_ReLU returns. Right: the slopes a backward signal is multiplied by.")

left, right = fig.row(2)
CHART_TOP = 168

# -- left: the function
body = fig.panel(left, rich("ReLU(", var("z"), ") = max(0, ", var("z"), ")"))
fig.text(body.x, body.y + 4, "Dots: the five inputs of section 2.3", "note")
ax = fig.line_chart(Box(left.x, CHART_TOP, left.w, left.bottom - CHART_TOP),
                    [dict(xs=Z.tolist(), ys=F.tolist(), color="output", label="", points=False)],
                    x=(-3, 3, [-3, -2, -1, 0, 1, 2, 3]), y=(-1, 3, [-1, 0, 1, 2, 3]),
                    x_label=rich("input ", var("z")), y_label="output", label_w=24)
with fig.data():
    for z, a in zip(INPUTS, OUTPUTS):
        fig.marker(*ax.to_px(z, a), "circle", "input", size=10)
    px, py = ax.to_px(-1.5, 0)
    fig.text(px, py - 12, "negative in, 0 out", "note", anchor="middle", snap=False)
    px, py = ax.to_px(1.6, 2.0)                    # above and left of the diagonal, which is at z = 2 here
    fig.text(px, py, "positive in, unchanged", "note", anchor="end", snap=False)
    px, py = ax.to_px(0, 0)
    fig.text(px + 8, py + 24, rich("kink at ", var("z"), " = 0"), "note", snap=False)

# -- right: the slopes
body = fig.panel(right, "Slopes of ReLU, sigmoid and tanh")
fig.text(body.x, body.y + 4, rich("ReLU's is 1 for every ", var("z"), " > 0; the sigmoid's is at most 0.25"),
         "note")
fmt = lambda t: num(t, 2) if t not in (0, 1) else num(t)  # noqa: E731
ax2 = fig.line_chart(Box(right.x, CHART_TOP, right.w, right.bottom - CHART_TOP),
                     [dict(xs=ZS.tolist(), ys=TANH.tolist(), color="input", label="", points=False),
                      dict(xs=ZS.tolist(), ys=SIG.tolist(), color="weight", label="", points=False)],
                     x=(-5, 5, [-4, -2, 0, 2, 4]), y=(-0.1, 1.1, [0, 0.25, 0.5, 0.75, 1]),
                     x_label=rich("input ", var("z")), y_label="slope", fmt_y=fmt, label_w=24)
with fig.data():
    # ReLU's slope: two flat pieces; at z = 0 it is taken as 0 (solid dot), not 1 (open dot)
    fig.edge(ax2.to_px(-5, 0), ax2.to_px(0, 0), color="gradient", width=1.5)
    fig.edge(ax2.to_px(0, 1), ax2.to_px(5, 1), color="gradient", width=1.5)
    fig.marker(*ax2.to_px(0, 0), "circle", "gradient", size=10)
    fig.marker(*ax2.to_px(0, 1), "circle", "gradient", size=10, hollow=True)
    px, py = ax2.to_px(3, 1)
    fig.text(px, py - 12, "ReLU", "label", color="gradient", anchor="middle", snap=False)
    # direct labels beside their own curves, clear of the other curve (checked below)
    for (z0, y0, anchor) in [(-1.1, 0.55, "end"), (2.4, 0.16, "start")]:
        assert TANH[np.searchsorted(ZS, z0)] < y0 - 0.15 or anchor == "start"
    assert sig_slope(2.4) < 0.16 - 0.06 and tanh_slope(2.4) < 0.16 - 0.06
    px, py = ax2.to_px(-1.1, 0.55)
    fig.text(px - 4, py, "tanh", "label", color="input", anchor="end", snap=False)
    px, py = ax2.to_px(2.4, 0.16)
    fig.text(px, py, "sigmoid", "label", color="weight", snap=False)

fig.caption("Past zero, ReLU passes a signal back unshrunk; the sigmoid scales it by 0.25 at best.")
fig.write()
