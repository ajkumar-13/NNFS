"""Post 30, section 3: one weight under each penalty alone, plain gradient steps, learning rate 0.1, lambda 0.01.

Run from anywhere:  python posts/30-l1-and-l2-regularisation/diagrams/src/02-one-weight-plain-steps.py
Writes posts/30-l1-and-l2-regularisation/diagrams/02-one-weight-plain-steps.svg. Takes about a second.

snippets/single_weight.py is imported (its main block does not run): its one_weight and plain_steps give the weight
after each of 1,000 plain steps from 0.0105, which is what the snippet prints from. Every printed row of the listing
of section 3 in index.md is asserted against these paths, as are the L1 range after step 10 (0.0005 either side of
zero, never 0) and the L2 factor 0.998 with 0.0105 * 0.998^t on every step.

Layout: two charts with the same weight axis, steps 0 to 20 left and 0 to 1,000 right; L1 in ink, L2 in blue.
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from single_weight import one_weight, plain_steps  # noqa: E402

INDEX = (POST / "index.md").read_text(encoding="utf-8")
START, LAM, LR, STEPS = 0.0105, 0.01, 0.1, 1000
assert ("takes one weight of 0.0105 and applies 1,000 plain gradient steps, `weights -= learning_rate * dweights`, "
        "with a learning rate $\\alpha = 0.1$ and $\\lambda = 0.01$") in INDEX

L1 = np.concatenate([[START], plain_steps(one_weight(START, l1=LAM), LR, STEPS)])     # index = step
L2 = np.concatenate([[START], plain_steps(one_weight(START, l2=LAM), LR, STEPS)])
for step in (1, 10, 11, 12, 100, 1000):
    row = f"{step:4d}   {L1[step]:12.6f}   {L2[step]:10.6f}"
    assert row in INDEX, row
TAIL = np.abs(L1[10:])
assert f"{TAIL.min():.6f}" == f"{TAIL.max():.6f}" == "0.000500" and not np.any(L1 == 0)
assert "L1 after step 10: |w| stays at 0.000500 to 0.000500; steps with w == 0: 0 of 1000" in INDEX
FACTOR = 1 - 2 * LR * LAM
assert np.isclose(FACTOR, 0.998) and np.allclose(L2, START * FACTOR ** np.arange(STEPS + 1))
assert "start * factor^1000 = 0.001418" in INDEX and f"{L2[1000]:.6f}" == "0.001418"
assert np.allclose(np.diff(L1[:11]), -LR * LAM)                  # L1 subtracts 0.001 per step up to step 10
assert "the neighbourhood of zero in a finite number of steps, ten here" in INDEX
BAND = float(np.max(L1[10:]))                                    # +0.0005; the lowest is -0.0005
assert np.isclose(BAND, 0.0005) and np.isclose(np.min(L1[10:]), -0.0005)

w, alpha, lam, t = var("w"), var("α"), var("λ"), var("t")
fig = Figure(
    "02-one-weight-plain-steps", "L1 subtracts a fixed step, L2 shrinks by a fixed factor",
    "Two charts of one weight against the step, the weight axis from minus 0.002 to 0.012, under each penalty "
    "alone with plain gradient steps, learning rate 0.1 and lambda 0.01, from a start of 0.0105. Left, steps 0 to "
    "20, a mark at every step of L1: L1 falls by 0.001 per step to 0.0005 at step 10, then jumps to minus 0.0005 "
    "at step 11, 0.0005 at step 12, and so on, never 0; L2 falls slowly, to 0.010292 at step 10. Right, steps 0 "
    "to 1,000: L2 follows 0.0105 times 0.998 to the power t, 0.008595 at step 100 and 0.001418 at step 1,000, "
    "both marked; L1 stays in a band from minus 0.0005 to 0.0005 from step 10 on.",
    subtitle=rich("One weight from 0.0105, its penalty the only gradient: plain steps, ", alpha, " = 0.1, ",
                  lam, " = 0.01."),
    data_w=True)

left, right = fig.row(2, y=104, h=376)
Y_AX = (-0.002, 0.012, [0, 0.004, 0.008, 0.012])
fy = lambda v: num(f"{v:g}")  # noqa: E731
LABEL_W = 120

# -- left: the first 20 steps, every L1 step marked
body = fig.panel(left, "Steps 0 to 20")
S20 = list(range(21))
ax = fig.line_chart(body,
                    [dict(xs=S20, ys=list(L2[:21]), color="blue", points=False),
                     dict(xs=S20, ys=list(L1[:21]), color="ink", shape="square", size=6)],
                    x=(0, 20, [0, 5, 10, 15, 20]), y=Y_AX, labels=False, label_w=LABEL_W,
                    x_label="step", y_label=rich("weight ", w), fmt_y=fy)
ax.text(20, L2[20], "L2", "label", color="blue", dx=12, dy=5)
ax.text(20, L1[20], "L1", "label", color="ink", dx=12, dy=5)
ax.text(10, L1[10], rich(num(0.0005, 4), " at step 10"), "note", dx=-4, dy=-14)
ax.text(11, L1[11], rich(num(-0.0005, 4), " at step 11"), "note", dx=-4, dy=24)
ax.text(5, L1[5], rich(num(-0.001, 3), " per step"), "note", dx=12, dy=5)

# -- right: all 1,000 steps; L1's jumps drawn as the band they fill
body = fig.panel(right, "Steps 0 to 1,000")
S = list(range(STEPS + 1))
ax = fig.line_chart(body,
                    [dict(xs=S, ys=list(L2), color="blue", points=False),
                     dict(xs=S[:11], ys=list(L1[:11]), color="ink", points=False)],
                    x=(0, 1000, [0, 250, 500, 750, 1000]), y=Y_AX, labels=False, label_w=LABEL_W,
                    x_label="step", y_label=rich("weight ", w), fmt_y=fy)
with fig.data():
    x0, y0 = ax.to_px(10, BAND)
    x1, y1 = ax.to_px(1000, -BAND)
    fig.fill(Box(x0, y0, x1 - x0, y1 - y0), "neutral-soft", fit=False)
    fig.outline(Box(x0, y0, x1 - x0, y1 - y0), "ink", width=1)
ax.text(500, BAND, rich("L1: ±", num(0.0005, 4), " from step 10 on"), "label", anchor="middle", dy=-8)
for k in (100, 1000):
    ax.point(k, L2[k], "diamond", "blue", size=10)
ax.text(100, L2[100], rich(f"{L2[100]:.6f} at step 100"), "value", color="blue", dx=12, dy=-6)
ax.text(1000, L2[1000], "L2", "label", color="blue", dx=12, dy=-3)
ax.text(1000, L2[1000], f"{L2[1000]:.6f}", "value", color="blue", dx=12, dy=15)
ax.text(500, START * FACTOR ** 500, rich("0.0105 · ", sup("0.998", t, italic=False)), "note", color="blue",
        dx=8, dy=-12)

fig.caption("L1 oversteps 0 on every step after the tenth; L2 is multiplied by 0.998 per step and never reaches 0.")
fig.write()
