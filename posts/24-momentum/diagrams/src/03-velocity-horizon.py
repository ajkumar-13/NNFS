"""Post 24, section 3: under a gradient that never changes the velocity grows to 1/(1 - beta) steps, and a
gradient k steps old counts beta^k.

Run from anywhere:  python posts/24-momentum/diagrams/src/03-velocity-horizon.py
Writes posts/24-momentum/diagrams/03-velocity-horizon.svg.
Both panels are computed here from the formulas of section 3: the size after t steps, (1 - beta^t)/(1 - beta) in
units of alpha * g, and the weight beta^k of a gradient k steps old. snippets/ravine.py is run (runpy, about a
second) and every value it prints for beta 0.5, 0.9 and 0.99 at t = 1, 2, 3, 10, 50 and 1000, the three limits and
the share 0.651 of the latest ten gradients, is asserted against the formula. The marked points are those printed.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, isup, num  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets"
sys.path.insert(0, str(SNIP))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path(str(SNIP / "ravine.py"), run_name="__main__")
OUT = buf.getvalue()

BETAS = (0.5, 0.9, 0.99)
TP = (1, 2, 3, 10, 50, 1000)                     # the steps ravine.py prints


def size(beta, t):
    return (1 - beta ** t) / (1 - beta)          # alpha g (1 + beta + ... + beta^(t-1)), in units of alpha g


for b in BETAS:
    shown = "  ".join(f"t={t}: {size(b, t):.3f}" for t in TP)
    assert f"beta {b:4.2f}  {shown}   1 / (1 - beta) = {1 / (1 - b):.0f}" in OUT, b
SHARE = 1 - 0.9 ** 10
assert f"share of the weight on the latest 10 gradients, beta 0.9: {SHARE:.3f}" in OUT and f"{SHARE:.3f}" == "0.651"
assert abs(sum(0.9 ** k for k in range(5000)) - 10) < 1e-9          # the weights add up to 1/(1 - beta)

BETA_S = "β"
COL = {0.5: "ink-muted", 0.9: "ink", 0.99: "ink-muted"}       # optimiser state: no role colour
SHAPE = {0.5: "circle", 0.9: "square", 0.99: "diamond"}

fig = Figure(
    "03-velocity-horizon", "A steady gradient grows the step to 1/(1 − β) times",
    "Left, a chart on logarithmic axes of the size of the velocity after t steps of a gradient that never changes, "
    "in units of one gradient step alpha g, for t from 1 to 1,000: (1 minus beta to the t)/(1 minus beta). With "
    "beta 0.5 it reaches 1.500 at t = 2 and settles at 2; with 0.9 it is 1.900 at t = 2, 6.513 at t = 10 and 9.948 "
    "at t = 50, and settles at 10; with 0.99 it is 9.562 at t = 10, 39.499 at t = 50 and 99.996 at t = 1,000, on "
    "its way to 100. Dotted lines mark 2, 10 and 100. Right, stems for the weight 0.9 to the k of a gradient k "
    "steps old, k from 0 to 29: the latest ten, k = 0 to 9, carry 0.651 of the total weight, and the weights add "
    "up to 10, so the velocity is a sum and not an average.",
    subtitle=rich("Section 3: the velocity under a gradient that never changes, and how much each past gradient "
                  "counts."),
    data_w=True)

# -- left: the size of the velocity against t, log-log
TS = list(range(1, 1001))
series = []
for b in BETAS:
    series.append(dict(xs=TS, ys=[size(b, t) for t in TS], color=COL[b], points=False,
                       label=rich(BETA_S, " = ", num(b))))
ax = fig.line_chart(Box(40, 112, 488, 364), series, x=(1, 1000, [1, 10, 100, 1000]), y=(1, 200, [1, 2, 10, 100]),
                    x_log=True, y_log=True, fmt_y=lambda v: num(v), fmt_x=lambda v: num(v),
                    x_label=rich("step ", var("t")),
                    y_label=rich("size of ", var("v"), ", in steps of ", var("α"), var("g")), label_w=88,
                    ref_lines=[dict(y=2), dict(y=10), dict(y=100)])
for b in BETAS:
    for t in TP:
        ax.point(t, size(b, t), SHAPE[b], COL[b], size=8)

# -- right: the weight of a gradient k steps old, beta = 0.9
K = list(range(30))
bx = fig.line_chart(Box(560, 112, 360, 364), [dict(xs=[0], ys=[1.0], color="gradient", points=False, label=None)],
                    x=(-1, 30, [0, 10, 20, 30]), y=(0, 1.05, [0, 0.5, 1]), fmt_y=lambda v: num(v),
                    x_label=rich("age ", var("k"), " of the gradient, in steps"),
                    y_label=rich("weight ", isup("β", "k"), ", ", BETA_S, " = 0.9"), label_w=8)
for k in K:
    c = "gradient" if k < 10 else "rule"
    bx.segment(k, 0, k, 0.9 ** k, color=c, width=1.5)
    bx.point(k, 0.9 ** k, "circle", c, size=8)
bx.text(11, 0.8, rich("latest ten: ", f"{SHARE:.3f}"), "value", color="gradient")
bx.text(11, 0.8, "of the total weight", "note", dy=20)
bx.text(11, 0.8, "the weights add up to 10", "note", dy=48)
ax.text(40, 10, "marks: the values printed at", "note", dy=44)
ax.text(40, 10, rich(var("t"), " = 1, 2, 3, 10, 50 and 1,000"), "note", dy=64)

fig.caption("Agreeing parts add up to 1/(1 − β) steps; at β = 0.9 the latest ten gradients carry 0.651 of the weight.")
fig.write()
