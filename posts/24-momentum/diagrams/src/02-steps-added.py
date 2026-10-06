"""Post 24, section 2: the first two steps on the ravine, added by hand without momentum and by the velocity with it.

Run from anywhere:  python posts/24-momentum/diagrams/src/02-steps-added.py
Writes posts/24-momentum/diagrams/02-steps-added.svg.
snippets/ravine.py is run here (runpy, about a second). Every vector drawn and every number in the two tables is a
step of its paths (start (10, 1), learning rate 0.019), recomputed from run() and asserted against the lines the
snippet prints: the step table, "steps 1 and 2 without momentum, added" and "step 2 with momentum = ...".
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, span, num  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets"
sys.path.insert(0, str(SNIP))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIP / "ravine.py"), run_name="__main__")
OUT = buf.getvalue()
run, Opt, pair, CURV = s["run"], s["Optimizer_SGD"], s["pair"], s["CURVATURE"]
ALPHA, BETA = 0.019, 0.9

plain = run(Opt(learning_rate=ALPHA), 2)
heavy = run(Opt(learning_rate=ALPHA, momentum=BETA), 2)
S1, S2 = plain[1] - plain[0], plain[2] - plain[1]
SUM = plain[2] - plain[0]
CARRY = BETA * (heavy[1] - heavy[0])
GSTEP = -ALPHA * CURV[0] * heavy[1]
H2 = heavy[2] - heavy[1]
assert f"   1   {pair(S1)} {pair(plain[1])}   {pair(heavy[1] - heavy[0])}" in OUT
assert f"   2   {pair(S2)} {pair(plain[2])}   {pair(H2)}" in OUT
assert f"steps 1 and 2 without momentum, added: {pair(SUM)}" in OUT
assert f"= {pair(CARRY)} + {pair(GSTEP)} = {pair(H2)}" in OUT
assert np.allclose(GSTEP, S2) and np.allclose(CARRY + GSTEP, H2)     # the same gradient step in both panels
assert (f"{heavy[1][0]:.2f}", f"{heavy[1][1]:.2f}") == ("9.81", "-0.90")
assert pair(SUM) == "(-0.3764, -0.1900)" and pair(H2) == "(-0.3574,  0.0000)"


def f4(v):
    return num(round(float(v), 4) + 0.0, 4)


G = "gradient"
fig = Figure(
    "02-steps-added", "Momentum adds the steps before it takes them",
    "Two vector diagrams on the ravine, along the floor drawn at four times the scale across, each with a table of "
    "the parts along and across. Left, no momentum, from (10, 1): step 1 (minus 0.1900 along, minus 1.9000 across) and step 2 "
    "(minus 0.1864, plus 1.7100) are gradient steps taken one after the other; added, they give minus 0.3764 along "
    "and minus 0.1900 across, so the across parts nearly cancel and the along parts double. Right, momentum 0.9, from (9.81, minus 0.90): "
    "step 2 is 0.9 times step 1 (minus 0.1710, minus 1.7100) plus the same gradient step (minus 0.1864, plus "
    "1.7100), which gives minus 0.3574 along and 0.0000 across. A key names the gradient steps (purple), the "
    "velocity carried over (grey), the step taken (black) and the sum never taken (dashed).",
    subtitle="The first two steps from (10, 1), learning rate 0.019. Along the floor drawn at 4 times the scale "
             "across.",
    height=720, data_w=True)

SA, SC = 480.0, 120.0                    # px per unit along (leftwards when negative) and across (down when negative)
left, right = fig.row(2)
TY = 432                                 # the tables' top
OX = 288                                 # the diagram origin, from the panel's left edge


def to(o, v):
    return (o[0] + SA * v[0], o[1] - SC * v[1])


def table(x, rows):
    body = [["", "along", "across"]] + [[name, f4(a), f4(c)] for name, (a, c) in rows]
    fig.fill(Box(x, TY + 3 * 32, 424, 32), "neutral-soft", fit=False)        # the result row
    fig.table(x, TY, body, [200, 112, 112], col_align=["start", "end", "end"])


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# -- left: two gradient steps, then their sum
lb = fig.panel(left, "No momentum: steps 1 and 2")
OL = (lb.x + OX, lb.y + 24)
p1 = to(OL, S1)
p2 = to(p1, S2)
assert p1[1] < TY - 24
with fig.data():
    fig.arrow(OL, p1, color=G)
    fig.arrow(p1, p2, color=G)
    fig.arrow(OL, p2, color="ink", dash="proj")
    fig.marker(OL[0], OL[1], "circle", "weight", size=8)
    fig.text(OL[0] + 12, OL[1] + 24, "from (10, 1)", "note", snap=False)
    m = mid(OL, p1)
    fig.text(m[0] + 12, m[1] + 5, "step 1", "label", color=G, snap=False)
    m = mid(p1, p2)
    fig.text(m[0] - 12, m[1] + 5, "step 2", "label", color=G, anchor="end", snap=False)
    fig.text((OL[0] + p2[0]) / 2, OL[1] - 12, "sum", "label", anchor="middle", snap=False)
table(lb.x, [("step 1", S1), ("step 2", S2), ("sum", SUM)])

# -- right: the carried velocity plus the same gradient step is step 2
rb = fig.panel(right, "Momentum 0.9: step 2")
OR = (rb.x + OX, rb.y + 24)
q1 = to(OR, CARRY)
q2 = to(q1, GSTEP)
AG2 = rich("−", var("α"), sub("g", "2"))
with fig.data():
    fig.arrow(OR, q1, color="arrow")
    fig.arrow(q1, q2, color=G)
    fig.arrow(OR, q2, color="ink")
    fig.marker(OR[0], OR[1], "circle", "weight", size=8)
    fig.text(OR[0] + 12, OR[1] + 24, rich("from (", f"{heavy[1][0]:.2f}", ", ", num(round(heavy[1][1], 2), 2), ")"),
             "note", snap=False)
    m = mid(OR, q1)
    fig.text(m[0] + 12, m[1] + 5, "0.9 × step 1", "label", snap=False)
    m = mid(q1, q2)
    fig.text(m[0] - 12, m[1] + 5, AG2, "label", color=G, anchor="end", snap=False)
    fig.text((OR[0] + q2[0]) / 2, OR[1] - 12, "step 2", "label", anchor="middle", snap=False)
table(rb.x, [("0.9 × step 1", CARRY), (AG2, GSTEP), ("step 2", H2)])

fig.legend(40, 616, [dict(color=G, label=rich("a gradient step, −", var("α"), var("g")), mark="line"),
                     dict(color="arrow", label="the velocity carried over", mark="line"),
                     dict(color="ink", label="the step taken", mark="line"),
                     dict(color="ink", label="added, never taken", mark="line", dash="proj")], direction="row")

fig.caption("The across parts cancel and the along parts add up; momentum does the adding inside one step.")
fig.write()
