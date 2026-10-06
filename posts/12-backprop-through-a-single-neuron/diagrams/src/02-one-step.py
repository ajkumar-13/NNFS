"""Post 12 section 6: one gradient-descent step with the post's learning rate 0.01, and what it does to z and L.

Run from anywhere:  python posts/12-backprop-through-a-single-neuron/diagrams/src/02-one-step.py
Writes posts/12-backprop-through-a-single-neuron/diagrams/02-one-step.svg.
The step is taken with the forward and backward functions of snippets/by_hand.py, which this script runs; the
new parameters, z and L are asserted against the lines that snippet prints (section 6 of the post). The terms of
z and their changes are computed here from the same arrays; each change is asserted to be -0.01 * 12 * x_i^2.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, sup, isub, num, span  # noqa: E402

PARTIAL, ALPHA, MINUS = "\u2202", "\u03b1", "\u2212"
SNIP = Path(__file__).resolve().parents[2] / "snippets" / "by_hand.py"
ns = runpy.run_path(str(SNIP), run_name="snippet")
with contextlib.redirect_stdout(io.StringIO()) as out:
    ns["main"]()
OUT = out.getvalue()

X = np.array([1.0, -2.0, 3.0])
WTS = np.array([-3.0, -1.0, 2.0])
B, Y, LR = 1.0, 0.0, 0.01                     # the post's learning rate (section 6)
z, yhat, loss = ns["forward"](WTS, B, X, Y)
up, dw, db = ns["backward"](z, yhat, X, Y)
NW, NB = WTS - LR * dw, B - LR * db
z2, yhat2, loss2 = ns["forward"](NW, NB, X, Y)
OLD, GRAD, NEW = [*WTS, B], [*dw, db], [*NW, NB]
for i in range(3):
    assert f"w_{i + 1}: {WTS[i]:5.2f} - 0.01 * {dw[i]:6.1f} = {NW[i]:5.2f}" in OUT
assert f"b  : {B:5.2f} - 0.01 * {db:6.1f} = {NB:5.2f}" in OUT
assert [round(v, 2) for v in NEW] == [-3.12, -0.76, 1.64, 0.88]
assert "z: 6.00 -> 4.20   L: 36.00 -> 17.64" in OUT
assert "measured: 0.70" in OUT and "measured: 0.49" in OUT
INS = [*X, 1.0]
T0 = [a * w for a, w in zip(INS, OLD)]        # the four terms of z before the step
T1 = [a * w for a, w in zip(INS, NEW)]        # and after it
DT = [b - a for a, b in zip(T0, T1)]
for a, d in zip(INS, DT):
    assert abs(d - (-LR * up * a * a)) < 1e-12           # each term falls by 0.01 * 12 * x_i^2
assert abs(sum(T1) - z2) < 1e-12 and round(sum(DT), 2) == -1.80 and round(z2 / z, 2) == 0.70
assert round(loss2 / loss, 2) == 0.49 and round(z2, 2) == 4.20 and round(loss2, 2) == 17.64


def dfrac(top, bottom):
    return rich(PARTIAL, top, "/", PARTIAL, bottom)


fig = Figure(
    "02-one-step", "One step with α = 0.01 takes the loss from 36 to 17.64",
    "On the left, the update w new = w old minus 0.01 times the gradient, row by row: w1 minus 3 with gradient "
    "12 becomes minus 3.12, w2 minus 1 with gradient minus 24 becomes minus 0.76, w3 2 with gradient 36 becomes "
    "1.64, and the bias 1 with gradient 12 becomes 0.88; the note says w2's gradient is negative, so the step "
    "raises it. On the right, a table of the four terms of z before and after the step: minus 3 to minus 3.12, "
    "2 to 1.52, 6 to 4.92 and 1 to 0.88, changes minus 0.12, minus 0.48, minus 1.08 and minus 0.12. Their sum "
    "z goes from 6 to 4.20, a change of minus 1.80, so 4.20 is 0.7 times 6; the loss L = z squared goes from 36 "
    "to 17.64, which is 0.49 times 36.",
    subtitle=rich("The four gradients of section 5, spent once with the update rule of post 09."))

TOP = 192                                     # first parameter row
C = 44                                        # row height, shared by the strips and the table
ROWY = [TOP + C * k + C // 2 for k in range(4)]
NAMES = [sub("w", "1"), sub("w", "2"), sub("w", "3"), var("b")]
LABEL_Y = TOP - C + 28                        # the table's header baseline; the strip labels share it

# ---------------------------------------------------------------- left: the update, row by row
fig.text(40, 128, rich("The update, ", ALPHA, " = 0.01"), "head")
XO, XG, XN = 80, 240, 352
fig.strip(XO, TOP, 4, vertical=True, cell_w=56, cell_h=C, values=[num(v) for v in OLD], font=16,
          fill=lambda k: "weight-soft")
fig.strip(XG, TOP, 4, vertical=True, cell_w=56, cell_h=C, values=[num(v) for v in GRAD], font=16,
          fill=lambda k: "gradient-soft")
fig.strip(XN, TOP, 4, vertical=True, cell_w=64, cell_h=C, values=[num(v, 2) for v in NEW], font=16,
          fill=lambda k: "weight-soft")
for k, y in enumerate(ROWY):
    fig.text(XO - 8, y + 5, NAMES[k], "label", anchor="end", color="weight")
    fig.op(XO + 56 + 20, y, "−")
    fig.text(XO + 56 + 40, y + 8, "0.01", "math")
    fig.op(XG - 12, y, "·")
    fig.op(XG + 56 + 24, y, "=")
fig.text(XO + 28, LABEL_Y, "old", "label", anchor="middle")
fig.text(XG + 28, LABEL_Y, "gradient", "label", anchor="middle", color="gradient")
fig.text(XN + 32, LABEL_Y, "new", "label", anchor="middle")
fig.note(Box(40, TOP, 0, 4 * C), rich(dfrac(var("L"), sub("w", "2")), " < 0, so the step raises ", sub("w", "2"), "."))

# ---------------------------------------------------------------- right: the terms of z before and after
XT = 456
fig.text(XT, 128, rich("The terms of ", var("z"), " before and after"), "head")
rows = [["term", "before", "after", "change"]]
for k in range(4):
    term = rich(sub("x", str(k + 1)), sub("w", str(k + 1))) if k < 3 else var("b")
    rows.append([term, num(T0[k]), num(T1[k], 2), num(DT[k], 2)])
rows.append([rich(var("z"), " (sum)"), num(z), num(z2, 2), num(sum(DT), 2)])
rows.append([rich(var("L"), " = ", sup("z", "2")), num(loss), num(loss2, 2), ""])
WIDTHS = [96, 72, 80, 80]
fig.table(XT, TOP - C, rows, WIDTHS, row_h=C, col_align=["start", "end", "end", "end"],
          highlight_row={5: "output", 6: "error"})
XR = XT + sum(WIDTHS) + 16
fig.text(XR, ROWY[0] + 4 * C + 5, rich("4.20 = 0.7 ", "·", " 6"), "label", color="output")
fig.text(XR, ROWY[0] + 5 * C + 5, rich("17.64 = 0.49 ", "·", " 36"), "label", color="error")

fig.caption(rich("Each term of ", var("z"), " falls by 0.12 ", "·", " ", isub("x", "i"), sup("", "2"),
                 " (the bias term by 0.12): 1.80 in all, so ", var("z"), " keeps 0.7 of its value."))
fig.write()
