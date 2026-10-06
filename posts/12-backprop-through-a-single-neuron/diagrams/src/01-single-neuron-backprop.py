"""Post 12 hero: the forward pass of one neuron left to right, and the four chain-rule factors right to left.

Run from anywhere:  python posts/12-backprop-through-a-single-neuron/diagrams/src/01-single-neuron-backprop.py
Writes posts/12-backprop-through-a-single-neuron/diagrams/01-single-neuron-backprop.svg.
Every number is computed with the forward, backward and numerical_gradients functions of snippets/by_hand.py,
which this script runs, and asserted against the lines that snippet prints (sections 3 to 5 of the post).
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, sup, isub, num, span, CDOT  # noqa: E402

PARTIAL, YHAT, MINUS = "\u2202", "\u0177", "\u2212"
SNIP = Path(__file__).resolve().parents[2] / "snippets" / "by_hand.py"
ns = runpy.run_path(str(SNIP), run_name="snippet")
with contextlib.redirect_stdout(io.StringIO()) as out:
    ns["main"]()
OUT = out.getvalue()

X = np.array([1.0, -2.0, 3.0])
WTS = np.array([-3.0, -1.0, 2.0])
B, Y = 1.0, 0.0
z, yhat, loss = ns["forward"](WTS, B, X, Y)
up, dw, db = ns["backward"](z, yhat, X, Y)
num_w, num_b = ns["numerical_gradients"](WTS, B, X, Y)
F1, F2, F3 = 2.0 * (yhat - Y), (1.0 if z > 0 else 0.0), 1.0
assert (z, yhat, loss) == (6.0, 6.0, 36.0)
assert "z = 6.00   yhat = ReLU(z) = 6.00   L = (yhat - y)^2 = 36.00" in OUT
assert (F1, F2, F3, up) == (12.0, 1.0, 1.0, 12.0) and F1 * F2 * F3 == up
assert "upstream gradient = factor 1 * factor 2 * factor 3 = 12.0" in OUT
assert list(dw) == [12.0, -24.0, 36.0] and db == 12.0
assert np.max(np.abs(num_w - dw)) < 1e-9 and abs(num_b - db) < 1e-9     # the central difference agrees
for i in range(3):
    assert f"dL/dw_{i + 1} = upstream * x_{i + 1} = {up:.1f} * {X[i]:4.1f} = {dw[i]:6.1f}" in OUT
INS = [*X, 1.0]                      # the bias's "input" is 1 (section 4.1)
PARAMS = [*WTS, B]
PRODS = [a * b for a, b in zip(INS, PARAMS)]
GRADS = [*dw, db]
assert PRODS == [-3.0, 2.0, 6.0, 1.0] and sum(PRODS) == z
assert GRADS == [up * a for a in INS]


def dfrac(top, bottom):
    return rich(PARTIAL, top, "/", PARTIAL, bottom)


fig = Figure(
    "01-single-neuron-backprop", "One upstream gradient, scaled by each input",
    "The forward pass on the top row, left to right: the inputs 1, minus 2, 3 and the bias input 1 times the "
    "weights minus 3, minus 1, 2 and the bias 1 give the products minus 3, 2, 6 and 1; their sum is z = 6, ReLU "
    "gives y hat = 6, and the squared error against the target 0 gives L = 36. The backward pass on the bottom row, "
    "right to left, in the gradient colour: factor 1, 2 times (y hat minus y), is 12; factor 2, the ReLU "
    "derivative at z > 0, is 1; factor 3, the sum, is 1; so the upstream gradient 12 reaches the parameters. "
    "Factor 4 is each parameter's input: the gradients are 12 times 1 = 12, 12 times minus 2 = minus 24, "
    "12 times 3 = 36 for the weights, and 12 times 1 = 12 for the bias.",
    subtitle=rich("The neuron of section 2: a ReLU and a squared error against the target ", var("y"), " = 0."),
    height=720)

C = 40                                             # cell side
XL, XP, XQ = 88, 168, 248                          # input, parameter and product strips
TOP = 176                                          # forward strips
GTOP = 440                                         # gradient strip
ROWY = [TOP + C * k + C // 2 for k in range(4)]    # forward row centres
YF = TOP + 2 * C                                   # the forward pipeline's centre line (256)
YB = GTOP + 2 * C                                  # the backward line (520)

# ---------------------------------------------------------------- forward
fig.text(40, 136, "Forward: values, left to right", "head")
gi = fig.strip(XL, TOP, 4, cell=C, vertical=True, values=INS, fill=lambda k: "input-soft")
gw = fig.strip(XP, TOP, 4, cell=C, vertical=True, values=PARAMS, fill=lambda k: "weight-soft")
gq = fig.strip(XQ, TOP, 4, cell=C, vertical=True, values=PRODS)
fig.text(XL + C / 2, TOP - 12, isub("x", "i"), "label", anchor="middle", color="input")
fig.text(XP + C / 2, TOP - 12, rich(isub("w", "i"), ", ", var("b")), "label", anchor="middle", color="weight")
fig.text(XQ + C / 2, TOP - 12, "product", "label", anchor="middle")
for k, lab in enumerate(["1", "2", "3", "bias"]):
    fig.text(XL - 8, ROWY[k] + 5, lab, "tick", anchor="end")
fig.text(XL - 8, TOP - 12, var("i"), "tick", anchor="end")
for k in range(4):
    fig.op(XP - 20, ROWY[k], "\u00b7")
    fig.op(XQ - 20, ROWY[k], "=")

# the four products feed the sum
SUM = Box(336, YF - 24, 96, 48)
RELU = Box(496, YF - 24, 112, 48)
SQ = Box(672, YF - 24, 136, 48)
for y in ROWY:
    fig.edge((XQ + C, y), (SUM.x, YF))
for bx, label in ((SUM, rich("\u03a3")), (RELU, "ReLU"),
                  (SQ, sup(rich("(", var(YHAT), " ", MINUS, " ", var("y"), ")"), "2", italic=False))):
    fig.card(bx.x, bx.y, bx.w, bx.h)
    fig.text(bx.cx, bx.cy + 6, label, "math", anchor="middle")
fig.arrow((SUM.right, YF), (RELU.x, YF))
fig.arrow((RELU.right, YF), (SQ.x, YF))
fig.arrow((SQ.right, YF), (SQ.right + 40, YF))
fig.text((SUM.right + RELU.x) / 2, YF - 12, rich(var("z"), " = ", num(z)), "value", anchor="middle", color="output")
fig.text((RELU.right + SQ.x) / 2, YF - 12, rich(var(YHAT), " = ", num(yhat)), "value", anchor="middle",
         color="output")
fig.text(SQ.right + 48, YF + 5, rich(var("L"), " = ", num(loss)), "value", color="error")

# ---------------------------------------------------------------- backward
fig.text(40, 400, "Backward: gradients, right to left", "head", color="gradient")


def back_arrow(p0, p1):
    """A backward arrow: the figure's one arrow, its shaft drawn over in the gradient colour."""
    fig.arrow(p0, p1)
    fig.edge(p0, (p1[0] + 12 if p1[0] < p0[0] else p1[0] - 12, p1[1]), color="gradient", width=1.5)


FACTORS = [(SQ, rich("2(", var(YHAT), " ", MINUS, " ", var("y"), ") = ", num(F1)),
            rich("factor 1: ", dfrac(var("L"), var(YHAT)))),
           (RELU, rich(num(F2), ", as ", var("z"), " > 0"), rich("factor 2: ", dfrac(var(YHAT), var("z")))),
           (SUM, num(F3), rich("factor 3: ", dfrac(var("z"), rich("(", isub("x", "i"), isub("w", "i"), ")"))))]
for bx, line, name in FACTORS:
    fb = Box(bx.x, YB - 24, bx.w, 48)
    fig.card(fb.x, fb.y, fb.w, fb.h, color="gradient")
    fig.text(fb.cx, fb.cy + 6, line, "math", anchor="middle")
    fig.text(fb.cx, fb.bottom + 28, name, "note", anchor="middle")
back_arrow((SQ.x, YB), (RELU.right, YB))
back_arrow((RELU.x, YB), (SUM.right, YB))
fig.text((SQ.x + RELU.right) / 2, YB - 12, num(F1), "value", anchor="middle", color="gradient")
fig.text((RELU.x + SUM.right) / 2, YB - 12, num(F1 * F2), "value", anchor="middle", color="gradient")

# the upstream gradient reaches the parameters; factor 4 is each parameter's input
gg = fig.strip(XP, GTOP, 4, cell=C, vertical=True, values=GRADS, fill=lambda k: "gradient-soft")
back_arrow((SUM.x, YB), (XP + C + 8, YB))
fig.text((SUM.x + XP + C) / 2 + 4, YB - 12, rich("upstream ", num(up)), "value", anchor="middle", color="gradient")
fig.text(XP + C / 2, GTOP - 12, "gradient", "label", anchor="middle", color="gradient")
for k, a in enumerate(INS):
    shown = num(a) if a >= 0 else rich("(", num(a), ")")
    fig.text(XP - 12, GTOP + C * k + C // 2 + 5, rich(num(up), " ", CDOT, " ", shown, " ="), "label", anchor="end")
fig.text(XP + C / 2, GTOP + 4 * C + 28, rich("factor 4: ", isub("x", "i"), ", or 1 for ", var("b")), "note",
         anchor="middle")

fig.caption("Factors 1 to 3 are shared by every parameter; only factor 4, the input, differs.")
fig.write()
