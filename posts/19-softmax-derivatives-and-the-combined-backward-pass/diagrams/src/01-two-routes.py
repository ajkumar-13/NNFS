"""Post 19, sections 2.3 and 3: the Jacobian route and the combined route give the same gradient for one sample.

Run from anywhere:  python posts/19-softmax-derivatives-and-the-combined-backward-pass/diagrams/src/01-two-routes.py
Writes posts/19-softmax-derivatives-and-the-combined-backward-pass/diagrams/01-two-routes.svg.
snippets/softmax_jacobian.py is run here (runpy): the softmax output [0.7, 0.2, 0.1], its Jacobian, the
cross-entropy gradient -y / y_hat for true class 0 and both results are the snippet's own arrays, and the lines
it prints (which the post quotes in sections 2.3 and 3) are checked against its output.

Layout: two rows that share their columns. Top, the Jacobian route: the row dvalues times the 3 x 3 Jacobian
equals the gradient with respect to the logits; the one non-zero entry of dvalues and the Jacobian row it picks
are outlined. Bottom, the combined route: y_hat minus the one-hot y gives the same row, in the same column.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, MINUS, rich, var, arr, hat, span, isub, num  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "softmax_jacobian.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

Y_HAT, J, Y, DV = s["y_hat"], s["jacobian"], s["y"], s["dvalues"]
RA, RB = DV @ J, Y_HAT - Y
assert Y_HAT.tolist() == [0.7, 0.2, 0.1] and Y.tolist() == [1.0, 0.0, 0.0]
assert ("[[ 0.210000 -0.140000 -0.070000]\n [-0.140000  0.160000 -0.020000]\n"
        " [-0.070000 -0.020000  0.090000]]") in OUT
assert "dvalues           = [-1.428571  0.000000  0.000000]" in OUT
assert "dvalues @ J       = [-0.300000  0.200000  0.100000]" in OUT
assert "y_hat - y         = [-0.300000  0.200000  0.100000]" in OUT
assert np.allclose(RA, RB) and np.allclose(RA, [-0.3, 0.2, 0.1])
assert np.allclose(J[0], 0.7 * np.array([0.3, -0.2, -0.1]))     # row 1 carries the factor y_hat_1 = 0.7


def two(v):
    return num(round(float(v), 2) + 0.0, 2)


JV = [[two(v) for v in row] for row in J]
assert JV == [["0.21", "−0.14", "−0.07"], ["−0.14", "0.16", "−0.02"], ["−0.07", "−0.02", "0.09"]]
DVV, RAV, RBV = [two(v) for v in DV], [two(v) for v in RA], [two(v) for v in RB]
assert DVV == ["−1.43", "0.00", "0.00"] and RAV == RBV == ["−0.30", "0.20", "0.10"]
YHV, YV = [two(v) for v in Y_HAT], [str(int(v)) for v in Y]
assert YHV == ["0.70", "0.20", "0.10"] and YV == ["1", "0", "0"]


def dLd(x):
    """The one-sample partial form of section 3, dL_i/dx."""
    return rich("∂", isub("L", "i"), "/∂", x)


YH, YB, ZB = hat(arr("y")), arr("y"), arr("z")

fig = Figure(
    "01-two-routes", "Two routes to one gradient: the Jacobian collapses",
    "One sample with softmax output 0.70, 0.20, 0.10 and true class 0. Top, the Jacobian route: the cross-entropy "
    "gradient minus y over y-hat, −1.43, 0.00, 0.00, times the 3 by 3 softmax Jacobian with rows 0.21, −0.14, "
    "−0.07; −0.14, 0.16, −0.02; −0.07, −0.02, 0.09 gives −0.30, 0.20, 0.10. The one non-zero entry, −1/0.7, and "
    "the first Jacobian row are outlined; that row is 0.7 times 0.3, −0.2, −0.1, so −1/0.7 times 0.7 times it gives "
    "−0.3, 0.2, 0.1. Bottom, the combined route: y-hat 0.70, "
    "0.20, 0.10 minus the one-hot label 1, 0, 0 gives the same −0.30, 0.20, 0.10, with no Jacobian and no "
    "division.",
    subtitle=rich("One sample: softmax output ", YH, " = [0.7, 0.2, 0.1], true class 0."),
    height=720, data_w=True)

CW, CH = 80, 56
XA, XB, XR = 40, 320, 600            # the three columns: left operand, right operand, result
YJ = 176                             # top of the Jacobian
YS = YJ + CH                         # the row strips of the top route sit on the Jacobian's middle row
YC = 552                             # the combined route's strips
ROW = "[0.3, " + MINUS + "0.2, " + MINUS + "0.1]"
assert np.allclose(J[0] / Y_HAT[0], [0.3, -0.2, -0.1])

# --- top: the Jacobian route
fig.text(40, 136, "Jacobian route", "head")
gd = fig.strip(XA, YS, 3, cell_w=CW, cell_h=CH, values=DVV, font=18, fill=lambda i: "gradient-soft")
gj = fig.grid(XB, YJ, 3, 3, cell_w=CW, cell_h=CH, values=JV, font=18)
gra = fig.strip(XR, YS, 3, cell_w=CW, cell_h=CH, values=RAV, font=18, fill=lambda i: "gradient-soft")
gd.window(0, 0, color="gradient")
gj.window(0, 0, 1, 3, color="gradient", form="grid")
fig.op((gd.box.right + XB) / 2, gd.box.cy, "·")
fig.op((gj.box.right + XR) / 2, gd.box.cy, "=")

fig.text(gd.box.cx, YS - 16, rich(dLd(YH), " = ", MINUS, YB, "/", YH), "label", anchor="middle")
fig.text(gj.box.cx, YJ - 16, "softmax Jacobian", "label", anchor="middle")
fig.text(gra.box.cx, YS - 16, dLd(ZB), "label", anchor="middle")
fig.text(gj.box.cx, gj.box.bottom + 28,
         rich("entry (", var("k"), ", ", var("j"), ") = ∂", isub(hat("y"), "k"), "/∂", isub("z", "j")), "note",
         anchor="middle")
fig.note(gd.box, [rich("its one non-zero entry, ", MINUS, "1/0.7,"), "picks the outlined row"])

# the cancellation, worked: the outlined row carries the factor 0.7
YM = gj.box.bottom + 76
fig.text(gj.box.cx, YM, rich("outlined row = 0.7 · ", ROW), "math", anchor="middle")
fig.text(gj.box.cx, YM + 32, rich("(", MINUS, "1/0.7) · 0.7 · ", ROW, " = [", MINUS, "0.3, 0.2, 0.1]"), "math",
         anchor="middle")

# --- bottom: the combined route
fig.text(40, YC - 56, "Combined route", "head")
gy = fig.strip(XA, YC, 3, cell_w=CW, cell_h=CH, values=YHV, font=18, fill=lambda i: "output-soft")
gl = fig.strip(XB, YC, 3, cell_w=CW, cell_h=CH, values=YV, font=18, fill=lambda i: "input-soft")
grb = fig.strip(XR, YC, 3, cell_w=CW, cell_h=CH, values=RBV, font=18, fill=lambda i: "gradient-soft")
fig.op((gy.box.right + XB) / 2, gy.box.cy, MINUS)
fig.op((gl.box.right + XR) / 2, gy.box.cy, "=")
fig.text(gy.box.cx, YC - 16, rich(YH, ", the softmax output"), "label", anchor="middle")
fig.text(gl.box.cx, YC - 16, rich(YB, ", one-hot"), "label", anchor="middle")
fig.text(grb.box.cx, YC - 16, rich(YH, " ", MINUS, " ", YB), "label", anchor="middle")
fig.note(grb.box, ["no Jacobian, no division"])

# the two results are the same array
fig.brace(gra.box.cy, grb.box.cy, gra.box.right + 16, label="same", side="right", vertical=True, kind="bracket")

fig.caption("Both routes give the same row; the combined one never builds the 3 × 3 matrix.")
fig.write()
