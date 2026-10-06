"""Post 14, section 3: where the upstream gradient dL/dZ comes from.

Run from anywhere:  python posts/14-matrices-in-backpropagation/diagrams/src/02-upstream-gradient.py
Writes posts/14-matrices-in-backpropagation/diagrams/02-upstream-gradient.svg.
snippets/single_sample.py is run here (runpy): every forward value (Z, A, Y, L) and every backward factor
(dL_dY, dY_dA, dA_dZ) and the product dL_dZ are the snippet's own arrays, checked against what it prints and
what section 3 states.

Layout: two lanes on the same columns. The forward lane runs left to right from Z to the loss; the backward
lane runs right to left under it, from the loss back to dL/dZ, each backward arrow labelled with the factor
it multiplies by (purple, above) and the step it undoes (muted, below).
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, arr, sup, isub, TIMES, CDOT  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "single_sample.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
Z, A, Y, L = s["Z"], s["A"], s["Y"], s["L"]
dL_dY, dY_dA, dA_dZ, dL_dZ = s["dL_dY"], s["dY_dA"], s["dA_dZ"], s["dL_dZ"]

# what section 3 prints and states
assert "Z: [[ 3.1  7.2 11.3]]  Y: 21.6  L: 466.56" in OUT
assert "dL_dZ: [[43.2 43.2 43.2]] (1, 3)" in OUT
assert np.array_equal(A, Z) and np.all(Z > 0)                 # every gate open
assert np.all(dY_dA == 1) and np.all(dA_dZ == 1)
assert round(dL_dY, 6) == 43.2 == round(2 * Y, 6)
assert np.array_equal(dL_dZ, dL_dY * dY_dA * dA_dZ)


def fmt(v):
    t = repr(round(float(v), 6))
    return t[:-2] if t.endswith(".0") else t


ZV, AV = [fmt(v) for v in Z[0]], [fmt(v) for v in A[0]]
DA = [fmt(v) for v in (dL_dY * dY_dA)[0]]
DZ = [fmt(v) for v in dL_dZ[0]]
assert ZV == AV == ["3.1", "7.2", "11.3"] and fmt(Y) == "21.6" and fmt(L) == "466.56"
assert DA == DZ == ["43.2"] * 3 and fmt(dL_dY) == "43.2"
YHAT = var("ŷ")


def dLd(x):
    return rich("∂", var("L"), "/∂", x)


fig = Figure(
    "02-upstream-gradient", "Everything after the layer arrives as one array",
    "Two lanes for the single sample. Forward, left to right: Z = 3.1, 7.2, 11.3; ReLU gives A = 3.1, 7.2, 11.3; "
    "the sum gives y-hat = 21.6; squaring gives the loss L = 466.56. Backward, right to left under it: the loss "
    "passes 2 y-hat = 43.2 to dL/dy-hat; the sum passes it on times 1, so dL/dA = 43.2, 43.2, 43.2; the ReLU "
    "gate, open for every z because every z is positive, multiplies by 1, so dL/dZ = 43.2, 43.2, 43.2, shape "
    "(1, 3), outlined. That array is all the layer's own backward pass receives.",
    subtitle=rich("The single sample of section 3: the three ReLU outputs are summed to ", YHAT, " and the loss is ", var("L"), " = ", sup("ŷ", "2"), "."))

CH = 48
XZ, XA, XY, XL = 136, 400, 656, 824              # left edges of the four columns
YF, YB = 160, 288                                # top of the forward lane, of the backward lane

fz = fig.strip(XZ, YF, 3, cell_w=56, cell_h=CH, values=ZV, fill=lambda k: "output-soft", font=16)
fa = fig.strip(XA, YF, 3, cell_w=56, cell_h=CH, values=AV, fill=lambda k: "output-soft", font=16)
fy = fig.strip(XY, YF, 1, cell_w=72, cell_h=CH, values=[fmt(Y)], fill=lambda k: "output-soft", font=16)
fl = fig.strip(XL, YF, 1, cell_w=96, cell_h=CH, values=[fmt(L)], fill=lambda k: "error-soft", font=16)
bz = fig.strip(XZ, YB, 3, cell_w=56, cell_h=CH, values=DZ, fill=lambda k: "gradient-soft", font=16)
ba = fig.strip(XA, YB, 3, cell_w=56, cell_h=CH, values=DA, fill=lambda k: "gradient-soft", font=16)
by = fig.strip(XY, YB, 1, cell_w=72, cell_h=CH, values=[fmt(dL_dY)], fill=lambda k: "gradient-soft", font=16)
assert fl.box.right == 920
bz.window(0, 2, color="gradient")

# lane names
fig.text(40, fz.box.cy + 6, "forward", "head")
fig.text(40, bz.box.cy + 6, "backward", "head", color="gradient")

# forward names above the strips, backward names under them
for g, name in ((fz, arr("Z")), (fa, arr("A")), (fy, YHAT), (fl, var("L"))):
    fig.text(g.box.cx, YF - 16, name, "head", anchor="middle")
for g, name in ((bz, dLd(arr("Z"))), (ba, dLd(arr("A"))), (by, dLd(YHAT))):
    fig.text(g.box.cx, YB + CH + 32, name, "head", anchor="middle", color="gradient")
fig.text(bz.box.cx, YB + CH + 56, "(1, 3), the upstream gradient", "note", anchor="middle")

# forward arrows, left to right, named by the step
for a, b, step in ((fz, fa, "ReLU"), (fa, fy, "sum"), (fy, fl, "square")):
    fig.arrow((a.box.right + 8, a.box.cy), (b.box.x - 8, b.box.cy))
    fig.text((a.box.right + b.box.x) / 2, a.box.cy - 12, step, "note", anchor="middle")

# backward arrows, right to left: the factor above in purple, the step it undoes below
fig.arrow((fl.box.cx, fl.box.bottom + 8), (by.box.right + 8, by.box.cy), via=[(fl.box.cx, by.box.cy)])
fig.text((by.box.right + fl.box.cx) / 2, by.box.cy - 12, rich(TIMES, " 2", YHAT), "label", anchor="middle",
         color="gradient")
fig.text((by.box.right + fl.box.cx) / 2, by.box.cy + 24, "square", "note", anchor="middle")
for a, b, factor, step in ((by, ba, rich(TIMES, " 1"), "sum"), (ba, bz, rich(TIMES, " gate"), "ReLU")):
    fig.arrow((a.box.x - 8, a.box.cy), (b.box.right + 8, b.box.cy))
    mid = (a.box.x + b.box.right) / 2
    fig.text(mid, a.box.cy - 12, factor, "label", anchor="middle", color="gradient")
    fig.text(mid, a.box.cy + 24, step, "note", anchor="middle")

# the three factors of section 3, as one worked line
ak, zk = isub("a", "k"), isub("z", "k")
fig.text(40, 448, rich(dLd(zk), " = ", dLd(YHAT), " ", CDOT, " ∂", YHAT, "/∂", ak, " ", CDOT, " ∂", ak, "/∂", zk,
                       " = 43.2 ", CDOT, " 1 ", CDOT, " 1 = 43.2"), "math")
fig.caption(rich("The layer's own backward pass needs only ", dLd(arr("Z")), ", not what produced it."))
fig.write()
