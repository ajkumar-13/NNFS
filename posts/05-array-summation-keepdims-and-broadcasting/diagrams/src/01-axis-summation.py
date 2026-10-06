"""Post 05 hero (sections 2 and 3): np.sum on the 3 x 3 array a removes the named axis; keepdims keeps it at size 1.

Run from anywhere:  python posts/05-array-summation-keepdims-and-broadcasting/diagrams/src/01-axis-summation.py
Writes posts/05-array-summation-keepdims-and-broadcasting/diagrams/01-axis-summation.svg.
Every sum and shape is computed here with NumPy and checked against what snippets/axis_keepdims.py prints.

Layout: on the left, a with its two axes; the keepdims=True results sit where their axis collapsed, the
(1, 3) column sums under a's columns and the (3, 1) row sums beside a's rows, in square cells like a's. On
the right, the default results, each level with a row on the left: the full sum, the bare number 45 of
shape (), with row 0 of a; both 1-D sums are flat strips, lower than the square cells, with the same shape
(3,), the row sums level with row 2 of a and the column sums level with the (1, 3) row.
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, span  # noqa: E402

SNIPPETS = Path(__file__).resolve().parents[2] / "snippets"


def run_snippet(name):
    """Execute snippets/<name>.py and return (module, everything it printed)."""
    spec = importlib.util.spec_from_file_location(name, SNIPPETS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    return mod, buf.getvalue()


snip, printed = run_snippet("axis_keepdims")
a = snip.a
assert a.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

total = np.sum(a)
s0, s1 = np.sum(a, axis=0), np.sum(a, axis=1)
k0, k1 = np.sum(a, axis=0, keepdims=True), np.sum(a, axis=1, keepdims=True)
assert np.shape(total) == () and s0.shape == s1.shape == (3,)
assert k0.shape == (1, 3) and k1.shape == (3, 1)
for arr in (total, s0, s1, k0, k1):                       # each value exactly as the snippet prints it
    assert f"{arr}\n" in printed, arr
assert "(3,) (3,)\n" in printed and "(1, 3) (3, 1)\n" in printed
assert (int(total), s0.tolist(), s1.tolist()) == (45, [12, 15, 18], [6, 15, 24])


A = span("a", mono=True)                # the code name, in the code face as in the post


def shape(t):
    return "(" + ", ".join(str(n) for n in t) + ("," if len(t) == 1 else "") + ")"


def ints(v):
    return ", ".join(str(int(x)) for x in np.ravel(v))


fig = Figure(
    "01-axis-summation", "The axis named in a reduction is the axis that disappears",
    f"The post's array a, shape {shape(a.shape)}, holding 1 to 9, with axis 0 running down its rows and axis 1 "
    f"across its columns. With keepdims=True the reduced axis stays at size 1: the column sums {ints(k0)} sit "
    f"under a's columns as a row of shape {shape(k0.shape)} (axis=0), and the row sums {ints(k1)} sit beside a's "
    f"rows as a column of shape {shape(k1.shape)} (axis=1). Without keepdims, np.sum(a, axis=0) gives {ints(s0)} "
    f"and np.sum(a, axis=1) gives {ints(s1)}, both drawn as flat strips, lower than the square cells, with the same shape "
    f"{shape(s0.shape)}, and np.sum(a) gives the scalar {int(total)}, shape (), drawn as a bare number. A note says "
    f"that flat strips are 1-D arrays and that the 1-D shape does not record which axis was summed.",
    subtitle=rich("np.sum on the post's array ", A, ", with keepdims=True on the left and without it on the right."))

C = 56                                  # cell side; values at 18, the size a 48 cell prints at
AX, AY = 136, 176                       # the array a
fig.text(40, 128, "keepdims=True: the axis stays, size 1", "head")
ga = fig.grid(AX, AY, 3, 3, C, values=a.tolist(), font=18)
# the two axes of a, as arrows along its top and its left edge
fig.arrow((AX, AY - 16), (ga.box.right, AY - 16))
fig.text(ga.box.cx, AY - 28, "axis 1", "label", anchor="middle")
fig.arrow((AX - 16, AY), (AX - 16, ga.box.bottom))
fig.text(AX - 28, ga.cell(1, 0).cy + 5, "axis 0", "label", anchor="end")
fig.text(AX - 28, ga.cell(0, 0).cy + 5, rich(A, ", (3, 3)"), "note", anchor="end")

# keepdims=True: square cells, as a's are; the column sums under the columns, the row sums beside the rows.
# The column's three labels stand one per cell beside it; the row's three stand on one line beside it.
gk1 = fig.grid(ga.box.right + 32, AY, 3, 1, C, values=k1.tolist(), fill=lambda i, j: "output-soft", font=18)
gk0 = fig.grid(AX, ga.box.bottom + 32, 1, 3, C, values=k0.tolist(), fill=lambda i, j: "output-soft", font=18)
fig.text(gk1.box.right + 16, gk1.cell(0, 0).cy + 5, "axis=1", "code")
fig.text(gk1.box.right + 16, gk1.cell(1, 0).cy + 5, "row sums", "note")
fig.text(gk1.box.right + 16, gk1.cell(2, 0).cy + 5, shape(k1.shape), "code", color="output")
fig.text(gk0.box.right + 16, gk0.box.cy + 5, "axis=0", "code")
fig.text(gk0.box.right + 80, gk0.box.cy + 5, "column sums", "note")
fig.text(gk0.box.right + 176, gk0.box.cy + 5, shape(k0.shape), "code", color="output")

# the default, each result level with the left-hand rows: the full sum with row 0 of a, the row sums (a flat
# strip) with row 2, the column sums (a flat strip) with the (1, 3) row; low, flat cells mark a 1-D array
RX, SH = 616, 40                        # 40 high against 56: on the 8 grid and centred on a row
fig.text(RX, 128, "Default: the axis is gone", "head")
fig.text(RX, AY, "np.sum(a)", "code")
fig.text(RX, ga.cell(0, 0).cy + 4, str(int(total)), "cell18")
fig.text(RX + 40, ga.cell(0, 0).cy + 4, shape(np.shape(total)), "code", color="output")
rows = [("np.sum(a, axis=1)", s1, ga.cell(2, 0).cy), ("np.sum(a, axis=0)", s0, gk0.box.cy)]
for call, vals, cy in rows:
    g = fig.strip(RX, int(cy - SH // 2), 3, C, height=SH, values=[int(v) for v in vals], fill=lambda k: "output-soft", font=16)
    fig.text(RX, g.box.y - 12, call, "code")
    fig.text(g.box.right + 16, cy + 5, shape(vals.shape), "code", color="output")
fig.note(g.box, ["Flat strips: 1-D arrays, both of shape (3,);", "nothing records which axis was summed."])
fig.caption("Kept at size 1, each sum stays lined up with the rows or columns it came from.")
fig.write()
