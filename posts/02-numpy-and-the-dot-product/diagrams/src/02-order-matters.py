"""Post 02 section 5: np.dot(a, B) and np.dot(B, a) give different vectors.

Run from anywhere:  python posts/02-numpy-and-the-dot-product/diagrams/src/02-order-matters.py
Writes posts/02-numpy-and-the-dot-product/diagrams/02-order-matters.svg.

No layer and no batch here, so no layout question: a plain vector and a square matrix. a is drawn as a row
when it comes first and as a column when it comes second, as section 6 reads a 1-D argument. Every value is
computed here from the arrays of snippets/order_matters.py and checked against the outputs the post prints.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, CDOT  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "order_matters.py"
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(SNIPPET))
a, B, C = ns["a"], ns["B"], ns["C"]
AB, BA = np.dot(a, B), np.dot(B, a)
assert AB.tolist() == [48, 54, 60] and BA.tolist() == [32, 50, 68]          # section 5's output
assert B.shape == (3, 3) and C.shape == (2, 3)


def worked(xs, ys, total):
    """'1 · 4 + 2 · 7 + 3 · 10 = 48' with the vector's factor first, as section 5's tables write it."""
    return " + ".join(f"{x} {CDOT} {y}" for x, y in zip(xs, ys)) + f" = {total}"


LEFT = worked(a, B[:, 0], AB[0])                 # a with column 1 of B
RIGHT = worked(B[0], a, BA[0])                   # row 1 of B with a
assert LEFT.endswith("= 48") and RIGHT.endswith("= 32")

fig = Figure(
    "02-order-matters", "Swap the arguments and the answer changes",
    "Two panels with the same vector a, holding 1, 2, 3, and the same 3 by 3 matrix B, holding 4 to 12 row by "
    "row. Left, np.dot(a, B): a drawn as a row meets each column of B and gives the row 48, 54, 60; column 1 of B "
    "is outlined and the worked line reads 1 times 4 plus 2 times 7 plus 3 times 10 equals 48. Right, "
    "np.dot(B, a): each row of B meets a drawn as a column and gives the column 32, 50, 68; row 1 of B is "
    "outlined and the worked line reads 4 times 1 plus 5 times 2 plus 6 times 3 equals 32. A caption line says "
    "both orders run only because B is square, and that with C of shape (2, 3) np.dot(a, C) raises.",
    subtitle=rich("Section 5's ", var("a"), " and ", var("B"), ". A 1-D first argument acts as a row; a 1-D second "
                  "argument as a column."))

CELL = 48
TOP, HEAD = 208, 176
XL, XR = 40, 600                                 # the left equation is 496 wide, the right one 304


def dot_mark(x, y):
    with fig.data():
        fig.add(f'<circle cx="{x}" cy="{y}" r="3" class="{fig._cls("f", "ink-muted")}"/>')


# -- left: a (row) . B = row; the operators sit on B's middle row
xa = XL
ga = fig.strip(xa, TOP + CELL, 3, CELL, values=a.tolist(), fill=lambda k: "input-soft")
xb = xa + 3 * CELL + 32
gb = fig.grid(xb, TOP, 3, 3, CELL, values=B.tolist(), fill=lambda i, j: "weight-soft")
xr = xb + 3 * CELL + 32
fig.strip(xr, TOP + CELL, 3, CELL, values=AB.tolist(), fill=lambda k: "output-soft", strong={0: "output"})
assert xr + 3 * CELL == 536
gb.window(0, 0, 3, 1, "weight")
dot_mark(xb - 16, TOP + 72)
fig.text(xr - 16, TOP + 80, "=", "op", anchor="middle")
fig.text(XL, HEAD, rich("np.dot(", var("a"), ", ", var("B"), "): vector first"), "head")
fig.text(XL, TOP + 3 * CELL + 32, rich(var("a"), " acts as a row and meets each column of ", var("B")), "label")
fig.text(XL, TOP + 3 * CELL + 64, LEFT, "math")

# -- right: B . a (column) = column
x0 = XR
assert x0 + 3 * CELL + 32 + CELL + 32 + CELL == 904
gb2 = fig.grid(x0, TOP, 3, 3, CELL, values=B.tolist(), fill=lambda i, j: "weight-soft")
xa2 = x0 + 3 * CELL + 32
fig.strip(xa2, TOP, 3, CELL, vertical=True, values=a.tolist(), fill=lambda k: "input-soft")
xr2 = xa2 + CELL + 32
fig.strip(xr2, TOP, 3, CELL, vertical=True, values=BA.tolist(), fill=lambda k: "output-soft", strong={0: "output"})
gb2.window(0, 0, 1, 3, "weight")
dot_mark(xa2 - 16, TOP + 72)
fig.text(xr2 - 16, TOP + 80, "=", "op", anchor="middle")
fig.text(x0, HEAD, rich("np.dot(", var("B"), ", ", var("a"), "): matrix first"), "head")
fig.text(x0, TOP + 3 * CELL + 32, rich(var("a"), " acts as a column and meets each row of ", var("B")), "label")
fig.text(x0, TOP + 3 * CELL + 64, RIGHT, "math")

fig.caption(rich("Both orders run only because ", var("B"), " is square; with ", var("C"),
                 " of shape (2, 3), np.dot(", var("a"), ", ", var("C"), ") raises."))
fig.write()
