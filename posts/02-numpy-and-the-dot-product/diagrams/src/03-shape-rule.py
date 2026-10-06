"""Post 02 section 6: the shape rule. Inner dimensions match and are summed away; outer dimensions survive.

Run from anywhere:  python posts/02-numpy-and-the-dot-product/diagrams/src/03-shape-rule.py
Writes posts/02-numpy-and-the-dot-product/diagrams/03-shape-rule.svg.

No layer here, so no layout question: two general matrices, A of shape (3, 4) and B of shape (4, 3), in both
orders. Inner sizes are set in purple, outer sizes in green, in every shape label. The product B A is drawn
colour-only, because the post prints only its shape. Values come from snippets/shape_rule.py.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Raw, rich, var, CDOT  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "shape_rule.py"
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(SNIPPET))
A, B = ns["A"], ns["B"]
AB, BA = np.dot(A, B), np.dot(B, A)
assert A.shape == (3, 4) and B.shape == (4, 3)
assert AB.tolist() == [[70, 80, 90], [158, 184, 210], [246, 288, 330]] and BA.shape == (4, 4)   # section 6
WORKED = " + ".join(f"{x} {CDOT} {y}" for x, y in zip(A[0], B[:, 0])) + f" = {AB[0, 0]}"
assert WORKED == f"1 {CDOT} 1 + 2 {CDOT} 4 + 3 {CDOT} 7 + 4 {CDOT} 10 = 70"

fig = Figure(
    "03-shape-rule", "Inner sizes must match; outer sizes make the result",
    "Two rows. Top, np.dot(A, B): A of shape (3, 4), holding 1 to 12 row by row, times B of shape (4, 3), "
    "holding 1 to 12 row by row, equals the 3 by 3 result 70, 80, 90; 158, 184, 210; 246, 288, 330. Row 1 of A "
    "and column 1 of B are outlined, 70 is bold, and the worked line reads 1 times 1 plus 2 times 4 plus 3 times 7 "
    "plus 4 times 10 equals 70. Bottom, np.dot(B, A): the same two matrices swapped give a 4 by 4 result, drawn "
    "without values. Under every grid its shape, with the inner sizes, the 4s in the top row and the 3s in the "
    "bottom row, in purple and the outer sizes in green. A last line states the rule in general: (m, n) times (n, p) "
    "gives (m, p).",
    subtitle=rich("Section 6's ", var("A"), " and ", var("B"), ". Both orders run here, and give results of "
                  "different shapes."),
    height=720)

CELL, GAP = 40, 32
XG, XN = 40, 592                                   # the equations, the notes column


def tint(s, color):
    return Raw(f'<tspan class="{fig._cls("f", fig._ink(color))}">{s}</tspan>')


def shape(r, c, kind):
    """'(3, 4)' with each size tinted: inner (summed away) purple, outer (kept) green."""
    if kind == "first":                   # the last size of a first argument is inner
        return rich("(", tint(r, "output"), ", ", tint(c, "purple"), ")")
    if kind == "second":                  # the first size of a second argument is inner
        return rich("(", tint(r, "purple"), ", ", tint(c, "output"), ")")
    return rich("(", tint(r, "output"), ", ", tint(c, "output"), ")")


def dot_mark(x, y):
    with fig.data():
        fig.add(f'<circle cx="{x}" cy="{y}" r="3" class="{fig._cls("f", "ink-muted")}"/>')


def equation(top, first, second, out, first_fill, second_fill, values):
    """first . second = out, tops aligned at top; shape labels on one baseline under the tallest grid."""
    g1 = fig.grid(XG, top, *first.shape, CELL, values=first.tolist(), fill=lambda i, j: first_fill)
    x1 = XG + first.shape[1] * CELL + GAP
    g2 = fig.grid(x1, top, *second.shape, CELL, values=second.tolist(), fill=lambda i, j: second_fill)
    x2 = x1 + second.shape[1] * CELL + GAP
    g3 = fig.grid(x2, top, *out.shape, CELL, values=out.tolist() if values else None,
                  fill=lambda i, j: "output-soft", strong={(0, 0): "output"} if values else None)
    ym = top + first.shape[0] * CELL // 2
    dot_mark(x1 - GAP // 2, ym)
    fig.text(x2 - GAP // 2, ym + 8, "=", "op", anchor="middle")
    base = top + 4 * CELL + 28
    for g, sh, kind in ((g1, first.shape, "first"), (g2, second.shape, "second"), (g3, out.shape, "out")):
        fig.text(g.box.cx, base, shape(*sh, kind), "math", anchor="middle")
    return g1, g2, g3


T1, T2 = 168, 432
ga, gb, gab = equation(T1, A, B, AB, "input-soft", "weight-soft", True)
ga.window(0, 0, 1, 4, "input")
gb.window(0, 0, 4, 1, "weight")
fig.text(XG, T1 - 16, rich("np.dot(", var("A"), ", ", var("B"), "): the 4s meet"), "head")
fig.text(XN, T1 + 48, rich("row 1 of ", var("A"), " with column 1 of ", var("B"), ":"), "label")
fig.text(XN, T1 + 80, WORKED, "math")

gb2, ga2, gba = equation(T2, B, A, BA, "weight-soft", "input-soft", False)
assert gab.box.x == gba.box.x == 384 and gba.box.right == 544
fig.text(XG, T2 - 16, rich("np.dot(", var("B"), ", ", var("A"), "): the 3s meet"), "head")
fig.text(XN, T2 + 48, "the same two matrices, swapped:", "label")
fig.text(XN, T2 + 80, "a different product, of shape (4, 4)", "label")

m, n, p = (var(s) for s in "mnp")
fig.text(XN, T2 + 152, "in general:", "label")
fig.text(XN, T2 + 188, rich("(", tint(m, "output"), ", ", tint(n, "purple"), ") ", CDOT, " (", tint(n, "purple"),
                            ", ", tint(p, "output"), ")  →  (", tint(m, "output"), ", ", tint(p, "output"), ")"),
         "math")
fig.caption("Inner sizes, in purple, must be equal and are summed away; outer sizes, in green, form the result.")
fig.write()
