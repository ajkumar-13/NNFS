"""Post 02 section 6: the shape rule. Inner dimensions match and are summed away; outer dimensions survive.

Run from anywhere:  python posts/02-numpy-and-the-dot-product/diagrams/src/03-shape-rule.py
Writes posts/02-numpy-and-the-dot-product/diagrams/03-shape-rule.svg.

No layer here, so no layout question: two general matrices, A of shape (3, 4) and B of shape (4, 3), in both
orders. The grids of each equation share their bottom edge, so every shape label sits 28 under its own grid
and all three share one baseline; rows of the first argument line up with rows of the result. Inner sizes are
purple, outer sizes green, and colour is never the only cue: a purple bracket joins the two inner sizes of
each equation, and the general rule at the right labels its inner pair and its outer pair in words. The
product B A is drawn colour-only, because the post prints only its shape. Whole arrays are bold; the calls
in the headings are code and stay upright. Values come from snippets/shape_rule.py.
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


def bold(s):
    """A whole array, bold upright as the notation guide writes it."""
    return Raw(f'<tspan class="b">{s}</tspan>')

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
    "bottom row, in purple and joined by a purple bracket labelled inner, and the outer sizes in green. At the right "
    "the rule in general, (m, n) times (n, p) gives (m, p), with a bracket over m and p labelled outer: form the "
    "result, and a bracket under the two n labelled inner: must be equal, summed away.",
    subtitle=rich("Section 6's ", bold("A"), " and ", bold("B"), ". Both orders run here, and give results of "
                  "different shapes."),
    height=720)

CELL, GAP = 40, 32
DOT_R = 2.5                                        # the product dot, as in every figure of the post
XG, XN = 40, 592                                   # the equations, the notes column
DIGIT = 9          # in a 17 px shape label "(3, 4)" each size sits about 9 off the label's centre (measured)


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
    """The product dot, on the axis of the "=" beside it (the "=" baseline is 8 lower)."""
    with fig.data():
        fig.add(f'<circle cx="{x}" cy="{y}" r="{DOT_R}" class="{fig._cls("f", "ink-muted")}"/>')


def bracket(x0, x1, y, color, up=False):
    """A square bracket in a hue joining two sizes of a label: legs 8 long from y, away from the text."""
    s = -1 if up else 1
    with fig.data():
        fig._path(f"M{x0:g},{y} V{y + s * 8} H{x1:g} V{y}", f"nofill w1 {fig._cls('s', color)}")


def equation(top, first, second, out, first_fill, second_fill, values):
    """first . second = out. The three grids share their bottom edge (top + 4 cells), so the shape labels sit
    28 under their own grids on one baseline; the operators sit on the first argument's middle."""
    bottom = top + 4 * CELL

    def at(mat):
        return bottom - mat.shape[0] * CELL

    g1 = fig.grid(XG, at(first), *first.shape, CELL, values=first.tolist(), fill=lambda i, j: first_fill)
    x1 = XG + first.shape[1] * CELL + GAP
    g2 = fig.grid(x1, at(second), *second.shape, CELL, values=second.tolist(), fill=lambda i, j: second_fill)
    x2 = x1 + second.shape[1] * CELL + GAP
    g3 = fig.grid(x2, at(out), *out.shape, CELL, values=out.tolist() if values else None,
                  fill=lambda i, j: "output-soft", strong={(0, 0): "output"} if values else None)
    assert g1.box.bottom == g2.box.bottom == g3.box.bottom == bottom
    ym = g1.box.cy
    dot_mark(x1 - GAP // 2, ym)
    fig.text(x2 - GAP // 2, ym + 8, "=", "op", anchor="middle")
    base = bottom + 28
    for g, sh, kind in ((g1, first.shape, "first"), (g2, second.shape, "second"), (g3, out.shape, "out")):
        fig.text(g.box.cx, base, shape(*sh, kind), "math", anchor="middle")
    # the two inner sizes, joined: the first label's last size and the second label's first size
    y = base + 6
    bracket(g1.box.cx + DIGIT, g2.box.cx - DIGIT, y, "purple")
    fig.text((g1.box.cx + g2.box.cx) / 2, y + 28, "inner", "note", anchor="middle", color="purple")
    return g1, g2, g3


T1, T2 = 152, 432
ga, gb, gab = equation(T1, A, B, AB, "input-soft", "weight-soft", True)
ga.window(0, 0, 1, 4, "input")
gb.window(0, 0, 4, 1, "weight")
fig.text(XG, T1 - 16, "np.dot(A, B): the 4s meet", "head")
fig.text(XN, T1 + 72, rich("row 1 of ", bold("A"), " with column 1 of ", bold("B"), ":"), "label")
fig.text(XN, T1 + 104, WORKED, "math")
assert ga.box.y == gab.box.y == T1 + CELL                # row 1 of A and row 1 of the result on one line

gb2, ga2, gba = equation(T2, B, A, BA, "weight-soft", "input-soft", False)
assert gab.box.x == gba.box.x == 384 and gba.box.right == 544
fig.text(XG, T2 - 16, "np.dot(B, A): the 3s meet", "head")
fig.text(XN, T2 + 16, "the same two matrices, swapped:", "label")
fig.text(XN, T2 + 40, "a different product, of shape (4, 4)", "label")

# the rule in general, set in pieces so each letter's place is known; brackets name the two pairs
m, n, p = (var(s) for s in "mnp")
R = T2 + 136                                        # the rule's baseline
X1 = XN + 20                                        # centres: (m, n), the dot, (n, p), the arrow, (m, p);
XD, X2, XA, X3 = X1 + 32, X1 + 64, X1 + 100, X1 + 136   # "(m, n)" is about 42 wide, so it starts at XN
fig.text(X1, R, rich("(", tint(m, "output"), ", ", tint(n, "purple"), ")"), "math", anchor="middle")
fig.text(XD, R, CDOT, "math", anchor="middle")
fig.text(X2, R, rich("(", tint(n, "purple"), ", ", tint(p, "output"), ")"), "math", anchor="middle")
fig.text(XA, R, "→", "math", anchor="middle")
fig.text(X3, R, rich("(", tint(m, "output"), ", ", tint(p, "output"), ")"), "math", anchor="middle")
M_OFF, N_OFF1, N_OFF2, P_OFF = -9, 11, -8, 9        # each letter's centre from its label's centre (measured)
bracket(X1 + M_OFF, X2 + P_OFF, R - 18, "output", up=True)
fig.text(XN, R - 40, "outer: form the result", "note", color="output")       # labels flush with the rule
bracket(X1 + N_OFF1, X2 + N_OFF2, R + 6, "purple")
fig.text(XN, R + 34, "inner: must be equal, summed away", "note", color="purple")
fig.write()
