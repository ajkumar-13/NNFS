"""Post 05 section 4: the broadcasting rules on four pairs of shapes, aligned axis by axis from the right.

Run from anywhere:  python posts/05-array-summation-keepdims-and-broadcasting/diagrams/src/03-shape-alignment.py
Writes posts/05-array-summation-keepdims-and-broadcasting/diagrams/03-shape-alignment.svg.
Every padded shape, stretch, clash and result is computed here with broadcast_shape from snippets/broadcasting.py
and with NumPy, and checked against the lines that snippet prints (section 4's table and section 5's bias line).

Layout: four panels, one pair each. A panel is a small table: one column per aligned axis, rows left, right and
result. A 1 added by padding has a dashed inner box; a size 1 that stretches is filled; a clash is outlined in red
and its result cell reads ValueError. The row labels are drawn once, left of the first panel.
"""
import contextlib
import importlib.util
import io
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Box, Figure, text_width  # noqa: E402

SNIPPETS = Path(__file__).resolve().parents[2] / "snippets"


def run_snippet(name):
    """Execute snippets/<name>.py and return (module, everything it printed)."""
    spec = importlib.util.spec_from_file_location(name, SNIPPETS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    return mod, buf.getvalue()


snip, printed = run_snippet("broadcasting")

PAIRS = [((300, 3), (1, 3)), ((3, 3), (3,)), ((3, 1), (3,)), ((5, 3), (5,))]
# section 5 prints the bias line; section 4's table prints the other three pairs, by hand and by NumPy
assert "(300, 3) (1, 3) (300, 3)\n" in printed
for left, right in PAIRS[1:]:
    line = (f"{str(left):8s} with {str(right):6s} -> by hand {str(snip.by_hand(left, right)):8s} "
            f"NumPy {snip.by_numpy(left, right)}\n")
    assert line in printed, line


def analyse(left, right):
    nd = max(len(left), len(right))
    pl, pr = (1,) * (nd - len(left)) + left, (1,) * (nd - len(right)) + right
    padded = {("L", k) for k in range(nd - len(left))} | {("R", k) for k in range(nd - len(right))}
    clash = [k for k in range(nd) if pl[k] != pr[k] and 1 not in (pl[k], pr[k])]
    stretched = {("L", k) for k in range(nd) if pl[k] == 1 and pr[k] != 1} | \
                {("R", k) for k in range(nd) if pr[k] == 1 and pl[k] != 1}
    result = snip.by_hand(left, right)
    assert result == snip.by_numpy(left, right)                         # rule 4 left to NumPy: same verdict
    if result != "error":
        assert result == tuple(max(m, n) for m, n in zip(pl, pr)) == np.broadcast_shapes(left, right)
    return pl, pr, padded, stretched, clash, result


INFO = [analyse(lft, rgt) for lft, rgt in PAIRS]
assert [i[5] for i in INFO] == [(300, 3), (3, 3), (3, 3), "error"]
assert INFO[3][4] == [1] and INFO[2][3] == {("L", 1), ("R", 0)}


def shape(t):
    return "(" + ", ".join(str(n) for n in t) + ("," if len(t) == 1 else "") + ")"


NOTES = [["The 1 stretches to 300:", "one bias row per sample."],
         ["(3,) is padded to (1, 3)", "and acts as a row."],
         ["Both operands stretch;", "neither had shape (3, 3)."],
         ["Axis 1: 3 against 5,", "and neither is 1."]]


def words(i):
    (lft, rgt), (pl, pr, padded, stretched, clash, result) = PAIRS[i], INFO[i]
    s = f"{shape(lft)} plus {shape(rgt)}"
    if padded:
        s += f", the 1-D shape padded to {shape(pr)}"
    s += f": aligned left {pl[0]}, {pl[1]} and right {pr[0]}, {pr[1]}; "
    s += "raises ValueError because on axis 1 the sizes 3 and 5 clash" if result == "error" \
        else f"result {shape(result)}"
    return s


fig = Figure(
    "03-shape-alignment", "Broadcasting compares two shapes axis by axis from the right",
    "Four panels, each a small table with one column per aligned axis, axis 0 and axis 1, and rows for the "
    "left shape, the right shape and the result. " + "; ".join(words(i) for i in range(4)) + ". A 1 added by "
    "padding has a dashed box, a size 1 that stretches is filled blue, the clash is outlined in red, the last "
    "panel's result cell reads ValueError on a red fill, and result shapes are filled green.",
    subtitle="Each column is one aligned axis. The shorter shape is padded with 1s on the left.")

# four panels share one row-label column on the left, so every grid sits flush under its heading
C, PW, PG = 56, 192, 16                    # cell side (values at 18), panel width, gutter
GY = 168                                   # grid top: 24 under the axis labels' row
PX = [104 + k * (PW + PG) for k in range(4)]
assert PX[-1] + PW == 920
for i, (px, (lft, rgt)) in enumerate(zip(PX, PAIRS)):
    pl, pr, padded, stretched, clash, result = INFO[i]
    fig.text(px, 136, f"{shape(lft)} + {shape(rgt)}", "head")
    nrows = 3 if result != "error" else 2
    vals = [list(pl), list(pr)] + ([list(result)] if result != "error" else [])

    def fill(r, c, padded=padded, stretched=stretched, nrows=nrows):
        if r == 2:
            return "output-soft"
        side = "L" if r == 0 else "R"
        return "blue-soft" if (side, c) in stretched else None
    g = fig.grid(px, GY, nrows, 2, C, values=vals, fill=fill, font=18)
    g.col_labels(["axis 0", "axis 1"])
    if i == 0:
        g.row_labels(["left", "right", "result"])
    with fig.data():
        for side, c in sorted(padded):
            b = g.cell(0 if side == "L" else 1, c)
            fig._rect(b.x + 6, b.y + 6, b.w - 12, b.h - 12, "lead", extra='data-fit="skip"')
    for c in clash:
        g.window(0, c, 2, 1, "error")
    if result == "error":                  # the result row holds the error, in the same slot as a shape
        fig.grid(px, GY + 2 * C, 1, 1, width=2 * C, height=C, values=[["ValueError"]],
                 fill=lambda r, c: "error-soft", font=16)
    fig.note(Box(px, GY + 3 * C, 0, 0), NOTES[i])

# the legend, one line under the panels, items packed from the first panel's left edge (not one per panel,
# which would read as belonging to that panel): the kit has no dashed legend mark, so it is drawn here
LY = 448
items = [("blue-soft", None, "a 1 that stretches"), (None, "lead", "a 1 added by padding"),
         (None, "error", "sizes that clash"), ("output-soft", None, "result shape")]
x = PX[0]
with fig.data():
    for fill_c, stroke, label in items:
        if stroke == "lead":
            fig._rect(x, LY - 13, 16, 16, "lead", extra='data-fit="skip"')
        elif stroke:
            fig._rect(x, LY - 13, 16, 16, f"w15 nofill {fig._cls('s', 'red')}", extra='data-fit="skip"')
        else:
            fig._rect(x, LY - 13, 16, 16, f"cell {fig._cls('f', fill_c)}", extra='data-fit="skip"')
        fig.text(x + 24, LY, label, "note")
        x += 24 + math.ceil((text_width(label, 14) + 32) / 8) * 8
fig.caption("Pad with 1s on the left, check each aligned pair, stretch every 1; a clash raises ValueError.")
fig.write()
