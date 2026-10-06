"""Post 34, section 2.1: three ways to write the sigmoid on seven logits, float64.

Run from anywhere:  python posts/34-sigmoid-and-binary-cross-entropy/diagrams/src/02-three-sigmoids.py
Writes posts/34-sigmoid-and-binary-cross-entropy/diagrams/02-three-sigmoids.svg. Takes about a second.

snippets/sigmoid_overflow.py is run here (runpy) and its printed table is parsed: every value in the grid is a
value it prints, and the lines it prints are checked against the listing in index.md. Which cells raised a warning
is measured here by calling each of the snippet's three functions on one logit at a time. The overflow bound
709.78 is the snippet's printed line. The two branches of the stable form are checked against
Activation_Sigmoid.forward in snippets/binary_classes.py.

Layout: a 3 by 7 grid, one row per form, one column per logit. The cells that return nan are filled red and
outlined; the cells that return a value with an overflow warning have a dashed inner outline; the stable row is
filled green. Two braces under the grid say which expression the stable form uses on each side of zero, and a key
under them names the three cell styles.
"""
import contextlib
import io
import re
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, MINUS, TIMES, rich, var, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIP = POST / "snippets"
sys.path.insert(0, str(SNIP))
INDEX = (POST / "index.md").read_text(encoding="utf-8")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(SNIP / "sigmoid_overflow.py"), run_name="snippet")
OUT = buf.getvalue()

for line in OUT.splitlines()[1:5] + OUT.splitlines()[7:9]:
    assert line in INDEX, line
Z = [-1000.0, -720.0, -40.0, 0.0, 40.0, 720.0, 1000.0]
head = re.search(r"^z +(.*)$", OUT, re.M).group(1).split()
assert [float(v) for v in head] == Z
FORMS = ("naive", "other", "stable")
TABLE = {}
for name in FORMS:
    m = re.search(rf"^{name} +((?:\S+ +){{7}})  warnings: (.*)$", OUT, re.M)
    TABLE[name] = m.group(1).split()
assert TABLE["naive"] == ["0", "0", "4.248e-18", "0.5", "1", "1", "1"]
assert TABLE["other"] == ["0", "2.032e-313", "4.248e-18", "0.5", "1", "nan", "nan"]
assert TABLE["stable"] == ["0", "2.032e-313", "4.248e-18", "0.5", "1", "1", "1"]
assert "float64: np.exp overflows above 709.78, so the naive form warns for z below -709.78" in OUT

FUNCS = {"naive": ns["sigmoid_naive"], "other": ns["sigmoid_other"], "stable": ns["sigmoid_stable"]}
WARN = {}
for name in FORMS:
    flags = []
    for z in Z:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = FUNCS[name](np.array([z]))[0]
        flags.append(bool(caught))
        assert f"{v:.4g}" == TABLE[name][Z.index(z)]
    WARN[name] = flags
assert WARN["naive"] == [True, True, False, False, False, False, False]
assert WARN["other"] == [False, False, False, False, False, True, True]
assert WARN["stable"] == [False] * 7

# the stable form's two branches, as Activation_Sigmoid.forward writes them
SRC = (SNIP / "binary_classes.py").read_text(encoding="utf-8")
assert "out[pos] = 1.0 / (1.0 + np.exp(-inputs[pos]))" in SRC and "pos = inputs >= 0" in SRC
assert "out[neg] = ex / (1.0 + ex)" in SRC and "ex = np.exp(inputs[neg])" in SRC


def show(t):
    if t in ("nan", "0", "1", "0.5"):
        return t
    m = re.fullmatch(r"(\d\.\d+)e-(\d+)", t)
    return rich(m.group(1), " ", TIMES, " ", sup("10", MINUS + m.group(2), italic=False))


VALUES = [[show(t) for t in TABLE[name]] for name in FORMS]
z = var("z")
EZ = sup("e", z)
EMZ = sup("e", rich(MINUS, z))
F_NAIVE = rich("1/(1 + ", EMZ, ")")
F_OTHER = rich(EZ, "/(1 + ", EZ, ")")

fig = Figure(
    "02-three-sigmoids", "A one-line sigmoid fails on one side; the stable one never",
    "A grid of three rows and seven columns, float64. Columns are the logits minus 1,000, minus 720, minus 40, 0, "
    "40, 720 and 1,000. Row 1, the naive form 1/(1 + e to the minus z): 0, 0, 4.248 times 10 to the minus 18, 0.5, "
    "1, 1, 1, with an overflow warning at minus 1,000 and minus 720. Row 2, the other form e to the z over (1 + e "
    "to the z): 0, 2.032 times 10 to the minus 313, 4.248 times 10 to the minus 18, 0.5, 1, nan, nan; the two nan "
    "cells at 720 and 1,000 are red and outlined. Row 3, the stable form, chosen by the sign of z: 0, 2.032 times "
    "10 to the minus 313, 4.248 times 10 to the minus 18, 0.5, 1, 1, 1, with no warning, filled green. Braces "
    "under the grid: for z below 0 the stable form uses e to the z over (1 + e to the z), for z of 0 or more 1/(1 "
    "+ e to the minus z). A key names the three cell styles: overflow warning with a value returned, nan with "
    "warnings, and no warning with a value returned. np.exp overflows above 709.78.",
    subtitle="Float64: np.exp overflows above 709.78, so each one-line form overflows on one side of 0.", data_w=True)

GX, GY, CW, CH = 192, 128, 104, 76
g = fig.grid(GX, GY, 3, 7, cell_w=CW, cell_h=CH, values=VALUES, font=13,
             fill=lambda i, j: "output-soft" if i == 2 else None)
g.col_labels([num(v) for v in Z], style="label")
fig.text(40, GY - 8, rich("logit ", z), "label")

# row labels: the name, and the expression under it
for i, (name, expr) in enumerate((("naive", F_NAIVE), ("other", F_OTHER), ("stable", "chosen by sign"))):
    c = g.cell(i, 0)
    fig.text(40, c.cy - 6, name, "label")
    fig.text(40, c.cy + 18, expr, "note")

# nan: the emphasised block; an overflow warning with a value: a dashed inner outline
g.window(1, 5, 1, 2, color="error", fill="error", form="grid")
for j, flag in enumerate(WARN["naive"]):
    if flag:
        fig.outline(g.cell(0, j).inset(4), "ink-muted", width=1.5, dash="lead")

# the stable form's two branches, under its row
yb = g.box.bottom + 12
fig.brace(g.cell(2, 0).x + 4, g.cell(2, 2).right - 4, yb, label=rich(z, " < 0: ", F_OTHER))
fig.brace(g.cell(2, 3).x + 4, g.cell(2, 6).right - 4, yb, label=rich(z, " ≥ 0: ", F_NAIVE))

# the key, one blank line under the brace labels: the three cell styles
fig.legend(GX, yb + 80, [dict(color="ink-muted", label="overflow warning, a value returned", mark="dashed-outline"),
                         dict(color="error-soft", label="nan, with warnings", mark="swatch"),
                         dict(color="output-soft", label="no warning, a value returned", mark="swatch")],
           direction="row")
fig.caption(rich("Choosing the expression by the sign of ", z, " keeps every argument of np.exp at or below 0."))
fig.write()
