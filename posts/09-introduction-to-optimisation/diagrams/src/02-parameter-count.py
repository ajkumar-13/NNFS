"""Post 09 sections 1 and 4: every parameter of the 3-neuron network, and both networks to one scale.

Run from anywhere:  python posts/09-introduction-to-optimisation/diagrams/src/02-parameter-count.py
Writes posts/09-introduction-to-optimisation/diagrams/02-parameter-count.svg.

The shapes are those of Layer_Dense(2, n_hidden) and Layer_Dense(n_hidden, 3), built here with the class of
the post's snippet, for n_hidden = 3 and 64; their sizes are asserted against section 1's table, the 387 of
section 4, and the parameter counts the two snippets print. The chances in the caption are (1/2) to the
power of each count, as section 4 computes them.
"""
import contextlib
import io
import runpy
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, arr, sup, var, num, TIMES, MINUS  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(SNIP / "random_perturbation.py"), run_name="snippet")
OUT = buf.getvalue()
Layer_Dense = ns["Layer_Dense"]


def arrays(n_hidden):
    d1, d2 = Layer_Dense(2, n_hidden), Layer_Dense(n_hidden, 3)
    return [("W", "1", d1.weights.shape), ("b", "1", d1.biases.shape),
            ("W", "2", d2.weights.shape), ("b", "2", d2.biases.shape)]


SMALL, WIDE = arrays(3), arrays(64)
size = lambda shape: shape[0] * shape[1]  # noqa: E731
assert [s for _, _, s in SMALL] == [(2, 3), (1, 3), (3, 3), (1, 3)]           # section 1's table
assert [size(s) for _, _, s in SMALL] == [6, 3, 9, 3]
assert [size(s) for _, _, s in WIDE] == [2 * 64, 64, 64 * 3, 3]               # section 4: 2·64 + 64 + 64·3 + 3
N_SMALL, N_WIDE = sum(size(s) for _, _, s in SMALL), sum(size(s) for _, _, s in WIDE)
assert (N_SMALL, N_WIDE) == (21, 387)
assert "  21 parameters" in OUT and "  387 parameters" in OUT                  # random_perturbation.py prints both
CHANCE_SMALL = Fraction(1, 2) ** N_SMALL
assert CHANCE_SMALL == Fraction(1, 2_097_152)
c = float(Fraction(1, 2) ** N_WIDE)
MANT, EXP = f"{c:.0e}".split("e")
assert (MANT, int(EXP)) == ("3", -117)                                         # section 4: about 3 × 10^-117

fig = Figure(
    "02-parameter-count", "Every parameter is one number to move",
    "Top: the 21 parameters of the network with 3 hidden neurons, one cell each, in their array shapes: W1 of "
    "shape (2, 3), 6 weights; b1 of shape (1, 3), 3 biases; W2 of shape (3, 3), 9 weights; b2 of shape (1, 3), "
    "3 biases; 6 plus 3 plus 9 plus 3 is 21. Bottom: both networks as bars on one axis from 0 to 400 "
    "parameters. With 3 hidden neurons the bar is 21 long; with 64 it is 387: 128 weights in W1, 64 biases in "
    "b1, 192 weights in W2 and 3 biases in b2. If every parameter had to land in the right half of its range, "
    "one random draw would succeed with chance 1 in 2,097,152 for 21 parameters and about 3 times 10 to the "
    "minus 117 for 387.",
    subtitle=rich("Layer_Dense(2, ", var("n"), ") then Layer_Dense(", var("n"), ", 3), with ", var("n"), " = 3 and ",
                  var("n"), " = 64."))

# -- top: the 21 cells, each array in its own shape, centred on one line, joined by + and =
fig.text(40, 120, "3 hidden neurons, one cell per parameter", "head")
CELL, OPW, BOT, SUM = 32, 56, 256, 288          # cell size, the gap between arrays, their common bottom, the sum
widths = [s[1] * CELL for _, _, s in SMALL]
total_w = sum(widths) + OPW * 4 + 24                  # four operators (three + and one =) and the total
x = (960 - total_w) // 2 // 8 * 8
boxes = []
for (name, k, shape), w in zip(SMALL, widths):
    g = fig.grid(x, BOT - shape[0] * CELL, shape[0], shape[1], cell=CELL, fill=lambda i, j: "weight-soft")
    boxes.append(g.box)
    x += w + OPW
# each array's name over it; under the arrays, one line that adds their sizes: 6 + 3 + 9 + 3 = 21
for bx, (name, k, shape) in zip(boxes, SMALL):
    fig.text(bx.cx, bx.y - 12, rich(arr(name, sub=k), "  ", str(shape)), "label", anchor="middle")
    fig.text(bx.cx, SUM, str(size(shape)), "math", anchor="middle", color="weight")
for a, b in zip(boxes, boxes[1:]):
    fig.op((a.right + b.x) / 2, SUM - 8, "+")
eq = boxes[-1].right + OPW / 2
fig.op(eq, SUM - 8, "=")
fig.text(eq + OPW / 2, SUM, str(N_SMALL), "math", bold=True)

# -- bottom: both networks on one axis, one unit per parameter
fig.text(40, 344, "Both networks to one scale", "head")
PX0, PX1, LO, HI = 216, 856, 0, 400
sx = lambda v: PX0 + (PX1 - PX0) * (v - LO) / (HI - LO)  # noqa: E731
ROWS = [("3 hidden neurons", SMALL, 376), ("64 hidden neurons", WIDE, 424)]
BH = 24
with fig.data():
    for t in range(0, 401, 100):
        fig.edge((sx(t), 360), (sx(t), 444), color="grid", width=0.75)
        fig.text(sx(t), 464, num(t), "tick", anchor="middle", snap=False)
    fig.edge((PX0, 360), (PX0, 444), color="ink-muted", width=1)
    for label, parts, cy in ROWS:
        fig.text(40, cy + 5, label, "label", snap=False)
        acc = 0
        for name, k, shape in parts:
            n = size(shape)
            seg = Box(sx(acc), cy - BH / 2, sx(acc + n) - sx(acc), BH)
            fig.fill(seg, "weight-soft", fit=False)
            fig.outline(seg, "weight", width=1)
            if parts is WIDE and seg.w > 64:
                fig.text(seg.cx, cy + 5, rich(arr(name, sub=k), "  ", num(n)), "label", anchor="middle",
                         snap=False)
            acc += n
        # the small bar's four segments are slivers at this scale: name their sizes after it, as the top panel does
        total = num(acc) if parts is WIDE else rich(" + ".join(str(size(s)) for _, _, s in parts), " = ", num(acc))
        fig.text(sx(acc) + 12, cy + 5, total, "value", color="weight", snap=False)
    last = Box(sx(N_WIDE - 3), 424 - BH / 2, sx(N_WIDE) - sx(N_WIDE - 3), BH)
fig.callout(last, rich(arr("b", sub="2"), "  3"), side="above", length=12)
fig.text(40, 464, "parameters", "note")
fig.caption(rich("Chance that all land in the right half: 1 in 2,097,152 for 21, about 3 ", TIMES, " ",
                 sup("10", MINUS + "117", italic=False), " for 387."))
fig.write()
