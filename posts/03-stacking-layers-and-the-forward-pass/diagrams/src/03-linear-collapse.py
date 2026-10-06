"""Post 03, section 8.1: the two layers of the post and the single layer they collapse to, on the post's numbers.

Run from anywhere:  python posts/03-stacking-layers-and-the-forward-pass/diagrams/src/03-linear-collapse.py
Writes posts/03-stacking-layers-and-the-forward-pass/diagrams/03-linear-collapse.svg.
Every value is computed by snippets/linear_collapse.py (run here) and checked against its printout in section 8.1.
Weights are drawn as this post stores them, one row per neuron: W1 (3, 4), W2 (3, 3), and for the single layer
W_star transposed = W2 W1, (3, 4), the transpose of the (4, 3) W_star that the snippet prints. The largest difference
between the two outputs is rounding that can vary between machines, so the figure states no number for it.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, Raw, rich, num, MINUS  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "linear_collapse.py"
with contextlib.redirect_stdout(io.StringIO()):
    S = runpy.run_path(str(SNIP))
W1, B1, W2, B2 = S["weights1"], S["biases1"], S["weights2"], S["biases2"]
WS, BS = S["weights_star"], S["biases_star"]
TWO, ONE = S["layer2_outputs"], S["collapsed"]
WST = WS.T                                           # the single layer stored one row per neuron


def fmt(v, d=6):
    """The value as the post prints it: shortest form after rounding, true minus."""
    return num(repr(round(float(v), d)))


# section 8.1's printout
assert [fmt(v, 4) for v in WS.ravel()] == [num(s) for s in
                                          "-0.18 0.0458 0.3108 0.0724 -0.4201 -0.9812 -0.0014 0.2251 0.3877 "
                                          "0.605 -0.8471 -0.9181".split()]
assert [fmt(v, 4) for v in BS] == [num(s) for s in "-0.97 1.195 0.745".split()]
assert [fmt(v) for v in TWO.ravel()] == [fmt(v) for v in ONE.ravel()] == [num(s) for s in
       "0.5031 -1.04185 -2.03875 0.2434 -2.7332 -5.7633 -0.99314 1.41254 -0.35655".split()]
assert np.max(np.abs(TWO - ONE)) < 1e-12                      # floating-point rounding only
assert np.allclose(WST, W2 @ W1)                               # section 4: W_star transposed = W2 W1
P1, P2, PS = W1.size + B1.size, W2.size + B2.size, WS.size + BS.size
assert (P1, P2, P1 + P2, PS) == (15, 12, S["two_layer_params"], S["one_layer_params"]) == (15, 12, 27, 15)


STAR = "\u2217"                         # the post's W_* and b_*: a subscript asterisk


def bold(base, s=None, t=False):
    out = f'<tspan class="b">{base}</tspan>'
    if t and s is not None:
        # the transpose sits straight over the subscript: T raised, then the subscript pulled back under it
        # by the width of a 13 px T, then back to the baseline
        out += (f'<tspan class="sub" dy="-9">T</tspan><tspan class="sub" dx="-7" dy="13">{s}</tspan>'
                f'<tspan dy="-4">\u200b</tspan>')
    elif t:
        out += '<tspan class="sub" dy="-9">T</tspan><tspan dy="9">\u200b</tspan>'
    elif s is not None:
        out += f'<tspan class="sub" dy="4">{s}</tspan><tspan dy="-4">\u200b</tspan>'
    return Raw(out)


def say(a, d=6):
    return ", ".join(fmt(v, d).replace(MINUS, "minus ") for v in np.asarray(a).ravel())


fig = Figure(
    "03-linear-collapse", "Two layers without an activation are one layer",
    f"Two rows, each ending in the output Z2 of shape (3, 3) for the post's batch X of shape (3, 4). Weights are "
    f"drawn one row per neuron. Upper row, two layers with 27 parameters: W1 of shape (3, 4), {say(W1)}, with b1, "
    f"{say(B1)}, 15 parameters; an arrow carrying Z1 of shape (3, 3); W2 of shape (3, 3), {say(W2)}, with b2, "
    f"{say(B2)}, 12 parameters; then Z2, {say(TWO)}. Lower row, one layer with 15 parameters: W star transposed, "
    f"which is W2 times W1, of shape (3, 4), {say(WST, 4)}, with b star, which is b1 times W2 transposed plus b2, "
    f"{say(BS, 4)}, 15 parameters; one arrow, labelled no hidden layer, with the two formulas under it, straight "
    f"to Z2. An equals sign between the two Z2 grids sits over the note: same values, up to rounding. "
    f"A caption line reads: with no activation "
    f"between them, the second layer adds parameters, not capability.",
    subtitle=rich("Both rows take the post's batch ", bold("X"), ", shape (3, 4). Values from "
                  "snippets/linear_collapse.py."),
    height=720)

C, WC, ZC, F = 48, 64, 72, 14            # bias and W2 cells, cells of the (3, 4) matrices, output cells, font
XW = 40                                  # the (3, 4) matrix of each row
XB = XW + 4 * WC + 8                     # its bias
XW2 = 432                                # layer 2
XB2 = XW2 + 3 * C + 8
XZ = 960 - 40 - 3 * ZC                   # the output
ROWS = ((136, 192), (424, 480))          # (row heading baseline, grid top) for the two rows


def layer(x, y, w, b, wd, wlabel, blabel):
    """One stored layer: its weights, one row per neuron, and its bias beside them, row for row."""
    gw = fig.grid(x, y, 3, w.shape[1], cell_w=wd, cell_h=C, values=lambda i, j: fmt(w[i, j], 4),
                  fill=lambda i, j: "weight-soft", font=F)
    gb = fig.strip(gw.box.right + 8, y, 3, C, vertical=True, values=[fmt(v, 4) for v in b],
                   fill=lambda k: "weight-soft", font=F)
    fig.text(x, y - 16, wlabel, "label", color="weight")
    fig.text(gb.box.x, y - 16, blabel, "label", color="weight")
    return gw, gb


def output(y, values):
    g = fig.grid(XZ, y, 3, 3, cell_w=ZC, cell_h=C, values=lambda i, j: fmt(values[i, j]),
                 fill=lambda i, j: "output-soft", font=F)
    fig.text(XZ, y - 16, rich(bold("Z", "2"), " (3, 3)"), "label", color="output")
    return g


# -- upper row: the two layers of the post
(hy, gy) = ROWS[0]
fig.text(40, hy, rich("Two layers: ", num(P1 + P2), " parameters"), "head")
g1, b1 = layer(XW, gy, W1, B1, WC, rich(bold("W", "1"), " (3, 4), one row per neuron"), bold("b", "1"))
g2, b2 = layer(XW2, gy, W2, B2, C, rich(bold("W", "2"), " (3, 3)"), bold("b", "2"))
z_two = output(gy, TWO)
cy = g1.box.cy
fig.arrow((b1.box.right + 8, cy), (g2.box.x - 8, cy), label=rich(bold("Z", "1"), " (3, 3)"))
fig.arrow((b2.box.right + 8, cy), (z_two.box.x - 8, cy))
fig.note(g1.box, rich("12 + 3 = ", num(P1), " parameters"))
fig.note(g2.box, rich("9 + 3 = ", num(P2), " parameters"))

# -- lower row: the single equivalent layer of section 4
(hy, gy) = ROWS[1]
fig.text(40, hy, rich("One layer: ", num(PS), " parameters"), "head")
gs, bs = layer(XW, gy, WST, BS, WC, rich(bold("W", STAR, t=True), " (3, 4), one row per neuron"),
               bold("b", STAR))
z_one = output(gy, ONE)
fig.arrow((bs.box.right + 8, gs.box.cy), (z_one.box.x - 8, gs.box.cy), label="no hidden layer")
fig.note(gs.box, rich("12 + 3 = ", num(PS), " parameters"))
# section 4's substitution, under the arrow that replaces layer 2
mx = (bs.box.right + z_one.box.x) / 2
fig.text(mx, gs.box.cy + 40, rich(bold("W", STAR, t=True), " = ", bold("W", "2"), " ", bold("W", "1")), "label",
         anchor="middle")
fig.text(mx, gs.box.cy + 68, rich(bold("b", STAR), " = ", bold("b", "1"), " ", bold("W", "2", t=True), " + ",
                                  bold("b", "2")), "label", anchor="middle")

# -- the two outputs agree: an equals sign between the two Z2 grids, the note under it
ex = z_two.box.cx
fig.text(ex, z_two.box.bottom + 48, "=", "op", anchor="middle")
fig.text(ex, z_two.box.bottom + 76, "Same values, up to rounding", "note", anchor="middle")
fig.caption("With no activation between them, the second layer adds parameters, not capability.")
fig.write()
