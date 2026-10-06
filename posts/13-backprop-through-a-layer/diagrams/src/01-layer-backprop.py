"""Post 13 hero: the forward values of the worked layer, the one upstream value 2y-hat that flows back to all three
neurons, and the fifteen gradients it becomes.

Run from anywhere:  python posts/13-backprop-through-a-layer/diagrams/src/01-layer-backprop.py
Writes posts/13-backprop-through-a-layer/diagrams/01-layer-backprop.svg.
Every number is computed by snippets/layer_backward.py (run here): its forward and backward functions on its
inputs, weights and biases, asserted against the lines the snippet prints (sections 4 and 5 of the post) and
against its central-difference check.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sub, isub, span, num  # noqa: E402

PARTIAL = "\u2202"
SNIP = Path(__file__).resolve().parents[2] / "snippets" / "layer_backward.py"
with contextlib.redirect_stdout(io.StringIO()) as out:
    S = runpy.run_path(str(SNIP))
OUT = out.getvalue()
X, WTS, BS = S["inputs"], S["weights"], S["biases"]
Z, A, Y, L = S["forward"](WTS, BS, X)
DZ, DW, DB = S["backward"](Z, Y, X)
UP = 2 * Y
GATES = [int(z > 0) for z in Z]

# the snippet's printout for sections 4 and 5
assert "Z = [   3.1    7.2   11.3]   A = [   3.1    7.2   11.3]   Y = 21.6   L = 466.56" in OUT
assert "upstream dL/dY = 2Y = 43.2   gates = [1, 1, 1]   dL/dZ = [  43.2   43.2   43.2]" in OUT
for k in range(3):
    assert f"neuron {k + 1}   dL/dW = [  43.2   86.4  129.6  172.8]   dL/db =  43.2" in OUT
assert "over 15 parameters: 6.4e-09" in OUT
assert X == [1.0, 2.0, 3.0, 4.0] and GATES == [1, 1, 1]
num_dW, num_db = S["numerical_gradients"](WTS, BS, X)
assert S["largest_gap"](DW, DB, num_dW, num_db) < 1e-8


def f1(v):
    """One decimal, as the post prints the layer's values."""
    return num(v, 1)


assert [f1(z) for z in Z] == ["3.1", "7.2", "11.3"] and f1(Y) == "21.6" and f"{L:.2f}" == "466.56"
assert f1(UP) == "43.2" and all(f1(d) == "43.2" for d in DZ + DB)


def d(top, bot):
    """The partial derivative top over bot, as the post writes it: a slash between two partials."""
    return rich(PARTIAL, top, "/", PARTIAL, bot)


YH = var("\u0177")          # y-hat, the layer sum
LV = var("L")

fig = Figure(
    "01-layer-backprop", "One upstream value reaches every gradient",
    "The worked layer drawn left to right. Four input nodes x1 to x4 hold 1, 2, 3 and 4 and are joined to three "
    "neuron cards. Neuron 1 reads z1 = 3.1, a1 = 3.1, gate 1 and dL/dz1 = 43.2; neuron 2 reads z2 = 7.2, "
    "a2 = 7.2, gate 1, dL/dz2 = 43.2; neuron 3 reads z3 = 11.3, a3 = 11.3, gate 1, dL/dz3 = 43.2. The three "
    "activations meet in the layer sum, y-hat = 21.6, whose derivative with respect to each activation is 1, "
    "and the sum feeds the loss L = y-hat squared = 466.56, whose derivative is 2 y-hat = 43.2. An arrow under "
    "the network runs right to left, labelled: backward, one value, 43.2, reaches all three neurons.Below, a 3 by 4 table of weight gradients, "
    "dL/dw_kj = dL/dz_k times x_j, with one row per neuron and one column per input, reads 43.2, 86.4, 129.6 "
    "and 172.8 in every row, and a column of bias gradients, dL/db_k, reads 43.2 three times. A key gives the colours: "
    "input, forward value, gradient, loss.",
    subtitle="The layer of section 4: four inputs, three ReLU neurons, target 0. All three gates are open.",
    height=720)

# -- columns of the network
XI, R_IN = 104, 20                       # input nodes
XC, WC, HC = 216, 232, 80                # neuron cards: left edge, width, height
XS, XL, R = 600, 768, 24                 # layer-sum node, loss node
YMID = 272
Y_IN = [YMID + 64 * (j - 1.5) for j in range(4)]           # 176, 240, 304, 368
TOPS = [YMID - HC // 2 + 88 * (k - 1) for k in range(3)]  # 144, 232, 320
CYS = [t + HC // 2 for t in TOPS]
assert all(v % 4 == 0 for v in Y_IN + TOPS + CYS)

YHEAD = 128
for x, s in ((XI, "Inputs"), (XC + WC // 2, "Neurons, one gate each"), (XS, "Layer sum"), (XL, "Loss")):
    fig.text(x, YHEAD, s, "head", anchor="middle")

# edges first: every input to every neuron, every neuron to the sum, the sum to the loss
for cy in CYS:
    for y in Y_IN:
        fig.edge((XI + R_IN, y), (XC, cy))
    fig.edge((XC + WC, cy), (XS - R, YMID))
fig.edge((XS + R, YMID), (XL - R, YMID))

for j, y in enumerate(Y_IN):
    fig.node(XI, y, r=R_IN, label=sub("x", str(j + 1)), color="input")
    fig.text(XI - R_IN - 12, y + 5, num(int(X[j])), "value", anchor="end", color="input")

for k, (top, cy) in enumerate(zip(TOPS, CYS)):
    i = str(k + 1)
    fig.card(XC, top, WC, HC, lines=[
        rich(span(rich(sub("z", i), " = ", f1(Z[k]), ",  ", sub("a", i), " = ", f1(A[k])), color="output")),
        rich(f"gate {GATES[k]},  ", span(rich(d(LV, sub("z", i)), " = ", f1(DZ[k])), color="gradient"))])

fig.node(XS, YMID, r=R, label="\u03a3", color="output")
fig.node(XL, YMID, r=R, label=LV, color="error")
# forward value above each node, its local derivative (the backward factor) below
fig.text(XS, YMID - 40, rich(YH, " = ", f1(Y)), "value", anchor="middle", color="output")
fig.text(XS, YMID + 52, rich(d(YH, isub("a", "k")), " = 1"), "value", anchor="middle", color="gradient")
fig.text(XL, YMID - 40, rich(LV, " = ", f"{L:.2f}"), "value", anchor="middle", color="error")
fig.text(XL, YMID + 52, rich(d(LV, YH), " = 2", YH, " = ", f1(UP)), "value", anchor="middle", color="gradient")

# the backward direction: one arrow under the network, right to left
YB = 432
fig.arrow((XL + 56, YB), (XC + 16, YB), label="backward: one value, 43.2, reaches all three neurons")

# -- the fifteen gradients: one row per neuron, one column per input, and the biases
GX, GY, CW, CH = XC, 528, 80, 40
g = fig.grid(GX, GY, 3, 4, cell_w=CW, cell_h=CH, values=lambda k, j: f1(DW[k][j]),
             fill=lambda k, j: "gradient-soft", font=16)
g.col_labels([span(rich(sub("x", str(j + 1)), " = ", num(int(X[j]))), color="input") for j in range(4)],
             style="label")
g.row_labels([f"neuron {k + 1}" for k in range(3)], style="note")
BX = GX + 4 * CW + 32
gb = fig.strip(BX, GY, 3, cell_w=CW, cell_h=CH, vertical=True, values=[f1(v) for v in DB],
               fill=lambda k: "gradient-soft", font=16)
gb.col_labels([span(d(LV, isub("b", "k")), color="gradient")], style="label")
fig.text(GX, GY - 48, rich("Weight gradients  ", d(LV, isub("w", "kj")), " = ", d(LV, isub("z", "k")), " \u00b7 ",
                           isub("x", "j")), "head")

# the key and the reading, right of the table
KX = BX + CW + 40
fig.legend(KX, GY + 4, [dict(color="input", label="input"), dict(color="output", label="forward value"),
                        dict(color="gradient", label="gradient"), dict(color="error", label="loss")])

fig.caption("Every column of the table is 43.2 times one input; each bias is 43.2 times 1.")
fig.write()
