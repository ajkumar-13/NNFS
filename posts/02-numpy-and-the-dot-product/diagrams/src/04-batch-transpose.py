"""Post 02 section 8: a batch through a layer, samples-first, and why the weights are transposed.

Run from anywhere:  python posts/02-numpy-and-the-dot-product/diagrams/src/04-batch-transpose.py
Writes posts/02-numpy-and-the-dot-product/diagrams/04-batch-transpose.svg.

Samples-first, as section 8 turns the third form round: every sample is a row of the inputs and of the
outputs. Layout (960 x 720): the arithmetic reads left to right as one line, inputs . weights.T + biases =
outputs, every operand centred on one axis with its operator between, so sample 1's row of inputs and its row
of outputs sit on the same line. Above, at the left, the weights as stored (one row per neuron) with an arrow
labelled transpose down onto weights.T, and beside them the error the call raises without the transpose.
Column labels of weights.T and the outputs sit under the grids so the arrow lands on weights.T itself.
Every value is computed here from snippets/batch_layer.py; the error text comes from snippets/silent_failures.py.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, num, CDOT  # noqa: E402

SNIPPETS = Path(__file__).resolve().parents[2] / "snippets"
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(SNIPPETS / "batch_layer.py"))
X, W, b = ns["inputs"], ns["weights"], ns["biases"]
OUT = np.dot(X, W.T) + b
assert np.allclose(OUT, [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])   # section 8's output
assert X.shape == (3, 4) and W.T.shape == (4, 3) and OUT.shape == (3, 3)

buf = io.StringIO()                              # the loud failure, as section 11 prints it
with contextlib.redirect_stdout(buf):
    runpy.run_path(str(SNIPPETS / "silent_failures.py"))
assert ("np.dot(inputs, weights): shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)"
        in buf.getvalue().splitlines())

D_W, D_X, D_Z = 2, 1, 3
assert np.allclose(np.round(OUT, D_Z), OUT)
TERMS = [(float(x), float(w)) for x, w in zip(X[0], W[0])]
assert abs(sum(x * w for x, w in TERMS) + b[0] - OUT[0, 0]) < 1e-12 and round(OUT[0, 0], 3) == 4.8


def factor(v, d):
    s = num(v, d)
    return f"({s})" if v < 0 else s


WORKED = (" + ".join(f"{factor(x, D_X)} {CDOT} {factor(w, D_W)}" for x, w in TERMS)
          + f" + {num(float(b[0]), 1)} = {num(float(OUT[0, 0]), D_Z)}")


def say(v, d):
    return num(float(v), d).replace("\u2212", "minus ")


def rows(M, d):
    return "; ".join(", ".join(say(v, d) for v in r) for r in M)


fig = Figure(
    "04-batch-transpose", "The transpose lets a batch meet the weights",
    "Section 8's batch, samples-first, read left to right as one line: the inputs, shape (3, 4), one row per "
    f"sample, {rows(X, D_X)}; a product dot; weights.T, shape (4, 3), one column per neuron, "
    f"{rows(W.T, D_W)}; a plus; the bias row, shape (3,), {', '.join(say(v, 1) for v in b)}, added to every row; an "
    "equals sign; and "
    f"the outputs, shape (3, 3), one row per sample and one column per neuron: {rows(OUT, D_Z)}. Above, at the left, "
    f"the weights as stored, shape (3, 4), one row per neuron, {rows(W, D_W)}, with an arrow labelled transpose "
    "down to weights.T. The row of sample 1 and the column of neuron 1 are outlined, and the output cell "
    f"{say(OUT[0, 0], D_Z)} is bold; the worked line reads 1.0 times 0.20 plus 2.0 times 0.80 plus 3.0 times minus "
    "0.50 plus 2.5 times 1.00, plus the bias 2.0, equals 4.800. A note beside the stored weights says that without "
    "the transpose, np.dot(inputs, weights) pairs (3, 4) with (3, 4), the 4 meets a 3, and NumPy raises ValueError.",
    subtitle="Section 8's batch, samples-first: every sample is a row. The weights are stored one row per neuron.",
    height=720)

CI, CW, CH, FONT = 48, 64, 48, 14                # one-decimal cells 48 wide, the rest 64 (room for "neuron 1"); all 48 high
CS_W = 56                                        # the stored weights: two decimals, no column labels
GAP = 32                                         # between operands; each operator sits in the middle
XI = 104                                         # inputs; their row labels sit in the 64 to their left
XT = XI + 4 * CI + GAP                           # weights.T
XB = XT + 3 * CW + GAP                           # biases, one decimal
XO = XB + 3 * CI + GAP                           # outputs
assert (XT, XB, XO) == (328, 552, 728) and XO + 3 * CW == 920        # the line fills the content width
T_T = 320                                        # top of weights.T, the tallest operand
AXIS = T_T + 4 * CH // 2                         # 416: every operand is centred on this line
T_X = AXIS - 3 * CH // 2                         # 344: inputs and outputs, 3 rows each
T_B = AXIS - CH // 2                             # 392: the bias row
T_S, CS = 152, CH                                # the stored weights: top, and the line's cell height

# -- the line: inputs . weights.T + biases = outputs
gx = fig.grid(XI, T_X, 3, 4, cell_w=CI, cell_h=CH, values=X.tolist(), decimals=D_X, font=FONT,
              fill=lambda i, j: "input-soft")
gx.row_labels([f"sample {k + 1}" for k in range(3)])
gx.window(0, 0, 1, 4, "input")

gt = fig.grid(XT, T_T, 4, 3, cell_w=CW, cell_h=CH, values=W.T.tolist(), decimals=D_W, font=FONT,
              fill=lambda i, j: "weight-soft")
gt.col_labels([f"neuron {k + 1}" for k in range(3)], side="below")
gt.window(0, 0, 4, 1, "weight")

gb = fig.grid(XB, T_B, 1, 3, cell_w=CI, cell_h=CH, values=[b.tolist()], decimals=1, font=FONT,
              fill=lambda i, j: "weight-soft")

go = fig.grid(XO, T_X, 3, 3, cell_w=CW, cell_h=CH, values=OUT.tolist(), decimals=D_Z, font=FONT,
              fill=lambda i, j: "output-soft", strong={(0, 0): "output"})
go.col_labels([f"neuron {k + 1}" for k in range(3)], side="below")
assert gx.box.cy == gt.box.cy == gb.box.cy == go.box.cy == AXIS
assert gx.cell(0, 0).y == go.cell(0, 0).y         # sample 1's row of inputs and its row of outputs on one line

fig.op(XT - GAP // 2, AXIS, CDOT)                # the operators sit on the line's axis
fig.op(XB - GAP // 2, AXIS, "+")
fig.op(XO - GAP // 2, AXIS, "=")

# names and shapes under the operands: under the grid, or under its column labels
NAME_X = gx.box.bottom + 48                      # 536: inputs and outputs share it (outputs carry column labels)
fig.text(gx.box.cx, NAME_X, "inputs, (3, 4)", "label", anchor="middle")
fig.text(go.box.cx, NAME_X, "outputs, (3, 3)", "label", anchor="middle")
fig.text(gt.box.cx, gt.box.bottom + 48, "weights.T, (4, 3)", "label", anchor="middle")
fig.text(gb.box.cx, gb.box.bottom + 28, "biases, (3,)", "label", anchor="middle")
fig.text(gb.box.cx, gb.box.bottom + 48, "added to every row", "note", anchor="middle")

# -- above: the weights as stored, one row per neuron, and the transpose that turns them into weights.T
gw = fig.grid(XI, T_S, 3, 4, cell_w=CS_W, cell_h=CS, values=W.tolist(), decimals=D_W, font=FONT,
              fill=lambda i, j: "weight-soft")
gw.row_labels([f"neuron {k + 1}" for k in range(3)])
fig.text(XI, T_S - 16, "weights, (3, 4), as stored", "head")
assert gw.box.right <= XT and gw.box.bottom + 24 == T_T   # the arrow turns down over weights.T
assert gx.box.y - gw.box.bottom == 48                     # one gap between the stored weights and the inputs
fig.arrow((gw.box.right + 8, gw.box.cy), (gt.box.cx, T_T), label="transpose", via=((gt.box.cx, gw.box.cy),))
XE = 560                                         # the loud failure, level with the stored weights' last row
fig.text(XE, gw.box.bottom - 28, "Without .T, np.dot(inputs, weights) pairs (3, 4)", "note")
fig.text(XE, gw.box.bottom - 8, "with (3, 4): the 4 meets a 3, and NumPy raises ValueError.", "note")

# -- below: one output cell worked out
YS = 600
fig.text(XI, YS, "sample 1 with neuron 1, plus its bias:", "label")
fig.text(XI, YS + 32, WORKED, "math")
fig.caption(rich("np.dot(inputs, weights.T) + biases: (3, 4) ", CDOT, " (4, 3) + (3,) → (3, 3)."))
fig.write()
