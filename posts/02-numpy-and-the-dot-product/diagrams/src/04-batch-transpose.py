"""Post 02 section 8: a batch through a layer, samples-first, and why the weights are transposed.

Run from anywhere:  python posts/02-numpy-and-the-dot-product/diagrams/src/04-batch-transpose.py
Writes posts/02-numpy-and-the-dot-product/diagrams/04-batch-transpose.svg.

Samples-first, as section 8 turns the third form round: every sample is a row of the inputs and of the
outputs. Layout (960 x 720): the stored weights (one row per neuron) top left, an arrow labelled .T to their
transpose; the transpose sits above the outputs and the inputs to their left, so a sample's row and a neuron's
column meet in one output cell; the bias row sits between the transpose and the outputs, added to every row.
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
    "Section 8's batch, samples-first. Top left, the stored weights, shape (3, 4), one row per neuron: "
    f"{rows(W, D_W)}. An arrow labelled transpose leads to weights.T, shape (4, 3), one column per neuron. Under the "
    f"transpose sits the bias row {', '.join(say(v, 1) for v in b)}, added to every row, and under that the "
    f"outputs, shape (3, 3): {rows(OUT, D_Z)}. Left of the outputs are the inputs, shape (3, 4), one row per "
    f"sample: {rows(X, D_X)}. The row of sample 1 and the column of neuron 1 are outlined and meet in the output "
    f"cell {say(OUT[0, 0], D_Z)}, which is bold; the worked line reads 1.0 times 0.20 plus 2.0 times 0.80 plus 3.0 "
    "times minus 0.50 plus 2.5 times 1.00, plus the bias 2.0, equals 4.800. A note says that without the "
    "transpose, np.dot(inputs, weights) pairs (3, 4) with (3, 4), the 4 meets a 3, and NumPy raises ValueError.",
    subtitle="Section 8's batch, samples-first: every sample is a row. The weights are stored one row per neuron.",
    height=720)

CW, CT, CH, FONT = 56, 64, 40, 14               # cells: inputs and stored weights 56 wide, the rest 64; all 40 high
XL, XR, XN = 120, 424, 648                       # left grids (row labels to their left), right grids, notes
T_W, T_B, T_O = 152, 336, 400                    # tops: the transpose, the bias row, the outputs

gw = fig.grid(XL, T_W, 3, 4, cell_w=CW, cell_h=CH, values=W.tolist(), decimals=D_W, font=FONT,
              fill=lambda i, j: "weight-soft")
gw.row_labels([f"neuron {k + 1}" for k in range(3)])
fig.text(XL, T_W - 16, rich("weights, ", "(3, 4)"), "head")

gt = fig.grid(XR, T_W, 4, 3, cell_w=CT, cell_h=CH, values=W.T.tolist(), decimals=D_W, font=FONT,
              fill=lambda i, j: "weight-soft")
gt.col_labels([f"neuron {k + 1}" for k in range(3)])
gt.window(0, 0, 4, 1, "weight")
fig.arrow((gw.box.right + 8, gw.box.cy), (XR - 8, gw.box.cy), label="transpose")
fig.text(XN, T_W + 24, rich("weights.T, ", "(4, 3)"), "head")
fig.text(XN, T_W + 52, "one column per neuron", "note")

fig.grid(XR, T_B, 1, 3, cell_w=CT, cell_h=CH, values=[b.tolist()], decimals=1, font=FONT,
         fill=lambda i, j: "weight-soft")
fig.text(XR - 16, T_B + 28, "+", "op", anchor="middle")
fig.text(XN, T_B + 28, rich("biases, ", "(3,): added to every row"), "label")

go = fig.grid(XR, T_O, 3, 3, cell_w=CT, cell_h=CH, values=OUT.tolist(), decimals=D_Z, font=FONT,
              fill=lambda i, j: "output-soft", strong={(0, 0): "output"})
fig.text(XN, T_O + 24, rich("outputs, ", "(3, 3)"), "head")
fig.text(XN, T_O + 52, "one row per sample,", "note")
fig.text(XN, T_O + 72, "one column per neuron", "note")

gx = fig.grid(XL, T_O, 3, 4, cell_w=CW, cell_h=CH, values=X.tolist(), decimals=D_X, font=FONT,
              fill=lambda i, j: "input-soft")
gx.row_labels([f"sample {k + 1}" for k in range(3)])
gx.window(0, 0, 1, 4, "input")
fig.text(XL, T_O - 16, rich("inputs, ", "(3, 4)"), "head")
with fig.data():                                 # sample 1's row runs on into its output row
    fig.leader((gx.box.right, gx.cell(0, 0).cy), (XR, go.cell(0, 0).cy))

YS = T_O + 3 * CH + 48                           # 568
fig.text(XL, YS, "sample 1 with neuron 1, plus its bias:", "label")
fig.text(XL, YS + 32, WORKED, "math")
fig.text(XL, YS + 72, rich("Without .T, np.dot(inputs, weights) pairs (3, 4) with (3, 4): the 4 meets a 3, "
                           "and NumPy raises ValueError."), "note")
fig.caption(rich("np.dot(inputs, weights.T) + biases: (3, 4) ", CDOT, " (4, 3) + (3,) \u2192 (3, 3)."))
fig.write()
