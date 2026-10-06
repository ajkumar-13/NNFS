"""Post 01, section 10: a batch of three samples through the layer in one call, Z = X W^T + b.

Run from anywhere:  python posts/01-neurons-and-layers/diagrams/src/03-batch-in-one-call.py
Writes posts/01-neurons-and-layers/diagrams/03-batch-in-one-call.svg.
Every number comes from snippets/layer_batch.py, which this script runs; the printed matrix is checked.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sub, arr, tr, num  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "layer_batch.py"
with contextlib.redirect_stdout(io.StringIO()) as out:
    ns = runpy.run_path(str(SNIP))
X = np.array(ns["inputs"])
WT = np.array(ns["weights"]).T
B = np.array(ns["biases"])
P = X @ WT
Z = np.dot(ns["inputs"], np.array(ns["weights"]).T) + ns["biases"]
assert out.getvalue().strip() == str(Z)
assert np.allclose(Z, [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])
N, N_IN = X.shape
N_NEU = WT.shape[1]
S, K = 1, 1                                             # the traced entry: sample 2, neuron 2
assert round(P[S, K], 10) == -4.81 and round(Z[S, K], 10) == -1.81


def say(v, d):
    return num(float(v), d).replace("\u2212", "minus ")


def rows(a, d):
    return "; ".join(", ".join(say(v, d) for v in r) for r in a)


WTN = tr(arr("W"))
fig = Figure(
    "03-batch-in-one-call", "A batch goes through the layer in one call",
    f"Above the grids, the equation Z equals X W transposed plus b and the call np.dot(X, W.T) + b. Four grids in a "
    f"row: X, 3 by 4, one sample per row, rows {rows(X, 1)}; times W transposed, 4 by 3, one "
    f"neuron per column, rows {rows(WT, 2)}; plus b broadcast to 3 by 3, every row 2.0, 3.0, 0.5, copied from the "
    f"stored b of shape (3,) drawn above it; equals Z, 3 by 3, rows {rows(Z, 3)}. Sample 2's row of X, neuron 2's "
    f"column of W transposed, b2 and Z entry (2, 2) are outlined: minus 4.81 plus 3.0 is minus 1.81.",
    subtitle="Section 10's batch: three samples, the layer of section 7, one np.dot call and one addition.")

TOP, C = 208, 48                                       # W^T (4 rows) from TOP; rows of 48 everywhere
TOP3 = TOP + (N_IN - N) * C // 2                        # the 3-row grids, centred on W^T: 232..376 against 208..400
assert TOP3 == 232
XX = 104
gx = fig.grid(XX, TOP3, N, N_IN, cell_w=40, cell_h=C, values=X.tolist(), decimals=1, font=16,
              fill=lambda i, j: "input-soft")
WX = gx.box.right + 32
gw = fig.grid(WX, TOP, N_IN, N_NEU, cell_w=72, cell_h=C, values=WT.tolist(), decimals=2, font=14,
              fill=lambda i, j: "weight-soft")
BX = gw.box.right + 32
gb = fig.grid(BX, TOP3, N, N_NEU, cell_w=40, cell_h=C, values=[B.tolist()] * N, decimals=1, font=16,
              fill=lambda i, j: "weight-soft")
ZX = gb.box.right + 32
gz = fig.grid(ZX, TOP3, N, N_NEU, cell_w=72, cell_h=C, values=Z.tolist(), decimals=3, font=16,
              fill=lambda i, j: "output-soft", strong={(S, K): "output"})
assert (WX, BX, ZX, gz.box.right) == (296, 544, 696, 912)
OPY = TOP + C * N_IN // 2                               # the operators' centre line: the middle of every grid
assert OPY == gx.box.cy == gw.box.cy
fig.op((gx.box.right + gw.box.x) / 2, OPY, "·")
fig.op((gw.box.right + gb.box.x) / 2, OPY, "+")
fig.op((gb.box.right + gz.box.x) / 2, OPY, "=")

gx.row_labels([f"sample {i + 1}" for i in range(N)], style="tick")
gx.col_labels([sub("x", str(j + 1)) for j in range(N_IN)])
for g in (gw, gz):
    g.col_labels([f"neuron {j + 1}" for j in range(N_NEU)])

# the equation and the call the grids spell out, in the space left of the stored bias
fig.text(40, 140, rich(arr("Z"), " = ", arr("X"), WTN, " + ", arr("b")), "math")
fig.text(40, 168, "np.dot(X, W.T) + b", "code")

# the stored bias, above its broadcast copy; its label shares the equation's baseline
sb = fig.strip(BX, 112, N_NEU, width=40 * N_NEU, height=C, values=B.tolist(), decimals=1, font=16,
               fill=lambda i: "weight-soft")
fig.text(BX - 16, 140, rich(arr("b"), " (3,)"), "label", anchor="end")
fig.arrow((sb.box.cx, sb.box.bottom), (gb.box.cx, gb.box.y), label="copied to every row", side="right")

gx.window(S, 0, 1, N_IN, "input")
gw.outline(0, K, N_IN, 1, color="weight", width=1.5)
gb.outline(S, K, 1, 1, color="weight", width=1.5)
gz.window(S, K, 1, 1, "output")

Y_NAME = TOP + C * N_IN + 28                            # one baseline under the tallest grid
for g, name, shape in ((gx, arr("X"), "(3, 4)"), (gw, WTN, "(4, 3)"), (gb, rich(arr("b"), ", broadcast"), "(3, 3)"),
                       (gz, arr("Z"), "(3, 3)")):
    fig.text(g.box.cx, Y_NAME, rich(name, " ", shape), "label", anchor="middle")
fig.text(gz.box.right, Y_NAME + 32, rich("Sample 2, neuron 2: ", num(float(P[S, K]), 2), " + ", num(float(B[K]), 1), " = ",
                               num(float(Z[S, K]), 2)), "label", anchor="end", color="output")
fig.caption(rich("Entry (", var("i"), ", ", var("j"), ") of ", arr("Z"), " is sample ", var("i"), " times neuron ",
                 var("j"), "'s weights, plus neuron ", var("j"), "'s bias."))
fig.write()
