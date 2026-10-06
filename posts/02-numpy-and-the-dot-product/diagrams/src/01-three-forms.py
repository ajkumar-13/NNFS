"""Post 02 hero: the three forms of np.dot, drawn with section 8's weights and inputs, neurons-first.

Run from anywhere:  python posts/02-numpy-and-the-dot-product/diagrams/src/01-three-forms.py
Writes posts/02-numpy-and-the-dot-product/diagrams/01-three-forms.svg.

Layout (960 x 720), neurons-first as section 3 frames the third form: W holds one neuron per row and the
second argument one sample per column. The two vector forms side by side across the top, each 408 wide and
flush with the two margins, 64 apart (twice the gap inside an equation), with its code, shapes and note
under it; the batch form across the bottom, its text beside the result. Within each equation the operands follow at one spacing with the
operators centred between, so no equation has a gap inside it, and column x of the first form sits right
above column 1 of X.T. No bias: the figure shows np.dot alone.
Every value is computed here from the arrays of snippets/batch_layer.py and checked against the post.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, arr, rich, num  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "batch_layer.py"
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(SNIPPET))
X, W, b = ns["inputs"], ns["weights"], ns["biases"]
assert X.shape == (3, 4) and W.shape == (3, 4)

ONE = float(np.dot(W[0], X[0]))                 # vector by vector: neuron 1, sample 1
LAYER = np.dot(W, X[0])                         # matrix by vector: 3 neurons, sample 1
BATCH = np.dot(W, X.T)                          # matrix by matrix, neurons-first: one column per sample
assert np.allclose(LAYER + b, [4.8, 1.21, 2.385])                     # section 8, "sample 1 alone"
assert np.allclose(BATCH.T + b, [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])
assert np.allclose(BATCH.T, np.dot(X, W.T))                           # section 8's last check
assert ONE == LAYER[0] == BATCH[0, 0] and round(ONE, 3) == 2.8

D_W, D_X, D_Z = 2, 1, 3                         # decimals: weights, inputs, outputs (all exact at these)
assert np.allclose(np.round(W, D_W), W) and np.allclose(np.round(X, D_X), X)
assert np.allclose(np.round(BATCH, D_Z), BATCH)


def say(v, d):
    return num(float(v), d).replace("\u2212", "minus ")


def says(vals, d):
    return ", ".join(say(v, d) for v in vals)


fig = Figure(
    "01-three-forms", "One function, three forms: the shapes decide",
    "Three equations, one per form of np.dot, drawn neurons-first with the weights and inputs of section 8 and no "
    "bias: the two vector forms side by side at the top, each with its call, shapes and meaning under it, and the "
    "matrix form across the bottom with its text beside the result. Vector by vector, np.dot(w, x): the weights of neuron 1, "
    f"{says(W[0], D_W)}, as a row, meet sample 1, {says(X[0], D_X)}, as a column, and give the scalar "
    f"{say(ONE, D_Z)}; shapes (4,) and (4,), one neuron and one sample. Matrix by vector, np.dot(W, x): the 3 by "
    f"4 weight matrix, rows {'; '.join(says(r, D_W) for r in W)}, meets the same column and gives the column "
    f"{says(LAYER, D_Z)}; shapes (3, 4) and (4,) give (3,), a layer of 3 neurons on one sample. Matrix by matrix, "
    f"np.dot(W, X.T): the same weights meet a 4 by 3 matrix whose columns are labelled samples 1 to 3 and give a 3 by 3 "
    f"result with one column per sample, rows {'; '.join(says(r, D_Z) for r in BATCH)}; shapes (3, 4) and (4, 3) "
    f"give (3, 3). Row 1 of the weights and column 1 of the samples are outlined, and {say(ONE, D_Z)} is bold in "
    "every result. A caption line says every form sums over the 4 inputs.",
    subtitle=rich("Section 8's layer before the bias, neurons-first: one neuron per row of ", arr("W"),
                  ", one sample per column."),
    height=720)

CW, CS, CH, FONT = 56, 64, 32, 14              # weight and result cells 56 wide, sample cells 64 (room for labels)
GAP = 32                                        # every operator sits in the middle of a gap this wide
PANEL = 4 * CW + GAP + CS + GAP + CW            # 408: one vector-form equation
X0, X1 = 40, 512                                # the two upper panels, flush with the two margins
GUTTER = X1 - (X0 + PANEL)                      # 64: twice the gap inside an equation, so the pair reads as two
TOP_UP, TOP_LOW = 160, 464                      # the upper pair; the batch row
assert PANEL == 408 and X1 + PANEL == 920 and GUTTER == 2 * GAP

ROWS = [
    dict(head="Vector by vector", code="np.dot(w, x)", shapes="(4,) · (4,) → scalar",
         notes=["one neuron, one sample"], first=W[:1], second=X[:1].T, out=np.array([[ONE]]),
         x=X0, top=TOP_UP, text="below"),
    dict(head="Matrix by vector", code="np.dot(W, x)", shapes="(3, 4) · (4,) → (3,)",
         notes=["a layer of 3 neurons, one sample"], first=W, second=X[:1].T, out=LAYER[:, None],
         x=X1, top=TOP_UP, text="below"),
    dict(head="Matrix by matrix", code="np.dot(W, X.T)", shapes="(3, 4) · (4, 3) → (3, 3)",
         notes=["a layer of 3 neurons,", "3 samples as columns"], first=W, second=X.T, out=BATCH,
         x=X0, top=TOP_LOW, text="beside"),
]

for k, r in enumerate(ROWS):
    x, top = r["x"], r["top"]
    fig.text(x, top - 16, r["head"], "head")
    a, s, z = r["first"], r["second"], r["out"]
    ga = fig.grid(x, top, *a.shape, cell_w=CW, cell_h=CH, values=a.tolist(), decimals=D_W, font=FONT,
                  fill=lambda i, j: "weight-soft")
    xs = ga.box.right + GAP                                   # the second argument follows the first
    gs = fig.grid(xs, top, *s.shape, cell_w=CS, cell_h=CH, values=s.tolist(), decimals=D_X, font=FONT,
                  fill=lambda i, j: "input-soft")
    xz = gs.box.right + GAP                                   # the result follows its second argument
    gz = fig.grid(xz, top, *z.shape, cell_w=CW, cell_h=CH, values=z.tolist(), decimals=D_Z, font=FONT,
                  fill=lambda i, j: "output-soft", strong={(0, 0): "output"})
    ymid = top + a.shape[0] * CH // 2                         # the operators sit on the first argument's middle
    fig.op(ga.box.right + GAP // 2, ymid, "·")
    fig.op(gs.box.right + GAP // 2, ymid, "=")
    if r["text"] == "below":                                  # under the tallest operand, the 4-row column
        xt, y0 = x, gs.box.bottom + 28
    else:                                                     # beside the result, from its first row
        xt, y0 = gz.box.right + GAP, top + 20
    fig.text(xt, y0, r["code"], "code")
    fig.text(xt, y0 + 24, r["shapes"], "label")
    for i, line in enumerate(r["notes"]):
        fig.text(xt, y0 + 48 + 20 * i, line, "note")
    gs.col_labels([f"sample {j + 1}" for j in range(s.shape[1])])
    if k == 0:
        assert gz.box.right == X0 + PANEL
    if k == 2:                                                # one dot product, picked out in the batch
        assert xs == ROWS[0]["x"] + 4 * CW + GAP              # column 1 of X.T sits under the first form's x
        ga.window(0, 0, 1, 4, "weight")
        gs.window(0, 0, 4, 1, "input")

fig.caption("Every form sums over the 4 inputs, where the first argument's last axis meets the second's first.")
fig.write()
