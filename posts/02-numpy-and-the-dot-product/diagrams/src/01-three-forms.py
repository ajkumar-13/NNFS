"""Post 02 hero: the three forms of np.dot, drawn with section 8's weights and inputs, neurons-first.

Run from anywhere:  python posts/02-numpy-and-the-dot-product/diagrams/src/01-three-forms.py
Writes posts/02-numpy-and-the-dot-product/diagrams/01-three-forms.svg.

Layout (960 x 720), neurons-first as section 3 frames the third form: W holds one neuron per row and the
second argument one sample per column. Three rows, one per form, each first argument at the left edge, the
second argument and the result in fixed columns, so the column x of row 2 is column 1 of X.T in row 3 and
the result 2.800 sits in the same place in all three rows. No bias: the figure shows np.dot alone.
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
from figkit import Figure, num  # noqa: E402

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
    "Three rows, one per form of np.dot, drawn neurons-first with the weights and inputs of section 8 and no "
    "bias. Vector by vector, np.dot(w, x): the weights of neuron 1, "
    f"{says(W[0], D_W)}, as a row, meet sample 1, {says(X[0], D_X)}, as a column, and give the scalar "
    f"{say(ONE, D_Z)}; shapes (4,) and (4,), one neuron and one sample. Matrix by vector, np.dot(W, x): the 3 by "
    f"4 weight matrix, rows {'; '.join(says(r, D_W) for r in W)}, meets the same column and gives the column "
    f"{says(LAYER, D_Z)}; shapes (3, 4) and (4,) give (3,), a layer of 3 neurons on one sample. Matrix by matrix, "
    f"np.dot(W, X.T): the same weights meet a 4 by 3 matrix whose columns are labelled samples 1 to 3 and give a 3 by 3 "
    f"result with one column per sample, rows {'; '.join(says(r, D_Z) for r in BATCH)}; shapes (3, 4) and (4, 3) "
    f"give (3, 3). Row 1 of the weights and column 1 of the samples are outlined, and {say(ONE, D_Z)} is bold in "
    "every result. A caption line says every form sums over the 4 inputs.",
    subtitle="Section 8's layer before the bias, neurons-first: one neuron per row of W, one sample per column.",
    height=720)

CW, CS, CH, FONT = 56, 64, 32, 14              # weight and result cells 56 wide, sample cells 64 (room for labels)
XW, XS, XZ, XT = 40, 312, 552, 752              # first argument, second argument, result, text column
OP1, OP2 = (XW + 4 * CW + XS) // 2, (XS + 3 * CS + XZ) // 2       # 288 and 528: the operator glyphs
TOPS = [144, 336, 528]
assert OP1 == 288 and OP2 == 528 and TOPS[2] + 4 * CH == fig.content.bottom

ROWS = [
    dict(head="Vector by vector", code="np.dot(w, x)", shapes="(4,) \u00b7 (4,) \u2192 scalar",
         notes=["one neuron, one sample"], first=W[:1], second=X[:1].T, out=np.array([[ONE]])),
    dict(head="Matrix by vector", code="np.dot(W, x)", shapes="(3, 4) \u00b7 (4,) \u2192 (3,)",
         notes=["a layer of 3 neurons,", "one sample"], first=W, second=X[:1].T, out=LAYER[:, None]),
    dict(head="Matrix by matrix", code="np.dot(W, X.T)", shapes="(3, 4) \u00b7 (4, 3) \u2192 (3, 3)",
         notes=["a layer of 3 neurons,", "3 samples as columns"], first=W, second=X.T, out=BATCH),
]

for k, (top, r) in enumerate(zip(TOPS, ROWS)):
    fig.text(XW, top - 16, r["head"], "head")
    a, s, z = r["first"], r["second"], r["out"]
    ga = fig.grid(XW, top, *a.shape, cell_w=CW, cell_h=CH, values=a.tolist(), decimals=D_W, font=FONT,
                  fill=lambda i, j: "weight-soft")
    gs = fig.grid(XS, top, *s.shape, cell_w=CS, cell_h=CH, values=s.tolist(), decimals=D_X, font=FONT,
                  fill=lambda i, j: "input-soft")
    fig.grid(XZ, top, *z.shape, cell_w=CW, cell_h=CH, values=z.tolist(), decimals=D_Z, font=FONT,
             fill=lambda i, j: "output-soft", strong={(0, 0): "output"})
    ymid = top + a.shape[0] * CH // 2 + 8                     # the operators sit on the first argument's middle
    with fig.data():                                          # the product dot, drawn: the glyph is too small
        fig.add(f'<circle cx="{OP1}" cy="{ymid - 8}" r="3" class="{fig._cls("f", "ink-muted")}"/>')
    fig.text(OP2, ymid, "=", "op", anchor="middle")
    fig.text(XT, top + 20, r["code"], "code")
    fig.text(XT, top + 44, r["shapes"], "label")
    for i, line in enumerate(r["notes"]):
        fig.text(XT, top + 68 + 20 * i, line, "note")
    gs.col_labels([f"sample {j + 1}" for j in range(s.shape[1])])
    if k == 2:                                                # one dot product, picked out in the batch
        ga.window(0, 0, 1, 4, "weight")
        gs.window(0, 0, 4, 1, "input")

fig.caption("Every form sums over the 4 inputs, where the first argument's last axis meets the second's first.")
fig.write()
