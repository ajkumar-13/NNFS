"""Post 08 section 5: the worked batch read with integer labels (indexing) and with one-hot labels (multiply, sum).

Run from anywhere:  python posts/08-loss-categorical-cross-entropy/diagrams/src/02-indexing-methods.py
Writes posts/08-loss-categorical-cross-entropy/diagrams/02-indexing-methods.svg.
The softmax rows and both label arrays are the ones snippets/worked_example.py defines (it is run here); the
product, the correct confidences and the losses are computed with NumPy and checked against the four lines
the snippet prints and the post quotes.

Layout: two rows that share one result column on the right. Top, integer labels: class_targets beside the
softmax rows, the cell each label picks outlined. Bottom, one-hot labels times the softmax rows equals the
product, whose wrong-class entries are 0. Both arrows end in the same correct_confidences, and -log of it
gives the three per-sample losses and their mean.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, MINUS, ODOT, rich, span, var  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "worked_example.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

P = np.asarray(s["softmax_outputs"])
T = list(s["class_targets"])
H = np.asarray(s["class_targets_onehot"])
assert P.tolist() == [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]] and T == [0, 1, 1]
assert np.array_equal(np.eye(3)[T], H)
picked = P[range(3), T]
product = H * P
summed = np.sum(product, axis=1)
losses = -np.log(summed)
mean = np.mean(losses)
lines = OUT.splitlines()
assert lines[:4] == [str(picked), str(summed), str(losses), str(mean)], lines[:4]
assert lines[:4] == ["[0.7 0.5 0.9]", "[0.7 0.5 0.9]", "[0.35667494 0.69314718 0.10536052]", "0.38506088005216804"]
assert np.array_equal(picked, summed)
LOSS3 = [f"{v:.3f}" for v in losses]
assert LOSS3 == ["0.357", "0.693", "0.105"] and f"{mean:.3f}" == "0.385"
assert f"batch loss, the mean of the three: {mean:.3f}" in OUT


def mono(t):
    return span(t, mono=True)


def two(v):
    return f"{v:.2f}"


fig = Figure(
    "02-indexing-methods", "Two label formats, one vector of true-class probabilities",
    "Two rows on the worked batch of softmax outputs 0.70, 0.10, 0.20; 0.10, 0.50, 0.40; 0.02, 0.90, 0.08. Top, "
    "integer labels: class_targets 0, 1, 1 picks row 0 column 0, row 1 column 1 and row 2 column 1, the cells "
    "outlined. Bottom, one-hot labels: the rows 1, 0, 0; 0, 1, 0; 0, 1, 0 times the softmax outputs element by "
    "element give 0.70, 0.00, 0.00; 0.00, 0.50, 0.00; 0.00, 0.90, 0.00. Both paths end in the same "
    "correct_confidences 0.70, 0.50, 0.90; minus log of it gives the losses " + ", ".join(LOSS3)
    + f" and the mean {mean:.3f}.",
    subtitle="The worked batch of section 4, true classes 0, 1, 1. Outlined: the probability on the true class.",
    height=720)

C = 48
XA, XB, XC = 40, 248, 456            # one-hot labels, softmax outputs, product
YA, YB = 208, 488                    # the two rows' grids
XR, XL, YR, SW = 680, 824, 344, 64   # result strip, loss strip, their top, their width

TRUE = lambda i, j: j == T[i]  # noqa: E731

# --- top row: integer labels
fig.text(40, 144, rich("Integer labels: ", mono("softmax_outputs[range(3), class_targets]")), "head")
ct = fig.strip(144, YA, 3, C, vertical=True, values=T, fill=lambda i: "input-soft")
fig.text(ct.box.cx, 176, mono("class_targets"), "code", anchor="middle")
ga = fig.grid(XB, YA, 3, 3, C, values=P.tolist(), decimals=2, fill=lambda i, j: "output-soft" if TRUE(i, j) else None)
for i in range(3):
    ga.outline(i, T[i], color="output", width=1.5)
fig.text(ga.box.cx, 176, mono("softmax_outputs"), "code", anchor="middle")
ga.col_labels(["0", "1", "2"])
ga.row_labels(["row 0", "row 1", "row 2"])

# --- bottom row: one-hot labels
fig.text(40, 424, rich("One-hot labels: multiply element-wise, then sum along ", mono("axis=1")), "head")
gh = fig.grid(XA, YB, 3, 3, C, values=H.tolist(), fill=lambda i, j: "input-soft" if H[i][j] else None)
fig.text(XA, 456, mono("class_targets_onehot"), "code")
gp = fig.grid(XB, YB, 3, 3, C, values=P.tolist(), decimals=2)
fig.text(gp.box.cx, 456, mono("softmax_outputs"), "code", anchor="middle")
gx = fig.grid(XC, YB, 3, 3, C, values=product.tolist(), decimals=2,
              fill=lambda i, j: "output-soft" if TRUE(i, j) else None)
fig.text(gx.box.cx, 456, "product", "label", anchor="middle")
for i in range(3):
    gx.outline(i, T[i], color="output", width=1.5)
for g in (gh, gp, gx):
    g.col_labels(["0", "1", "2"])
fig.op((XA + 3 * C + XB) / 2, gh.box.cy, ODOT)
fig.op((XB + 3 * C + XC) / 2, gh.box.cy, "=")

# --- the shared result and the losses
gr = fig.strip(XR, YR, 3, C, vertical=True, width=SW, values=[two(v) for v in summed], font=18,
               fill=lambda i: "output-soft")
fig.text(gr.box.cx, YR - 16, mono("correct_confidences"), "code", anchor="middle")
gl = fig.strip(XL, YR, 3, C, vertical=True, width=SW, values=LOSS3, font=18, fill=lambda i: "error-soft")
fig.text(gl.box.cx, YR - 16, mono("sample_losses"), "code", anchor="middle")
fig.text(gl.box.cx, gl.box.bottom + 28, rich("mean ", span(f"{mean:.3f}", bold=True)), "label", anchor="middle")

# the top arrow turns at 624, left of the correct_confidences label it would otherwise cross; the bottom
# arrow turns 16 further right, so it leaves the product with a 32-unit stub before it turns
ELBOW_TOP, ELBOW = 624, 640
fig.arrow((ga.box.right + 8, ga.box.cy), (gr.box.x, gr.cell(0).cy), via=[(ELBOW_TOP, ga.box.cy), (ELBOW_TOP, gr.cell(0).cy)])
fig.text(ga.box.right + 16, ga.box.cy - 12, "pick", "note")
fig.arrow((gx.box.right + 8, gx.box.cy), (gr.box.x, gr.cell(2).cy), via=[(ELBOW, gx.box.cy), (ELBOW, gr.cell(2).cy)])
fig.text(gx.box.right + 16, gx.box.cy + 24, "sum, axis=1", "note")
fig.arrow((gr.box.right + 8, gr.box.cy), (gl.box.x - 8, gr.box.cy), label=rich(MINUS, "log"))

fig.caption("Either path gives the same three probabilities; only they enter the loss.")
fig.write()
