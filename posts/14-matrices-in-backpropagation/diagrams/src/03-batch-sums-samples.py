"""Post 14, section 6: on a batch, the matrix product contracts the batch axis and so sums the samples.

Run from anywhere:  python posts/14-matrices-in-backpropagation/diagrams/src/03-batch-sums-samples.py
Writes posts/14-matrices-in-backpropagation/diagrams/03-batch-sums-samples.svg.
snippets/batch.py is run here (runpy): X, the made-up upstream gradient, the product and the three per-sample
rows are the snippet's own arrays, checked against the lines it prints and section 6 quotes.

Layout: top, the product (dL/dZ)^T X = dL/dW in a row, the shapes under the grids and a bracket joining the
two batch axes, the ones the product contracts. Bottom, row 1 of the result rebuilt one sample at a time:
each sample's input row, drawn under the rows of X, scaled by that sample's upstream value for neuron 1, and
their sum, which is row 1 of the product.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, arr, tr, span, num, CDOT, TIMES  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "batch.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
X, dZ, dW = s["X"], s["dL_dZ"], s["dL_dW"]
N, m = dZ.shape
n = X.shape[1]
assert (N, m, n) == (3, 3, 4) and dW.shape == (m, n)

# what section 6 prints and states
assert "Weight gradients:\n[[ 0.5 20.1 10.9  4.1]\n [ 0.5 20.1 10.9  4.1]\n [ 0.5 20.1 10.9  4.1]]" in OUT
assert "sample 1, row of neuron 1: [1.  2.  3.  2.5]" in OUT
assert "sample 2, row of neuron 1: [ 4. 10. -2.  4.]" in OUT
assert "sample 3, row of neuron 1: [-4.5  8.1  9.9 -2.4]" in OUT
assert "sum of the three, row of neuron 1: [ 0.5 20.1 10.9  4.1]" in OUT
assert "largest difference from the matrix product: 0.0" in OUT
PARTS = [np.outer(dZ[i], X[i])[0] for i in range(N)]          # neuron 1's row from each sample
assert np.max(np.abs(sum(PARTS) - dW[0])) < 1e-12
assert [dZ[i, 0] for i in range(N)] == [1.0, 2.0, 3.0]
assert abs(1 * 2.0 + 2 * 5.0 + 3 * 2.7 - dW[0, 1]) < 1e-12 and f"{dW[0, 1]:.1f}" == "20.1"


def dLd(x):
    return rich("∂", var("L"), "/∂", x)


def vals(a):
    """One decimal throughout a grid of X or of a row of contributions, as the post writes them."""
    return [[num(round(float(v), 1), 1) for v in r] for r in np.atleast_2d(a)]


def say(a):
    return ", ".join(num(round(float(v), 1), 1).replace("\u2212", "minus ") for v in np.ravel(a))


fig = Figure(
    "03-batch-sums-samples", "Contracting the batch axis adds up the samples",
    "Top: the batch of section 6.1 as one product. dL/dZ transposed, shape (m, N) = (3, 3), one row per neuron and "
    "one column per sample, holds 1, 2, 3 in every row; times X, shape (N, n) = (3, 4), one row per sample, "
    f"holding {say(X)}; equals dL/dW, shape (m, n) = (3, 4), every row {say(dW[0])}. A bracket joins the two N "
    "axes: the batch axis, which the product contracts and sums away. Row 1 of dL/dZ transposed and row 1 of "
    "dL/dW are outlined. Bottom: row 1 rebuilt one sample at a time, each under the columns of X: 1 times row 1 "
    f"of X gives {say(PARTS[0])}; 2 times row 2 gives {say(PARTS[1])}; 3 times row 3 gives {say(PARTS[2])}; "
    f"their sum is {say(dW[0])}, row 1 of the product.",
    subtitle="The batch of section 6.1: three samples, four inputs, and a made-up upstream gradient of 1, 2 and 3.",
    height=720)

C = 48
XA, XX, XR = 120, 328, 584                 # left edges: (dL/dZ)^T, X, dL/dW
YT = 160                                   # top of the three grids

ga = fig.grid(XA, YT, m, N, C, values=dZ.T.astype(int).tolist(), fill=lambda i, j: "gradient-soft", font=16)
gx = fig.grid(XX, YT, N, n, C, values=vals(X), fill=lambda i, j: "input-soft", font=16)
gr = fig.grid(XR, YT, m, n, C, values=vals(dW), fill=lambda i, j: "gradient-soft", font=16,
              strong={(0, 1): "gradient"})
fig.op((ga.box.right + gx.box.x) / 2, ga.box.cy, CDOT)
fig.op((gx.box.right + gr.box.x) / 2, ga.box.cy, "=")
ga.window(0, 0, 1, N, "gradient")
gr.window(0, 0, 1, n, "gradient")
ga.row_labels([f"neuron {k + 1}" for k in range(m)], style="note")

# names above, what the axes are under, shapes and the bracket of the batch axes below that
for g, name in ((ga, tr(rich("(", dLd(arr("Z")), ")"))), (gx, arr("X")), (gr, dLd(arr("W")))):
    fig.text(g.box.x, YT - 20, name, "head")
YN = ga.box.bottom + 28
fig.text(ga.box.x, YN, "a column per sample", "note")
fig.text(gx.box.x, YN, "a row per sample", "note")
fig.text(gr.box.x, YN, "the shape of the weights", "note")
YS = YN + 36
mm, NN, nn = var("m"), var("N"), var("n")
fig.text(ga.box.cx, YS, rich("(", mm, ", ", NN, ")"), "math", anchor="middle")
fig.text(gx.box.cx, YS, rich("(", NN, ", ", nn, ")"), "math", anchor="middle")
fig.text(gr.box.cx, YS, rich("(", mm, ", ", nn, ")"), "math", anchor="middle")
N_OFF = 11                                 # the N of "(m, N)" sits about 11 right of the label's centre
fig.brace(ga.box.cx + N_OFF, gx.box.cx - N_OFF, YS + 8, kind="bracket")
fig.text((ga.box.cx + gx.box.cx) / 2, YS + 40, rich("the batch axis, ", NN, " = 3: contracted, summed"), "note",
         anchor="middle")

# bottom: row 1 of the result, one sample at a time, each under the columns of X
YH = 440
fig.text(40, YH, rich("Row 1 of ", dLd(arr("W")), ", one sample at a time"), "head")
H, STEP = 40, 48
rows = []
for i in range(N):
    y = YH + 16 + i * STEP
    g = fig.strip(XX, y, n, cell_w=C, cell_h=H, values=vals(PARTS[i])[0], fill=lambda j: "gradient-soft", font=16)
    fig.text(XX - 16, y + 25, rich(span(f"{int(dZ[i, 0])}", color="gradient"), f" {TIMES} row {i + 1} of ", arr("X")),
             "label", anchor="end")
    rows.append(g)
ysum = YH + 16 + N * STEP + 8
gs = fig.strip(XX, ysum, n, cell_w=C, cell_h=H, values=vals(dW[0])[0], fill=lambda j: "gradient-soft", font=16)
gs.outline(0, n - 1, color="gradient", width=1.5)
fig.text(XX - 16, ysum + 25, "sum", "label", anchor="end")
fig.text(gs.box.right + 16, ysum + 25, rich("= row 1 of ", dLd(arr("W"))), "label")
assert gs.box.bottom <= 656, gs.box.bottom

fig.caption(rich("Input 2 of neuron 1: 1 ", CDOT, " 2.0 + 2 ", CDOT, " 5.0 + 3 ", CDOT, " 2.7 = 20.1."))
fig.write()
