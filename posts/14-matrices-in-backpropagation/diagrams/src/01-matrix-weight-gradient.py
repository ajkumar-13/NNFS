"""Post 14, sections 2 and 4: the twelve weight gradients of post 13 as one outer product.

Run from anywhere:  python posts/14-matrices-in-backpropagation/diagrams/src/01-matrix-weight-gradient.py
Writes posts/14-matrices-in-backpropagation/diagrams/01-matrix-weight-gradient.svg.
snippets/single_sample.py is run here (runpy): the upstream gradient, the input row and the (3, 4) result are
the snippet's own arrays, and the printed lines the post quotes are checked against its output.

Layout: the product drawn as an outer product, the column (dL/dZ transposed) beside the result and the input
row X above it, so entry (k, j) of the result sits where row k of the column meets entry j of the row. One
entry, neuron 2 and input 3, is outlined in all three places and worked out on the right, with the shape
arithmetic and the code line under it.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sub, arr, tr, span, CDOT  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "single_sample.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
X, dZ, dW, W = s["X"], s["dL_dZ"], s["dL_dW"], s["weights"]

# what section 4 prints, and what the post states
assert X.shape == (1, 4) and dZ.shape == (1, 3) and dW.shape == (3, 4) == W.shape
assert "dL_dZ: [[43.2 43.2 43.2]] (1, 3)" in OUT
assert ("[[ 43.2  86.4 129.6 172.8]\n [ 43.2  86.4 129.6 172.8]\n [ 43.2  86.4 129.6 172.8]]") in OUT
assert np.array_equal(dW, dZ.T @ X)


def fmt(v):
    """The value as NumPy prints it here: at most one decimal is ever needed."""
    t = repr(round(float(v), 6))
    return t[:-2] if t.endswith(".0") else t


COL = [fmt(v) for v in dZ[0]]
ROW = [fmt(v) for v in X[0]]
assert COL == ["43.2"] * 3 and ROW == ["1", "2", "3", "4"]
RES = [[fmt(v) for v in r] for r in dW]
assert RES == [["43.2", "86.4", "129.6", "172.8"]] * 3

K, J = 1, 2                                    # neuron 2, input 3 (0-based indices)
assert fmt(dZ[0, K] * X[0, J]) == RES[K][J] == "129.6"


def dLd(x):
    """dL/d(x) in the partial form."""
    return rich("∂", var("L"), "/∂", x)


UP_T = tr(rich("(", dLd(arr("Z")), ")"))

fig = Figure(
    "01-matrix-weight-gradient", "One outer product holds all twelve weight gradients",
    "The single sample of post 13 drawn as an outer product. On the left, the upstream gradient dL/dZ transposed, "
    "a column of shape (3, 1) holding 43.2 for neurons 1, 2 and 3. Above the result, the input row X, shape "
    "(1, 4), holding x1 to x4 = 1, 2, 3, 4. The result dL/dW, shape (3, 4), the shape of the weights, has one "
    "row per neuron and one column per input, and every row reads 43.2, 86.4, 129.6, 172.8. Neuron 2's 43.2, "
    "the input 3 and the entry 129.6 are outlined, and the right side works it out: dL/dw23 = dL/dz2 times x3 = "
    "43.2 times 3 = 129.6. Under it the shapes, (3, 1) times (1, 4) gives (3, 4), and the code line "
    "dL_dW = dL_dZ.T @ X.",
    subtitle=rich("The single sample of section 4: entry (", var("k"), ", ", var("j"), ") is row ", var("k"), " of the column times entry ", var("j"), " of the row."))

CW, CH = 80, 56
XC, XR = 136, 224                    # the column, the result (and the row above it)
YX, YR = 136, 208                    # top of the input row, of the result

gx = fig.strip(XR, YX, 4, cell_w=CW, cell_h=CH, values=ROW, fill=lambda j: "input-soft", font=18)
gc = fig.strip(XC, YR, 3, cell_w=CW, cell_h=CH, vertical=True, values=COL, fill=lambda k: "gradient-soft",
               font=18)
gr = fig.grid(XR, YR, 3, 4, cell_w=CW, cell_h=CH, values=RES, fill=lambda i, j: "gradient-soft", font=18,
              strong={(K, J): "gradient"})
assert gc.box.bottom == gr.box.bottom and gx.box.right == gr.box.right

# the one entry: its two factors and the result cell
gc.window(K, K, color="gradient")
gx.window(J, J, color="input")
gr.outline(K, J, color="gradient", width=1.5)

# names and shapes: X to the left of its row, the column and the result named under them
fig.text(XC + CW, YX + 29, rich(arr("X"), "  (1, 4)"), "label", anchor="end", color="input")
gx.col_labels([sub("x", str(j + 1)) for j in range(4)], style="label")
gc.row_labels([f"neuron {k + 1}" for k in range(3)], style="note")
YL = gr.box.bottom + 32
fig.text(gc.box.cx, YL, UP_T, "head", anchor="middle")
fig.text(gc.box.cx, YL + 24, "(3, 1)", "note", anchor="middle")
fig.text(XR, YL, dLd(arr("W")), "head")
fig.text(XR, YL + 24, "(3, 4): one row per neuron, one column per input", "note")

# the worked entry, the shapes and the code, in a column on the right
XT = 608
fig.text(XT, YX + 16, "Neuron 2, input 3", "head")
fig.text(XT, YX + 52, rich(dLd(sub("w", "23")), " = ", dLd(sub("z", "2")), " ", CDOT, " ", sub("x", "3")), "math")
fig.text(XT, YX + 84, rich("= 43.2 ", CDOT, " 3 = ", span("129.6", bold=True)), "math")
fig.text(XT, YR + 72, "Shapes", "head")
fig.text(XT, YR + 108, rich("(3, 1) ", CDOT, " (1, 4) → (3, 4)"), "math")
fig.text(XT, YR + 132, "the shape of the stored weights", "note")
fig.text(XT, YL, "Code", "head")
fig.code_block(XT, YL + 28, ["dL_dW = dL_dZ.T @ X"])

fig.caption("Each weight gradient is one upstream value times one input, nothing summed.")
fig.write()
