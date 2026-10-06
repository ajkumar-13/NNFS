"""Post 15, sections 3 and 4: the four sums over paths as one matrix product, on the layer of posts 13 and 14.

Run from anywhere:  python posts/15-gradients-with-respect-to-inputs/diagrams/src/03-matrix-product.py
Writes posts/15-gradients-with-respect-to-inputs/diagrams/03-matrix-product.svg.
snippets/input_gradient.py is run here (runpy): the upstream row, the weights, the product, the column sums and the
central differences of section 7.1 are read from it and asserted against the lines it prints and the numbers the post
states (64.80, 77.76, 90.72, 103.68; column sums 1.5, 1.8, 2.1, 2.4; largest gap 1.0e-08).
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sub, arr, tr, span, sup, CDOT, TIMES  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "input_gradient.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

weights, dZ, dX, numeric = s["weights"], s["dL_dZ"], s["dL_dX"], s["numeric"]
assert dZ.shape == (1, 3) and weights.shape == (3, 4) and dX.shape == (1, 4)
UP = [f"{v:.1f}" for v in dZ[0]]
RES = [f"{v:.2f}" for v in dX[0]]
COLSUM = [f"{v:.1f}" for v in weights.sum(axis=0)]
assert UP == ["43.2"] * 3 and COLSUM == ["1.5", "1.8", "2.1", "2.4"]
assert RES == ["64.80", "77.76", "90.72", "103.68"]
assert "dL_dZ @ weights          = [64.80, 77.76, 90.72, 103.68]   shape (1, 3) @ (3, 4) = (1, 4)" in OUT
assert np.allclose(dX[0], dZ[0, 0] * weights.sum(axis=0))         # equal upstream entries: 43.2 x column sum
NUM = [f"{v:.6f}" for v in numeric[0]]
assert NUM == ["64.800000", "77.760000", "90.720000", "103.680000"]
for j in range(4):
    assert f"central difference {numeric[0, j]:10.6f}" in OUT
GAP = np.max(np.abs(numeric - dX))
assert f"largest gap: {GAP:.1e}" in OUT and f"{GAP:.1e}" == "1.0e-08"


def dL(of):
    return rich("∂", var("L"), "/∂", of)


fig = Figure(
    "03-matrix-product", "The four sums over paths are one matrix product",
    "The upstream row dL/dZ, shape (1, 3), holding 43.2, 43.2, 43.2, times the array weights, which holds W "
    "transposed, shape (3, 4), with one row per neuron: 0.1, 0.2, 0.3, 0.4; 0.5, 0.6, 0.7, 0.8; 0.9, 1.0, 1.1, 1.2. "
    "Column 1, the weights leaving x1, is outlined, and so is the first entry of the result dL/dX, shape (1, 4): "
    "64.80, 77.76, 90.72, 103.68. Under the weights, the column sums 1.5, 1.8, 2.1, 2.4; each result is 43.2 times "
    "its column sum. A line below gives the central differences of section 7.1, 64.800000, 77.760000, 90.720000 "
    "and 103.680000, with a largest gap of 1.0e-08.",
    subtitle=rich("The layer of posts 13 and 14 at ", var("x"), " = (1, 2, 3, 4), one sample. Outlined: the path "
                  "sum for ", sub("x", "1"), "."))

C = 48
YG = 208                                   # top of the weights grid; the two rows sit on its middle row
up = fig.strip(40, YG + C, 3, cell_w=64, cell_h=C, values=UP, font=16, fill=lambda i: "gradient-soft")
fig.text(up.box.cx, up.box.y - 32, rich(dL(arr("Z")), "  (1, 3)"), "label", anchor="middle")
up.col_labels([sub("z", str(k + 1)) for k in range(3)])

fig.op(264, YG + 1.5 * C, CDOT)
w = fig.grid(296, YG, 3, 4, cell_w=56, cell_h=C, values=[[f"{v:.1f}" for v in r] for r in weights], font=16,
             fill=lambda i, j: "weight-soft")
w.window(0, 0, 3, 1, "weight")
fig.text(w.box.cx, w.box.y - 32, rich(span("weights", mono=True), " = ", tr(arr("W")), "  (3, 4)"), "label", anchor="middle")
w.col_labels([sub("x", str(j + 1)) for j in range(4)])

fig.op(552, YG + 1.5 * C, "=")
res = fig.strip(584, YG + C, 4, cell_w=80, cell_h=C, values=RES, font=16, fill=lambda i: "gradient-soft")
res.window(0, 0, color="gradient")
fig.text(res.box.cx, res.box.y - 32, rich(dL(arr("X")), "  (1, 4)"), "label", anchor="middle")
res.col_labels([sub("x", str(j + 1)) for j in range(4)])

# -- column sums under the weights, and what they explain under the result
YS = w.box.bottom + 28
for j, v in enumerate(COLSUM):
    fig.text(w.cell(0, j).cx, YS, v, "label", anchor="middle", color="weight")
fig.text(w.box.x - 16, YS, "column sums", "note", anchor="end")
fig.text(res.box.x, res.box.bottom + 28, rich("each entry = 43.2 ", TIMES, " its column sum"), "note")
fig.text(res.box.x, res.box.bottom + 48, rich("the summed axis: ", var("m"), " = 3 neurons"), "note")

# -- the check of section 7.1
fig.text(40, YS + 48, rich("Central difference on each input, ", var("h"), " = ", sup("10", "−5", italic=False), ":  ",
                       ",  ".join(NUM), ";  largest gap 1.0 ", TIMES, " ", sup("10", "−8", italic=False)), "note")

fig.caption(rich("Column ", var("j"), " of the weights holds the weights leaving input ", var("j"), "; one dot product per column."))
fig.write()
