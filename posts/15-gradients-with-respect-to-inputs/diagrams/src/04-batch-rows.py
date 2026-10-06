"""Post 15, section 5: over a batch the input gradient keeps one row per sample.

Run from anywhere:  python posts/15-gradients-with-respect-to-inputs/diagrams/src/04-batch-rows.py
Writes posts/15-gradients-with-respect-to-inputs/diagrams/04-batch-rows.svg.
snippets/batch_and_handoff.py is run here (runpy): the per-sample outputs, the upstream gradient, the weights, the
batched product, both checks and the shapes of the two shared gradients are read from it and asserted against the
lines it prints and the numbers the post states in sections 5 and 7.2.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sub, isub, arr, tr, span, sup, CDOT, TIMES  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "batch_and_handoff.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

X, weights, Y, dZ, dX, N = s["X"], s["weights"], s["Y"], s["dL_dZ"], s["dL_dX"], s["N"]
dweights, dbiases = s["dweights"], s["dbiases"]
assert X.shape == (3, 4) and dZ.shape == (3, 3) and dX.shape == (3, 4) and N == 3
YV = [f"{v:.2f}" for v in Y[:, 0]]
assert YV == ["18.00", "15.30", "8.22"] and "Y per sample: 18.00, 15.30, 8.22" in OUT
assert np.allclose(dZ, 2 * Y / N)                                  # every gate open: 2 y_i / N in each column
UP = [[f"{v:.2f}" for v in r] for r in dZ]
assert UP == [["12.00"] * 3, ["10.20"] * 3, ["5.48"] * 3]
assert str(np.round(dZ, 4)) in OUT and str(np.round(dX, 4)) in OUT
RES = [[f"{v:.3f}" for v in r] for r in dX]
assert RES == [["18.000", "21.600", "25.200", "28.800"], ["15.300", "18.360", "21.420", "24.480"],
               ["8.220", "9.864", "11.508", "13.152"]]
assert "largest gap to a central difference on all 12 inputs: 2.8e-09" in OUT
assert "largest gap to the three rows computed one sample at a time: 3.6e-15" in OUT
assert dweights.shape == (4, 3) and dbiases.shape == (1, 3)
assert "shape (4, 3)" in OUT and "shape (1, 3)" in OUT


def dL(of):
    return rich("∂", var("L"), "/∂", of)


fig = Figure(
    "04-batch-rows", "Over a batch the input gradient keeps one row per sample",
    "The batch of three samples of section 5. Left, the upstream gradient dL/dZ, shape (3, 3), one row per sample: "
    "12.00, 10.20 and 5.48 in every column, which is 2 times y-hat over N for y-hat = 18.00, 15.30 and 8.22. Times "
    "the array weights, W transposed, shape (3, 4), it gives dL/dX, shape (3, 4): 18.000, 21.600, 25.200, 28.800; "
    "15.300, 18.360, 21.420, 24.480; 8.220, 9.864, 11.508, 13.152. Row 1 of the upstream gradient and row 1 of "
    "the result are outlined. Notes: row i of the result uses only row i of dL/dZ; the rows computed one sample at "
    "a time differ by at most 3.6e-15 and a central difference on all 12 inputs by 2.8e-09; the weight gradient "
    "(4, 3) and the bias gradient (1, 3) instead sum over the three samples.",
    subtitle=rich("The batch of section 5, ", var("N"), " = 3, through the layer of posts 13 and 14. Every ReLU "
                  "gate is open."))

C = 48
YG = 184
up = fig.grid(136, YG, 3, 3, cell_w=64, cell_h=C, values=UP, font=16, fill=lambda i, j: "gradient-soft")
up.window(0, 0, 1, 3, "gradient")
fig.text(up.box.cx, YG - 40, rich(dL(arr("Z")), "  (3, 3)"), "label", anchor="middle")
up.col_labels([sub("z", str(k + 1)) for k in range(3)])
# the row labels share the cells' baseline (cell centre + round(0.35 * 16) = + 6), off the 4 grid, so in data()
with fig.data():
    for i, v in enumerate(YV):
        fig.text(up.box.x - 12, up.cell(i, 0).cy + 6, rich(sub("ŷ", str(i + 1)), " = ", v), "label", anchor="end",
                 snap=False)

fig.op(360, YG + 1.5 * C, CDOT)
w = fig.grid(392, YG, 3, 4, cell_w=56, cell_h=C, values=[[f"{v:.1f}" for v in r] for r in weights], font=16,
             fill=lambda i, j: "weight-soft")
fig.text(w.box.cx, YG - 40, rich(span("weights", mono=True), " = ", tr(arr("W")), "  (3, 4)"), "label",
         anchor="middle")
w.col_labels([sub("x", str(j + 1)) for j in range(4)])

fig.op(648, YG + 1.5 * C, "=")
res = fig.grid(680, YG, 3, 4, cell_w=60, cell_h=C, values=RES, font=15, fill=lambda i, j: "gradient-soft")
res.window(0, 0, 1, 4, "gradient")
fig.text(res.box.cx, YG - 40, rich(dL(arr("X")), "  (3, 4)"), "label", anchor="middle")
res.col_labels([sub("x", str(j + 1)) for j in range(4)])

# -- what the rows mean, and how the shared gradients differ
YN = up.box.bottom + 28
fig.text(up.box.x, YN, rich("row ", var("i"), ": 2", isub("ŷ", "i"), " / ", var("N")), "note")
fig.text(res.box.x, YN, rich("row ", var("i"), " uses only row ", var("i")), "note")
fig.text(res.box.x, YN + 20, rich("of ", dL(arr("Z"))), "note")
fig.text(40, 424, rich("Rows computed one sample at a time: largest gap 3.6 ", TIMES, " ", sup("10", "−15", italic=False),
                       ".  Central difference on all 12 inputs: 2.8 ", TIMES, " ", sup("10", "−9", italic=False), "."),
         "note")
fig.text(40, 452, rich("The shared gradients sum over the samples instead: ", span("dweights", mono=True), " (4, 3), ",
                       span("dbiases", mono=True), " (1, 3)."), "note")

fig.caption(rich("The batch axis ", var("N"), " is outside the product and survives; the neuron axis ", var("m"),
                 " is summed away."))
fig.write()
