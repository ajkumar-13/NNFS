"""Post 20, section 3: the first array the backward pass hands on, its column sums, and the ReLU's gated copy.

Run from anywhere:  python posts/20-assembling-full-backpropagation/diagrams/src/02-first-gradients.py
Writes posts/20-assembling-full-backpropagation/diagrams/02-first-gradients.svg.
snippets/assemble.py is run here (runpy). The left grid and the strip are loss_activation.dinputs and
dense2.dbiases, printed as the snippet prints them (six decimals) and asserted against its printout; the right grid
is activation1.dinputs, read from the snippet's arrays (the snippet prints only its count of zeros, which is
asserted), with the closed gates taken from dense1.output <= 0 as the ReLU's backward takes them.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, arr, hat, span, var, num  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "assemble.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

y = s["y"]
G = s["loss_activation"].dinputs
DB2 = s["dense2"].dbiases
A1 = s["activation1"].dinputs
CLOSED = s["dense1"].output <= 0
N = len(y)
assert list(y) == [0, 1, 2, 1] and G.shape == (4, 3) and A1.shape == (4, 3)


def fmt(v):
    return num(float(f"{v:.6f}") + 0.0, 6)


# the snippet's printout of section 3, entry by entry
with np.printoptions(precision=6, suppress=True):
    assert "loss_activation.dinputs\n" + str(G) + "\n" in OUT
    assert "dense2.dbiases " + str(DB2) + "\n" in OUT
assert [fmt(v) for v in G[0]] == ["−0.166638", "0.083295", "0.083343"]
assert [fmt(v) for v in DB2[0]] == ["0.083399", "−0.166745", "0.083346"]
assert np.allclose(DB2, G.sum(axis=0, keepdims=True))
assert np.all(np.abs(G[np.arange(N), y] + 1 / 6) < 1e-3)               # close to -1/6 at the label
off = np.ones_like(G, dtype=bool)
off[np.arange(N), y] = False
assert np.all(np.abs(G[off] - 1 / 12) < 1e-4)                            # close to 1/12 elsewhere
K = int(CLOSED.sum())
assert K == 5 and int(np.sum(A1 == 0)) == 5 and np.all(A1[CLOSED] == 0) and np.all(A1[~CLOSED] != 0)
assert np.array_equal(A1[~CLOSED], s["dense2"].dinputs[~CLOSED])      # the ReLU copies the rest
assert f"closed ReLU gates: {K} of {CLOSED.size}; zeros in activation1.dinputs: {K}" in OUT

fig = Figure(
    "02-first-gradients", "Near −1/6 at each label, 1/12 elsewhere, 0 at closed gates",
    "Left: loss_activation.dinputs, the first array of the backward pass, four rows for the samples with labels "
    "y = 0, 1, 2, 1 and three columns for the classes: row 1 is -0.166638, 0.083295, 0.083343; row 2 0.083333, "
    "-0.166668, 0.083336; row 3 0.08334, 0.083327, -0.166667; row 4 0.083365, -0.166699, 0.083335. The cell of "
    "each sample's label is shaded green and holds about -1/6; the others hold about 1/12. Summed down each column "
    "they give dense2.dbiases: 0.083399, -0.166745, 0.083346, where class 1 is the label of two samples. Right: "
    "activation1.dinputs, the array the ReLU hands to dense1, four samples by three hidden neurons: row 1 "
    "-0.001795, 0.000648, 0; row 2 0, 0.001314, 0; row 3 0, 0, -0.000004; row 4 0.000958, 0.001314, 0.000801. "
    "The five zeros, shaded red, are the closed gates, where dense1's output was at most 0.",
    subtitle=rich("The backward pass of section 3 on the four samples. Six decimals, as the script prints them."),
    data_w=True)

fig.legend(40, 120, [dict(color="output-soft", label="the label's class", mark="swatch"),
                     dict(color="negative-soft", label=rich("closed gate, ", arr("Z", sub="1"), " ≤ 0"),
                          mark="swatch")],
           direction="row", gap=32)

CW, CH = 96, 40
YG = 192
# -- left: loss_activation.dinputs, its row labels the labels y, and its column sums
LX = 104
fig.text(40, 160, "loss_activation.dinputs", "code")
fig.text(236, 160, rich("= (", hat(arr("y")), " − ", arr("y"), ")/", var("N")), "label")
gl = fig.grid(LX, YG, 4, 3, cell_w=CW, cell_h=CH, font=14, values=lambda i, j: fmt(G[i, j]),
              fill=lambda i, j: "output-soft" if j == y[i] else None)
gl.col_labels([f"class {j}" for j in range(3)])
gl.row_labels([rich(var("y"), f" = {v}") for v in y])
YS = YG + 4 * CH + 56
fig.arrow((LX + 1.5 * CW, gl.box.bottom + 8), (LX + 1.5 * CW, YS - 8), label="sum down each column",
          color="gradient")
gs = fig.grid(LX, YS, 1, 3, cell_w=CW, cell_h=CH, font=14, values=lambda i, j: fmt(DB2[0, j]))
fig.text(gs.box.right + 16, YS + 25, "dense2.dbiases", "code")

# -- right: activation1.dinputs, zero where the gate is closed
RX = 600
fig.text(536, 160, "activation1.dinputs", "code")
fig.text(712, 160, rich("handed to ", span("dense1.backward", mono=True)), "note")
gr = fig.grid(RX, YG, 4, 3, cell_w=CW, cell_h=CH, font=14,
              values=lambda i, j: "0" if A1[i, j] == 0 else fmt(A1[i, j]),
              fill=lambda i, j: "negative-soft" if CLOSED[i, j] else None)
gr.col_labels([f"neuron {j + 1}" for j in range(3)])
gr.row_labels([f"sample {i + 1}" for i in range(N)])
fig.note(gr.box, [f"{K} of {CLOSED.size} gates closed: {K} zeros;", f"the other {CLOSED.size - K} entries are copied from"])
fig.text(RX, gr.box.bottom + 68, "dense2.dinputs", "code", color="ink-muted")

fig.caption(rich("The division by ", var("N"), f" = {N} happens once, in the first backward call; the ReLU only copies or zeroes."))
fig.write()
