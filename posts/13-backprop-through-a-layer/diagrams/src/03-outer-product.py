"""Post 13 section 8.2: the weight-gradient matrix is the column dL_dZ times the row of inputs, an outer product.

Run from anywhere:  python posts/13-backprop-through-a-layer/diagrams/src/03-outer-product.py
Writes posts/13-backprop-through-a-layer/diagrams/03-outer-product.svg.
dL_dZ, the inputs and the (3, 4) result come from snippets/layer_backward.py (run here); the result is
recomputed by the line the post shows, dL_dZ.reshape(-1, 1) * inputs, and asserted equal to the snippet's loop,
to np.outer and to the (3, 1) @ (1, 4) product, as the snippet's section 8.2 printout states.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, span, num, text_width  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "layer_backward.py"
with contextlib.redirect_stdout(io.StringIO()) as out:
    S = runpy.run_path(str(SNIP))
OUT = out.getvalue()
dL_dZ = np.array(S["dL_dZ"])
inputs = np.array(S["inputs"])
dL_dW = dL_dZ.reshape(-1, 1) * inputs                 # the line of section 8.2
assert dL_dW.shape == (3, 4)
assert np.array_equal(dL_dW, np.array(S["dL_dW"]))    # the loop of section 5.1
assert np.array_equal(dL_dW, np.outer(dL_dZ, inputs))
assert np.array_equal(dL_dW, dL_dZ.reshape(3, 1) @ inputs.reshape(1, 4))
for line in ("shape of the broadcast result: (3, 4)", "loop equals broadcast: True",
             "broadcast equals np.outer: True", "broadcast equals the matrix product: True"):
    assert line in OUT, line
LINE = "dL_dW = dL_dZ.reshape(-1, 1) * inputs"
assert LINE in " ".join(Path(SNIP).parents[1].joinpath("index.md").read_text(encoding="utf-8").split())
K, J = 0, 3                                           # the traced cell: row 1, column 4
assert num(dL_dZ[K], 1) == "43.2" and inputs[J] == 4 and num(dL_dW[K, J], 1) == "172.8"


def f1(v):
    return num(v, 1)


def code(s):
    return span(s, mono=True)


fig = Figure(
    "03-outer-product", "The weight gradients are an outer product",
    "The column dL_dZ, shape (3, 1), holds 43.2, 43.2 and 43.2 and sits left of the result; the row inputs, "
    "shape (4,), holds 1, 2, 3 and 4 and sits above it, so row k of the column and column j of the row meet at "
    "entry (k, j) of dL_dW, shape (3, 4), whose rows each read 43.2, 86.4, 129.6 and 172.8. The first entry of "
    "the column, the fourth entry of the row and the cell where they meet, 172.8, are outlined, and two short "
    "arrows lead from the column's entry into row 1 and from the row's entry into column 4.Text on the left gives the line dL_dW = dL_dZ.reshape(-1, 1) * inputs, "
    "the shapes (3, 1) times (4,) giving (3, 4), and the two other forms that give the same array, "
    "np.outer(dL_dZ, inputs) and the matrix product of a (3, 1) column and a (1, 4) row. "
    "The caption works the cell out: 43.2 times 4 = 172.8.",
    subtitle="Section 8.2: the line that computes dL_dW, run on the gradients of section 5.")

CW, CH = 96, 56
GX, GY = 536, 232                                 # the result
RX, RY = GX, GY - CH - 40                           # the inputs row, above the result
CX = GX - CW - 40                                   # the dL_dZ column, left of the result

gr = fig.strip(RX, RY, 4, cell_w=CW, cell_h=CH, values=[num(int(v)) for v in inputs],
               fill=lambda j: "input-soft", font=18)
gc = fig.strip(CX, GY, 3, cell_w=CW, cell_h=CH, vertical=True, values=[f1(v) for v in dL_dZ],
               fill=lambda k: "gradient-soft", font=18)
gw = fig.grid(GX, GY, 3, 4, cell_w=CW, cell_h=CH, values=lambda k, j: f1(dL_dW[k, j]),
              fill=lambda k, j: "gradient-soft", font=18)

# names and shapes: the row's over it, the column's over it, the result's under it
fig.text(RX, RY - 16, rich(code("inputs"), "  (4,)"), "label", color="input")
fig.text(CX, GY - 16, rich(code("dL_dZ"), "  (3, 1)"), "label", color="gradient")
fig.text(GX, gw.box.bottom + 28, rich(code("dL_dW"), "  (3, 4)"), "label", color="gradient")

# the traced cell and its two factors
gc.window(K, K, color="gradient")
gr.window(J, J, color="input")
hit = gw.cell(K, J)
gw.outline(K, J, color="gradient", width=1.5)
# short arrows in the gaps only: the column's entry into row 1, the row's entry into column 4
fig.arrow((gc.box.right + 8, hit.cy), (GX - 8, hit.cy))
fig.arrow((hit.cx, gr.box.bottom + 8), (hit.cx, hit.y - 8))

# the left column, spanning the same height as the arrays: the line, the shapes, the other two forms
LX = 40
assert LX + text_width("dL_dZ.reshape(-1, 1) * inputs", 14, mono=True) <= CX - 32
fig.text(LX, RY + 24, "The broadcast", "head")
fig.code_block(LX, RY + 56, ["dL_dZ.reshape(-1, 1) * inputs"])
fig.text(LX, RY + 88, rich("(3, 1) column \u00d7 (4,) row \u2192 (3, 4)"), "note")
fig.text(LX, GY + 72, "Same array", "head")
fig.code_block(LX, GY + 104,["np.outer(dL_dZ, inputs)", "dL_dZ.reshape(3, 1) @", "    inputs.reshape(1, 4)"])

fig.caption(rich("Row 1 of the column meets column 4 of the row: 43.2 \u00b7 4 = 172.8."))
fig.write()
