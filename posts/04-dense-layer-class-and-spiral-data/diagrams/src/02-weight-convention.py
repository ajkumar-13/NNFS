"""Post 04 section 4: the same weights in the old layout (one row per neuron) and the new (one column per neuron).

Run from anywhere:  python posts/04-dense-layer-class-and-spiral-data/diagrams/src/02-weight-convention.py
Writes posts/04-dense-layer-class-and-spiral-data/diagrams/02-weight-convention.svg.
The weights, the shapes and the output row come from snippets/weight_convention.py (the first layer of post 03).
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "weight_convention.py"
out = io.StringIO()
with contextlib.redirect_stdout(out):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OLD, NEW, Y = s["weights_old"], s["weights_new"], s["output_new"]
assert OLD.shape == (3, 4) and NEW.shape == (4, 3) and np.array_equal(NEW, OLD.T)
assert np.array_equal(s["output_old"], Y) and "outputs identical: True" in out.getvalue()
ROW = [f"{v:g}" for v in Y[0]]
assert ROW == ["4.8", "1.21", "2.385"]                      # section 4: "whose first row is 4.8, 1.21, 2.385"


def say(v):
    return f"minus {abs(v):.2f}" if v < 0 else f"{v:.2f}"


fig = Figure(
    "02-weight-convention", "Same weights, two layouts: a row or a column per neuron",
    "Two grids of the same 12 weights, the first layer of post 03. On the left, weights_old of shape (3, 4), one "
    "row per neuron and one column per input, used as np.dot(inputs, weights_old.T) + biases. On the right, "
    "weights_new of shape (4, 3), one row per input and one column per neuron, used as np.dot(inputs, "
    "weights_new) + biases. Neuron 1's weights, " + ", ".join(say(v) for v in OLD[0]) + ", are the first row on "
    "the left and the first column on the right, both shaded and outlined. An arrow labelled transpose joins "
    "the two. Both calls give the same (3, 3) output, whose first row is " + ", ".join(ROW) + ".",
    subtitle="The first layer of post 03, 3 neurons over 4 inputs, as weight_convention.py stores it.")

CW, CH = 64, 48
NEURONS = [f"neuron {k + 1}" for k in range(3)]
INPUTS = [f"input {j + 1}" for j in range(4)]

# -- left: the old layout, centred on the right grid's height
fig.text(40, 128, "Posts 01 to 03: a row per neuron", "head")
g_old = fig.grid(128, 200, 3, 4, cell_w=CW, cell_h=CH, values=OLD.tolist(), decimals=2, font=16,
                 fill=lambda i, j: "weight-soft" if i == 0 else None)
g_old.window(0, 0, 1, 4, "weight")
g_old.row_labels(NEURONS)
g_old.col_labels(INPUTS)

# -- right: the new layout
fig.text(520, 128, "Post 04 on: a column per neuron", "head")
g_new = fig.grid(632, 176, 4, 3, cell_w=CW, cell_h=CH, values=NEW.tolist(), decimals=2, font=16,
                 fill=lambda i, j: "weight-soft" if j == 0 else None)
g_new.window(0, 0, 4, 1, "weight")
g_new.row_labels(INPUTS)
g_new.col_labels(NEURONS)
assert g_old.box.cy == g_new.box.cy

fig.arrow((g_old.box.right + 24, g_old.box.cy), (g_new.box.x - 72, g_new.box.cy), label="transpose")

# -- under each grid: its shape and its forward call, on one baseline across the two panels
for x, name, shape, call in [(40, "weights_old", "(3, 4)", "np.dot(inputs, weights_old.T) + biases"),
                             (520, "weights_new", "(4, 3)", "np.dot(inputs, weights_new) + biases")]:
    fig.text(x, 416, f"{name}.shape == {shape}", "code", color="weight")
    fig.text(x, 444, call, "code")

fig.caption("Both calls return the same (3, 3) output; its first row is " + ", ".join(ROW) + ".")
fig.write()
