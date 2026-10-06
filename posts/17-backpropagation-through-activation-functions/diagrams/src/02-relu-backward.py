"""Post 17 section 2: Activation_ReLU as a class card, the worked example going forward and coming back.

Run from anywhere:  python posts/17-backpropagation-through-activation-functions/diagrams/src/02-relu-backward.py
Writes posts/17-backpropagation-through-activation-functions/diagrams/02-relu-backward.svg.
snippets/relu_backward.py is run here (runpy). Its Activation_ReLU is called on the worked example of section 2.1,
Z = [[1, -2, 3]] and dvalues = [[5, 6, 7]], and every value drawn is asserted against the lines the snippet prints.
The code lines in the card are the class as the post lists it in section 2.2, without its comments; they are
asserted to be lines of the snippet's source.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, arr, num  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "relu_backward.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    S = runpy.run_path(str(SNIP), run_name="snippet")
OUT = buf.getvalue()
SRC = SNIP.read_text(encoding="utf-8")

Z = np.array([[1.0, -2.0, 3.0]])
DV = np.array([[5.0, 6.0, 7.0]])
DV_BEFORE = DV.copy()
relu = S["Activation_ReLU"]()
relu.forward(Z)
relu.backward(DV)
A, DX = relu.output, relu.dinputs
assert np.array_equal(DV, DV_BEFORE)                            # the caller's array is left alone
assert A.tolist() == [[1.0, 0.0, 3.0]] and DX.tolist() == [[5.0, 0.0, 7.0]]
for line in ("inputs  Z       : [[ 1. -2.  3.]]", "output  A       : [[1. 0. 3.]]",
             "dvalues dL/dA   : [[5. 6. 7.]]", "dinputs dL/dZ   : [[5. 0. 7.]]",
             "dvalues after the call, unchanged: [[5. 6. 7.]]"):
    assert line in OUT, line
N, M_ = Z.shape
assert (N, M_) == (1, 3)

CODE = ["def forward(self, inputs):",
        "    self.inputs = inputs",
        "    self.output = np.maximum(0, inputs)",
        "",
        "def backward(self, dvalues):",
        "    self.dinputs = dvalues.copy()",
        "    self.dinputs[self.inputs <= 0] = 0"]
for c in CODE:
    if c:
        assert any(l.strip().startswith(c.strip()) for l in SRC.splitlines()), c


def d(top, bot):
    return rich("\u2202", top, "/\u2202", bot)


def vals(a):
    return [num(int(v)) for v in a[0]]


fig = Figure(
    "02-relu-backward", "ReLU's backward is a masked copy of dvalues",
    "A card for the class Activation_ReLU with its forward method above its backward method; the line self.inputs = "
    "inputs carries a tag saying cached. Forward, left to right in the default arrow colour: inputs Z, shape (1, 3), "
    "holding 1, minus 2, 3, enter the card and the output A, shape (1, 3), holding 1, 0, 3, leaves it. "
    "Backward, right to left in purple: dvalues dL/dA, shape (1, 3), holding 5, 6, 7, enters the card and dinputs "
    "dL/dZ, shape (1, 3), holding 5, 0, 7, leaves it; the 0 is outlined under the input minus 2, which is not "
    "positive. A note says the caller's dvalues still holds 5, 6, 7 after the call. A key shows a grey arrow for "
    "forward, a purple arrow for backward, and the cached tag: stored by forward, read by backward.",
    subtitle=rich("The worked example of section 2.1: one sample, so every array has shape (", var("N"), ", ",
                  var("m"), ") = (1, 3)."),
    data_w=True)

# -- the card, centred
CX, CY = 304, 200
body = fig.card(CX, CY, None, None, heading="Activation_ReLU", lines=CODE, style="code", fit=True)
card = Box(CX, CY, body.w + 32, body.bottom + 16 - CY)
base = body.y + 16                                   # first code baseline
YF = base + 20                                       # the forward row: the cached line's middle
YB = base + 5 * 24 - 4                              # the backward row: the copy line's middle
# the cached tag at the end of "self.inputs = inputs"
TAG = Box(560, base + 24 - 18, 64, 24)
fig.fill(TAG, "output-soft", radius=4)
fig.outline(TAG, "output-line", radius=4)
fig.text(TAG.cx, TAG.y + 17, "cached", "note", anchor="middle", color="output")

C, CW = 40, 56
XL, XR = 40, card.right + 96


def strip(x, yc, values, fill, head, side):
    g = fig.strip(x, yc - C / 2, 3, cell_w=CW, cell_h=C, values=values, font=16, fill=fill)
    y = g.box.y - 16 if side == "above" else g.box.bottom + 28
    fig.text(g.box.cx, y, head, "label", anchor="middle")
    return g


gz = strip(XL, YF, vals(Z), lambda i: "negative-soft" if Z[0, i] <= 0 else "input-soft",
           rich("inputs ", arr("Z"), "  (1, 3)"), "above")
ga = strip(XR, YF, vals(A), lambda i: "output-soft" if A[0, i] > 0 else None,
           rich("output ", arr("A"), "  (1, 3)"), "above")
gv = strip(XR, YB, vals(DV), lambda i: "gradient-soft", rich("dvalues ", d(var("L"), arr("A")), "  (1, 3)"), "below")
gd = strip(XL, YB, vals(DX), lambda i: "gradient-soft" if DX[0, i] else None,
           rich("dinputs ", d(var("L"), arr("Z")), "  (1, 3)"), "below")
gd.window(1, 1, color="negative")

fig.arrow((gz.box.right, YF), (card.x, YF))
fig.arrow((card.right, YF), (ga.box.x, YF))
fig.arrow((gv.box.x, YB), (card.right, YB), color="gradient")
fig.arrow((card.x, YB), (gd.box.right, YB), color="gradient")

fig.text(gd.box.x, gd.box.bottom + 52, rich("0 where the cached input was not positive"), "note")
fig.text(gv.box.right, gv.box.bottom + 52, "unchanged after the call", "note", anchor="end")

# the key
KY = 160                                  # the key, between the subtitle and the strips
fig.arrow((40, KY - 5), (80, KY - 5))
fig.text(88, KY, "forward pass", "note")
fig.arrow((224, KY - 5), (184, KY - 5), color="gradient")
fig.text(232, KY, "backward pass", "note")
KEY_TAG = Box(360, KY - 17, 64, 24)
fig.fill(KEY_TAG, "output-soft", radius=4)
fig.outline(KEY_TAG, "output-line", radius=4)
fig.text(KEY_TAG.cx, KY, "cached", "note", anchor="middle", color="output")
fig.text(KEY_TAG.right + 8, KY, "stored by forward, read by backward", "note")

fig.caption(rich("The mask passes 5 and 7 at full size and stops the 6 at the input ", num(-2), "."))
fig.write()
