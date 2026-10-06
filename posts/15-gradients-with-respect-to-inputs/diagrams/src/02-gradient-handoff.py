"""Post 15, section 1 (and sections 6 and 7.3): two gradients stay in a layer, the input gradient travels on.

Run from anywhere:  python posts/15-gradients-with-respect-to-inputs/diagrams/src/02-gradient-handoff.py
Writes posts/15-gradients-with-respect-to-inputs/diagrams/02-gradient-handoff.svg.
snippets/batch_and_handoff.py is run here (runpy): every shape in the figure is read from the arrays of its
section 7.3 (the two-layer network on the batch of three samples) and asserted against the line it prints. The two
bias gradients, which the snippet's section 7.3 does not compute, are computed here with the bias line of section 6.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, arr, span, sup, var, ODOT  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "batch_and_handoff.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

X, W, W2 = s["X"], s["W"], s["W2"]
dvalues2, dweights2, dinputs2 = s["dvalues2"], s["dweights2"], s["dinputs2"]
dvalues1, dweights1, dinputs1 = s["dvalues1"], s["dweights1"], s["dinputs1"]
dbiases2 = np.sum(dvalues2, axis=0, keepdims=True)          # the bias line of section 6, for each layer
dbiases1 = np.sum(dvalues1, axis=0, keepdims=True)
SH = {k: v.shape for k, v in dict(dvalues2=dvalues2, dweights2=dweights2, dbiases2=dbiases2, dinputs2=dinputs2,
                                     dvalues1=dvalues1, dweights1=dweights1, dbiases1=dbiases1,
                                     dinputs1=dinputs1).items()}
assert SH == dict(dvalues2=(3, 2), dweights2=(3, 2), dbiases2=(1, 2), dinputs2=(3, 3),
                  dvalues1=(3, 3), dweights1=(4, 3), dbiases1=(1, 3), dinputs1=(3, 4))
assert "layer 2 receives dvalues of shape (3, 2) and passes back dinputs of shape (3, 3)" in OUT
assert "layer 1 input gradient, shape (3, 4)" in OUT
assert X.shape == (3, 4) and W.shape == (4, 3) and W2.shape == (3, 2)      # 4 -> 3 -> 2, N = 3
assert dweights1.shape == W.shape and dweights2.shape == W2.shape and dinputs1.shape == X.shape


def shape(t):
    return f"({t[0]}, {t[1]})"


def code(t, color="gradient"):
    return span(t, color=color, mono=True)


fig = Figure(
    "02-gradient-handoff", "Two gradients stay in a layer, one travels on",
    "The backward pass of the two-layer network of section 7.3 on a batch of three samples, drawn right to left. "
    "The loss hands dvalues2 of shape (3, 2) to layer 2, which has 3 inputs and 2 neurons. Layer 2 keeps dweights2 "
    "(3, 2) and dbiases2 (1, 2) for the optimiser and passes dinputs2 (3, 3) back. Through layer 1's ReLU gate, "
    "multiplied element by element by Z1 > 0, it becomes dvalues1 (3, 3). Layer 1, with 4 inputs and 3 neurons, "
    "keeps dweights1 (4, 3) and dbiases1 (1, 3) for the optimiser and passes dinputs1 (3, 4) back to the data X, "
    "where nothing reads it.",
    subtitle="The two-layer network of section 7.3 on the batch of three samples. The backward pass runs right to left.")

YC, CH = 136, 96                     # the chain's cards: top and height
MID = YC + CH // 2
DATA, L1, GATE, L2, LOSS = (40, 72), (200, 144), (432, 80), (600, 144), (832, 88)


def box(xw, heading, line, color=None):
    x, w = xw
    fig.card(x, YC, w, CH, color=color)
    fig.text(x + w / 2, YC + 40, heading, "head", anchor="middle")
    fig.text(x + w / 2, YC + 68, line, "note", anchor="middle")


box(DATA, rich("Data ", arr("X")), shape(X.shape), color="input")
box(L1, "Layer 1", "4 inputs, 3 neurons")
box(GATE, "ReLU", rich(ODOT, " (", arr("Z", sub="1"), " > 0)"))
box(L2, "Layer 2", "3 inputs, 2 neurons")
box(LOSS, "Loss", rich("mean of ", var("ŷ"), " ", sup("", "2")))


def hop(src, dst, name, shp, emph=False):
    """A backward arrow from the left edge of src to the right edge of dst, the gradient named over it."""
    x0, x1 = src[0] - 8, dst[0] + dst[1] + 8
    fig.arrow((x0, MID), (x1, MID))
    cx = (x0 + x1) / 2
    fig.text(cx, MID - 12, code(name), "code", anchor="middle")
    fig.text(cx, MID + 24, shape(shp), "note", anchor="middle")


hop(LOSS, L2, "dvalues2", SH["dvalues2"])
hop(L2, GATE, "dinputs2", SH["dinputs2"])
hop(GATE, L1, "dvalues1", SH["dvalues1"])
hop(L1, DATA, "dinputs1", SH["dinputs1"])

# -- what stays: each layer's weight and bias gradients go down to the optimiser
YS = 336
for (x, w), k in ((L1, "1"), (L2, "2")):
    cx = x + w / 2
    fig.arrow((cx, YC + CH + 8), (cx, YS - 8))
    fig.text(cx + 12, (YC + CH + YS) / 2 + 4, "stays", "note")
    cw = 208
    fig.card(cx - cw / 2, YS, cw, 112)
    for i, name in enumerate((f"dweights{k}", f"dbiases{k}")):
        fig.text(cx - cw / 2 + 24, YS + 40 + 24 * i, code(name), "code")
        fig.text(cx + cw / 2 - 24, YS + 40 + 24 * i, shape(SH[name]), "label", anchor="end")
    fig.text(cx - cw / 2 + 24, YS + 92, "to the optimiser", "note")

# -- the one that travels: the hand-off, and the end of the line
fig.note(Box(GATE[0] - 24, YC + CH, GATE[1] + 48, 0),
         [rich("layer 2's ", code("dinputs2")), rich("is layer 1's ", code("dvalues1"))])
fig.note(Box(DATA[0], YC + CH, DATA[1], 0), ["nothing", "reads it"])

fig.caption("Only the input gradient leaves a layer; it is the upstream gradient of the layer before.")
fig.write()
