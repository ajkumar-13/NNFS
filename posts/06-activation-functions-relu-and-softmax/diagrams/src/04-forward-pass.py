"""Post 06 section 5: the forward pass Dense -> ReLU -> Dense -> Softmax on the spiral batch.

Run from anywhere:  python posts/06-activation-functions-relu-and-softmax/diagrams/src/04-forward-pass.py
Writes posts/06-activation-functions-relu-and-softmax/diagrams/04-forward-pass.svg.
snippets/forward_pass.py is run (runpy). Every shape is read from the arrays it builds (X, each layer's
weights and each object's output) and asserted against the shape table the snippet prints; the ReLU counts
and the row sums are computed from the same arrays and asserted against its printout.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, arr, span, num  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "forward_pass.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    S = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
X, d1, a1, d2, a2 = S["X"], S["dense1"], S["activation1"], S["dense2"], S["activation2"]

ARRAYS = [("X", X), ("dense1.output", d1.output), ("activation1.output", a1.output),
          ("dense2.output", d2.output), ("activation2.output", a2.output)]
for name, a in ARRAYS:                                     # the snippet's shape table, line by line
    assert f"{name:<19}{str(a.shape):<10}{a.dtype}" in OUT, name
    assert a.dtype == np.float32
assert [a.shape for _, a in ARRAYS] == [(300, 2)] + [(300, 3)] * 4
assert d1.weights.shape == (2, 3) and d2.weights.shape == (3, 3)

NEG, ZERO, POS = (int(np.sum(d1.output < 0)), int(np.sum(d1.output == 0)), int(np.sum(d1.output > 0)))
ZEROS_OUT, SIZE = int(np.sum(a1.output == 0)), a1.output.size
assert (NEG, ZERO, POS, ZEROS_OUT, SIZE) == (447, 9, 444, 456, 900) and ZEROS_OUT == NEG + ZERO
assert f"negative {NEG}  exactly zero {ZERO}  positive {POS}" in OUT
assert f"equal to zero: {ZEROS_OUT} of {SIZE}" in OUT
SUMS = np.sum(a2.output, axis=1)
assert np.allclose(SUMS, 1.0, atol=1e-6) and "all rows sum to 1 within 1e-6: True" in OUT
N = X.shape[0]


def shape(a):
    return rich("(", num(a.shape[0]), ", ", num(a.shape[1]), ")")


def mono(s):
    return rich(span(s, mono=True))


fig = Figure(
    "04-forward-pass", "Four objects in a row, each reading the last one's output",
    f"A left-to-right pipeline over the spiral batch of {N} samples. Five array nodes: X, shape (300, 2); "
    "dense1.output, Z1, (300, 3); activation1.output, A1, (300, 3); dense2.output, Z2, the logits, (300, 3); and "
    "activation2.output, y hat, the probabilities, (300, 3). Under each arrow between two nodes, a card for the "
    "object that makes the next array: dense1 = Layer_Dense(2, 3), Z1 = X W1 + b1 with W1 of shape (2, 3); "
    f"activation1 = Activation_ReLU, A1 = max(0, Z1), which sets {ZEROS_OUT} of the {SIZE} entries to 0 "
    f"({NEG} negative and {ZERO} exactly zero); dense2 = Layer_Dense(3, 3), Z2 = A1 W2 + b2 with W2 of shape "
    "(3, 3); activation2 = Activation_Softmax, y hat = softmax(Z2) per row, every row summing to 1.",
    subtitle=rich("The spiral batch of ", num(N), " samples; every array is float32. Shapes as the snippet prints them."))

R = 24
CY = 200                                         # node centres
CX = [112, 296, 480, 664, 848]                   # 184 apart
NODES = [(arr("X"), "input"), (arr("Z", sub="1"), None), (arr("A", sub="1"), None), (arr("Z", sub="2"), None),
         (arr("\u0177"), "output")]
CARD_TOP, CARD_H, CARD_W = 264, 184, 176
HS = "\u2009"                                    # a thin space between the factors of a product

# -- the arrays: name and shape above each node, the node, an arrow to the next
for (name, a), cx, (label, color) in zip(ARRAYS, CX, NODES):
    fig.text(cx, 136, name, "code13", anchor="middle")
    fig.text(cx, 160, shape(a), "tick", anchor="middle")
for a, b in zip(CX, CX[1:]):
    fig.arrow((a + R + 4, CY), (b - R - 4, CY))
for cx, (label, color) in zip(CX, NODES):
    fig.node(cx, CY, r=R, label=label, color=color)

# -- the objects: one card under each arrow
CARDS = [
    ("dense1", "Layer_Dense(2, 3)",
     rich(arr("Z", sub="1"), " = ", arr("X"), HS, arr("W", sub="1"), " + ", arr("b", sub="1")),
     [rich(arr("W", sub="1"), ": ", shape(d1.weights)), f"{d1.weights.shape[1]} hidden neurons"]),
    ("activation1", "Activation_ReLU",
     rich(arr("A", sub="1"), " = max(0, ", arr("Z", sub="1"), ")"),
     [rich(num(ZEROS_OUT), " of ", num(SIZE), " set to 0"), f"{NEG} negative, {ZERO} exactly 0"]),
    ("dense2", "Layer_Dense(3, 3)",
     rich(arr("Z", sub="2"), " = ", arr("A", sub="1"), HS, arr("W", sub="2"), " + ", arr("b", sub="2")),
     [rich(arr("W", sub="2"), ": ", shape(d2.weights)), "the logits"]),
    ("activation2", "Activation_Softmax",
     rich(arr("\u0177"), " = softmax(", arr("Z", sub="2"), ")"),
     ["the probabilities", "each row sums to 1"]),
]
for (name, cls, formula, facts), a, b in zip(CARDS, CX, CX[1:]):
    mid = (a + b) / 2
    x0 = mid - CARD_W / 2
    fig.leader((mid, CY + 8), (mid, CARD_TOP))
    body = fig.card(x0, CARD_TOP, CARD_W, CARD_H, heading=mono(name))
    fig.text(body.x, body.y + 16, cls, "code13")
    fig.text(body.x, body.y + 52, formula, "label")
    for k, s in enumerate(facts):
        fig.text(body.x, body.y + 88 + 20 * k, s, "note")

fig.caption("Only the dense layers change a shape, and only its second number.")
fig.write()
