"""Post 16, sections 1 and 2: Layer_Dense with forward above backward, what comes in and what goes out.

Run from anywhere:  python posts/16-coding-backpropagation/diagrams/src/01-dense-backward-class.py
Writes posts/16-coding-backpropagation/diagrams/01-dense-backward-class.svg.
snippets/backward_classes.py is run here (runpy): the code lines on the card are asserted to be lines of the
snippet's Layer_Dense (comments dropped), and the shape rule the figure states (every gradient has the shape of
the array it belongs to) is asserted on the snippet's section 3 layer and section 6 chain, so the symbolic shapes
(N, n), (N, m), (n, m) and (1, m) are checked against two concrete batches.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, span  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "backward_classes.py"
with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")

# -- the card's code: the snippet's Layer_Dense without comments, constructor left out
SRC = [l.split("#")[0].rstrip() for l in SNIPPET.read_text(encoding="utf-8").splitlines()]
FWD = ["    def forward(self, inputs):",
       "        self.inputs = inputs",
       "        self.output = np.dot(inputs, self.weights) + self.biases"]
BWD = ["    def backward(self, dvalues):",
       "        self.dweights = np.dot(self.inputs.T, dvalues)",
       "        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)",
       "        self.dinputs = np.dot(dvalues, self.weights.T)"]
start = SRC.index("class Layer_Dense:")
body = SRC[start:SRC.index("class Activation_ReLU:")]
for line in FWD + BWD:
    assert line in body, line

# -- the shape rule on two concrete layers: N samples, n inputs, m neurons
for layer, (N, n, m) in ((s["layer"], (3, 4, 3)), (s["dense1"], (3, 4, 3)), (s["dense2"], (3, 3, 2))):
    assert layer.inputs.shape == layer.dinputs.shape == (N, n)
    assert layer.output.shape == (N, m)
    assert layer.weights.shape == layer.dweights.shape == (n, m)
    assert layer.biases.shape == layer.dbiases.shape == (1, m)
assert s["dloss"].shape == s["dense2"].output.shape                 # dvalues has the shape of output


def shp(a, b):
    """(N, n) with italic letters and an upright 1."""
    f = lambda t: var(t) if isinstance(t, str) else str(t)
    return rich("(", f(a), ", ", f(b), ")")


def code(t, color=None):
    return span(t, mono=True, color=color)


G = "gradient"
fig = Figure(
    "01-dense-backward-class", "One forward call, one backward call, three gradients out",
    "A card with the two methods of Layer_Dense, forward above backward. forward stores self.inputs = inputs, "
    "tagged cached because backward reads it, and computes self.output = np.dot(inputs, self.weights) + "
    "self.biases. backward takes dvalues and computes self.dweights = np.dot(self.inputs.T, dvalues), "
    "self.dbiases = np.sum(dvalues, axis=0, keepdims=True) and self.dinputs = np.dot(dvalues, self.weights.T). "
    "Grey arrows, the forward pass, run left to right: inputs of shape (N, n) in, output of shape (N, m) out. "
    "Purple arrows, the backward pass, run right to left: dvalues of shape (N, m) in from the next layer, dinputs "
    "of shape (N, n) out to the previous layer, which reads it as its dvalues. dweights, shape (n, m), and "
    "dbiases, shape (1, m), stay on the layer for the optimiser. Each gradient has the shape of its array.",
    subtitle=rich("The class of section 2, for a batch of ", var("N"), " samples, ", var("n"), " inputs and ",
                  var("m"), " neurons."),
    data_w=True)

# -- the card: forward block, a blank line, backward block
CX, CY, CW = 216, 104, 536
LINES = ["class Layer_Dense:"] + FWD + [""] + BWD
LINES[6] = rich("        ", code("self.dweights", G), code(" = np.dot(self.inputs.T, dvalues)"))
LINES[7] = rich("        ", code("self.dbiases", G), code(" = np.sum(dvalues, axis=0, keepdims=True)"))
LINES[8] = rich("        ", code("self.dinputs", G), code(" = np.dot(dvalues, self.weights.T)"))
card = fig.card(CX, CY, CW, None, lines=LINES, style="code", fit=True)
CARD = Box(CX, CY, CW, card.bottom + 16 - CY)
BASE = [card.y + 16 + 24 * k for k in range(len(LINES))]          # baselines of the code lines


def mid(k0, k1):
    """The vertical middle of code lines k0 to k1 (a 14 px line sits 5 above its baseline)."""
    return (BASE[k0] + BASE[k1]) / 2 - 4


# the cached tag beside self.inputs = inputs
TX, TY = card.x + 32 + 176, BASE[2] - 16
tag = Box(TX, TY, 64, 24)
fig.fill(tag, "output-soft", radius=4)
fig.outline(tag, "output-line", radius=4)
fig.text(tag.cx, BASE[2] - 1, "cached", "tick", anchor="middle", color="output")

# -- forward, left to right, beside the forward block; backward, right to left, beside the backward block
YF, YB = mid(1, 3), mid(5, 8)
LX0, LX1 = 56, CX - 8
RX0, RX1 = CARD.right + 8, 904


def arrow_label(x0, x1, y, name, a, b, color=None):
    cx = (x0 + x1) / 2
    fig.text(cx, y - 12, code(name, color), "code", anchor="middle")
    fig.text(cx, y + 24, shp(a, b), "label", anchor="middle")


fig.arrow((LX0, YF), (LX1, YF))
arrow_label(LX0, LX1, YF, "inputs", "N", "n")
fig.arrow((RX0, YF), (RX1, YF))
arrow_label(RX0, RX1, YF, "output", "N", "m")
fig.arrow((RX1, YB), (RX0, YB), color=G)
arrow_label(RX0, RX1, YB, "dvalues", "N", "m", G)
fig.arrow((LX1, YB), (LX0, YB), color=G)
arrow_label(LX0, LX1, YB, "dinputs", "N", "n", G)
fig.text(LX0, YB + 52, "to the previous layer,", "note")
fig.text(LX0, YB + 72, rich("as its ", code("dvalues")), "note")
fig.text(RX1, YB + 52, "from the next layer", "note", anchor="end")

# -- what stays: dweights and dbiases, a short step down to the optimiser's card
YS = CARD.bottom + 48
assert YS + 76 <= 476                                              # the last row ends inside the content zone
fig.arrow((CARD.cx, CARD.bottom + 8), (CARD.cx, YS - 8), color=G)
fig.text(CARD.cx + 12, (CARD.bottom + YS) / 2 + 5, "stays on the layer, for the optimiser", "note")
OW = 304
ob = fig.card(CARD.cx - OW / 2, YS, OW, None, lines=["", ""], fit=True)
for k, (name, a, b, of) in enumerate((("dweights", "n", "m", "weights"), ("dbiases", 1, "m", "biases"))):
    y = ob.y + 16 + 24 * k
    fig.text(ob.x + 8, y, code(name, G), "code")
    fig.text(ob.x + 104, y, shp(a, b), "label")
    fig.text(ob.right - 8, y, code(of), "code", anchor="end", color="ink-muted")
    fig.text(ob.right - 12 - 8.5 * len(of), y, "as", "note", anchor="end")

# -- the key, on the same row: the two arrow colours left, the tag right
KY = ob.y + 16                                                     # the card's first baseline
fig.legend(40, KY, [dict(color="arrow", label="forward: arrays", mark="line"),
                    dict(color=G, label="backward: gradients", mark="line")], direction="column")
key = Box(736, KY - 16, 64, 24)                                 # the tag again, as the key's third item
fig.fill(key, "output-soft", radius=4)
fig.outline(key, "output-line", radius=4)
fig.text(key.cx, KY - 1, "cached", "tick", anchor="middle", color="output")
fig.text(key.right + 12, KY, "kept by forward", "label")
fig.text(key.right + 12, KY + 24, "for backward", "label")

fig.caption("Every gradient has the shape of the array it belongs to.")
fig.write()
