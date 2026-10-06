"""Post 22, sections 1.1 and 2: the Optimizer_SGD class and its one call on the numbers checked by hand.

Run from anywhere:  python posts/22-gradient-descent-optimiser/diagrams/src/02-update-params.py
Writes posts/22-gradient-descent-optimiser/diagrams/02-update-params.svg.
The card's code is read from snippets/optimizer_sgd.py (every line asserted to be the class's, in order).
snippets/one_update.py is run here (runpy, about 1 s): the arrays after the update are its own, the arrays before
are its literals (asserted in its source), and the three lines it prints for the first update are asserted.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, span, MINUS, CDOT  # noqa: E402

SNIPPETS = Path(__file__).resolve().parents[2] / "snippets"
sys.path.insert(0, str(SNIPPETS))

# -- the class as the snippet writes it
CLASS = ["class Optimizer_SGD:",
         "    def __init__(self, learning_rate=1.0):",
         "        self.learning_rate = learning_rate",
         "",
         "    def update_params(self, layer):",
         "        layer.weights -= self.learning_rate * layer.dweights",
         "        layer.biases -= self.learning_rate * layer.dbiases"]
SRC = (SNIPPETS / "optimizer_sgd.py").read_text(encoding="utf-8")
cls = SRC[SRC.index("class Optimizer_SGD:"):SRC.index('if __name__ == "__main__":')]
pos = 0
for line in CLASS:
    if line:
        pos = cls.index(line, pos)
assert cls.count("\n    def ") == 2 and cls.count("-=") == 2       # nothing else in the class

# -- the hand-checked update
ONE = SNIPPETS / "one_update.py"
ONE_SRC = ONE.read_text(encoding="utf-8")
for lit in ("weights=np.array([[0.5, -1.0], [2.0, 0.0]])", "biases=np.array([[0.1, -0.2]])",
            "dweights=np.array([[1.0, -2.0], [0.0, 4.0]])", "dbiases=np.array([[-1.0, 0.5]])",
            "Optimizer_SGD(learning_rate=0.1)"):
    assert lit in ONE_SRC, lit
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(ONE), run_name="snippet")
OUT = buf.getvalue()
layer = s["layer"]
LR = 0.1
W0, B0 = np.array([[0.5, -1.0], [2.0, 0.0]]), np.array([[0.1, -0.2]])
DW, DB = layer.dweights, layer.dbiases
W1, B1 = layer.weights, layer.biases
assert np.allclose(W1, W0 - LR * DW) and np.allclose(B1, B0 - LR * DB)
assert "weights [[0.4, -0.8], [2.0, -0.4]]" in OUT
assert "biases  [[0.2, -0.25]]" in OUT
assert "same array object as before the update: True" in OUT
assert f"{W0[0, 0] - LR * DW[0, 0]:.1f}" == "0.4" and DW[1, 0] == 0 and W1[1, 0] == 2.0


def code(t, color=None):
    return span(t, mono=True, color=color)


P, G = "weight", "gradient"
fig = Figure(
    "02-update-params", "One call moves every entry against its own gradient",
    "A card with the class Optimizer_SGD: __init__ stores self.learning_rate, and update_params(layer) runs "
    "layer.weights -= self.learning_rate * layer.dweights and layer.biases -= self.learning_rate * "
    "layer.dbiases. The parameters are orange, the gradients purple. A key says the optimiser reads dweights "
    "and dbiases and writes weights and biases in place. Below, the call with learning rate 0.1 on the numbers "
    "of section 2: weights 0.5, minus 1.0, 2.0, 0.0 minus 0.1 times dweights 1.0, minus 2.0, 0.0, 4.0 gives "
    "0.4, minus 0.8, 2.0, minus 0.4; biases 0.10, minus 0.20 minus 0.1 times dbiases minus 1.00, 0.50 gives "
    "0.20, minus 0.25. Notes beside the rows work two entries, 0.5 minus 0.1 times 1.0 = 0.4 and minus 1.0 minus "
    "0.1 times minus 2.0 = minus 0.8, say that a gradient of 0 leaves 2.0 where it was, and that -= writes into "
    "the arrays the layer holds, the same objects as before the call.",
    subtitle="The class of section 2, and one call on numbers that can be checked by hand.",
    height=720, data_w=True)

# -- the card: the class, parameters orange and gradients purple
LINES = list(CLASS)
LINES[5] = rich("        ", code("layer.weights", P), code(" -= self.learning_rate * "), code("layer.dweights", G))
LINES[6] = rich("        ", code("layer.biases", P), code(" -= self.learning_rate * "), code("layer.dbiases", G))
card = fig.card(40, 104, 536, None, lines=LINES, style="code", fit=True)
CARD_BOTTOM = card.bottom + 16

# the key, right of the card, one item per role
KX, KY = 616, card.y + 16
fig.text(KX, KY, "The optimiser", "head")
fig.legend(KX, KY + 36, [dict(color=P, label="writes the parameters in place"),
                         dict(color=G, label="reads the gradients")], direction="column", step=28)
fig.text(KX, KY + 116, "and stores one number,", "note")
fig.text(KX, KY + 140, code("self.learning_rate"), "code")

# -- the worked call: before, minus 0.1 times the gradient, equals after
YH = -(-(CARD_BOTTOM + 48) // 8) * 8
fig.text(40, YH, rich(code("optimizer.update_params(layer)"), "  with learning rate 0.1"), "head")
CW, CH = 64, 48
XG = [40, 296, 488]                         # the three arrays of a row: before, gradient, after
YW = YH + 64                                # the weights row (2 x 2)
YB = YW + 2 * CH + 56                       # the biases row (1 x 2)
XN = 664                                    # the notes, right of the rows


def row(y, before, grad, after, names, decimals):
    n = before.shape[0]
    for x, vals, fill, name, when in zip(XG, (before, grad, after), (P, G, P), names, ("before", None, "after")):
        fig.grid(x, y, n, 2, cell_w=CW, cell_h=CH, values=vals.tolist(), decimals=decimals, font=18,
                 fill=lambda i, j, f=fill: f + "-soft")
        fig.text(x, y - 12, code(name, fill), "code")
        if when:
            fig.text(x + 2 * CW, y - 12, when, "note", anchor="end")
    cy = y + n * CH / 2
    fig.op(XG[0] + 2 * CW + 32, cy, "-")
    fig.text(XG[0] + 2 * CW + 72, cy + 6, "0.1", "math", anchor="middle")
    fig.op(XG[0] + 2 * CW + 100, cy, "·")
    fig.op(XG[1] + 2 * CW + 32, cy, "=")
    return cy


cw = row(YW, W0, DW, W1, ["weights", "dweights", "weights"], 1)
cb = row(YB, B0, DB, B1, ["biases", "dbiases", "biases"], 2)

# the notes beside the rows: two entries worked, the zero gradient, the in-place write
fig.text(XN, YW + 29, rich("0.5 ", MINUS, " 0.1 ", CDOT, " 1.0 = 0.4"), "label")
fig.text(XN, YW + 53, rich(MINUS, "1.0 ", MINUS, " 0.1 ", CDOT, " (", MINUS, "2.0) = ", MINUS, "0.8"), "label")
fig.text(XN, YW + 85, "a gradient of 0 leaves 2.0", "note")
fig.text(XN, YW + 105, "where it was", "note")
fig.text(XN, YB + 4, rich(code("-="), " writes into the arrays"), "note")
fig.text(XN, YB + 24, "the layer holds: the same", "note")
fig.text(XN, YB + 44, "objects as before the call", "note")

fig.caption("Each entry moves by the learning rate times its own gradient, in the opposite direction.")
fig.write()
