"""Post 04 sections 5 and 6: the Layer_Dense class as a blueprint, and the two instances section 6 makes from it.

Run from anywhere:  python posts/04-dense-layer-class-and-spiral-data/diagrams/src/03-dense-layer-class.py
Writes posts/04-dense-layer-class-and-spiral-data/diagrams/03-dense-layer-class.svg.
The code lines are the class of snippets/dense_layer.py; every shape and parameter count is read from the
instances that snippet builds (runpy), and asserted against the section 6.1 table.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "dense_layer.py"
out = io.StringIO()
with contextlib.redirect_stdout(out):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
X, d1, d2 = s["X"], s["dense1"], s["dense2"]
P1, P2 = d1.weights.size + d1.biases.size, d2.weights.size + d2.biases.size
assert X.shape == (300, 2) and d1.weights.shape == (2, 3) and d1.biases.shape == (1, 3)
assert d1.output.shape == (300, 3) and d2.weights.shape == (3, 3) and d2.output.shape == (300, 3)
assert (P1, P2) == (9, 12) and d1.weights is not d2.weights
assert "dense1 and dense2 share a weight array: False" in out.getvalue()

# The class as dense_layer.py writes it, line by line (indent, text); the double space before "=" is dropped
# because SVG collapses runs of spaces.
SRC = SNIPPET.read_text(encoding="utf-8")
CODE = [(0, "class Layer_Dense:"),
        (1, "def __init__(self, n_inputs, n_neurons):"),
        (2, "self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)"),
        (2, "self.biases = np.zeros((1, n_neurons))"),
        (1, "def forward(self, inputs):"),
        (2, "self.output = np.dot(inputs, self.weights) + self.biases")]
for _, line in CODE:
    assert line in SRC.replace("biases  =", "biases ="), line


def shape(a):
    return f"({a.shape[0]}, {a.shape[1]})"


fig = Figure(
    "03-dense-layer-class", "One class, two instances, each with its own weights",
    "At the top, a card holding the six lines of the Layer_Dense class: __init__, which runs once when an "
    "instance is created, sets self.weights to 0.01 times np.random.randn(n_inputs, n_neurons) and self.biases "
    "to zeros of shape (1, n_neurons); forward, which runs on every call, stores np.dot(inputs, self.weights) "
    "plus self.biases in self.output. Below, the two instances of section 6 in a chain: X of shape (300, 2) "
    "goes into dense1 = Layer_Dense(2, 3), with weights (2, 3) and biases (1, 3); dense1.output, (300, 3), goes "
    "into dense2 = Layer_Dense(3, 3), with weights (3, 3) and biases (1, 3); dense2.output is (300, 3). Dashed "
    f"lines join the class to both instances. dense1 holds {P1} parameters and dense2 {P2}.",
    subtitle="The class of section 5 and the two layers section 6 builds from it on the spiral data.",
    height=720)

# -- the blueprint
fig.text(40, 128, "The class: a blueprint", "head")
card = Box(40, 144, 880, 168)
fig.panel(card, None, card=True)
for k, (ind, line) in enumerate(CODE):
    fig.text(64 + 32 * ind, 176 + 24 * k, line, "code")
fig.text(896, 200, "runs once, when an instance is made", "note", anchor="end")
fig.text(896, 272, "runs on every call", "note", anchor="end")

# -- the two instances, left to right along the data
fig.text(40, 368, "Two instances", "head")
CT, CH = 416, 192                                    # instance cards: top and height
TOP, BH = 432, 160                                   # the data bands: 160 high, 16 per column
CY = TOP + BH // 2
assert CY == CT + CH // 2


def band(x, cols, color, name, arr):
    b = fig.grid(x, TOP, 1, cols, cell_w=16, cell_h=BH, fill=lambda i, j: color + "-soft")
    fig.text(b.box.cx, CT - 16, name, "code", anchor="middle", color=color)
    fig.text(b.box.cx, TOP + BH + 24, shape(arr), "tick", anchor="middle")
    return b.box


def instance(x, name, layer, call, fwd):
    c = Box(x, CT, 280, CH)
    fig.panel(c, None, card=True)
    fig.text(x + 16, CT + 28, f"{name} = {call}", "code")
    fig.text(x + 16, CT + 52, fwd, "code")
    n_in = layer.weights.shape[0]
    w = fig.grid(x + 16, CT + 96, n_in, 3, cell=24, fill=lambda i, j: "weight-soft")
    b = fig.grid(x + 152, CT + 96, 1, 3, cell=24, fill=lambda i, j: "weight-soft")
    for g, attr, arr in [(w, ".weights", layer.weights), (b, ".biases", layer.biases)]:
        fig.text(g.box.x, CT + 84, attr, "code", color="weight")
        fig.text(g.box.x + 80 if attr == ".weights" else g.box.x + 72, CT + 84, shape(arr), "tick")
    return c


bx = band(48, 2, "input", "X", X)
c1 = instance(112, "dense1", d1, "Layer_Dense(2, 3)", "dense1.forward(X)")
o1 = band(424, 3, "output", "dense1.output", d1.output)
c2 = instance(504, "dense2", d2, "Layer_Dense(3, 3)", "dense2.forward(dense1.output)")
o2 = band(816, 3, "output", "dense2.output", d2.output)
for a, b in [(bx, c1), (c1, o1), (o1, c2), (c2, o2)]:
    fig.arrow((a.right + 4, CY), (b.x, CY))

# instances made from the class
for c in (c1, c2):
    fig.leader((c.cx, card.bottom), (c.cx, c.y))

fig.caption(f"dense1 holds {P1} parameters and dense2 {P2}; only dense1.output passes from one to the other.")
fig.write()
