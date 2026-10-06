"""Post 23, sections 4 and 5: the class Optimizer_SGD with decay, and the calls the training loop makes to it.

Run from anywhere:  python posts/23-learning-rate-decay/diagrams/src/03-optimizer-class.py
Writes posts/23-learning-rate-decay/diagrams/03-optimizer-class.svg.

The card's code is the class of snippets/decay_schedule.py, line for line (asserted, and asserted to be the listing of
section 4 in index.md and the class of train_with_decay.py). The five calls on the left are the constructor line and
the four update lines of the loop of section 5 (asserted against index.md and train_with_decay.py). The symbols beside
the constructor's attributes are those of section 2 (alpha_0, alpha_t, d, t); the class is run here to check that
pre_update_params writes alpha_0 / (1 + d t) and that post_update_params alone moves the counter.
"""
import runpy
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, isub, span  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIP = POST / "snippets"
INDEX = (POST / "index.md").read_text(encoding="utf-8")
TRAIN = (SNIP / "train_with_decay.py").read_text(encoding="utf-8")
SRC = (SNIP / "decay_schedule.py").read_text(encoding="utf-8").splitlines()
CLASS = SRC[SRC.index("class Optimizer_SGD:"):SRC.index("def rates_of_a_run(decay, updates=10001, learning_rate=1.0):")]
while not CLASS[-1].strip():
    CLASS.pop()
assert len(CLASS) == 19
BLOCK = "\n".join(CLASS)
assert "```python\n" + BLOCK + "\n```" in INDEX                    # the listing of section 4
assert BLOCK in TRAIN                                             # the class the documented run uses

CALLS = ["optimizer.pre_update_params()", "optimizer.update_params(dense1)", "optimizer.update_params(dense2)",
         "optimizer.post_update_params()"]
CTOR = "optimizer = Optimizer_SGD(learning_rate=1.0, decay=1e-3)"
LOOP = "\n".join("    " + c for c in CALLS)
assert "    # Update with decay.\n" + LOOP in INDEX and LOOP in TRAIN
assert CTOR in INDEX and CTOR in TRAIN

# the class does what the symbols say: alpha_t from t before the updates, t + 1 once per step after them
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(SNIP / "decay_schedule.py"), run_name="snippet")


class Layer:
    def __init__(self):
        self.weights, self.biases, self.dweights, self.dbiases = 0.0, 0.0, 1.0, 1.0


opt, layers = ns["Optimizer_SGD"](learning_rate=1.0, decay=1e-3), [Layer(), Layer()]
for step in range(3):
    opt.pre_update_params()
    assert opt.current_learning_rate == 1.0 / (1.0 + 1e-3 * step) and opt.iterations == step
    for layer in layers:
        opt.update_params(layer)
        assert opt.iterations == step                              # update_params leaves the counter alone
    opt.post_update_params()
assert opt.iterations == 3 and opt.learning_rate == 1.0

W, G = "weight", "gradient"
code = lambda t, color=None: span(t, mono=True, color=color)  # noqa: E731
LINES = list(CLASS)
assert LINES[14] == "        layer.weights -= self.current_learning_rate * layer.dweights"
assert LINES[15] == "        layer.biases  -= self.current_learning_rate * layer.dbiases"
LINES[14] = rich("        ", code("layer.weights", W), code(" -= self.current_learning_rate * "),
                 code("layer.dweights", G))
LINES[15] = rich("        ", code("layer.biases", W), code("  -= self.current_learning_rate * "),
                 code("layer.dbiases", G))
DEF = {k: LINES[k].strip() for k in (2, 8, 13, 17)}
assert DEF == {2: "def __init__(self, learning_rate=1.0, decay=0.0):", 8: "def pre_update_params(self):",
               13: "def update_params(self, layer):", 17: "def post_update_params(self):"}
ATTR = {3: "self.learning_rate", 4: "self.current_learning_rate", 5: "self.decay", 6: "self.iterations"}
for k, a in ATTR.items():
    assert LINES[k].strip().startswith(a + " ")

fig = Figure(
    "03-optimizer-class", "The rate is set once per step and applied once per layer",
    "A card with the class Optimizer_SGD of section 4. The constructor sets learning_rate, alpha_0, "
    "current_learning_rate, alpha_t, decay, d, and iterations, t, to 0. pre_update_params computes "
    "current_learning_rate = learning_rate / (1.0 + decay * iterations). update_params subtracts "
    "current_learning_rate times layer.dweights from layer.weights and times layer.dbiases from layer.biases, "
    "parameters in orange and gradients in purple. post_update_params adds 1 to iterations. On the left, arrows run "
    "from the calls of the training loop to the methods: the constructor once before the loop, then in every step "
    "pre_update_params once, update_params twice, for dense1 and dense2, and post_update_params once.",
    subtitle="The class of section 4 with the symbols of section 2, and the calls the loop of section 5 makes to it.",
    height=720, data_w=True)

CX, CY, CW = 360, 104, 560
body = fig.card(CX, CY, CW, None, lines=LINES, style="code", fit=True)
CARD = Box(CX, CY, CW, body.bottom + 16 - CY)
BASE = [body.y + 16 + 24 * k for k in range(len(LINES))]          # the code lines' baselines
MID = lambda k: BASE[k] - 5  # noqa: E731                         # the middle of a 14 px line

# -- the symbols of section 2 beside the constructor's attributes, in ink: optimiser state has no role colour
SYM = {3: sub("α", "0"), 4: isub("α", "t"), 5: var("d"), 6: var("t")}
for k, s in SYM.items():
    fig.text(CARD.right - 24, BASE[k], s, "label", anchor="end")

# -- how often each method runs, as a tag on its def line
TAG_W = 168


def tag(k, text):
    box = Box(CARD.right - 16 - TAG_W, BASE[k] - 17, TAG_W, 24)
    fig.fill(box, "surface", radius=4)
    fig.outline(box, "rule", radius=4)
    fig.text(box.cx, BASE[k] - 1, text, "tick", anchor="middle", color="ink-muted")


tag(8, "once per step")
tag(13, "once per layer")
tag(17, "once per step")

# -- the calls, left, each with an arrow to its method
BX, BW = 40, 280


def call(lines, cy):
    """A call box centred on cy: one or two lines of code."""
    h = 16 + 24 * len(lines) - 8
    box = Box(BX, cy - h / 2, BW, h)
    fig.fill(box, "neutral-soft", radius=4)
    fig.outline(box, "border", radius=4)
    with fig.data():
        for i, s in enumerate(lines):          # leading spaces would collapse: indent by x, 8 per level
            lead = len(s) - len(s.lstrip(" "))
            fig.text(BX + 12 + 8 * (lead // 4),box.y + 4 + 17 + 24 * i, s.lstrip(" "), "code", snap=False)
    return box


with fig.data():
    c = call(["optimizer = Optimizer_SGD(", "    learning_rate=1.0, decay=1e-3)"], (MID(2) + MID(4)) / 2)
    fig.arrow((c.right, MID(2)), (CX, MID(2)))
    p = call([CALLS[0]], MID(8))
    fig.arrow((p.right, MID(8)), (CX, MID(8)))
    for i, dy in enumerate((-16, 16)):
        u = call([CALLS[1 + i]], MID(13) + dy)
        fig.arrow((u.right, u.cy), (CX, MID(13)))
    q = call([CALLS[3]], MID(17))
    fig.arrow((q.right, MID(17)), (CX, MID(17)))

fig.text(BX, MID(8) - 32, "every step, in this order:", "note")
fig.text(BX, c.y - 12, "once, before the loop:", "note")
assert c.y - 12 - 14 >= 104

# -- the key, under the card
KY = CARD.bottom + 40
assert KY <= 656
fig.legend(CX, KY, [dict(color=W, label="parameters"), dict(color=G, label="gradients")], direction="row")
fig.text(CX + 264, KY, "optimiser state in ink", "label")

fig.caption(rich("Every layer of a step reads the same ", isub("α", "t"), "; the counter moves after the last one."))
fig.write()
