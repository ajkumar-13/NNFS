"""The project's architecture figure: the 784-128-128-10 network, its shapes and its parameters.

Run from the project directory:  uv run python docs/diagrams/src/architecture.py
Writes docs/diagrams/architecture.svg.

Every number in the figure is read from mnist_from_scratch.model.Network: the layer sizes, the
parameter count of each dense layer, the L2 strength of each layer, and the dropout rate. Nothing
here is a training result.
"""

import sys
from itertools import pairwise
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
sys.path.insert(0, str(PROJECT / "src"))
from figkit import Figure, num, rich, var  # noqa: E402

from mnist_from_scratch.model import Network  # noqa: E402

net = Network()
LAYERS = net.dense_layers
SIZES = [layer.weights.shape for layer in LAYERS]
PARAMS = [layer.weights.size + layer.biases.size for layer in LAYERS]
L2 = [layer.weight_regularizer_l2 for layer in LAYERS]
DROP = round(1 - net.dropout1.rate, 6)
TOTAL = net.parameter_count()
if SIZES != [(784, 128), (128, 128), (128, 10)] or PARAMS != [100480, 16512, 1290]:
    raise SystemExit("the network is not the documented 784-128-128-10; redraw the figure")
if TOTAL != 118282 or L2 != [5e-4, 5e-4, 0] or DROP != 0.1 or net.dropout2.rate != 0.9:
    raise SystemExit("the network is not the documented baseline; rewrite the description below")

DESC = (
    "The network as four columns joined by arrows, left to right. Input: a batch of N MNIST "
    "images, each 28 by 28 pixels flattened to 784 values, shape (N, 784). Hidden layer 1: "
    "Dense(784, 128), ReLU, Dropout(0.1); output shape (N, 128); L2 penalty 0.0005 on its "
    "weights; 100,480 parameters. Hidden layer 2: Dense(128, 128), ReLU, Dropout(0.1); output "
    "shape (N, 128); L2 penalty 0.0005; 16,512 parameters. Output layer: Dense(128, 10) and "
    "Softmax; output shape (N, 10); no penalty; 1,290 parameters. A brace under the three "
    "layers gives the total, 118,282 parameters."
)


def shape(width):
    return rich("(", var("N"), f", {width})")


def dense(size):
    return f"Dense({size[0]}, {size[1]})"


fig = Figure(
    "architecture",
    f"The first layer holds {num(PARAMS[0])} of the {num(TOTAL)} parameters",
    DESC,
    subtitle="The 784-128-128-10 network, as built by mnist_from_scratch.model.Network.",
)

# -- columns: a card per stage, with an arrow between neighbours
Y_HEAD, CARD_Y, CARD_H = 136, 152, 112
CARDS = [(136, 120), (304, 168), (520, 168), (736, 184)]  # (x, width)
HEADS = ["Input", "Hidden layer 1", "Hidden layer 2", "Output layer"]
LINES = [
    ["28 × 28 pixels", "flattened"],
    [dense(SIZES[0]), "ReLU", f"Dropout({DROP})"],
    [dense(SIZES[1]), "ReLU", f"Dropout({DROP})"],
    [dense(SIZES[2]), "Softmax"],
]
CENTRES = [x + w // 2 for x, w in CARDS]
MID = CARD_Y + CARD_H // 2

for k, ((x, w), head, lines) in enumerate(zip(CARDS, HEADS, LINES, strict=True)):
    fig.text(CENTRES[k], Y_HEAD, head, "head", anchor="middle")
    role = "input" if k == 0 else "weight"
    classes = f"{fig._cls('f', role + '-soft')} {fig._cls('s', role + '-line')} w1"
    fig._rect(x, CARD_Y, w, CARD_H, classes, rx=8)
    first = MID + 4 - 12 * (len(lines) - 1)
    for j, line in enumerate(lines):
        fig.text(CENTRES[k], first + 24 * j, line, "label" if k == 0 else "code", anchor="middle")
for (x, w), (x_next, _) in pairwise(CARDS):
    fig.arrow((x + w + 8, MID), (x_next - 8, MID))

# -- what each stage outputs, what it is penalised by, and what it holds
Y_SHAPE, Y_L2, Y_PARAM, Y_BRACE = 304, 340, 376, 392
fig.text(40, Y_SHAPE, "output shape", "note")
fig.text(40, Y_L2, "L2 penalty", "note")
fig.text(40, Y_PARAM, "parameters", "note")
for k, width in enumerate([784, 128, 128, 10]):
    colour = "input" if k == 0 else "output" if k == 3 else None
    fig.text(CENTRES[k], Y_SHAPE, shape(width), "value", anchor="middle", color=colour)
for k, strength in enumerate(L2, start=1):
    text = f"{strength:.4f}" if strength else "none"
    fig.text(CENTRES[k], Y_L2, text, "value" if strength else "note", anchor="middle")
for k, count in enumerate(PARAMS, start=1):
    fig.text(CENTRES[k], Y_PARAM, num(count), "value", anchor="middle", color="weight")

LEFT, RIGHT = CARDS[1][0], CARDS[3][0] + CARDS[3][1]
fig.brace(LEFT, RIGHT, Y_BRACE, side="below")
fig.text(
    (LEFT + RIGHT) / 2,
    Y_BRACE + 32,
    f"{num(TOTAL)} parameters",
    "value",
    anchor="middle",
    color="weight",
    bold=True,
)
fig.caption("Dropout acts in training only. Weights and biases are counted; L2 is on weights.")
fig.write()
