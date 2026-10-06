"""Post 35, section 2: the three new layers each reuse one set of weights across positions, with the counts the post
states.

Run from anywhere:  python posts/35-whats-next/diagrams/src/03-weights-shared-across-positions.py
Writes posts/35-whats-next/diagrams/03-weights-shared-across-positions.svg.

Every number is computed here from the formula the post states and asserted against the post's text:
  - section 2.1: the first layer of nn-p01 maps 784 pixels to 128 neurons, 784 * 128 + 128 = 100,480 parameters;
    8 filters of 5 x 5 on the one-channel image have 8 * 25 + 8 = 208. The grid is the 784 pixels as 28 x 28, the
    two shaded windows one 5 x 5 filter at two positions.
  - section 2.2: h_t = tanh(x_t W_x + h_(t-1) W_h + b); with 10 features and 64 hidden units,
    10 * 64 + 64 * 64 + 64 = 4,800 parameters for 5 steps and for 500. Three steps are drawn.
  - section 2.3: one score per pair of positions, 100 positions give 10,000 scores and 1,000 give 1,000,000;
    each row of the score matrix goes through the row-wise softmax. Six positions are drawn.
Roles: the shared weights orange (weight), the inputs x_t blue (input), the softmax row green (output); the hidden
state and the pixels are ink.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, arr, var, num, dims  # noqa: E402

POST = Path(__file__).resolve().parents[2]
RAW = (POST / "index.md").read_text(encoding="utf-8")
INDEX = re.sub(r"[`*]", "", RAW)


def said(*pieces):
    for p in pieces:
        assert p in INDEX, p


# -- the counts, from the post's formulas
PIX, NEURONS = 784, 128
DENSE = PIX * NEURONS + NEURONS
FILTERS, K = 8, 5
CONV = FILTERS * K * K + FILTERS
SIDE = 28
assert SIDE * SIDE == PIX and (DENSE, CONV) == (100_480, 208)
said("maps 784 pixels to 128 neurons and has $784 \\cdot 128 + 128 = 100{,}480$ parameters",
     "8 filters of size $5 \\times 5$ on the same one-channel image has $8 \\cdot 25 + 8 = 208$, whatever the size")
FEAT, HID = 10, 64
RNN = FEAT * HID + HID * HID + HID
assert RNN == 4_800
said("with 10 features and 64 hidden units it is $10 \\cdot 64 + 64 \\cdot 64 + 64 = 4{,}800$ for 5 steps and for 500",
     "The same $\\mathbf{W}_x$, $\\mathbf{W}_h$ and $\\mathbf{b}$ are used at every step",
     "\\mathbf{h}_t = \\tanh(\\mathbf{x}_t \\mathbf{W}_x + \\mathbf{h}_{t-1} \\mathbf{W}_h + \\mathbf{b})")
SCORES = {n: n * n for n in (100, 1_000)}
assert SCORES == {100: 10_000, 1_000: 1_000_000}
said("one score per pair of positions: 100 positions give 10,000 scores and 1,000 give 1,000,000",
     "the row-wise softmax of post 06", "### 2.1. Convolutional layers: the same weights at every place",
     "### 2.2. Recurrent layers: the same weights at every timestep",
     "### 2.3. Attention and transformers: every position reads every other",
     "Each of the three reuses one set of weights across positions")

W, IN, OUT = "weight", "input", "output"
fig = Figure(
    "03-weights-shared-across-positions", "Each new layer reuses one set of weights",
    "Three panels. Convolution: a 28 by 28 grid of the 784 pixels with one 5 by 5 filter shaded orange at two "
    "positions, the same 25 weights at both; a table: a dense layer from 784 to 128 has 100,480 parameters, 8 "
    "filters of 5 by 5 have 208. Recurrence: three steps, each input x1, x2, x3 in blue feeding a hidden state h1, "
    "h2, h3 through W_x, and each hidden state feeding the next through W_h, orange arrows with the same labels at "
    "every step; a table: with 10 features and 64 hidden units, 4,800 parameters for 5 steps and 4,800 for 500. "
    "Attention: a 6 by 6 grid of scores, one per pair of positions, one row shaded green as one row-wise softmax; "
    "a table: 100 positions give 10,000 scores and 1,000 positions give 1,000,000. A shared weight's gradient is "
    "the sum of the contributions from every position.",
    subtitle="Section 2: the parameters do not grow with the input; the attention scores do, one per pair of positions.",
    data_w=True)

c1, c2, c3 = fig.row(3)
COLW = [160, 112]
TABLE_Y = 392

# -- convolution: the 784 pixels as 28 x 28, one 5 x 5 filter at two positions
b1 = fig.panel(c1, "Convolution, section 2.1")
CELL = 7
WINS = [(3, 4), (16, 17)]
inwin = lambda i, j: any(r <= i < r + K and c <= j < c + K for r, c in WINS)  # noqa: E731
g = fig.grid(b1.x, b1.y, SIDE, SIDE, cell=CELL, fill=lambda i, j: "weight-soft" if inwin(i, j) else None,
             lines_on_top=True)
for r, c in WINS:
    g.outline(r, c, K, K, W, width=1.5)
assert g.box.bottom <= 340
fig.text(b1.x, 356, rich("one filter, the same ", num(K * K), " weights,"), "note")
fig.text(b1.x, 376, "at two positions", "note")
fig.table(b1.x, TABLE_Y, [["layer", "parameters"], [rich("dense, ", num(PIX), " to ", num(NEURONS)), num(DENSE)],
                          [rich(num(FILTERS), " filters, ", dims(K, K)), num(CONV)]], COLW, row_h=28)

# -- recurrence: three steps, the same W_x and W_h at each
b2 = fig.panel(c2, "Recurrence, section 2.2")
HW, HH, STEP = 56, 40, 108
hs, xs = [], []
for t in range(3):
    x = b2.x + t * STEP
    hb = Box(x, 176, HW, HH)
    fig.card(hb.x, hb.y, hb.w, hb.h, radius=4)
    fig.text(hb.cx, hb.y + 26, arr("h", sub=str(t + 1)), "label", anchor="middle")
    hs.append(hb)
    xn = fig.node(hb.cx, 300, r=20, label=arr("x", sub=str(t + 1)), color=IN)
    xs.append(xn)
assert hs[-1].right == b2.right
for t in range(3):
    fig.arrow((hs[t].cx, xs[t].y), (hs[t].cx, hs[t].bottom), color=W)
    fig.text(hs[t].cx + 8, 252, arr("W", sub=var("x")), "label", color=W)
for t in range(2):
    fig.arrow((hs[t].right, hs[t].cy), (hs[t + 1].x, hs[t + 1].cy), color=W)
    fig.text((hs[t].right + hs[t + 1].x) / 2, hs[t].cy - 12, arr("W", sub=var("h")), "label", anchor="middle", color=W)
fig.text(b2.x, 356, rich("the same ", arr("W", sub=var("x")), ", ", arr("W", sub=var("h")), " and ", arr("b")), "note")
fig.text(b2.x, 376, "at every step", "note")
fig.table(b2.x, TABLE_Y, [[rich(num(FEAT), " features, ", num(HID), " units"), "parameters"],
                          ["5 steps", num(RNN)], ["500 steps", num(RNN)]], COLW, row_h=28)

# -- attention: one score per pair of positions, each row one softmax
b3 = fig.panel(c3, "Attention, section 2.3")
N = 6
ROW = 2
a = fig.grid(b3.x, b3.y, N, N, cell=32, fill=lambda i, j: "output-soft" if i == ROW else None)
a.outline(ROW, 0, 1, N, OUT, width=1.5)
fig.text(b3.x, 356, "one score per pair of positions;", "note")
fig.text(b3.x, 376, "shaded: one row-wise softmax", "note")
fig.table(b3.x, TABLE_Y, [["positions", "scores"], [num(100), num(SCORES[100])], [num(1_000), num(SCORES[1_000])]],
          COLW, row_h=28)
assert TABLE_Y + 3 * 28 <= 476

said("so its gradient is the sum of the contributions from all of them, the rule of post 11")
fig.caption("A shared weight's gradient is the sum of the contributions from every position, the rule of post 11.")
fig.write()
