"""Post 35, sections 1 and 8: the calls every layer and every optimiser answer, one step of the loop that makes them,
and the three things section 8 says do change.

Run from anywhere:  python posts/35-whats-next/diagrams/src/02-the-skeleton-is-fixed.py
Writes posts/35-whats-next/diagrams/02-the-skeleton-is-fixed.svg.

Nothing here is a measurement. Every label is the post's own wording, asserted against index.md (markdown emphasis
and backticks removed):
  - the layer card: section 1, "Every layer has a forward that stores its output and a backward that receives
    dvalues and stores dinputs (post 16)"; the layers built are section 1's list (Layer_Dense, ReLU and softmax,
    the sigmoid, dropout); the layers after the series are sections 2 and 3.1 (convolution, recurrence, attention,
    and normalisation, which section 3.1 calls a layer);
  - the optimiser card: section 1, "every optimiser answers the same three calls: pre_update_params,
    update_params(layer) and post_update_params (post 23, section 8)", the six optimisers of section 1, and the
    summary table's "another formula in pre_update_params" for a schedule;
  - the loop: section 1, "The loop that calls them in order and then hands every layer with parameters to an
    optimiser is post 22's";
  - the three changes: section 8 and pitfall 2.
Arrows run from a step of the loop to the method it calls, as in post 23's class card. dvalues and dinputs are
gradients and take the gradient colour; everything else is ink.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, span, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
RAW = (POST / "index.md").read_text(encoding="utf-8")
INDEX = re.sub(r"[`*]", "", RAW)
G = "gradient"


def said(*pieces):
    """Every piece is the post's own wording (a capital at the start of a label aside)."""
    for p in pieces:
        assert p in INDEX or (p[0].lower() + p[1:]) in INDEX, p


said("Every layer has a forward that stores its output and a backward that receives dvalues and stores dinputs "
     "(post 16)",
     "The loop that calls them in order and then hands every layer with parameters to an optimiser is [post 22]",
     "every optimiser answers the same three calls: pre_update_params, update_params(layer) and "
     "post_update_params (post 23, section 8)",
     "Layers: Layer_Dense (post 04), ReLU and softmax (post 06), the sigmoid (posts 17 and 34), dropout (post 31)",
     "Optimisers: gradient descent, decay, momentum, AdaGrad, RMSProp and Adam (posts 22 to 27)",
     "Batch normalisation (Ioffe and Szegedy, 2015) is a layer",
     "### 2.1. Convolutional layers", "### 2.2. Recurrent layers", "### 2.3. Attention",
     "another formula in pre_update_params",
     "Three things do change",
     "a layer with shared weights sums gradient contributions inside its backward",
     "Training and evaluation modes, and carried state, are the loop's business (section 8)",
     "dropout in post 31 and batch normalisation here",
     "takes a training argument",
     "A recurrent layer carries state between calls, which has to be reset between unrelated sequences")


def sans(s, style="label"):
    size, weight = (16, 600) if style == "head" else (14, 400)
    return text_width(s, size, weight=weight) / 1.05          # the calibrated estimate carries 5 percent slack


def width(s, style):
    return text_width(s, 14, mono=True) if style == "code" else sans(s, style)


def runs(x, y, parts, space=0):
    """One line of mixed runs [(text, style, colour)], laid end to end, space units between them; code names are
    texts of their own."""
    cx = x
    with fig.data():
        for s, style, color in parts:
            fig.text(cx, y, s, style, color=color, snap=False)
            cx += width(s, style) + space
    return cx


fig = Figure(
    "02-the-skeleton-is-fixed", "A layer answers two calls, an optimiser three",
    "Three columns. Left, a card Any layer: forward stores its output; backward receives dvalues and stores "
    "dinputs, both in the gradient colour; built in this series: Layer_Dense, ReLU, softmax, the sigmoid, dropout; "
    "after it: convolution, recurrence, attention, normalisation. Centre, one step of the loop of post 22 as five "
    "boxes joined by downward arrows: forward of every layer in order, backward of every layer, "
    "pre_update_params(), update_params(layer) for every layer with parameters, post_update_params(). Arrows run "
    "from the first two boxes to the layer card's two methods and from the last three to the three methods of the "
    "right card, Any optimiser, which lists gradient descent, decay, momentum, AdaGrad, RMSProp and Adam, and "
    "notes that a schedule is another formula in pre_update_params. Below, three things do change (section 8): a "
    "layer with shared weights sums gradient contributions inside its backward; training and evaluation modes, for "
    "dropout and batch normalisation, through a training argument; carried state, which a recurrent layer resets "
    "between unrelated sequences.",
    subtitle="Section 1: the layer contract of post 16, one step of post 22's loop, the optimiser calls of post 23.",
    height=720, data_w=True)

LX, LW = 40, 248                  # the layer card
MX, MW = 344, 272                 # the loop
RX, RW = 672, 248                 # the optimiser card
TOP, BOT = 152, 488
PAD = 16
HEAD_Y = 136

# -- the loop: five steps, top to bottom
fig.text(MX, HEAD_Y, "One step of the loop, post 22", "head")
STEPS = [(168, 40), (224, 40), (280, 40), (336, 56), (408, 40)]
boxes = []
for y, h in STEPS:
    b = Box(MX, y, MW, h)
    fig.card(b.x, b.y, b.w, b.h, radius=4)
    boxes.append(b)
B = [b.y + 24 for b in boxes]                                 # one-line baselines (a 14 px line centred)
runs(MX + 12, B[0], [("forward", "code", None), ("  every layer, in order", "note", None)])
runs(MX + 12, B[1], [("backward", "code", None), ("  every layer", "note", None)])
fig.text(MX + 12, B[2], "pre_update_params()", "code")
fig.text(MX + 12, boxes[3].y + 24, "update_params(layer)", "code")
fig.text(MX + 12, boxes[3].y + 44, "every layer with parameters", "note")
fig.text(MX + 12, B[4], "post_update_params()", "code")
fig.chain(boxes)

# -- the layer card, methods at the heights of the steps that call them
fig.text(LX, HEAD_Y, "Any layer, post 16", "head")
fig.card(LX, TOP, LW, BOT - TOP)
x = LX + PAD
fig.text(x, B[0], "forward", "code")
fig.text(x, B[0] + 20, "stores its output", "note")
fig.text(x, B[1], "backward", "code")
for i, (verb, name) in enumerate((("receives", "dvalues"), ("stores", "dinputs"))):   # the names in one column
    fig.text(x, B[1] + 20 + 20 * i, verb, "note")
    fig.text(x + 68, B[1] + 20 + 20 * i, name, "code", color=G)
yb = B[1] + 76
fig.text(x, yb, "built in this series", "note")
fig.text(x, yb + 24, rich(span("Layer_Dense", mono=True), ", ReLU,"), "label")
fig.text(x, yb + 44, "softmax, the sigmoid,", "label")
fig.text(x, yb + 64, "dropout", "label")
fig.text(x, yb + 96, "after it", "note")
fig.text(x, yb + 120, "convolution, recurrence,", "label")
fig.text(x, yb + 140, "attention, normalisation", "label")
assert yb + 140 + 20 <= BOT, yb
for k in (0, 1):
    fig.arrow((MX, boxes[k].cy), (LX + LW, boxes[k].cy))

# -- the optimiser card, the built list above the three methods
fig.text(RX, HEAD_Y, "Any optimiser, post 23", "head")
fig.card(RX, TOP, RW, BOT - TOP)
x = RX + PAD
fig.text(x, TOP + 32, "built in this series", "note")
fig.text(x, TOP + 56, "gradient descent, decay,", "label")
fig.text(x, TOP + 76, "momentum, AdaGrad,", "label")
fig.text(x, TOP + 96, "RMSProp and Adam", "label")
fig.text(x, B[2], "pre_update_params()", "code")
fig.text(x, B[2] + 20, "a schedule: another formula", "note")
fig.text(x, boxes[3].cy + 4, "update_params(layer)", "code")
fig.text(x, B[4], "post_update_params()", "code")
for k in (2, 3, 4):
    fig.arrow((MX + MW, boxes[k].cy), (RX, boxes[k].cy))
for s, style in [("pre_update_params()", "code"), ("update_params(layer)", "code"), ("a schedule: another formula", "note"),
                 ("gradient descent, decay,", "label")]:
    assert width(s, style) <= RW - 2 * PAD, s

# -- what does change, section 8
RY = 528
fig.text(LX, RY, "Three things do change, section 8", "head")
fig.legend(RX, RY, [dict(color=G, label="gradients")])
CH = [
    (LX, LW, ["A layer with shared weights", "sums gradient contributions", rich("inside its ", span("backward", mono=True))]),
    (MX, MW, ["Training and evaluation modes", "dropout, batch normalisation:",
              None]),
    (RX, RW, ["Carried state", "a recurrent layer, reset between", "unrelated sequences"]),
]
said("A layer with shared weights", "sums gradient contributions", "inside its backward", "Training and evaluation modes",
     "a training argument", "Carried state", "reset between unrelated sequences")
for cx, cw, lines in CH:
    fig.card(cx, RY + 16, cw, 84)
    for i, s in enumerate(lines):
        if s is None:                     # "a training argument": the code name as a text of its own
            yy = RY + 16 + 28 + 20 * i            # "a" is 8 wide; the code face is at most 8.4 a character
            with fig.data():
                fig.text(cx + PAD, yy, "a", "note", snap=False)
                fig.text(cx + PAD + 12, yy, "training", "code", snap=False)
                fig.text(cx + PAD + 12 + 68 + 4, yy, "argument", "note", snap=False)
        else:
            fig.text(cx + PAD, RY + 16 + 28 + 20 * i, s, "label" if i == 0 else "note")
assert RY + 16 + 84 <= 656

fig.caption("A new layer brings a forward and a backward; a new optimiser brings three calls.")
fig.write()
