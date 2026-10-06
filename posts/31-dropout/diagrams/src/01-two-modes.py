"""Post 31, sections 4 and 5: one Layer_Dropout at p = 0.2 in its two modes, on the post's five activations of 1.

Run from anywhere:  python posts/31-dropout/diagrams/src/01-two-modes.py
Writes posts/31-dropout/diagrams/01-two-modes.svg. Takes about a second.

snippets/layer_dropout.py is run as it is and its printout gives the mask of 0 and 1, the stored mask of 0 and
1 / (1 - p), the output and its sum, and the dinputs of the backward pass; every printed line used is asserted
against the listings of sections 4 and 5 in index.md. The dvalues of the backward pass are read from the
snippet's source, and dvalues * mask is checked against the printed dinputs. The evaluation row runs the post's
own Layer_Dropout (snippets/network.py) on the same activations with training=False.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, num, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIP = POST / "snippets"
INDEX = (POST / "index.md").read_text(encoding="utf-8")
SRC = (SNIP / "layer_dropout.py").read_text(encoding="utf-8")
OUT = subprocess.run([sys.executable, str(SNIP / "layer_dropout.py")], cwd=str(SNIP), capture_output=True,
                     text=True, check=True, env={"PYTHONDONTWRITEBYTECODE": "1", **__import__("os").environ}).stdout


def line(prefix):
    m = re.search(rf"^{re.escape(prefix)}.*$", OUT, re.M)
    assert m, prefix
    return m.group(0)


def arr(s):
    return np.array([float(v) for v in s.strip("[] ").split()])


# -- section 4: the draw, printed and in the post
L_MASK, L_SCALED = line("mask of 0 and 1:"), line("mask / (1 - p):")
assert L_MASK in INDEX and L_SCALED in INDEX
P = 0.2
assert "p = 0.2\n" in SRC and "Five activations of 1 at $p = 0.2$" in INDEX
A = np.ones(5)
M = arr(re.search(r"\[(.*?)\]", L_MASK).group(0))
M_SCALED = arr(re.search(r"\[(.*?)\]", L_SCALED).group(0))
SUM_RAW = float(re.search(r"sum of a \* mask: (\S+)", L_MASK).group(1))
SUM_OUT = float(re.search(r"sum: (\S+)$", L_SCALED).group(1))
assert M.tolist() == [1, 1, 1, 1, 0] and np.allclose(M_SCALED, M / (1 - P))
assert SUM_RAW == (A * M).sum() == 4.0 and SUM_OUT == (A * M_SCALED).sum() == 5.0
OUT_TRAIN = A * M_SCALED
L_LAYER = line("Layer_Dropout(0.2):")
assert np.allclose(arr(re.search(r"\[(.*?)\]", L_LAYER).group(0)), OUT_TRAIN)        # the class gives the same
assert "The four survivors carry 1.25 each and the sum is back at 5, as the first row of the figure" in INDEX

# -- section 5: backward with the stored mask
L_BM, L_DIN = line("mask:   "), line("dinputs:")
assert f"mask:    [1.25 1.25 1.25 1.25 0.  ]\ndinputs: [ 0.125 -0.25   0.375  0.5   -0.   ]" in INDEX
assert L_BM == "mask:    [1.25 1.25 1.25 1.25 0.  ]" and L_DIN == "dinputs: [ 0.125 -0.25   0.375  0.5   -0.   ]"
DVALUES = np.array(eval(re.search(r"layer\.backward\(np\.array\(\[\[(.*?)\]\]\)\)", SRC).group(1).join("[]")))
DINPUTS = arr(L_DIN.split(":", 1)[1])
assert np.allclose(DVALUES * M_SCALED, DINPUTS) and DVALUES.tolist() == [0.1, -0.2, 0.3, 0.4, -0.5]

# -- section 5: evaluation is the identity (the post's class, run here on the same activations)
assert "output: [7. 7. 7. 7. 7.]  equals the input: True  random stream untouched: True" in INDEX
sys.path.insert(0, str(SNIP))
from network import Layer_Dropout  # noqa: E402

layer = Layer_Dropout(P)
state = np.random.get_state()[1].copy()
layer.forward(A[None, :], training=False)
OUT_EVAL = layer.output[0]
assert np.array_equal(OUT_EVAL, A) and np.array_equal(state, np.random.get_state()[1])
assert not hasattr(layer, "binary_mask")                                              # no draw at all


def say(vals, d):
    return ", ".join(num(float(v), d).replace("−", "minus ") for v in vals)


fig = Figure(
    "01-two-modes", "Training masks and rescales; evaluation passes through",
    "Three rows of five cells for one Layer_Dropout with drop rate p = 0.2, the draw of section 4. Training, "
    f"forward: activations a of {say(A, None)} times the stored mask, the mask of 0 and 1 ({say(M, None)}) "
    f"divided by 1 minus p = 0.8, that is {say(M_SCALED, 2)}, gives the output {say(OUT_TRAIN, 2)}, which sums to "
    f"{SUM_OUT:g} as the input does; the fifth entry is dropped. Training, backward: dvalues {say(DVALUES, 1)} "
    f"times the same stored mask gives dinputs {say(DINPUTS, 3)}. Evaluation with training=False: no mask is "
    f"drawn and the output is a copy of the input, {say(OUT_EVAL, None)}.",
    subtitle=rich("Five activations of 1, drop rate ", var("p"), " = 0.2, so survivors are divided by 1 − ",
                  var("p"), " = 0.8."),
    height=720, data_w=True)

CELL = 48
FONT = 14                                   # every strip prints at 14, so 0.00 and 0.000 keep room for the dashed mark
XS = [40, 360, 680]                         # three strips of 5 × 48 = 240; operators centred in the 80 gaps
OPX = [320, 640]
ROW = [136, 312, 488]                       # heading baselines; strips start 48 below
DROP = int(np.flatnonzero(M == 0)[0])
COPY = "self.output = inputs.copy()"
assert COPY in (SNIP / "network.py").read_text(encoding="utf-8") and COPY in INDEX


def heading(y, title, code):
    fig.text(40, y, title, "head")
    fig.text(40 + int(text_width(title, 16, weight=600)) // 4 * 4 + 16, y, code, "code")


def strip(x, y, vals, d, color, dropped=False, font=FONT):
    g = fig.strip(x, y, 5, cell=CELL, values=[num(float(v), d) for v in vals], font=font,
                  fill=lambda k: ("surface" if dropped and k == DROP else f"{color}-soft"))
    if dropped:
        fig.outline(g.cell(0, DROP).inset(4), "ink-muted", width=1.5, dash="lead")
    return g


def label(x, y, s, style="note"):
    fig.text(x, y - 12, s, style)


def ops(y, a, b):
    fig.op(OPX[0], y + CELL // 2, a)
    fig.op(OPX[1], y + CELL // 2, b)


# -- row 1: training forward
sy = ROW[0] + 48
heading(ROW[0], "Training, forward", "training=True")
label(XS[0], sy, rich("activations ", var("a")))
label(XS[1], sy, rich("stored mask: 0 or 1 / (1 − ", var("p"), ")"))
label(XS[2], sy, "output to the next layer")
strip(XS[0], sy, A, None, "input")
strip(XS[1], sy, M_SCALED, 2, "neutral", dropped=True)
strip(XS[2], sy, OUT_TRAIN, 2, "output", dropped=True)
ops(sy, "⊙", "=")
fig.text(XS[0], sy + CELL + 28, f"sum {A.sum():g}", "note")
fig.text(XS[1], sy + CELL + 28, rich("drawn as ", " ".join(f"{int(v)}" for v in M), ", then divided by 0.8"), "note")
fig.text(XS[2], sy + CELL + 28, f"sum {SUM_OUT:g} on this draw, {A.sum():g} on average", "note")

# -- row 2: training backward, the same mask
sy = ROW[1] + 48
heading(ROW[1], "Training, backward", "backward(dvalues)")
fig.text(XS[0], sy - 12, "dvalues", "code")
label(XS[1], sy, "the same stored mask")
fig.text(XS[2], sy - 12, "dinputs", "code")
strip(XS[0], sy, DVALUES, 1, "gradient")
strip(XS[1], sy, M_SCALED, 2, "neutral", dropped=True)
strip(XS[2], sy, DINPUTS, 3, "gradient", dropped=True, font=13)    # "−0.250" needs 13 in a 48 cell
ops(sy, "⊙", "=")
fig.text(XS[1], sy + CELL + 28, "no new draw: the forward pass stored it", "note")
fig.text(XS[2], sy + CELL + 28, "a dropped entry passes back 0", "note")

# -- row 3: evaluation, the identity
sy = ROW[2] + 48
heading(ROW[2], "Evaluation", "training=False")
label(XS[0], sy, rich("activations ", var("a")))
label(XS[1], sy, "no mask: the layer copies its input")
label(XS[2], sy, "output, equal to the input")
strip(XS[0], sy, A, None, "input")
mid = Box(XS[1], sy, 5 * CELL, CELL)
fig.fill(mid, "neutral-soft", fit=False)
fig.outline(mid, "rule", width=1)
fig.text(mid.cx, sy + 29, COPY, "code", anchor="middle")
strip(XS[2], sy, OUT_EVAL, None, "output")
fig.arrow((XS[0] + 5 * CELL + 12, sy + CELL // 2), (XS[1] - 12, sy + CELL // 2))
fig.arrow((XS[1] + 5 * CELL + 12, sy + CELL // 2), (XS[2] - 12, sy + CELL // 2))
fig.text(XS[0], sy + CELL + 28, f"sum {A.sum():g}", "note")
fig.text(XS[2], sy + CELL + 28, f"sum {OUT_EVAL.sum():g}, the same on every pass", "note")

fig.caption("Dashed: the entry this draw dropped. A new training pass draws a new mask.")
fig.write()
