"""Post 07, sections 2 and 3: the four objects of the forward pass, the arrays they hand on, and what the script prints.

Run from anywhere:  python posts/07-coding-the-complete-forward-pass/diagrams/src/01-pipeline.py
Writes posts/07-coding-the-complete-forward-pass/diagrams/01-pipeline.svg.
snippets/forward_pass.py is run here (runpy): the column count of every band is read from the array it stands for,
the code lines are asserted to be lines of the snippet, and the output lines are the snippet's own printout.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, arr, span, text_width  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "forward_pass.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue().splitlines()
X, d1, a1, d2, a2 = s["X"], s["dense1"], s["activation1"], s["dense2"], s["activation2"]
P = a2.output

# the printout of section 3, line by line
assert OUT == ["[[0.33333334 0.33333334 0.33333334]",
               " [0.3333332  0.3333332  0.33333364]",
               " [0.3333329  0.33333293 0.3333342 ]",
               " [0.3333326  0.33333263 0.33333477]",
               " [0.33333233 0.3333324  0.33333528]]"], OUT
assert X.shape == (300, 2) and P.shape == (300, 3)
assert np.all(P[0] == np.float32(1 / 3))                       # row 0, the origin: exactly 1/3
FAR = float(np.max(np.abs(P[1:5] - 1 / 3)))
assert FAR < 2e-6                                              # "no more than two millionths from 1/3"
assert np.allclose(P.sum(axis=1), 1)
LO, HI = f"{P.min():.6f}", f"{P.max():.6f}"
assert (LO, HI) == ("0.333297", "0.333402")                   # section 5's extremes of the whole output

# the build and forward lines of the snippet, without their comments (internal spacing kept)
SRC = SNIPPET.read_text(encoding="utf-8").splitlines()
BUILD = ["dense1      = Layer_Dense(2, 3)", "activation1 = Activation_ReLU()",
         "dense2      = Layer_Dense(3, 3)", "activation2 = Activation_Softmax()"]
CALLS = ["dense1.forward(X)", "activation1.forward(dense1.output)",
         "dense2.forward(activation1.output)", "activation2.forward(dense2.output)"]
for line in BUILD + CALLS:
    assert any(l.split("#")[0].rstrip() == line for l in SRC), line

# the chain: each array as a band, one 16-wide column per column of the array
CHAIN = [("X", X, arr("X"), "input"),
         ("dense1.output", d1.output, arr("Z", sub="1"), "output"),
         ("activation1.output", a1.output, arr("A", sub="1"), "output"),
         ("dense2.output", d2.output, arr("Z", sub="2"), "output"),
         ("activation2.output", a2.output, arr("ŷ"), "output")]
OPS = [("dense1", "weight", rich(arr("X"), arr("W", sub="1"), " + ", arr("b", sub="1"))),
       ("activation1", None, "max(0, ·)"),
       ("dense2", "weight", rich(arr("A", sub="1"), arr("W", sub="2"), " + ", arr("b", sub="2"))),
       ("activation2", None, "softmax per row")]
COLS = [a.shape[1] for _, a, _, _ in CHAIN]
assert COLS == [2, 3, 3, 3, 3] and all(a.shape[0] == 300 for _, a, _, _ in CHAIN)

fig = Figure(
    "01-pipeline", "Four objects and four calls turn points into probabilities",
    "Top: the forward pass as a chain of five arrays drawn as bands, one column per feature, each named in code above "
    "and in the notation below: X with 2 columns, dense1.output (Z1) with 3, activation1.output (A1) with 3, "
    "dense2.output (Z2, the logits) with 3, and activation2.output (y-hat, the probabilities) with 3. Between them "
    "the four objects: dense1 computes X W1 + b1, activation1 takes max(0, .) of each entry, dense2 computes "
    "A1 W2 + b2, and activation2 takes the softmax per row. Bottom left, a card with the script's eight lines: "
    "dense1 = Layer_Dense(2, 3), activation1 = Activation_ReLU(), dense2 = Layer_Dense(3, 3), activation2 = "
    "Activation_Softmax(), then dense1.forward(X), activation1.forward(dense1.output), "
    "dense2.forward(activation1.output) and activation2.forward(dense2.output). Bottom right, what "
    "print(activation2.output[:5]) prints: 0.33333334 three times, then 0.3333332, 0.3333332, 0.33333364; "
    "0.3333329, 0.33333293, 0.3333342; 0.3333326, 0.33333263, 0.33333477; 0.33333233, 0.3333324, 0.33333528. Row 0, "
    "the origin, is exactly one third in every entry; rows 1 to 4 are within 0.000002 of one third. A note adds that "
    f"over all 300 rows every entry lies between {LO} and {HI} and every row sums to 1.",
    subtitle="The script of section 3 on the 300 spiral points: the arrays it hands on, its lines, and what it prints.",
    height=720)

# -- the chain
X0, GAP, CW = 56, 144, 16
TOP, BH = 152, 144
MID = TOP + BH // 2
xs = [X0]
for c in COLS[:-1]:
    xs.append(xs[-1] + c * CW + GAP)
assert xs == [56, 232, 424, 616, 808]
bands = []
for x, c, (code, _, math, role) in zip(xs, COLS, CHAIN):
    g = fig.grid(x, TOP, 1, c, cell_w=CW, cell_h=BH, fill=lambda i, j, r=role: r + "-soft")
    b = g.box
    fig.text(b.cx, TOP - 16, code, "code", anchor="middle", color=role)
    fig.text(b.cx, b.bottom + 28, math, "math", anchor="middle")
    assert b.cx + text_width(code, 14, mono=True) / 2 <= 920
    bands.append(b)
for (name, color, op), a, b in zip(OPS, bands, bands[1:]):
    fig.arrow((a.right + 8, MID), (b.x - 8, MID))
    cx = (a.right + b.x) / 2
    fig.text(cx, MID - 12, name, "code", anchor="middle", color=color)
    fig.text(cx, MID + 28, op, "label", anchor="middle")
fig.text(bands[3].cx, bands[3].bottom + 52, "the logits", "note", anchor="middle")
fig.text(bands[4].cx, bands[4].bottom + 52, "the probabilities", "note", anchor="middle")

# -- the script and its printout
YH, YC, CH = 408, 424, 232
fig.text(40, YH, "The script: four objects, four calls", "head")
left = Box(40, YC, 352, CH)
fig.card(left.x, left.y, left.w, left.h)
fig.code_block(56, YC + 32, BUILD, size=14)
fig.code_block(56, YC + 136, CALLS, size=14)

fig.text(424, YH, rich("What it prints: ", span("activation2.output[:5]", mono=True)), "head")
right = Box(424, YC, 496, CH)
fig.card(right.x, right.y, right.w, right.h)
code = fig.code_block(440, YC + 32, OUT, size=14, indent=8)
assert code.right <= 760, code.right
# row 0 and rows 1 to 4, annotated on the right of the printout
BX = 760
fig.text(BX + 16, YC + 32, "the origin, 1/3 each", "note")
fig.brace(YC + 44, YC + 140, BX, side="right", kind="bracket", vertical=True)
fig.text(BX + 16, YC + 88, "rows 1 to 4:", "note")
fig.text(BX + 16, YC + 108, "within 0.000002", "note")
fig.text(BX + 16, YC + 128, "of 1/3", "note")
fig.text(440, YC + 184, f"All 300 rows: every entry between {LO} and {HI},", "note")
fig.text(440, YC + 204, "and every row sums to 1.", "note")

fig.caption("Each object stores its result in .output, and the next call reads it from there.")
fig.write()
