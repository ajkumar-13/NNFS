"""Post 21, sections 4 to 8: the fifteen-line script, and what each of its twelve statements leaves behind.

Run from anywhere:  python posts/21-coding-the-full-backpropagation/diagrams/src/01-forward-backward-script.py
Writes posts/21-coding-the-full-backpropagation/diagrams/01-forward-backward-script.svg.
snippets/forward_backward.py is run here (runpy; it takes well under a second): the code lines are the snippet's
script with its end-of-line comments dropped, asserted to be fifteen lines, three comments and twelve statements;
every shape is read from the arrays the snippet leaves behind and asserted against the tables it prints, and the
loss, ln 3 and the count of closed gates are its own printout, asserted to be the lines the post shows.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, arr, var, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIPPET = POST / "snippets" / "forward_backward.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
BODY = (POST / "index.md").read_text(encoding="utf-8")

# -- the script: from "# Network." to the last backward call, end-of-line comments dropped
SRC = SNIPPET.read_text(encoding="utf-8").splitlines()
i0 = SRC.index("# Network.")
i1 = next(i for i, l in enumerate(SRC) if l.startswith("dense1.backward(activation1.dinputs)"))
SCRIPT = [l if l.startswith("#") else l.split("  #")[0].rstrip() for l in SRC[i0:i1 + 1]]
counted = [l for l in SCRIPT if l]
assert len(counted) == 15 and sum(l.startswith("#") for l in counted) == 3     # fifteen lines, three comments
assert len(SCRIPT) == 17 and SCRIPT[5] == "" and SCRIPT[11] == ""             # two blank lines between groups
for l in SRC[i0:i1 + 1]:
    assert l in BODY, l                                                       # the listing of section 8

d1, a1, d2, la = s["dense1"], s["activation1"], s["dense2"], s["loss_activation"]


def shape(t):
    return f"({', '.join(str(v) for v in t)})"


# parameters and their gradients, as the shape table of section 7 prints them
for name, layer in (("dense1", d1), ("dense2", d2)):
    for kind in ("weights", "biases"):
        p, g = getattr(layer, kind), getattr(layer, "d" + kind)
        assert p.shape == g.shape
        line = f"{name + '.' + kind:<16} {str(p.shape):<8} {name + '.d' + kind:<17} {str(g.shape):<8} True"
        assert line in OUT and line in BODY, line
assert (d1.weights.shape, d1.biases.shape, d2.weights.shape, d2.biases.shape) == ((2, 3), (1, 3), (3, 3), (1, 3))
SIZES = [d1.weights.size, d1.biases.size, d2.weights.size, d2.biases.size]
assert SIZES == [6, 3, 9, 3] and sum(SIZES) == 21 and "gradient entries in all: 21" in OUT

# forward outputs, the loss, and the gradients each backward call passes on
FWD = {"Z1": d1.output.shape, "A1": a1.output.shape, "Z2": d2.output.shape}
assert set(FWD.values()) == {(300, 3)} and la.output.shape == (300, 3)
LOSS, LN3 = f"{s['loss']:.7f}", f"{np.log(3):.7f}"
assert f"loss {LOSS}   ln 3 = {LN3}" in OUT and (LOSS, LN3) == ("1.0986104", "1.0986123")
assert f"loss {LOSS}   ln 3 = {LN3}" in BODY
CLOSED = int(np.sum(d1.output <= 0))
assert f"closed ReLU gates: {CLOSED} of 900; zeros in activation1.dinputs: {CLOSED}" in OUT and CLOSED == 456
DIN = {"loss_activation": la.dinputs.shape, "dense2": d2.dinputs.shape, "activation1": a1.dinputs.shape,
       "dense1": d1.dinputs.shape}
for name, t in DIN.items():
    assert f"{name + '.dinputs':<24} {str(t):<10} " in OUT
assert [DIN[k] for k in ("loss_activation", "dense2", "activation1", "dense1")] == [(300, 3)] * 3 + [(300, 2)]

W, G = "weight", "gradient"

fig = Figure(
    "01-forward-backward-script", "Fifteen lines leave four gradient arrays, 21 numbers",
    "The script of section 8 as a numbered listing of fifteen lines in three groups, each opened by a comment: "
    "# Network., four objects; # Forward., four forward calls; # Backward., four backward calls. Beside each line, "
    "what it leaves behind. Network: dense1 = Layer_Dense(2, 3) creates weights (2, 3) and biases (1, 3), dense2 = "
    "Layer_Dense(3, 3) creates weights (3, 3) and biases (1, 3), both in orange; activation1 and loss_activation "
    "have no parameters. Forward: Z1, A1 and Z2, each (300, 3), and the loss 1.0986104 beside ln 3 = 1.0986123. "
    "Backward: loss_activation.backward stores dinputs (300, 3); dense2.backward stores dweights (3, 3) and dbiases "
    "(1, 3); activation1.backward stores dinputs (300, 3) with 456 zeros at closed gates; dense1.backward stores "
    "dweights (2, 3) and dbiases (1, 3), the gradients in purple. Each gradient array lines up with its parameter's "
    "shape. A note says each backward call also stores dinputs for the next call, and a key gives the 21 numbers: "
    "6 + 3 + 9 + 3 parameters and as many gradient entries.",
    subtitle="The script of section 8 on the 300 spiral points, and what each line leaves behind.",
    height=720, data_w=True)

# -- the listing: line numbers outside the code, the two blank lines unnumbered
CX, CY, LEAD = 40, 104, 24
NX = CX + 28                                      # line numbers end here
TX = CX + 40                                      # code starts here
MONO = 8.25                                       # the code face's advance at 14 px, as Chrome renders it (8.1)
code_w = max(len(l) for l in SCRIPT) * MONO
CW = int(-(-(TX - CX + code_w + 16) // 8) * 8)
CH = 32 + 16 * LEAD + 20
CH = int(-(-CH // 4) * 4)
fig.card(CX, CY, CW, CH)
BASE = [CY + 32 + LEAD * k for k in range(len(SCRIPT))]
n = 0
for k, line in enumerate(SCRIPT):
    if not line:
        continue
    n += 1
    fig.text(NX, BASE[k], str(n), "tick", anchor="end")
    fig.text(TX, BASE[k], line, "code", color="ink-muted" if line.startswith("#") else None)
assert n == 15

# -- what each line leaves behind, one column right of the listing, on the line's own baseline
AX = CX + CW + 16
assert 920 - AX >= 216


def pieces(k, items, x0=None, y=None):
    """A row of (text, style, colour) runs on line k's baseline (or at x0, y), each run after the one before."""
    x = AX if x0 is None else x0
    y = BASE[k] if y is None else y
    for t, style, color in items:
        fig.text(x, y, t, style, color=color)
        # the next run starts after this one as rendered: the kit's estimates run long by about 4 percent
        # for code and 10 percent for the sans face, which would leave uneven gaps
        w = len(t) * MONO if style == "code" else 0.9 * text_width(t, 14, weight=400)
        x += w + 6
    assert x - 6 <= 920, (k, x)


def param_row(k, layer, prefix, color):
    """weights (n, m), biases (1, m), named in code, the shapes in the label face."""
    pieces(k, [(prefix + "weights", "code", color), (shape(getattr(layer, prefix + "weights").shape) + ",", "label", None),
               (prefix + "biases", "code", color), (shape(getattr(layer, prefix + "biases").shape), "label", None)])


K = {}
for k, l in enumerate(SCRIPT):
    if l and not l.startswith("#"):
        K.setdefault(l.split(" ")[0].split(".")[0].split("(")[0], []).append(k)
# K maps an object to its three lines: created, forward, backward ("loss" is the forward loss line)
assert K["dense1"] == [1, 7, 16] and K["dense2"] == [3, 9, 14] and K["activation1"] == [2, 8, 15]
assert K["loss_activation"] == [4, 13] and K["loss"] == [10]

fig.text(AX, BASE[0], "Left behind", "head")
# network
param_row(1, d1, "", W)
param_row(3, d2, "", W)
fig.text(AX, BASE[2], "no parameters", "note")
fig.text(AX, BASE[4], "no parameters", "note")
# forward
for k, a, key in ((7, arr("Z", sub="1"), "Z1"), (8, arr("A", sub="1"), "A1"), (9, arr("Z", sub="2"), "Z2")):
    fig.text(AX, BASE[k], rich(a, "  ", shape(FWD[key])), "label")
fig.text(AX, BASE[10], rich("loss ", LOSS), "value", color="ink")
fig.text(AX, BASE[11], rich("ln 3 = ", LN3), "note")
# backward
pieces(13, [("dinputs", "code", G), (shape(DIN["loss_activation"]), "label", None)])
param_row(14, d2, "d", G)
pieces(15, [("dinputs", "code", G), (shape(DIN["activation1"]) + ", " + f"{CLOSED} zeros", "label", None)])
param_row(16, d1, "d", G)

# -- under the listing: the dinputs note, then the key with the counts
YN = CY + CH + 28
pieces(None, [("Every backward call also stores", "note", None), ("dinputs", "code", G),
              ("for the call after it.", "note", None)], x0=CX, y=YN)
YK = YN + 48
fig.legend(CX, YK, [dict(color=W, label="parameters: 6 + 3 + 9 + 3 = 21 numbers"),
                    dict(color=G, label="gradients: 21 entries, shape for shape")], direction="row", gap=48)
fig.caption("Each gradient array has the shape of the parameter it belongs to.")
fig.write()
