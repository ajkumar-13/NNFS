"""Post 01, section 6: a layer of three neurons over four inputs, as wiring and as z = Wx + b.

Run from anywhere:  python posts/01-neurons-and-layers/diagrams/src/02-a-layer.py
Writes posts/01-neurons-and-layers/diagrams/02-a-layer.svg.
Every number comes from snippets/layer_by_hand.py (section 7), which this script runs.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, Raw, rich, var, sub, num, CDOT, SIGMA, TIMES  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "layer_by_hand.py"
with contextlib.redirect_stdout(io.StringIO()) as out:
    ns = runpy.run_path(str(SNIP))
X, W, B, Z = ns["inputs"], ns["weights"], ns["biases"], ns["outputs"]
assert out.getvalue().strip() == "[4.8, 1.21, 2.385]"
N_IN, N_NEU = len(X), len(W)
N_W, N_B = N_IN * N_NEU, N_NEU
assert (N_W, N_B, N_W + N_B) == (12, 3, 15)
K = 1                                                   # the neuron traced in colour: neuron 2
SUMK = sum(x * w for x, w in zip(X, W[K]))
assert round(SUMK, 10) == -1.79 and round(Z[K], 10) == 1.21


def bold(s):
    return Raw(f'<tspan class="b">{s}</tspan>')


def say(v, d):
    return num(v, d).replace("\u2212", "minus ")


fig = Figure(
    "02-a-layer", "A layer is several neurons reading the same inputs",
    f"Two panels for the layer of section 7. On the left, the wiring: four inputs x1 to x4 each connect to all three "
    f"neurons, twelve lines, and each neuron box reads sigma plus its own bias b1, b2 or b3 and gives z1, z2 or z3. "
    f"Neuron 2's four lines are drawn in the weight colour. On the right, the same layer as z equals W x plus b: W is "
    f"a 3 by 4 grid, one row per neuron, rows {'; '.join(', '.join(say(v, 2) for v in r) for r in W)}; x is "
    f"{', '.join(say(v, 1) for v in X)}; b is {', '.join(say(v, 1) for v in B)}; z is "
    f"{', '.join(say(v, 3) for v in Z)}. Row 2 of W, b2 and z2 are outlined: minus 1.79 plus 3.0 is 1.21. "
    f"3 times 4 is 12 weights, plus 3 biases, 15 parameters.",
    subtitle="Section 7's layer: three neurons over four inputs, each neuron with its own weights and its own bias.")

left, right = fig.row((5, 6))
lb = fig.panel(left, "Wiring: every input reaches every neuron")
rb = fig.panel(right, rich("The same layer as ", bold("z"), " = ", bold("W"), bold("x"), " + ", bold("b")))

ROW0, RH = 168, 64                                      # neuron rows: 168..360, one per row of W
ROWS = [ROW0 + RH * k + RH // 2 for k in range(N_NEU)]  # 200, 264, 328

# -- left: inputs, twelve lines, three neuron boxes
IX, IW, IH = 72, 48, 32
IY = [ROW0 - 8 + 56 * i for i in range(N_IN)]           # 160, 216, 272, 328 (cells 32 high)
gin = [fig.grid(IX, y, 1, 1, cell_w=IW, cell_h=IH, fill={(0, 0): "input-soft"}) for y in IY]
for i, g in enumerate(gin):
    fig.text(g.box.cx, g.box.cy + 5, sub("x", str(i + 1)), "label", anchor="middle")
NX, NW, NH = 256, 104, 40
with fig.data():
    for k in range(N_NEU):
        for i, g in enumerate(gin):
            hot = k == K
            cls = f"nofill {'w15' if hot else 'w1'} {fig._cls('s', 'weight' if hot else 'rule')}"
            if not hot:
                fig._path(f"M{g.box.right},{g.box.cy} L{NX},{ROWS[k]}", cls)
    for i, g in enumerate(gin):                         # neuron 2's lines last, on top
        fig._path(f"M{g.box.right},{g.box.cy} L{NX},{ROWS[K]}",
                  f"nofill w15 {fig._cls('s', 'weight')}")
for k in range(N_NEU):
    bx = Box(NX, ROWS[k] - NH // 2, NW, NH)
    fig.panel(bx, None, card=True)
    if k == K:
        with fig.data():
            fig._rect(bx.x, bx.y, bx.w, bx.h, f"nofill w15 {fig._cls('s', 'weight')}", rx=8, extra='data-fit="skip"')
    fig.text(bx.cx, bx.cy + 6, rich(SIGMA, " + ", sub("b", str(k + 1))), "label", anchor="middle")
    fig.text(bx.right + 16, bx.cy + 5, sub("z", str(k + 1)), "label")
fig.note(Box(IX, ROW0, 0, RH * N_NEU), rich(f"{N_IN} inputs {TIMES} {N_NEU} neurons = {N_W} lines, one weight each"))

# -- right: z = W x + b, rows of W aligned with the neuron rows
WX, WCW = rb.x, 48                                      # W: 4 columns of 48
gw = fig.grid(WX, ROW0, N_NEU, N_IN, cell_w=WCW, cell_h=RH, values=W, decimals=2, font=14,
              fill=lambda i, j: "weight-soft")
XX = gw.box.right + 16
gx = fig.strip(XX, ROW0, N_IN, vertical=True, width=40, height=RH * N_NEU, values=X, decimals=1, font=14,
               fill=lambda i: "input-soft")
BX = XX + 40 + 40
gb = fig.strip(BX, ROW0, N_NEU, vertical=True, width=40, height=RH * N_NEU, values=B, decimals=1, font=14,
               fill=lambda i: "weight-soft")
ZX = BX + 40 + 40
gz = fig.strip(ZX, ROW0, N_NEU, vertical=True, width=72, height=RH * N_NEU, values=Z,
               decimals=3, font=14, fill=lambda i: "output-soft", strong={K: "output"})
cy = ROW0 + RH * N_NEU // 2
fig.text(XX + 40 + 20, cy + 8, "+", "op", anchor="middle")
fig.text(BX + 40 + 20, cy + 8, "=", "op", anchor="middle")
gw.window(K, 0, 1, N_IN, "weight")
gb.outline(K, K, color="weight", width=1.5)
gz.window(K, K, color="output")
for g, name, shape in ((gw, "W", f"({N_NEU}, {N_IN})"), (gx, "x", f"({N_IN},)"), (gb, "b", f"({N_NEU},)"),
                       (gz, "z", f"({N_NEU},)")):
    fig.text(g.box.cx, g.box.bottom + 24, rich(bold(name), " ", shape), "label", anchor="middle")
fig.text(WX, 424, rich("Neuron 2: ", num(SUMK, 2), " + ", num(B[K], 1), " = ", num(Z[K], 2)), "label", color="output")
fig.text(WX, 444, rich("row 2 of ", bold("W"), " times ", bold("x"), ", then its bias ", sub("b", "2")), "note")
gw.col_labels([sub("x", str(j + 1)) for j in range(N_IN)])

fig.caption(rich(f"{N_NEU} {TIMES} {N_IN} = {N_W} weights + {N_B} biases = {N_W + N_B} parameters."))
fig.write()
