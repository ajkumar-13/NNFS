"""Post 03 hero: the post's 4 -> 3 -> 3 network, the array on each set of connections, and the size of each layer.

Run from anywhere:  python posts/03-stacking-layers-and-the-forward-pass/diagrams/src/01-two-layer-network.py
Writes posts/03-stacking-layers-and-the-forward-pass/diagrams/01-two-layer-network.svg.
The sizes and the parameter counts come from the arrays of snippets/two_layer_forward.py.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, Raw, rich, var, sub, num, TIMES  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "two_layer_forward.py"
with contextlib.redirect_stdout(io.StringIO()):
    S = runpy.run_path(str(SNIP))
W1, B1, W2, B2 = S["weights1"], S["biases1"], S["weights2"], S["biases2"]
M1, N_IN = W1.shape                      # 3 neurons, 4 weights each (one row per neuron)
M2, M1b = W2.shape                       # 3 neurons, 3 weights each
assert (M1, N_IN, M2, M1b) == (3, 4, 3, 3) and M1b == M1
P1, P2 = W1.size + B1.size, W2.size + B2.size
assert (P1, P2, P1 + P2) == (15, 12, 27)          # section 6


def bold(base, s=None, t=False):
    """A whole array: bold upright capital (or bold b), optional subscript, optional transpose."""
    out = f'<tspan class="b">{base}</tspan>'
    if t and s is not None:
        # the transpose sits straight over the subscript: T raised, then the subscript pulled back under it
        # by the width of a 13 px T, then back to the baseline
        out += (f'<tspan class="sub" dy="-9">T</tspan><tspan class="sub" dx="-7" dy="13">{s}</tspan>'
                f'<tspan dy="-4">\u200b</tspan>')
    elif t:
        out += '<tspan class="sub" dy="-6">T</tspan><tspan dy="6">\u200b</tspan>'
    elif s is not None:
        out += f'<tspan class="sub" dy="4">{s}</tspan><tspan dy="-4">\u200b</tspan>'
    return Raw(out)


def shape(*t):
    parts = [var(v) if isinstance(v, str) else num(v) for v in t]
    return rich("(", *sum(([p, ", "] for p in parts), [])[:-1], ",)" if len(t) == 1 else ")")


fig = Figure(
    "01-two-layer-network", "Layer 1's output is layer 2's input",
    f"The post's network drawn left to right: {N_IN} input nodes x1 to x4, {M1} neurons in layer 1, the hidden "
    f"layer, and {M2} neurons in layer 2, the output layer. Every neuron is joined to every node of the column "
    f"before it; the {N_IN} connections into the first neuron of layer 1 and the {M1} into the first neuron of "
    f"layer 2 are drawn in the weight colour. Above the connections are the arrays they carry: X of shape (N, 4), "
    f"then Z1 of shape (N, 3); a brace over the three output arrows labels them Z2 of shape (N, 3), the output. "
    f"Under layer 1: W1 of shape (3, 4), 3 neurons "
    f"with 4 weights each, b1 of shape (3,), 12 plus 3 is {P1} parameters, and Z1 = X W1 transposed + b1. Under "
    f"layer 2: W2 of shape (3, 3), 3 neurons with 3 weights each, b2 of shape (3,), 9 plus 3 is {P2} parameters, "
    f"and Z2 = Z1 W2 transposed + b2. The whole network has {P1} plus {P2}, which is {P1 + P2} parameters. A key at "
    f"the top right reads: into neuron 1, {N_IN} weights in layer 1 and {M1} weights in layer 2. A caption line reads: each "
    f"layer runs the same call; only the sizes change.",
    subtitle=rich("The post's network: 4 inputs, 3 neurons in each layer, a batch of ", var("N"), " samples."),
    height=720)

R = 24                                   # node radius
CX = (136, 392, 648)                     # input column, layer 1, layer 2
STEP = 80
YMID = 336
ys = lambda n: [YMID + STEP * (k - (n - 1) / 2) for k in range(n)]  # noqa: E731
Y_IN, Y_L1, Y_L2 = ys(N_IN), ys(M1), ys(M2)
OUT_X = 760                              # where the output arrows end

# -- connections: every neuron reads every node of the column before; neuron 1 of each layer in the weight colour
plain, hot = [], []
for (xa, ya_list), (xb, yb_list) in (((CX[0], Y_IN), (CX[1], Y_L1)), ((CX[1], Y_L1), (CX[2], Y_L2))):
    for k, yb in enumerate(yb_list):
        for ya in ya_list:
            seg = f"M{xa + R},{ya:.0f} L{xb - R},{yb:.0f}"
            (hot if k == 0 else plain).append(seg)
w_cls = fig._cls("s", "weight")
with fig.data():
    fig._path("".join(plain), "rule1")
    fig._path("".join(hot), f"{w_cls} w15 nofill")

# -- nodes
f_in, s_in = fig._cls("f", "input-soft"), fig._cls("s", "input")
f_n, s_n = fig._cls("f", "surface"), fig._cls("s", "ink-muted")
with fig.data():
    for k, y in enumerate(Y_IN):
        fig.add(f'<circle cx="{CX[0]}" cy="{y:.0f}" r="{R}" class="{f_in} {s_in} w15"/>')
    for x, ylist in ((CX[1], Y_L1), (CX[2], Y_L2)):
        for k, y in enumerate(ylist):
            stroke = w_cls if k == 0 else s_n
            fig.add(f'<circle cx="{x}" cy="{y:.0f}" r="{R}" class="{f_n} {stroke} w15"/>')
for k, y in enumerate(Y_IN):
    fig.text(CX[0], y + 5, sub("x", str(k + 1)), "label", anchor="middle")

# -- the output leaves layer 2
for y in Y_L2:
    fig.arrow((CX[2] + R, y), (OUT_X, y))

# -- the array each set of connections carries
TOP = 136
for x, name, shp, per in ((264, bold("X"), shape("N", N_IN), f"{N_IN} values per sample"),
                          (520, bold("Z", "1"), shape("N", M1), f"{M1} values per sample")):
    fig.text(x, TOP, rich(name, "  ", shp), "label", anchor="middle")
    fig.text(x, TOP + 24, per, "note", anchor="middle")
# the output: a brace over the three arrow tips, labelled in the right column, centred on the middle output
fig.brace(Y_L2[0] - 16, Y_L2[-1] + 16, OUT_X + 8, side="right", vertical=True)
XR = OUT_X + 24                          # the right column's left edge
fig.text(XR, YMID - 4, rich(bold("Z", "2"), "  ", shape("N", M2)), "label")
fig.text(XR, YMID + 20, "the output", "note")

# -- one block per column: what the layer is, its weights, its parameter count, its call
YB = 528
fig.text(CX[0], YB, "Input", "head", anchor="middle")
fig.text(CX[0], YB + 28, f"{N_IN} features", "note", anchor="middle")
for x, k, m, n, p, call in (
        (CX[1], "1", M1, N_IN, P1, rich(bold("Z", "1"), " = ", bold("X"), " ", bold("W", "1", t=True), " + ",
                                         bold("b", "1"))),
        (CX[2], "2", M2, M1, P2, rich(bold("Z", "2"), " = ", bold("Z", "1"), " ", bold("W", "2", t=True), " + ",
                                       bold("b", "2")))):
    fig.text(x, YB, f"Layer {k}: {'hidden' if k == '1' else 'output'}", "head", anchor="middle")
    fig.text(x, YB + 28, rich(bold("W", k), " ", shape(m, n), ",  ", bold("b", k), " ", shape(m)),
             "label", anchor="middle", color="weight")
    fig.text(x, YB + 52, f"{m} neurons, {n} weights each", "note", anchor="middle")
    fig.text(x, YB + 76, rich(f"{m * n} + {m} = ", num(p), " parameters"), "note", anchor="middle")
    fig.text(x, YB + 112, call, "math", anchor="middle")
fig.text(848, YB, "Network", "head", anchor="middle")
fig.text(848, YB + 28, rich(num(P1), " + ", num(P2), " = ", num(P1 + P2)), "value", anchor="middle")
fig.text(848, YB + 52, "parameters", "note", anchor="middle")

# -- what the weight colour marks: the key heads the right column, on the baseline of the array labels
fig.legend(XR, TOP, [dict(color="weight", label="Into neuron 1", mark="line")])
fig.note(Box(XR, TOP, 0, 0), [f"{N_IN} weights in layer 1", f"{M1} weights in layer 2"])
fig.caption("Each layer runs the same call; only the sizes change.")
fig.write()
