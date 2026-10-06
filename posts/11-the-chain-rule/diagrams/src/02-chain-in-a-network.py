"""Post 11 sections 6 and 8.2: one sample through the 2-3-3 network, the four local-derivative tables measured
under the functions they belong to, and their product, the gradient for W1.

Run from anywhere:  python posts/11-the-chain-rule/diagrams/src/02-chain-in-a-network.py
Writes posts/11-the-chain-rule/diagrams/02-chain-in-a-network.svg.
snippets/network_chain.py is run here: the forward values, the four tables and their product are computed with
the snippet's own functions (local_derivatives, dense1, relu, dense2, softmax_cross_entropy) exactly as its main()
does, and every number drawn is asserted against the lines main() prints and the post quotes.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sup, num, arr, span, CDOT  # noqa: E402

PARTIAL = "\u2202"
SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "network_chain.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
    s["main"]()
OUT = buf.getvalue()

X, W1 = s["X"], s["W1"]
w1 = W1.reshape(-1)
z1 = s["dense1"](w1)
a1 = s["relu"](z1)
z2 = s["dense2"](a1)
L = s["softmax_cross_entropy"](z2)[0]
LD = s["local_derivatives"]
T_loss = LD(s["softmax_cross_entropy"], z2)
T_d2 = LD(s["dense2"], a1)
T_relu = LD(s["relu"], z1)
T_d1 = LD(s["dense1"], w1)
chain = (T_loss @ T_d2 @ T_relu @ T_d1).reshape(2, 3)
direct = LD(lambda v: s["softmax_cross_entropy"](s["dense2"](s["relu"](s["dense1"](v)))), w1).reshape(2, 3)
GAP = float(np.max(np.abs(chain - direct)))

# every value against the printout
assert list(X) == [1.0, -2.0]
assert "z1 = x W1 + b1  = [ 1.2000 -1.6000  1.7000]" in OUT and np.allclose(z1, [1.2, -1.6, 1.7])
assert "a1 = ReLU(z1)   = [1.2000 0.0000 1.7000]" in OUT and np.allclose(a1, [1.2, 0, 1.7])
assert "z2 = a1 W2 + b2 = [-0.6600  1.3900  0.8400]" in OUT and np.allclose(z2, [-0.66, 1.39, 0.84])
assert f"L               = {L:.6f}" in OUT and f"{L:.3f}" == "2.584"
assert "[[-0.9245  0.5863  0.3383]]" in OUT and [f"{v:.4f}" for v in T_loss[0]] == ["-0.9245", "0.5863", "0.3383"]
assert T_d2.shape == (3, 3) and np.allclose(T_d2, s["W2"].T) and np.allclose(T_d2, np.round(T_d2, 1), atol=1e-8)
assert T_relu.shape == (3, 3) and np.allclose(T_relu, np.diag([1, 0, 1]), atol=1e-8)
assert T_d1.shape == (3, 6) and np.allclose(T_d1, np.round(T_d1), atol=1e-8)
assert np.allclose(T_d1, np.kron(X, np.eye(3)), atol=1e-8)       # only the inputs 1 and -2
assert "[[-0.2255  0.0000  1.1500]\n [ 0.4510  0.0000 -2.3000]]" in OUT
assert [[f"{v:.4f}" for v in row] for row in chain + 0.0] == [["-0.2255", "0.0000", "1.1500"],
                                                             ["0.4510", "0.0000", "-2.3000"]]
assert f"largest gap between the two: {GAP:.1e}" in OUT and f"{GAP:.1e}" == "6.0e-11"


def d(top, bottom):
    """A partial derivative written as the post writes it: dL/dz2 with the arrays bold."""
    return rich(PARTIAL, top, "/", PARTIAL, bottom)


def b(name, k=None):
    return arr(name, sub=k) if k else arr(name)


fig = Figure(
    "02-chain-in-a-network", "Four functions, four factors, one gradient",
    "Top row, forward, left to right: the sample x = (1, minus 2) passes Dense 1 to give z1 = (1.2, minus 1.6, 1.7), "
    "ReLU to give a1 = (1.2, 0.0, 1.7), Dense 2 to give z2 = (minus 0.66, 1.39, 0.84), and softmax with "
    "cross-entropy to give the loss L = 2.584. Under each function, the table of its local derivatives, measured "
    "by central differences at those values, with backward arrows running right to left: dL/dz2 of shape (1, 3) is "
    "minus 0.9245, 0.5863, 0.3383, the softmax output with 1 subtracted at class 0; dz2/da1 of shape (3, 3) holds "
    "the entries of W2: 0.3, 0.7, minus 0.6; minus 0.2, 0.1, 0.9; 0.5, minus 0.4, 0.2; da1/dz1 of shape (3, 3) "
    "has 1, 0, 1 on its diagonal, the middle neuron switched off; dz1/dW1 of shape (3, 6) holds only the inputs 1 "
    "and minus 2. Bottom: dL/dW1 = dL/dz2 times dz2/da1 times da1/dz1 times dz1/dW1, reshaped to (2, 3), is "
    "minus 0.2255, 0.0000, 1.1500; 0.4510, 0.0000, minus 2.3000, its middle column zero. A central difference "
    "through the whole network gives the same numbers, the largest gap 6.0 times 10 to the minus 11.",
    subtitle=rich("Section 8.2: ", b("x"), " = (1, ", num(-2), "), true class 0, in the 2-3-3 network; "
                  "every table measured by central differences."),
    height=720)

# -- forward row: five value strips with the four functions between them
YF, C = 184, 48
VAL = [("x", None, [num(int(v)) for v in X], "input", 40, C),
       ("z", "1", [num(float(v), 1) for v in z1], "output", 200, C),
       ("a", "1", [num(float(v), 1) for v in a1], "output", 424, C),
       ("z", "2", [num(float(v), 2) for v in z2], "output", 640, C),
       ("L", None, [f"{L:.3f}"], "error", 848, 72)]
FUN = ["Dense 1", "ReLU", "Dense 2", "softmax and CE"]
fig.text(40, 128, "Forward: values, left to right", "head")
strips = []
for name, k, vals, role, x, cw in VAL:
    g = fig.strip(x, YF, len(vals), cell_w=cw, cell_h=40, values=vals, font=15, fill=lambda i, r=role: r + "-soft")
    label = var("L") if name == "L" else b(name, k)
    fig.text(g.box.cx, g.box.bottom + 28, label, "math", anchor="middle")
    strips.append(g)
centres = []
for g0, g1, fn in zip(strips, strips[1:], FUN):
    fig.arrow((g0.box.right, YF + 20), (g1.box.x, YF + 20))
    cx = (g0.box.right + g1.box.x) / 2
    centres.append(cx)
    fig.text(cx, YF - 16, fn, "label", anchor="middle")
assert centres == [168, 384, 604, 816], centres
# the switched-off neuron: its weighted sum is negative and ReLU passes 0
strips[1].outline(1, 1, color="error", width=1.5)
strips[2].outline(1, 1, color="error", width=1.5)
fig.legend(640, 128, [dict(color="error", label="neuron 2, switched off", mark="outline")], direction="row")

# -- backward row: each function's table under it, the arrows running from the loss end
YB = 376
fig.text(40, 296, "Backward: one table of local derivatives per function", "head", color="gradient")
TABLES = [
    (d(b("z", "1"), b("W", "1")), T_d1, 32, 32, lambda v: num(int(round(v))), [rich("the inputs 1 and ", num(-2))]),
    (d(b("a", "1"), b("z", "1")), T_relu, 32, 32, lambda v: num(int(round(v))), ["1 or 0 on the diagonal"]),
    (d(b("z", "2"), b("a", "1")), T_d2, 40, 32, lambda v: num(float(v), 1), [rich("the entries of ", b("W", "2"))]),
    (d(var("L"), b("z", "2")), T_loss, 64, 32, lambda v: num(float(v), 4), ["softmax output,", "1 subtracted at class 0"]),
]
grids = []
for (head, T, cw, ch, fmt, note), cx in zip(TABLES, centres):
    rows, cols = T.shape
    w = cols * cw
    x = cx - w / 2
    y = YB + (96 - rows * ch) / 2
    g = fig.grid(x, y, rows, cols, cell_w=cw, cell_h=ch, values=lambda i, j, T=T, f=fmt: f(T[i, j] + 0.0),
                 font=14, fill=lambda i, j: "gradient-soft")
    fig.text(cx, YB - 40, head, "math", anchor="middle", color="gradient")
    fig.text(cx, YB - 16, f"({rows}, {cols})", "note", anchor="middle")
    for k, line in enumerate(note):
        fig.text(cx, YB + 96 + 28 + 20 * k, line, "note", anchor="middle")
    grids.append(g)
# the ReLU table's zero for the middle neuron
grids[1].outline(1, 1, color="error", width=1.5)
for g_right, g_left in zip(grids[:0:-1], grids[-2::-1]):
    fig.arrow((g_right.box.x - 8, YB + 48), (g_left.box.right + 8, YB + 48))

# -- the product: the four tables in the order of the formula, reshaped to the shape of W1
YR = 568
fig.text(40, YR + 46, rich(d(var("L"), b("W", "1")), " = ", d(var("L"), b("z", "2")), " ", CDOT, " ",
                           d(b("z", "2"), b("a", "1")), " ", CDOT, " ", d(b("a", "1"), b("z", "1")), " ", CDOT, " ",
                           d(b("z", "1"), b("W", "1")), " ="), "math")
gr = fig.grid(424, YR, 2, 3, cell_w=72, cell_h=40, values=lambda i, j: num(float(chain[i, j] + 0.0), 4), font=16,
              fill=lambda i, j: "gradient-soft")
gr.outline(0, 1, 2, 1, color="error", width=1.5, form="grid")
fig.text(gr.box.cx, YR - 16, "reshaped to (2, 3)", "note", anchor="middle")
fig.text(gr.box.right + 24, YR + 16, "middle column 0:", "note")
fig.text(gr.box.right + 24, YR + 36, "neuron 2 is off", "note")
fig.text(gr.box.right + 24, YR + 64, "central difference: the same,", "note")
fig.text(gr.box.right + 24, YR + 88, rich("largest gap 6.0 \u00d7 10", sup("", "\u221211", italic=False)), "note")

fig.caption("Read the lower row right to left: four functions, four factors, one gradient entry per weight.")
fig.write()
