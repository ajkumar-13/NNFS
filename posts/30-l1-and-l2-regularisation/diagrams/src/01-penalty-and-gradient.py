"""Post 30 hero (section 2): the L1 and L2 penalties of one weight and their gradients, at lambda = 0.01.

Run from anywhere:  python posts/30-l1-and-l2-regularisation/diagrams/src/01-penalty-and-gradient.py
Writes posts/30-l1-and-l2-regularisation/diagrams/01-penalty-and-gradient.svg. Takes about a second.

The penalties are the formulas of section 2, lambda * |w| and lambda * w^2, at the strength section 2 uses,
lambda = 0.01, each formula asserted against index.md. The gradients are not typed in: Layer_Dense of
snippets/network.py is given the weights of the chart, a zero input and a zero data gradient, so its backward
stores the penalty terms alone (as single_weight.py does), and those are drawn. That gives the class's +lambda at
w = 0 as well. The gradient table that network.py prints (section 2) is asserted against the same class, and the
crossing at |w| = 0.5 against it.

Layout: two charts side by side, the penalty left and its gradient right; L1 in ink, L2 in blue.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from network import Layer_Dense  # noqa: E402

INDEX = (POST / "index.md").read_text(encoding="utf-8")
LAM = 0.01
assert "$$L_{\\text{reg}}^{\\text{L1}} = \\lambda \\sum_m |w_m|$$" in INDEX
assert "$$L_{\\text{reg}}^{\\text{L2}} = \\lambda \\sum_m w_m^2 \\qquad \\frac{\\partial L_{\\text{reg}}^{\\text{L2}}}" \
       "{\\partial w_m} = 2 \\lambda \\, w_m$$" in INDEX
assert "`snippets/network.py` prints both at $\\lambda = 0.01$" in INDEX


def class_gradient(ws, l1=0.0, l2=0.0):
    """dweights of Layer_Dense(1, n) with these weights, a zero input and a zero data gradient: the penalty alone."""
    ws = np.asarray(ws, dtype=float)
    layer = Layer_Dense(1, ws.size, weight_regularizer_l1=l1, weight_regularizer_l2=l2)
    layer.weights = ws.reshape(1, -1).copy()
    layer.forward(np.zeros((1, 1)))
    layer.backward(np.zeros((1, ws.size)))
    return layer.dweights[0]


# -- network.py's table of section 2, against the class and the formulas
OUT = subprocess.run([sys.executable, str(POST / "snippets" / "network.py")], cwd=str(POST.parents[1]),
                     capture_output=True, text=True, check=True).stdout
TABLE = re.findall(r"^ *([\d.]+) +([\d.]+) +([\d.]+)$", OUT, re.M)
assert [r[0] for r in TABLE] == ["0.01", "0.1", "0.5", "1", "10", "100"], TABLE
for w, g1, g2 in TABLE:
    w = float(w)
    assert np.isclose(class_gradient([w], l1=LAM)[0], float(g1)) and np.isclose(float(g1), LAM * np.sign(w))
    assert np.isclose(class_gradient([w], l2=LAM)[0], float(g2)) and np.isclose(float(g2), 2 * LAM * w)
    md = f"| {w:g} | {float(g1):g} | {float(g2):g} |"
    assert md in INDEX, md
assert TABLE[2] == ("0.5", "0.01", "0.01")                       # the crossing: equal at |w| = 0.5
assert "Below $|w| = 0.5$ the L1 gradient is the larger one, and above it the L2 gradient is." in INDEX

W = np.round(np.linspace(-1.5, 1.5, 301), 6)
P1, P2 = LAM * np.abs(W), LAM * W ** 2
NEG, POS = W[W < 0], W[W > 0]
G1_NEG, G1_POS = class_gradient(NEG, l1=LAM), class_gradient(POS, l1=LAM)
G1_ZERO = class_gradient([0.0], l1=LAM)[0]
G2 = class_gradient(W, l2=LAM)
assert np.allclose(G1_NEG, -LAM) and np.allclose(G1_POS, LAM) and G1_ZERO == LAM      # +lambda at w = 0
assert np.allclose(G2, 2 * LAM * W)
assert np.isclose(class_gradient([0.5], l2=LAM)[0], class_gradient([0.5], l1=LAM)[0])
assert np.isclose(class_gradient([-0.5], l2=LAM)[0], class_gradient([-0.5], l1=LAM)[0])
assert "a weight of exactly zero receives $+\\lambda$" in INDEX

w, lam = var("w"), var("λ")
fig = Figure(
    "01-penalty-and-gradient", "L1 pulls every weight equally hard, L2 in proportion",
    "Two charts over one weight w from minus 1.5 to 1.5, at a strength lambda of 0.01. Left, the penalty: the L1 "
    "penalty lambda times |w| is a V with its corner at 0, rising to 0.015 at |w| = 1.5; the L2 penalty lambda "
    "times w squared is a parabola, rising to 0.0225. Right, the gradient of each penalty as the dense layer's "
    "backward computes it: L1 is minus 0.01 for every negative weight and plus 0.01 for every positive one, with "
    "a dot at plus 0.01 for w = 0, the class's sign there; L2 is the line 2 lambda w from minus 0.03 to 0.03. The "
    "two gradients are equal at w = 0.5 and at w = minus 0.5, both marked.",
    subtitle=rich("Both penalties on one weight ", w, ", at a strength ", lam, " = 0.01."),
    data_w=True)

left, right = fig.row(2, y=104, h=376)
X_AX = (-1.5, 1.5, [-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
LABEL_W = 112
fx = lambda v: num(v)  # noqa: E731

body = fig.panel(left, "Penalty on the weight")
ax = fig.line_chart(body,
                    [dict(xs=list(W), ys=list(P1), color="ink", points=False),
                     dict(xs=list(W), ys=list(P2), color="blue", points=False)],
                    x=X_AX, y=(0, 0.025, [0, 0.01, 0.02]), labels=False, label_w=LABEL_W,
                    x_label=rich("weight ", w), y_label="penalty", fmt_x=fx, fmt_y=lambda v: f"{v:g}")
ax.text(1.5, LAM * 1.5, rich("L1, ", lam, "|", w, "|"), "label", color="ink", dx=12, dy=5)
ax.text(1.5, LAM * 1.5 ** 2, rich("L2, ", lam, sup("w", "2")), "label", color="blue", dx=12, dy=5)

body = fig.panel(right, "Gradient of the penalty")
ax = fig.line_chart(body,
                    [dict(xs=list(NEG), ys=list(G1_NEG), color="ink", points=False),
                     dict(xs=list(POS), ys=list(G1_POS), color="ink", points=False),
                     dict(xs=list(W), ys=list(G2), color="blue", points=False)],
                    x=X_AX, y=(-0.03, 0.03, [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]), labels=False,
                    label_w=LABEL_W, x_label=rich("weight ", w), y_label="gradient",
                    fmt_x=fx, fmt_y=lambda v: num(f"{v:g}"))
ax.text(1.5, LAM, rich("L1, ", lam, " sign(", w, ")"), "label", color="ink", dx=12, dy=5)
ax.text(1.5, 2 * LAM * 1.5, rich("L2, 2", lam, w), "label", color="blue", dx=12, dy=5)
ax.point(0, -LAM, "circle", "ink", size=8, hollow=True)
ax.point(0, G1_ZERO, "circle", "ink", size=8)
ax.text(0, G1_ZERO, rich("+", lam, " at ", w, " = 0"), "note", anchor="end", dx=-12, dy=5)
for v in (-0.5, 0.5):
    ax.point(v, 2 * LAM * v, "diamond", "blue", size=10)
ax.text(0.5, LAM, rich("equal at ", w, " = 0.5"), "note", dx=4, dy=24)
ax.text(-0.5, -LAM, rich("and at ", w, " = ", num(-0.5)), "note", dx=4, dy=24)

fig.caption(rich("Below |", w, "| = 0.5 the L1 gradient is the larger, above it the L2 gradient."))
fig.write()
