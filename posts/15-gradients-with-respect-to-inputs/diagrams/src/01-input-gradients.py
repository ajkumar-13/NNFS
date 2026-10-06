"""Post 15 hero (section 2): a weight lies on one path to the loss, an input on one path per neuron.

Run from anywhere:  python posts/15-gradients-with-respect-to-inputs/diagrams/src/01-input-gradients.py
Writes posts/15-gradients-with-respect-to-inputs/diagrams/01-input-gradients.svg.
snippets/input_gradient.py is run here (runpy): the inputs, the weights, the upstream gradient 43.2 and the two
worked results (dL/dw11 = 43.2 and dL/dx1 = 64.80) are read from it and asserted against the lines it prints and
the numbers the post states in sections 2 and 4.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Raw, CDOT, SIGMA, rich, var, sub, isub, span  # noqa: E402


def wk1():
    """w with the subscript k1: the neuron index k italic (a variable), the input index 1 upright (a digit)."""
    return rich(var("w"), Raw('<tspan class="sub" dy="4"><tspan class="i">k</tspan>1</tspan><tspan dy="-4">​</tspan>'))

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "input_gradient.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

X, weights, dZ, dX = s["X"], s["weights"], s["dL_dZ"], s["dL_dX"]
assert X.tolist() == [[1.0, 2.0, 3.0, 4.0]] and weights.shape == (3, 4)
assert np.allclose(dZ, [[43.2, 43.2, 43.2]])
G = f"{dZ[0, 0]:.1f}"                                        # the upstream entry of every neuron
W1 = [f"{weights[k, 0]:.1f}" for k in range(3)]              # the weights leaving x1
assert G == "43.2" and W1 == ["0.1", "0.5", "0.9"]
DW11 = dZ[0, 0] * X[0, 0]
assert f"for contrast, a weight has one path: dL/dw11 = dL/dz1 * x1 = 43.2 * 1 = {DW11:.1f}" in OUT
assert f"{DW11:.1f}" == "43.2"
DX1 = f"{dX[0, 0]:.2f}"
assert "dL/dx1 = 43.2 * 0.1 + 43.2 * 0.5 + 43.2 * 0.9 = 64.80" in OUT and DX1 == "64.80"
assert np.isclose(dX[0, 0], np.sum(dZ[0] * weights[:, 0]))


def dL(of):
    """The partial derivative of the loss with respect to `of`, as the post writes it: dL/d(of)."""
    return rich("∂", var("L"), "/∂", of)


def grad(t):
    return span(t, color="gradient")


def wt(t):
    return span(t, color="weight")


fig = Figure(
    "01-input-gradients", "A weight has one path to the loss, an input has three",
    "Two copies of the layer of posts 13 and 14: four input nodes x1 to x4 joined to three neurons z1 to z3, every "
    f"neuron with the upstream gradient {G} beside it. Left, the weight w11: one path, from x1 into z1, is drawn in "
    f"the gradient colour, and its gradient is dL/dz1 times x1, {G} times 1 = {DW11:.1f}; the upstream values of "
    "z2 and z3 are muted, with a bracket noting that w11 is not in z2 or z3. Right, the input x1: three "
    "paths, into z1, z2 and z3, are drawn in the gradient colour, a column beside the neurons lists the weights "
    "on them, wk1 = 0.1, 0.5 and 0.9, and its gradient is the "
    f"sum over k of dL/dzk times wk1, {G} times 0.1 plus {G} times 0.5 plus {G} times 0.9 = {DX1}.",
    subtitle=rich("The layer of posts 13 and 14 at ", var("x"), " = (1, 2, 3, 4). Every ReLU gate is open, so each "
                  "neuron's ", dL(isub("z", "k")), " = ", G, "."))

R = 20
YS_IN = [184, 240, 296, 352]
YS_N = [212, 268, 324]
LEFT, RIGHT = fig.row(2)


def layer(panel, paths, heading, used=(0, 1, 2)):
    """One copy of the layer: edges (the highlighted paths in the gradient colour, last), nodes, upstream values."""
    x0 = panel.x
    cx_in, cx_n = x0 + 40, x0 + 232
    fig.text(x0, 136, heading, "head")
    for k_hl in (False, True):
        for j, yi in enumerate(YS_IN):
            for k, yn in enumerate(YS_N):
                hl = (j, k) in paths
                if hl == k_hl:
                    fig.edge((cx_in + R, yi), (cx_n - R, yn), *(("gradient", 1.5) if hl else ("rule", 1)))
    for j, yi in enumerate(YS_IN):
        fig.node(cx_in, yi, r=R, label=sub("x", str(j + 1)), color="input", width=2.5 if (j, 0) in paths and len(paths) > 1 else 1.5)
    for k, yn in enumerate(YS_N):
        fig.node(cx_n, yn, r=R, label=sub("z", str(k + 1)), color="output")
        if k in used:
            fig.text(cx_n + 40, yn + 5, G, "value", color="gradient")
        else:
            fig.text(cx_n + 40, yn + 5, G, "note")
    fig.text(cx_n + 40, 176, dL(isub("z", "k")), "label", color="gradient")
    return cx_in, cx_n


# -- left: the weight w11, one path
a_in, a_n = layer(LEFT, {(0, 0)}, rich("A weight: ", sub("w", "11"), " lies on one path"), used=(0,))
fig.brace(YS_N[1] - 16, YS_N[2] + 16, a_n + 96, side="right", kind="bracket", vertical=True)
fig.text(a_n + 112, (YS_N[1] + YS_N[2]) / 2 - 4, rich(sub("w", "11"), " is not"), "note")
fig.text(a_n + 112, (YS_N[1] + YS_N[2]) / 2 + 16, rich("in ", sub("z", "2"), " or ", sub("z", "3")), "note")
fig.text((a_in + a_n) / 2, 184, wt(sub("w", "11")), "label", anchor="middle")
fig.text(LEFT.x, 420, rich(dL(sub("w", "11")), " = ", dL(sub("z", "1")), " ", CDOT, " ", sub("x", "1")), "math")
fig.text(LEFT.x, 452, rich(grad(G), " ", CDOT, " ", span(f"{X[0, 0]:.0f}", color="input"), " = ", span(f"{DW11:.1f}", color="gradient", bold=True)), "math")

# -- right: the input x1, three paths; the weight on each path in a column beside its neuron
b_in, b_n = layer(RIGHT, {(0, 0), (0, 1), (0, 2)}, rich("An input: ", sub("x", "1"), " lies on three paths"))
fig.text(b_n + 128, 176, wk1(), "label", color="weight")
for k, yn in enumerate(YS_N):
    fig.text(b_n + 128, yn + 5, W1[k], "value", color="weight")
sum_k = rich(SIGMA, Raw('<tspan class="sub i" dy="4">k</tspan><tspan dy="-4">​</tspan>'))
fig.text(RIGHT.x, 420, rich(dL(sub("x", "1")), " = ", sum_k, " ", dL(isub("z", "k")), " ", CDOT, " ",
                            wk1()), "math")
terms = []
for k, w in enumerate(W1):
    terms += ([" + "] if k else []) + [grad(G), f" {CDOT} ", wt(w)]
fig.text(RIGHT.x, 452, rich(*terms, " = ", span(DX1, color="gradient", bold=True)), "math")

fig.caption("Each path adds one product; the input's gradient adds one per neuron it feeds.")
fig.write()
