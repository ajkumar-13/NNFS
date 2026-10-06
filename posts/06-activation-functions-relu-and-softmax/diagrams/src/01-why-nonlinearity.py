"""Post 06 hero (section 1.1): the post's two small layers run without and with a ReLU between them.

Run from anywhere:  python posts/06-activation-functions-relu-and-softmax/diagrams/src/01-why-nonlinearity.py
Writes posts/06-activation-functions-relu-and-softmax/diagrams/01-why-nonlinearity.svg.
snippets/why_nonlinearity.py is run (runpy) for its weights, biases and nine outputs. This script then evaluates
both networks itself, from those weights, on a fine grid of inputs, and asserts: that the fine curves agree with
the nine printed outputs; that the polyline through the nine points is the curve exactly (every kink falls on one
of them); where the kinks are (x = -b / w of each hidden neuron); and the slopes the snippet prints.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, isub, arr, num, MINUS  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "why_nonlinearity.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    S = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
W1, B1, W2, B2 = S["W1"], S["b1"], S["W2"], S["b2"]
XS = S["X"].ravel()
LIN, RELU = S["no_activ"].ravel(), S["with_relu"].ravel()
W_STAR, B_STAR = float(S["W_star"].ravel()[0]), float(S["b_star"].ravel()[0])
N_PARAMS = W1.size + B1.size + W2.size + B2.size
assert N_PARAMS == 10 and XS.tolist() == [-1.0 + 0.5 * k for k in range(9)]
assert (W_STAR, B_STAR) == (1.0, -2.0) and "W_star = [1.]  b_star = [-2.]" in OUT
for x, a, c in zip(XS, LIN, RELU):                         # the table of section 1.1, row by row
    assert f"{x:5.1f}   {a:13.1f}   {a:9.1f}   {c:9.1f}" in OUT


def two_layers(x, activation):
    """The snippet's network on a column of inputs, evaluated here from its own weights."""
    z1 = np.dot(np.asarray(x, float).reshape(-1, 1), W1) + B1
    return (np.dot(activation(z1), W2) + B2).ravel()


fine = np.linspace(-1.0, 3.0, 801)
lin_f = two_layers(fine, lambda z: z)
relu_f = two_layers(fine, lambda z: np.maximum(0, z))
assert np.array_equal(two_layers(XS, lambda z: z), LIN) and np.array_equal(two_layers(XS, lambda z: np.maximum(0, z)), RELU)
assert np.allclose(lin_f, W_STAR * fine + B_STAR)                     # one straight line, x - 2
assert np.allclose(relu_f, np.interp(fine, XS, RELU))                 # the nine-point polyline is the curve
KINKS = sorted((-B1 / W1).ravel().tolist())                            # where each hidden neuron's z crosses 0
assert KINKS == [0.0, 1.0, 2.0]
SEG = [(-1, 0), (0, 1), (1, 2), (2, 3)]                                # the pieces between the kinks
SLOPES = [float(np.interp(b, XS, RELU) - np.interp(a, XS, RELU)) / (b - a) for a, b in SEG]
assert SLOPES == [0.0, 1.0, -1.0, 1.0]
assert "slopes without an activation: [1.]" in OUT and "slopes with ReLU:             [-1.  0.  1.]" in OUT
assert "f(0) = 0.0  f(2) = 0.0  their midpoint = 0.0  but f(1) = 1.0" in OUT


def slope_txt(s):
    return rich("slope ", num(int(s)))


# The output as the second layer forms it from the hidden activations a_k: output = sum of W2[k] a_k + b2.
w2 = [int(v) for v in W2.ravel()]
assert w2 == [1, -2, 2] and float(B2.ravel()[0]) == 0.0
assert SLOPES[1:] == np.cumsum(w2).tolist()       # past each kink the slope gains that neuron's weight out
terms = []
for k, w in enumerate(w2):
    a = sub("a", str(k + 1))
    if k == 0:
        terms.append(rich(a) if w == 1 else rich(num(w), a))
    else:
        terms.append(rich(" ", "+" if w > 0 else MINUS, " ", str(abs(w)) if abs(w) != 1 else "", a))
OUTPUT_EQ = rich("output = ", *terms)


fig = Figure(
    "01-why-nonlinearity", "Two layers make a line; with ReLU between them, a zigzag",
    "Two line charts of the output against the input x, from minus 1 to 3, for the post's network of one input, "
    f"three hidden neurons and one output, {N_PARAMS} parameters, with the nine outputs of the snippet as dots. "
    "Left, without an activation: the two layers collapse to one layer with W star = 1 and b star = minus 2, and "
    "the output is the straight line x minus 2, from minus 3 to 1, written output = x minus 2, slope 1, in the "
    "empty upper left. Right, with ReLU between "
    "the same layers: the output is 0 up to x = 0, rises to 1 at x = 1, falls to 0 at x = 2 and rises to 1 at "
    "x = 3, slopes 0, 1, minus 1 and 1. Diamonds mark the three kinks, at x = 0, 1 and 2, one per hidden neuron, "
    "where its weighted sum crosses zero. Under the zigzag: output = a1 minus 2 a2 plus 2 a3, with a k = "
    "max(0, x + b k) and b1 = (0, minus 1, minus 2).",
    subtitle=rich("One input ", var("x"), ", three hidden neurons, one output: the same ", num(N_PARAMS),
                  " parameters in both panels."))

left, right = fig.row(2)
X_AX = (-1, 3, [-1, 0, 1, 2, 3])
Y_AX = (-3, 1, [-3, -2, -1, 0, 1])
CHART_TOP = 168
LABEL_W = 24                                     # no end labels: the equations name the curves


def chart(panel, heading, note, ys, label):
    body = fig.panel(panel, heading)
    fig.text(body.x, body.y + 4, note, "note")
    box = Box(panel.x, CHART_TOP, panel.w, panel.bottom - CHART_TOP)
    return fig.line_chart(box, [dict(xs=XS.tolist(), ys=ys.tolist(), color="output", label=label)],
                          x=X_AX, y=Y_AX, x_label=rich("input ", var("x")), y_label="output",
                          label_w=LABEL_W)


ax_l = chart(left, "Without an activation",
             rich("The two collapse to one layer: ", arr("W", sub="\u2217"), " = ", num(W_STAR), ", ",
                  arr("b", sub="\u2217"), " = ", num(B_STAR)),
             LIN, None)
ax_r = chart(right, "With ReLU between the layers",
             rich("Diamonds: kinks at ", var("x"), " = 0, 1, 2, one per hidden neuron"),
             RELU, None)

with fig.data():
    # left: the one line's equation and slope, in the empty upper left, as the right panel writes its output
    ax_l.text(-0.85, 0.45, rich("output = ", var("x"), " ", MINUS, " 2"), "math")
    ax_l.text(-0.85, 0.05, slope_txt(W_STAR), "note")
    # right: the kinks, then each piece's slope on one baseline under the zigzag
    for k in KINKS:
        kx, ky = ax_r.to_px(k, float(np.interp(k, XS, RELU)))
        fig.marker(kx, ky, "diamond", "weight", size=12)
    for (a, b), s in zip(SEG, SLOPES):
        px, py = ax_r.to_px((a + b) / 2, -0.75)
        fig.text(px, py, slope_txt(s), "note", anchor="middle", snap=False)
    # right, in the space under the zigzag: the output as the second layer forms it, and the hidden neurons
    px, py = ax_r.to_px(1.0, -1.4)                 # the two lines as one block, close under the slopes
    fig.text(px, py, OUTPUT_EQ, "math", anchor="middle", snap=False)
    px, py = ax_r.to_px(1.0, -1.85)
    fig.text(px, py, rich(isub("a", "k"), " = max(0, ", var("x"), " + ", isub("b", "k"), "),  ",
                          arr("b", sub="1"), " = (", ", ".join(num(v) for v in B1.ravel()), ")"), "note",
             anchor="middle", snap=False)

fig.caption(rich("Without the activation the two layers are the line ", var("x"), " ", MINUS,
                 " 2; ReLU gives one bend per hidden neuron."))
fig.write()
