"""Post 25, sections 1 and 2: on L = w1^2/100 + w2^2, one learning rate against AdaGrad's per-parameter rates.

Run from anywhere:  python posts/25-adagrad/diagrams/src/01-one-rate-two-weights.py
Writes posts/25-adagrad/diagrams/01-one-rate-two-weights.svg.
snippets/adagrad.py is imported here (runpy, without its main block): its Optimizer_SGD, Optimizer_Adagrad and
toy_layer follow the two weights from (1, 1) for 100 steps with alpha = 0.1, one step at a time, on the loss of
section 1 (gradient [0.02, 2.0] * w, the snippet's bowl). The curves are those steps. The dots sit at the steps
the snippet prints, and every printed value is asserted against the listings of sections 1 and 2 in index.md,
which the series lint checks against the snippet's output.

Layout: two charts side by side, the weight against the step, gradient descent left and AdaGrad right.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path(str(POST / "snippets" / "adagrad.py"), run_name="snippet")
INDEX = (POST / "index.md").read_text(encoding="utf-8")

SCALES = np.array([[2 / 100, 2.0]])              # L = w1^2 / 100 + w2^2, so dL/dw = SCALES * w
assert "$$L(w_1, w_2) = \\frac{w_1^2}{100} + w_2^2$$" in INDEX
ALPHA, STEPS = 0.1, 100


def path(optimizer):
    """The two weights after each of STEPS updates, starting at (1, 1); the toy layer at the end."""
    layer = s["toy_layer"]([1.0, 1.0])
    out = [layer.weights[0].copy()]
    for _ in range(STEPS):
        layer.dweights = SCALES * layer.weights
        optimizer.pre_update_params()
        optimizer.update_params(layer)
        optimizer.post_update_params()
        out.append(layer.weights[0].copy())
    return np.array(out), layer


GD, _ = path(s["Optimizer_SGD"](learning_rate=ALPHA))
AG, AG_LAYER = path(s["Optimizer_Adagrad"](learning_rate=ALPHA))

# -- the printed lines of sections 1 and 2, reproduced from the same steps
assert np.allclose(SCALES * np.array([[1.0, 1.0]]), [[0.02, 2.0]])
assert (f"gradient descent, alpha =  0.1: first steps {1 - GD[1, 0]:.4g} and {1 - GD[1, 1]:.4g}; after 100 steps "
        f"w1 = {GD[100, 0]:.4f}, w2 = {GD[100, 1]:.4g}") in INDEX
for k in (1, 10, 100):
    assert f"after {k:3d} steps w1 = {AG[k, 0]:.6f}, w2 = {AG[k, 1]:.6f}" in INDEX, k
assert (f"after 100 steps w1 = {AG[100, 0]:.6f}, w2 = {AG[100, 1]:.6f}, caches {AG_LAYER.weight_cache[0, 0]:.4g} "
        f"and {AG_LAYER.weight_cache[0, 1]:.4g}") in INDEX
assert (f"{GD[100, 0]:.4f}", f"{GD[100, 1]:.4g}", f"{AG[100, 0]:.6f}") == ("0.8186", "2.037e-10", "0.027387")
assert (f"{GD[1, 1]:.4g}", f"{AG[10, 0]:.4f}") == ("0.8", "0.5537")           # the two notes
assert round(100 * (1 - GD[100, 0])) == 18                        # "w1 has covered 18 percent of its way"
assert np.max(np.abs(AG[:, 0] - AG[:, 1])) < 1e-5                 # the two AdaGrad weights travel together
C1, C2 = f"{AG_LAYER.weight_cache[0, 0]:.4g}", f"{AG_LAYER.weight_cache[0, 1]:.4g}"
assert (C1, C2) == ("0.004011", "40.11")
assert round(AG_LAYER.weight_cache[0, 1] / AG_LAYER.weight_cache[0, 0]) == 10000   # the caption: 10,000 times
assert "The caches differ by a factor of 10,000, their roots by 100" in INDEX

W1, W2 = sub("w", "1"), sub("w", "2")
fig = Figure(
    "01-one-rate-two-weights", "One rate strands the slow weight; AdaGrad moves both alike",
    "Two charts of the weights w1 and w2 of the loss w1 squared over 100 plus w2 squared, against the step, from "
    "the start (1, 1), where the gradients are 0.02 and 2.0, for 100 steps with a learning rate of 0.1. Dots mark "
    "the steps the script prints. Gradient descent, left: w2 falls to 0.8 after one step and is at 2.037 times 10 "
    "to the minus 10 after 100, while w1 moves 0.002 in the first step and ends at 0.8186. AdaGrad, right: the "
    "two weights take the same path, 0.9 after one step, 0.5537 after 10, and both end at 0.027387; their caches "
    "after 100 steps are 0.004011 and 40.11.",
    subtitle=rich(var("L"), " = ", sup(W1, "2", italic=False), "/100 + ", sup(W2, "2", italic=False),
                  " from (1, 1), where the gradients are 0.02 and 2.0. Both runs: ", var("α"), " = 0.1, 100 steps."),
    data_w=True)

left, right = fig.row(2, y=104, h=336)
X_AX = (0, STEPS, [0, 25, 50, 75, 100])
Y_AX = (0, 1, [0, 0.25, 0.5, 0.75, 1])
STEP_X = list(range(STEPS + 1))
LABEL_W = 112


def chart(box, heading, data, dots):
    body = fig.panel(box, heading)
    ax = fig.line_chart(body,
                        [dict(xs=STEP_X, ys=list(data[:, 1]), color="ink", points=False),
                         dict(xs=STEP_X, ys=list(data[:, 0]), color="weight", points=False)],
                        x=X_AX, y=Y_AX, labels=False, label_w=LABEL_W, x_label="step",
                        y_label="value of the weight", fmt_y=lambda v: f"{v:g}")
    for k in dots:
        ax.point(k, data[k, 1], "square", "ink", size=10, hollow=True)
        ax.point(k, data[k, 0], "circle", "weight", size=8)
    return ax


ax = chart(left, rich("Gradient descent, ", var("α"), " = 0.1"), GD, (1, 100))
ax.text(100, GD[100, 0], rich(W1, " = 0.8186"), "value", color="weight", dx=12, dy=5)
ax.text(100, GD[100, 1], rich(W2, " = 2.0 × ", sup("10", num(-10), italic=False)), "value", color="ink",
        dx=12, dy=5)
ax.text(1, GD[1, 1], rich(W2, " = 0.8 after one step"), "note", dx=12, dy=5)

ax = chart(right, rich("AdaGrad, ", var("α"), " = 0.1"), AG, (1, 10, 100))
ax.text(100, AG[100, 0], "0.027387", "value", color="weight", dx=12, dy=5)
ax.text(100, AG[100, 0], rich(W1, " and ", W2), "note", dx=12, dy=-17)
ax.text(10, AG[10, 0], rich("0.5537 after 10 steps"), "note", dx=12, dy=-6)
ax.text(50, 0.85, rich("caches after 100 steps:"), "note", anchor="middle")
ax.text(50, 0.85, rich(W1, " ", C1, ", ", W2, " ", C2), "note", anchor="middle", dy=20)

fig.legend(96, 468, [dict(color="weight", label=rich(W1, ", gradient 0.02 at the start"), mark="circle"),
                     dict(color="ink", label=rich(W2, ", gradient 2.0 at the start"), mark="square", hollow=True),
                     ], direction="row", gap=40)
fig.caption("Dots: the steps the script prints. The caches differ 10,000 times, their roots 100 times.")
fig.write()
