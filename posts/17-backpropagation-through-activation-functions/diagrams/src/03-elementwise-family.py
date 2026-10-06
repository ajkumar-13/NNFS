"""Post 17 section 3: ReLU, sigmoid and tanh with their slopes, and the one backward line they share.

Run from anywhere:  python posts/17-backpropagation-through-activation-functions/diagrams/src/03-elementwise-family.py
Writes posts/17-backpropagation-through-activation-functions/diagrams/03-elementwise-family.svg.
The curves f and f' are evaluated here from the formulas of the post's section 3 table (whose rows are asserted
against index.md): max(0, z) with slope 1 for z > 0 and 0 otherwise, sigma(z) = 1 / (1 + e^-z) with slope
sigma(z)(1 - sigma(z)), and tanh(z) with slope 1 - tanh^2(z). The tables under the charts come from the snippet's
three classes (snippets/elementwise_backward.py, run here) on Z = [1, -2, 3] and dvalues = [5, 6, 7]; they are
asserted against the formulas and against the lines the snippet prints for section 3.
"""
import contextlib
import io
import re
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num  # noqa: E402

PRIME, MINUS, SIG = "\u2032", "\u2212", "\u03c3"
POST = Path(__file__).resolve().parents[2]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    E = runpy.run_path(str(POST / "snippets" / "elementwise_backward.py"), run_name="snippet")
OUT = buf.getvalue()
INDEX = (POST / "index.md").read_text(encoding="utf-8")
for row in (r"| **ReLU** | $\max(0, z)$ | $1$ if $z > 0$, else $0$ | mask: zero where $z \le 0$ |",
            r"| **Sigmoid** | $\sigma(z) = 1 / (1 + e^{-z})$ | $\sigma(z) \, (1 - \sigma(z))$ | multiply by $a \, (1 - a)$ |",
            r"| **Tanh** | $(e^z - e^{-z}) / (e^z + e^{-z})$ | $1 - \tanh^2(z)$ | multiply by $1 - a^2$ |"):
    assert row in INDEX, row


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


FUNCS = {   # name: (f, f')
    "ReLU": (lambda z: np.maximum(0, z), lambda z: np.where(z > 0, 1.0, 0.0)),
    "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
}
CLASSES = {"ReLU": "Activation_ReLU", "sigmoid": "Activation_Sigmoid", "tanh": "Activation_Tanh"}

Z = np.array([[1.0, -2.0, 3.0]])
DV = np.array([[5.0, 6.0, 7.0]])
TABLE = {}
for name, (f, fp) in FUNCS.items():
    act = E[CLASSES[name]]()
    act.forward(Z)
    act.backward(DV)
    assert np.allclose(act.output, f(Z)) and np.allclose(act.dinputs, DV * fp(Z))
    slope = act.dinputs[0] / DV[0]
    line = (f"{name:8} {np.array2string(slope, precision=6, floatmode='fixed'):>31}   "
            f"{np.array2string(act.dinputs[0], precision=5, floatmode='fixed'):>31}")
    assert line in OUT, line
    TABLE[name] = (slope, act.dinputs[0])
assert [f"{v:.6f}" for v in TABLE["sigmoid"][0]] == ["0.196612", "0.104994", "0.045177"]
assert [f"{v:.5f}" for v in TABLE["tanh"][1]] == ["2.09987", "0.42390", "0.06906"]
ZS = np.linspace(-4, 4, 161)
assert abs(FUNCS["sigmoid"][1](np.array(0.0)) - 0.25) < 1e-15 and FUNCS["tanh"][1](np.array(0.0)) == 1.0


def fmt3(v, integer):
    return num(int(round(v))) if integer else num(float(f"{v:.3f}"), 3)


FP = rich(var("f"), " ", PRIME)        # a thin space keeps the prime clear of the italic f


def fz(s):
    return rich(s, "(", var("z"), ")")


fig = Figure(
    "03-elementwise-family", "One backward line, three slopes",
    "Three panels, each plotting an activation f(z) in green and its slope f prime(z) in purple for z from minus 4 "
    "to 4, with the slope marked at the inputs z = 1, minus 2 and 3. ReLU: f = max(0, z); its slope steps from 0 "
    "to 1 at z = 0, with a solid dot at 0 and an open dot at 1 for z = 0 and a label f prime(0) = 0; for z < 0 the green f is drawn over the purple slope, both 0. Sigmoid: its slope a(1 minus a), with a = sigma(z), peaks at 0.25. Tanh: its slope 1 minus "
    "a squared, with a = tanh(z), peaks at 1. Under each chart a table gives, for z = 1, minus 2, 3, the slope and dinputs = "
    "dvalues times the slope with dvalues = 5, 6, 7. ReLU: slopes 1, 0, 1 and dinputs 5, 0, 7. Sigmoid: 0.197, 0.105, "
    "0.045 and 0.983, 0.630, 0.316. Tanh: 0.420, 0.071, 0.010 and 2.100, 0.424, 0.069. A line at the bottom gives the "
    "code all three share, dinputs = dvalues * f_prime(Z). Table values are rounded to 3 decimals.",
    subtitle=rich("Green: the activation ", fz(var("f")), ". Purple: its slope ", fz(FP), ". Dots: the slope at ",
                  var("z"), " = 1, ", MINUS, "2, 3."),
    height=720, data_w=True)

panels = fig.row(3)
SPEC = {
    "ReLU": ("ReLU", rich(FP, " = 1 if ", var("z"), " > 0, else 0"), (-0.5, 4, [0, 1, 2, 3, 4]), True),
    "sigmoid": ("Sigmoid", rich(FP, " = ", var("a"), "(1 ", MINUS, " ", var("a"), "),  ", var("a"), " = ", SIG, "(",
                                var("z"), ")"), (-0.125, 1, [0, 0.25, 0.5, 0.75, 1]), False),
    "tanh": ("Tanh", rich(FP, " = 1 ", MINUS, " ", sup(var("a"), "2"), ",  ", var("a"), " = tanh(", var("z"), ")"),
             (-1, 1, [-1, -0.5, 0, 0.5, 1]), False),
}
CHART_TOP, CHART_H = 160, 248
TABLE_Y = CHART_TOP + CHART_H + 40
for box, (name, (f, fp)) in zip(panels, FUNCS.items()):
    head, rule, (ylo, yhi, yt), integer = SPEC[name]
    body = fig.panel(box, head)
    fig.text(body.x, body.y + 12, rule, "note")
    fy = lambda t: f"{t:g}".replace("-", MINUS)  # noqa: E731
    series = [dict(xs=ZS.tolist(), ys=np.clip(f(ZS), ylo, yhi).tolist(), color="output", points=False,
                   label=var("f"))]
    if name != "ReLU":
        series.append(dict(xs=ZS.tolist(), ys=fp(ZS).tolist(), color="gradient", points=False, label=FP))
    ax = fig.line_chart(Box(box.x, CHART_TOP, box.w, CHART_H), series, x=(-4, 4, [-4, -2, 0, 2, 4]),
                        y=(ylo, yhi, yt), x_label=var("z"), fmt_y=fy, label_w=24)
    with fig.data():
        if name == "ReLU":                       # the step, in two pieces; 0 at z = 0, as the series uses
            ax.segment(-4, 0, 0, 0, "gradient")
            ax.segment(0, 1, 4, 1, "gradient")
            ax.segment(-4, 0, 0, 0, "output")    # f drawn again over f' where both are 0
            ax.point(0, 0, "circle", "gradient", size=10)                 # the slope used at z = 0
            ax.point(0, 1, "circle", "gradient", size=10, hollow=True)    # not used
            ax.text(1.2, 0.25, rich(FP, "(0) = 0"), "note", color="gradient", dy=5)
            px, py = ax.to_px(4, 1)
            fig.text(px + 12, py + 5, FP, "note", color="gradient", snap=False)
        for z, s in zip(Z[0], fp(Z[0])):
            ax.point(z, s, "circle", "gradient", size=10)
    slope, din = TABLE[name]
    rows = [[var("z"), fz(FP), ""]]
    for z, s, dv in zip(Z[0], slope, din):
        rows.append([num(int(z)), fmt3(s, integer), fmt3(dv, integer)])
    t = fig.table(box.x + 56, TABLE_Y, rows, [56, 72, 88], row_h=32, head_style="label",
                  col_align=["end", "end", "end"])
    fig.text(t.right - 8, TABLE_Y + 21, "dinputs", "code", anchor="end")   # the code name, in the code face

fig.text(40, TABLE_Y - 16, "Table values rounded to 3 decimals.", "note")

# the shared line
YC = TABLE_Y + 4 * 32 + 48
fig.text(480, YC, "dinputs = dvalues * f_prime(Z)", "code", anchor="middle")
fig.text(480, YC + 24, "dvalues = (5, 6, 7) in all three tables; only f_prime changes", "note", anchor="middle")

fig.caption("A new element-wise activation costs one derivative; the backward line stays the same.")
fig.write()
