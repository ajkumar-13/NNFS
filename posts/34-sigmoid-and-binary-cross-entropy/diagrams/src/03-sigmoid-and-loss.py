"""Post 34, sections 2 and 3.1: the sigmoid and its slope, and binary cross-entropy against the logit with the clip.

Run from anywhere:  python posts/34-sigmoid-and-binary-cross-entropy/diagrams/src/03-sigmoid-and-loss.py
Writes posts/34-sigmoid-and-binary-cross-entropy/diagrams/03-sigmoid-and-loss.svg. Takes about a second.

The curves are the formulas the post states, evaluated with the snippets' own functions: the sigmoid is
Activation_Sigmoid of snippets/binary_classes.py, its slope that class's backward (sigma (1 - sigma)), and the loss
the clipped and the from-logits forms of snippets/loss_forms.py (both formulas asserted against index.md). The
marked points are the values the two snippets print, which index.md quotes in sections 2 and 3.1: the five logits
of section 2, and the seven logits of section 3.1 for label 1. The clip bound 16.118 is loss_forms.py's printed
value. The mirror statement (label 0 at z equals label 1 at -z) is checked on the printed rows.

Layout: two charts side by side. Left, sigma(z) (output hue) and its slope (gradient hue) over z from -6 to 6.
Right, the loss of one sample with label 1 over z from -40 to 40: the exact loss, dashed, and the clipped loss
the class reports, solid red, which leaves the exact one where the clip starts.
"""
import contextlib
import io
import re
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, MINUS, rich, var, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIP = POST / "snippets"
sys.path.insert(0, str(SNIP))
INDEX = (POST / "index.md").read_text(encoding="utf-8")


def run(name, run_name="snippet"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns = runpy.run_path(str(SNIP / name), run_name=run_name)
    return ns, buf.getvalue()


BC, OUT_BC = run("binary_classes.py", "__main__")
LF, OUT_LF = run("loss_forms.py")

# -- the formulas, as the post states them
assert "$$\\hat{y} = \\sigma(z) = \\frac{1}{1 + e^{-z}}$$" in INDEX
assert "**Slope.** $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$" in INDEX
assert "$$L_i = -\\bigl[\\, y \\log \\hat{y} + (1 - y) \\log(1 - \\hat{y}) \\,\\bigr]$$" in INDEX
assert "gives $L_i = \\log(1 + e^{z}) - y z$" in INDEX


def sigmoid(z):
    a = BC["Activation_Sigmoid"]()
    a.forward(np.asarray(z, dtype=float).reshape(1, -1))
    a.backward(np.ones_like(a.output))
    return a.output[0], a.dinputs[0]


Z1 = np.linspace(-6, 6, 241)
S1, D1 = sigmoid(Z1)
assert np.allclose(D1, S1 * (1 - S1)) and np.allclose(S1, 1 / (1 + np.exp(-Z1)))

# section 2's printed points
P_Z = [-5.0, -2.0, 0.0, 2.0, 5.0]
ps, pd = sigmoid(P_Z)
for line in ("z                    [-5.0000 -2.0000  0.0000  2.0000  5.0000]",
             "sigma(z)             [0.0067 0.1192 0.5000 0.8808 0.9933]",
             "slope                [0.0066 0.1050 0.2500 0.1050 0.0066]"):
    assert line in OUT_BC and line in INDEX, line
assert [f"{v:.4f}" for v in ps] == ["0.0067", "0.1192", "0.5000", "0.8808", "0.9933"]
assert [f"{v:.4f}" for v in pd] == ["0.0066", "0.1050", "0.2500", "0.1050", "0.0066"]

# -- the loss for label 1
bce_clipped, bce_from_logits = LF["bce_clipped"], LF["bce_from_logits"]
Z2 = np.linspace(-40, 40, 801)
EXACT = bce_from_logits(Z2, 1.0)
CLIP = bce_clipped(sigmoid(Z2)[0], 1.0)
assert np.allclose(EXACT, np.log1p(np.exp(Z2)) - Z2, atol=1e-12)           # log(1 + e^z) - y z at y = 1
BOUND = float(np.log((1 - 1e-7) / 1e-7))
CAP = float(-np.log(1e-7))
assert f"{BOUND:.3f}" == f"{CAP:.3f}" == "16.118"
assert "the clip changes y_hat when |z| exceeds log((1 - 1e-7) / 1e-7) = 16.118" in OUT_LF
assert "largest loss the clipped form can report, -log(1e-7): 16.118" in OUT_LF
assert np.isclose(CLIP.max(), CAP)

# section 3.1's printed rows, label 1 and label 0
rows = {}
for m in re.finditer(r"^y = (\d)  (unclipped|clipped|from logits) +((?:\s+\S+){7})", OUT_LF, re.M):
    rows[(m.group(1), m.group(2))] = m.group(3).split()
T_Z = [-40.0, -17.0, -5.0, 0.0, 5.0, 17.0, 40.0]
assert rows[("1", "clipped")] == ["16.12", "16.12", "5.007", "0.6931", "0.006715", "1e-07", "1e-07"]
assert rows[("1", "from logits")] == ["40", "17", "5.007", "0.6931", "0.006715", "4.14e-08", "4.248e-18"]
assert rows[("1", "unclipped")][-1] == "nan"
for form in ("unclipped", "clipped", "from logits"):
    if form != "unclipped":
        assert rows[("0", form)] == rows[("1", form)][::-1], form       # label 0 at z is label 1 at -z
for t in rows:
    line = next(l for l in OUT_LF.splitlines() if l.startswith(f"y = {t[0]}  {t[1]}"))
    assert line.split("   warnings")[0].rstrip() in INDEX, line
TC = bce_clipped(sigmoid(T_Z)[0], 1.0)
TE = bce_from_logits(np.array(T_Z), 1.0)
assert [f"{v:.4g}" for v in TC] == rows[("1", "clipped")]
assert [f"{v:.4g}" for v in TE] == rows[("1", "from logits")]

z = var("z")
fig = Figure(
    "03-sigmoid-and-loss", "The sigmoid squashes the logit; the clip caps the loss",
    "Two charts. Left, over z from minus 6 to 6: the sigmoid rises from near 0 to near 1 and is 0.5 at z = 0; its "
    "slope sigma times 1 minus sigma peaks at 0.25 at z = 0. Points mark the printed values at z = minus 5, minus "
    "2, 0, 2 and 5: sigma 0.0067, 0.1192, 0.5000, 0.8808, 0.9933 and slope 0.0066, 0.1050, 0.2500, 0.1050, "
    "0.0066. Right, the binary cross-entropy of one sample with label 1 over z from minus 40 to 40: the exact "
    "loss, dashed, falls along a line from 40 at z = minus 40 to near 0; the clipped loss the class reports, "
    "solid red, follows it from z = minus 16.118 on and stays at 16.118 below that. Points mark the printed "
    "clipped values 16.12, 16.12, 5.007, 0.6931, 0.006715, 1e-7, 1e-7 at z = minus 40, minus 17, minus 5, 0, 5, "
    "17, 40, and hollow points the exact 40 and 17 where the two differ; without the clip z = 40 gives nan. With "
    "label 0 the curve is mirrored.",
    subtitle="Float64. Right: one sample with label 1; with label 0 the curve is mirrored.", data_w=True)

left, right = fig.row(2, y=104, h=376)

body = fig.panel(left, rich("The sigmoid and its slope"))
ax = fig.line_chart(body,
                    [dict(xs=list(Z1), ys=list(S1), color="output", points=False),
                     dict(xs=list(Z1), ys=list(D1), color="gradient", points=False)],
                    x=(-6, 6, [-6, -4, -2, 0, 2, 4, 6]), y=(0, 1, [0, 0.25, 0.5, 0.75, 1]), labels=False,
                    label_w=72, x_label=rich("logit ", z), fmt_x=num, fmt_y=lambda v: f"{v:g}")
for a, c in zip(P_Z, pd):
    ax.point(a, c, "diamond", "gradient", size=10)
for a, b in zip(P_Z, ps):
    ax.point(a, b, "circle", "output", size=8)
ax.text(6, float(S1[-1]), rich("σ(", z, ")"), "label", color="output", dx=12, dy=5)
ax.text(6, float(D1[-1]), "slope", "label", color="gradient", dx=12, dy=5)
ax.text(0, 0.5, "0.5", "note", anchor="end", dx=-10, dy=-4)
ax.text(0, 0.25, "peak 0.25", "note", dx=10, dy=-6)

body = fig.panel(right, rich("Binary cross-entropy, label 1"))
ax = fig.line_chart(body,
                    [dict(xs=list(Z2), ys=list(EXACT), color="ink-muted", dash="proj", points=False),
                     dict(xs=list(Z2), ys=list(CLIP), color="error", points=False)],
                    x=(-40, 40, [-40, -20, 0, 20, 40]), y=(0, 40, [0, 10, 20, 30, 40]), labels=False,
                    label_w=24, x_label=rich("logit ", z), y_label="loss", fmt_x=num, fmt_y=num)
for a, b in zip(T_Z, TC):
    ax.point(a, b, "circle", "error", size=8)
for a, b, c in zip(T_Z, TE, TC):
    if not np.isclose(b, c, rtol=1e-3, atol=1e-6):
        ax.point(a, b, "circle", "ink-muted", size=8, hollow=True)
ax.text(-40, CAP, "cap 16.118", "note", color="error", dx=8, dy=24)
ax.text(-BOUND, CAP, rich("clip starts at ", z, " = ", MINUS, "16.118"), "note", dx=14, dy=0)
ax.text(40, 0, rich("unclipped at ", z, " = 40: nan"), "note", anchor="end", dx=-4, dy=-14)
# the key, in the empty upper right of the plot
kx, ky = ax.to_px(-10, 37)
kx, ky = round(kx / 4) * 4, round(ky / 4) * 4
fig.legend(kx, ky, [dict(color="ink-muted", label="exact, from the logit", mark="dash"),
                    dict(color="error", label="clipped, as the class reports", mark="line")])

fig.caption("The class reports at most 16.118 however wrong the logit is; the exact loss keeps rising.")
fig.write()
