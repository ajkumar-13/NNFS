"""Post 07, section 5: the entries shrink at every layer, so the logits are ten-thousandths and softmax gives 1/3.

Run from anywhere:  python posts/07-coding-the-complete-forward-pass/diagrams/src/03-why-uniform.py
Writes posts/07-coding-the-complete-forward-pass/diagrams/03-why-uniform.svg.
snippets/uniform_baseline.py is run here (runpy); every value drawn is read from its arrays and asserted to appear,
as printed, in its output (blocks 1 and 2) or in the text of section 5.
"""
import contextlib
import io
import math
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, span, sub, sup, var, num, MINUS  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "uniform_baseline.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

# block 1: largest and mean |entry| of each stage, as printed
STAGES = [("X", s["X"]), ("dense1.output", s["dense1"].output), ("activation1.output", s["activation1"].output),
          ("dense2.output", s["dense2"].output)]
ROWS = []
for name, a in STAGES:
    big, mean = float(np.max(np.abs(a))), float(np.mean(np.abs(a)))
    printed = "dense2.output (logits)" if name == "dense2.output" else name
    assert f"   {printed:<23} largest |entry| {big:.5f}   mean |entry| {mean:.6f}" in OUT, name
    ROWS.append((name, mean, big, f"{big:.5f}", f"{mean:.6f}"))
assert [r[3] for r in ROWS] == ["0.97939", "0.01746", "0.01731", "0.00023"]

# block 2: row 99, the least uniform row
P, Z = s["probabilities"], s["dense2"].output
R = int(np.argmax(np.max(P, axis=1)))
assert R == 99 and f"row {R}, the least uniform" in OUT
LOGITS = [f"{v:.7f}" for v in Z[R]]
PROBS = [f"{v:.6f}" for v in P[R]]
assert LOGITS == ["-0.0002333", "-0.0002187", "0.0000813"]          # section 5's text
assert PROBS[0] == f"{P.min():.6f}" == "0.333297" and PROBS[2] == f"{P.max():.6f}" == "0.333402"
GAP = float(Z[R].max() - Z[R].min())
assert f"largest logit gap {GAP:.7f}, exp of that gap {np.exp(GAP):.6f}" in OUT
GAP_S, EXP_S = f"{GAP:.7f}", f"{np.exp(GAP):.6f}"
assert (GAP_S, EXP_S) == ("0.0003146", "1.000315")
assert f"largest probability over smallest {EXP_S}" in OUT


def say(v):
    return v.replace("-", "minus ")


fig = Figure(
    "03-why-uniform", "Logits in ten-thousandths give probabilities near 1/3",
    "Left, a dot plot on a log scale from 10 to the minus 5 to 10 of the size of the entries at each stage of the "
    "forward pass on the 300 spiral points, the mean absolute entry as a grey circle and the largest as a "
    "diamond in the text colour: X, mean " + ROWS[0][4] + ", largest " + ROWS[0][3] + "; dense1.output, mean " + ROWS[1][4] +
    ", largest " + ROWS[1][3] + "; activation1.output, mean " + ROWS[2][4] + ", largest " + ROWS[2][3] +
    "; dense2.output, the logits, mean " + ROWS[3][4] + ", largest " + ROWS[3][3] + ". Right, row 99, the least "
    "uniform row of the output: logits " + ", ".join(say(v) for v in LOGITS) + " for classes 0, 1 and 2, and "
    "probabilities " + ", ".join(PROBS) + ", drawn as three bars on a scale from 0 to 1 that look equal. A note gives "
    "the largest logit gap, " + GAP_S + ", and e to that gap, " + EXP_S + ", the largest probability over the smallest.",
    subtitle="snippets/uniform_baseline.py: the entries of every stage on the 300 points, and row 99 of the output.")

left, right = fig.row((11, 9))

# -- left: the entries shrink
lb = fig.panel(left, "The entries shrink at every layer")
# the axis runs one decade past 1 so that X's largest entry, 0.979, sits inside the plot and not on its edge;
# the largest entries are ink diamonds, so green stays the colour of the probabilities on the right
LO, HI = 1e-5, 10.0
TICKS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
assert all(LO < r[1] and r[2] < 1 for r in ROWS)
ax = fig.dot_plot(Box(lb.x, lb.y, lb.w, 276), [(span(n, mono=True), [m, b]) for n, m, b, _, _ in ROWS],
                  LO, HI, TICKS, colors=("rule", "ink"), shapes=("circle", "diamond"), x_log=True,
                  label_w=160, pad_right=32, axis_label="size of an entry, log scale")
with fig.data():
    for i, (_, _, b, bs, _) in enumerate(ROWS):
        fig.text(ax.sx(b) + 12, ax.sy(i) + 5, bs, "value", color="ink", snap=False)
fig.legend(lb.x + 160, lb.y + 308, [dict(color="rule", label="mean |entry|", mark="circle"),
                                    dict(color="ink", label="largest |entry|", mark="diamond")], direction="row")

# -- right: row 99, its logits, then its probabilities on a 0 to 1 scale
rb = fig.panel(right, "Row 99, the least uniform row")
t = fig.table(rb.x, rb.y, [["class", "logit"]] + [[str(k), num(v)] for k, v in enumerate(LOGITS)], [64, 120])
NX = t.right + 24
for k, line in enumerate(["largest logit gap:", GAP_S, rich(var("e"), " to that gap: ", EXP_S, ","),
                          "the largest probability", "over the smallest"]):
    fig.text(NX, rb.y + 52 + 20 * k + (8 if k > 1 else 0), line, "note")
TL = {0: "0", 0.5: "0.5", 1: "1"}
fmt = lambda v: TL[v] if v in TL else f"{v:.6f}"  # noqa: E731
fig.text(rb.x, rb.y + 168, "Softmax of the row", "head")
fig.bar_chart(Box(rb.x, rb.y + 184, rb.w, 156), [(f"class {k}", float(P[R][k]), "output") for k in range(3)],
              0, 1, [0, 0.5, 1], fmt=fmt, label_w=72, value_w=80, axis_label="probability")
fig.write()
