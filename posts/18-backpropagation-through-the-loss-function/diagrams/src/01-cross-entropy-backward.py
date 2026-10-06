"""Post 18 hero, sections 2 and 3: the cross-entropy gradient of the worked batch, one division and one 1/N.

Run from anywhere:  python posts/18-backpropagation-through-the-loss-function/diagrams/src/01-cross-entropy-backward.py
Writes posts/18-backpropagation-through-the-loss-function/diagrams/01-cross-entropy-backward.svg.
snippets/loss_backward.py is run here (runpy): the labels, the predictions and the (3, 3) result are the snippet's
own arrays, the per-sample step is -y_true / y_pred on those arrays, and the printed lines the post quotes in
sections 2.2, 3.1 and 7 are checked against the snippet's output.

Layout: four grids in one row, labels divided by predictions gives each sample's gradient, divided by N gives the
batch gradient. Under them the two lines of backward that do the two steps, and the third sample worked out.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, arr, hat, span, MINUS, TIMES  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "loss_backward.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

Y, P, D = s["y_true"], s["y_pred"], s["dinputs"]
N = len(P)
assert Y.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
assert P.tolist() == [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]] and N == 3
PER = -Y / P                                            # section 2.2, row by row
assert np.array_equal(D, PER / N)
assert np.array_equal(s["from_integers"], D) and np.array_equal(s["from_onehot"], D)
assert "[[-1.42857143  0.          0.        ]]" in OUT                      # section 2.2, the first row
assert ("[[-0.47619048  0.          0.        ]\n [ 0.         -0.55555556  0.        ]\n"
        " [ 0.          0.         -0.66666667]]") in OUT                       # section 3.1
assert "same array from one-hot labels: True" in OUT
assert "non-zero entries per row: [1 1 1]" in OUT

# the two lines of backward, as the snippet has them
SRC = SNIPPET.read_text(encoding="utf-8").splitlines()
CODE = ["self.dinputs = -y_true / dvalues", "self.dinputs = self.dinputs / samples"]
for line in CODE:
    assert any(l.strip() == line for l in SRC), line


def f3(v):
    """Three decimals with the true minus; an exact zero prints as 0, as the post writes it."""
    return "0" if v == 0 else f"{v:.3f}".replace("-", MINUS)


def f1(v):
    return "0" if v == 0 else f"{v:g}"


PER_S = [[f3(v) for v in r] for r in PER]
D_S = [[f3(v) for v in r] for r in D]
assert [PER_S[i][i] for i in range(3)] == [f"{MINUS}1.429", f"{MINUS}1.667", f"{MINUS}2.000"]
assert [D_S[i][i] for i in range(3)] == [f"{MINUS}0.476", f"{MINUS}0.556", f"{MINUS}0.667"]
TRUE = lambda i, j: Y[i][j] == 1  # noqa: E731
I = 2                                                   # the third sample, the largest gradient
assert int(np.argmax(-D.min(axis=1))) == I and P[I, I] == P[range(3), range(3)].min()

YV, YH = arr("y"), hat(arr("y"))
dLdy = rich("∂", var("L"), "/∂", YH)

fig = Figure(
    "01-cross-entropy-backward", "One entry per row survives the division",
    "The worked batch of section 3.1 as four 3 by 3 grids. The one-hot labels y, with a 1 at classes 0, 1 and 2 "
    "in rows 1 to 3, divided element by element by the predictions y-hat, rows 0.7, 0.2, 0.1; 0.1, 0.6, 0.3; "
    "0.2, 0.3, 0.5, and negated, give each sample's gradient: minus 1.429, minus 1.667 and minus 2.000 on the "
    "true classes and 0 everywhere else. Divided by N = 3 they give dL/dy-hat, minus 0.476, minus 0.556 and "
    "minus 0.667, with zeros elsewhere. Below, the two lines of backward that do the two steps, and the third "
    "sample worked out: minus 1 over 3 times 0.5 is minus 0.667, the largest entry, because 0.5 is the smallest "
    "probability on a true class.",
    subtitle=rich("The batch of section 3.1, ", var("N"), " = 3. The labels are one-hot, so only the true-class "
                  "entry of each row is non-zero."),
    data_w=True)

C, CW = 48, 64                       # label and prediction cells; the wider gradient cells
YG = 152                             # top of the grids
XS = (40, 256, 472, 728)             # the four grids' left edges

gy = fig.grid(XS[0], YG, 3, 3, cell=C, values=Y.tolist(), fill=lambda i, j: "input-soft" if TRUE(i, j) else None)
gp = fig.grid(XS[1], YG, 3, 3, cell=C, values=[[f1(v) for v in r] for r in P],
              fill=lambda i, j: "output-soft" if TRUE(i, j) else None)
gs = fig.grid(XS[2], YG, 3, 3, cell_w=CW, cell_h=C, values=PER_S, font=16,
              fill=lambda i, j: "gradient-soft" if TRUE(i, j) else None)
gd = fig.grid(XS[3], YG, 3, 3, cell_w=CW, cell_h=C, values=D_S, font=16,
              fill=lambda i, j: "gradient-soft" if TRUE(i, j) else None)
assert gd.box.right == 920

# headings, one baseline across the row
YT = YG - 16
fig.text(XS[0], YT, rich("Labels ", YV), "head", color="input")
fig.text(XS[1], YT, rich("Predictions ", YH), "head", color="output")
fig.text(XS[2], YT, rich(MINUS, YV, " ÷ ", YH, ", per sample"), "head", color="gradient")
fig.text(XS[3], YT, rich(dLdy, ", the batch"), "head", color="gradient")

# the operators between the grids
MID = YG + 3 * C // 2
fig.op((gy.box.right + gp.box.x) / 2, MID, "÷")
fig.arrow((gp.box.right + 8, MID), (gs.box.x - 8, MID))
fig.arrow((gs.box.right + 8, MID), (gd.box.x - 8, MID), label=rich("÷ ", var("N")))

# the third sample's prediction and result, outlined
gp.outline(I, I, color="output", width=1.5)
gd.window(I, I, color="gradient")

fig.note(gy.box, "one-hot rows")
fig.note(gp.box, "the softmax output")
fig.note(gs.box, rich("section 2: ", MINUS, "1 ÷ the true-class ", var("ŷ")))
fig.note(gd.box, rich(var("N"), " = 3: the shape of ", YH, ", (3, 3)"))

# the two lines of code, and the worked entry
YC = 352
card = fig.card(40, YC, 352, None, heading=rich("The two steps in ", span("backward", mono=True)),
                lines=CODE, style="code", fit=True)
XW = 472
fig.text(XW, YC + 32, "The third sample, true class 2", "head")
fig.text(XW, YC + 68, rich(MINUS, "1 ÷ (3 ", TIMES, " 0.5) = ", span(f3(D[I, I]), bold=True)), "math")
fig.text(XW, YC + 96, "the smallest true-class probability gives the largest gradient", "note")

fig.caption("Every other entry is 0; the softmax backward of post 19 spreads the one entry over all logits.")
fig.write()
