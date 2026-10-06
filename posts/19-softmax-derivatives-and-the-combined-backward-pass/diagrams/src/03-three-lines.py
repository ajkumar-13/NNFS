"""Post 19, section 4: the three lines of the combined backward on the worked batch of post 08.

Run from anywhere:  python posts/19-softmax-derivatives-and-the-combined-backward-pass/diagrams/src/03-three-lines.py
Writes posts/19-softmax-derivatives-and-the-combined-backward-pass/diagrams/03-three-lines.svg.
snippets/worked_example.py is run here (runpy): the softmax output, the labels and dinputs are the snippet's own
arrays, the middle state (after the subtraction, before the division) is recomputed from them, and the three lines
the snippet prints (dinputs, its row sums, the unchanged first row of softmax_output) are checked against its output.

Layout: the three lines of code on top, numbered; under them the labels y_true and the array after each line,
left to right: the copy, the copy with 1 subtracted at the true class (those cells outlined), and dinputs after the
division by 3.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, MINUS, rich, arr, hat, span, num  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "worked_example.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
SRC = SNIPPET.read_text(encoding="utf-8")

P, T, D = s["softmax_output"], s["y_true"], s["dinputs"]
N = len(T)
assert P.tolist() == [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]] and T.tolist() == [0, 1, 1]
MID = P.copy()
MID[range(N), T] -= 1
assert np.allclose(MID / N, D)
assert OUT.startswith("[[-0.1         0.03333333  0.06666667]\n [ 0.03333333 -0.16666667  0.13333333]\n"
                      " [ 0.00666667 -0.03333333  0.02666667]]\n")
assert "row sums: [0. 0. 0.]" in OUT and "softmax_output unchanged: [0.7 0.1 0.2]" in OUT

# the three lines, as the snippet writes them (its comments dropped)
LINES = ["dinputs = softmax_output.copy()",
         "dinputs[range(len(y_true)), y_true] -= 1",
         "dinputs /= len(y_true)"]
for line in LINES:
    assert line in SRC, line


def fmt(a, d):
    return [[num(round(float(v), d) + 0.0, d) for v in row] for row in a]


PV, MV, DV = fmt(P, 2), fmt(MID, 2), fmt(D, 3)
assert MV == [["−0.30", "0.10", "0.20"], ["0.10", "−0.50", "0.40"], ["0.02", "−0.10", "0.08"]]
assert DV == [["−0.100", "0.033", "0.067"], ["0.033", "−0.167", "0.133"], ["0.007", "−0.033", "0.027"]]

YH, YB = hat(arr("y")), arr("y")

fig = Figure(
    "03-three-lines", "Three lines turn the softmax output into dinputs",
    "The worked batch of three samples and three classes, true classes 0, 1, 1. Line 1 copies the softmax output, "
    "rows 0.70, 0.10, 0.20; 0.10, 0.50, 0.40; 0.02, 0.90, 0.08. Line 2 subtracts 1 at each true class, the outlined "
    "cells, giving y-hat minus y: −0.30, 0.10, 0.20; 0.10, −0.50, 0.40; 0.02, −0.10, 0.08. Line 3 divides by "
    "the batch size 3: dinputs is −0.100, 0.033, 0.067; 0.033, −0.167, 0.133; 0.007, −0.033, 0.027, and each "
    "row sums to 0. The softmax output itself is unchanged.",
    subtitle=rich("The batch of section 4: three samples, three classes, integer labels ",
                  span("y_true", mono=True), " = [0, 1, 1]."),
    data_w=True)

# --- the code, numbered: the snippet's three lines and the same three in the class of section 5
CLASS = ["self.dinputs = dvalues.copy()",
         "self.dinputs[range(samples), y_true] -= 1",
         "self.dinputs /= samples"]
CLASS_SRC = (SNIPPET.parent / "combined_class.py").read_text(encoding="utf-8")
for line in CLASS:
    assert line in CLASS_SRC, line
for x, w, head, lines in ((40, 400, "Section 4", LINES), (472, 448, "In the class, section 5", CLASS)):
    fig.card(x, 112, w, 136)
    fig.text(x + 16, 144, head, "head")
    fig.code_block(x + (40 if x == 40 else 16), 176, lines)
for k in range(3):
    fig.text(56, 176 + 24 * k, str(k + 1), "tick")

CW, CH = 72, 40
YG = 304
XT, X1, X2, X3 = 40, 120, 392, 664
gt = fig.strip(XT, YG, 3, cell_w=48, cell_h=CH, vertical=True, values=[str(int(v)) for v in T],
               fill=lambda i: "input-soft", font=16)
g1 = fig.grid(X1, YG, 3, 3, cell_w=CW, cell_h=CH, values=PV, font=16, fill=lambda i, j: "output-soft")
g2 = fig.grid(X2, YG, 3, 3, cell_w=CW, cell_h=CH, values=MV, font=16, fill=lambda i, j: "gradient-soft")
g3 = fig.grid(X3, YG, 3, 3, cell_w=CW, cell_h=CH, values=DV, font=16, fill=lambda i, j: "gradient-soft")
for i in range(N):
    g1.outline(i, int(T[i]), color="output", width=1.5)
    g2.outline(i, int(T[i]), color="gradient", width=1.5)

fig.text(gt.box.cx, YG - 16, span("y_true", mono=True), "code", anchor="middle")
fig.text(X1, YG - 16, rich("after line 1: ", YH), "head")
fig.text(X2, YG - 16, rich("after line 2: ", YH, " ", MINUS, " ", YB), "head")
fig.text(X3, YG - 16, rich("after line 3: ", span("dinputs", mono=True)), "head")

fig.arrow((g1.box.right + 8, g1.box.cy), (X2 - 8, g1.box.cy), label=rich(MINUS, "1"))
fig.arrow((g2.box.right + 8, g2.box.cy), (X3 - 8, g2.box.cy), label="÷ 3")

fig.note(g1.box, ["a copy: the softmax", "output is unchanged"])
fig.note(g2.box, ["outlined: each row's true class"])
fig.note(g3.box, ["each row sums to 0"])

fig.caption("With integer labels the one-hot matrix is never built: one entry per row changes.")
fig.write()
