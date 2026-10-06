"""Post 19, section 3.2 (and section 8): the two reasons the combined backward is used, work and no division.

Run from anywhere:  python posts/19-softmax-derivatives-and-the-combined-backward-pass/diagrams/src/02-why-combined.py
Writes posts/19-softmax-derivatives-and-the-combined-backward-pass/diagrams/02-why-combined.svg.
The counts on the left are computed here from K and checked against the three lines snippets/softmax_jacobian.py
prints for section 3.2. The case on the right is case 5 of snippets/what_can_go_wrong.py (logits [0, -800, 0],
true class 1): the arrays are recomputed here with the classes of snippets/combined_class.py and checked against
the lines the snippet prints, which the post quotes in section 8.

Layout: two panels. Left, a table of the numbers each route computes for one sample at K = 3, 1,000 and 50,000.
Right, the one sample whose true-class probability is exactly 0: the softmax output, the Jacobian route's
-y / y_hat with its minus infinity and the nan row it ends in, and the combined route's finite y_hat - y.
"""
import contextlib
import io
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, MINUS, rich, var, arr, hat, sup, span, num  # noqa: E402

SNIPPETS = Path(__file__).resolve().parents[2] / "snippets"
sys.path.insert(0, str(SNIPPETS))
from combined_class import (Activation_Softmax, Activation_Softmax_Loss_CategoricalCrossentropy,  # noqa: E402
                            Loss_CategoricalCrossentropy)

DEFAULT_PRINT = np.get_printoptions()


def run(name):
    """Run one snippet as a script would, NumPy's print options reset first, and return what it prints."""
    np.set_printoptions(**DEFAULT_PRINT)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runpy.run_path(str(SNIPPETS / name), run_name="snippet")
    return buf.getvalue()


# --- the counts, from K
KS = (3, 1000, 50000)
ROWS = [(f"{k:,}", f"{k ** 2:,}", f"{k:,}") for k in KS]
OUT1 = run("softmax_jacobian.py")
for k in KS:
    assert f"K = {k:>6,}  Jacobian entries {k ** 2:>13,}  combined entries {k:>6,}" in OUT1
assert ROWS[-1] == ("50,000", "2,500,000,000", "50,000")
assert all(k ** 2 // k == k for k in KS)                 # the ratio is K itself

# --- the underflowed probability, through both routes
logits, label = np.array([[0.0, -800.0, 0.0]]), np.array([1])
softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.forward(logits, label)
softmax_loss.backward(softmax_loss.output, label)
activation, loss_fn = Activation_Softmax(), Loss_CategoricalCrossentropy()
activation.forward(logits)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    loss_fn.backward(activation.output, label)
    activation.backward(loss_fn.dinputs)
P, DV, JR, CB = activation.output[0], loss_fn.dinputs[0], activation.dinputs[0], softmax_loss.dinputs[0]
assert P.tolist() == [0.5, 0.0, 0.5]
assert DV[0] == 0 and DV[2] == 0 and DV[1] == -np.inf
assert np.isnan(JR).all() and CB.tolist() == [0.5, -1.0, 0.5]
OUT2 = run("what_can_go_wrong.py")
for line in ("   softmax output: [[0.5 0.  0.5]]", "   Jacobian route: [[nan nan nan]]",
             "   combined class: [[ 0.5 -1.   0.5]]"):
    assert line in OUT2.splitlines(), line


def show(v):
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return MINUS + "∞" if v < 0 else "∞"
    return num(float(v) + 0.0, 2)


PV, DVV, JRV, CBV = ([show(v) for v in a] for a in (P, DV, JR, CB))
assert PV == ["0.50", "0.00", "0.50"] and DVV == ["0.00", MINUS + "∞", "0.00"]
assert JRV == ["nan"] * 3 and CBV == ["0.50", "−1.00", "0.50"]

YH, YB, KK = hat(arr("y")), arr("y"), var("K")

fig = Figure(
    "02-why-combined", "The Jacobian route does more work and can return nan",
    "Two panels. Left, the numbers each route computes for one sample: at 3 classes 9 for the Jacobian route "
    "against 3 for the combined route, at 1,000 classes 1,000,000 against 1,000, and at 50,000 classes, the size "
    "of a language-model vocabulary, 2,500,000,000 against 50,000; the ratio is K. Right, logits 0, −800, 0 with "
    "true class 1 give the softmax output 0.50, 0.00, 0.50. The Jacobian route divides by it: minus y over y-hat "
    "is 0.00, minus infinity, 0.00, and the product with the Jacobian is nan, nan, nan. The combined route gives "
    "y-hat minus y = 0.50, −1.00, 0.50.",
    subtitle="Section 3.2's two reasons, counted and run, not timed.", data_w=True)

left, right = fig.row([5, 6])

# --- left: the work
body = fig.panel(left, "Numbers built per sample")
tb = fig.table(body.x, body.y, [[rich("classes ", KK), rich("Jacobian, ", sup(KK, "2")), rich("combined, ", KK)],
                                *ROWS], [96, 160, 120], row_h=48, col_align=["end", "end", "end"])
fig.note(tb, ["3 classes: the spiral data", "50,000: a language-model vocabulary",
              rich("The ratio is ", KK, " itself.")])

# --- right: the division by an underflowed probability
body = fig.panel(right, rich("A true-class probability of exactly 0"))
CW, CH = 64, 48
X0 = body.x
X1 = body.right - 3 * CW
Y1, Y2, Y3 = body.y + 24, body.y + 128, body.y + 232
gp = fig.strip(X0, Y1, 3, cell_w=CW, cell_h=CH, values=PV, font=16, fill=lambda i: "output-soft")
fig.text(X0, Y1 - 12, rich(YH, " for logits [0, ", MINUS, "800, 0], true class 1"), "label")

gd = fig.strip(X0, Y2, 3, cell_w=CW, cell_h=CH, values=DVV, font=16, fill=lambda i: "gradient-soft")
gn = fig.strip(X1, Y2, 3, cell_w=CW, cell_h=CH, values=JRV, font=16, fill=lambda i: "error-soft")
gd.window(1, 1, color="error")
fig.text(X0, Y2 - 12, rich("Jacobian route: ", MINUS, YB, "/", YH), "label")
fig.text(X1, Y2 - 12, "times the Jacobian", "label")
fig.arrow((gd.box.right + 8, gd.box.cy), (gn.box.x - 8, gd.box.cy))
fig.note(gn.box, [rich(MINUS, "∞ · 0 is nan")])

gc = fig.strip(X0, Y3, 3, cell_w=CW, cell_h=CH, values=CBV, font=16, fill=lambda i: "gradient-soft")
fig.text(X0, Y3 - 12, rich("Combined route: ", YH, " ", MINUS, " ", YB), "label")
fig.note(gc.box, ["finite, no division"])

fig.caption(rich("The combined route builds ", KK, " numbers per sample and never divides by a probability."))
fig.write()
