"""Post 06 section 4.2: softmax of the logits [1000, 1001, 999], naive and with the per-row maximum subtracted.

Run from anywhere:  python posts/06-activation-functions-relu-and-softmax/diagrams/src/03-softmax-stability.py
Writes posts/06-activation-functions-relu-and-softmax/diagrams/03-softmax-stability.svg.
snippets/softmax.py is run (runpy) for its naive_softmax and Activation_Softmax. This script then runs both on
the row itself: np.exp and the naive formula really overflow here (NumPy's RuntimeWarnings are recorded, and
their text is drawn), and the stable class really returns the probabilities. Every drawn value, warning and
limit is asserted against item 3 and item 4 of the snippet's printout.
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
from figkit import Figure, Box, rich, var, span, num, MINUS  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "softmax.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    S = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

BIG = np.array([[1000.0, 1001.0, 999.0]])


def run(function, argument):
    """function(argument) with NumPy's warnings recorded, as the snippet's run_and_report does."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(argument)
    return result, [f"{w.category.__name__}: {w.message}" for w in caught]


# -- naive: run here, overflow and all
EXP_NAIVE, w_exp = run(np.exp, BIG)
P_NAIVE, w_naive = run(S["naive_softmax"], BIG)
assert np.isinf(EXP_NAIVE).all() and np.isnan(P_NAIVE).all()
assert w_exp == ["RuntimeWarning: overflow encountered in exp"]
assert w_naive == ["RuntimeWarning: overflow encountered in exp", "RuntimeWarning: invalid value encountered in divide"]
assert f"3. np.exp of {BIG[0]} = {EXP_NAIVE[0]}\n" in OUT and f"   naive softmax    {P_NAIVE[0]}\n" in OUT
for m in w_naive:
    assert f"     {m}\n" in OUT

# -- stable: the post's class
MAX = float(np.max(BIG, axis=1, keepdims=True)[0, 0])
SHIFTED = BIG - np.max(BIG, axis=1, keepdims=True)
EXP_STABLE = np.exp(SHIFTED)
soft = S["Activation_Softmax"]()
with warnings.catch_warnings():
    warnings.simplefilter("error")                     # the stable path raises no warning at all
    soft.forward(BIG)
P_STABLE = soft.output
assert MAX == 1001.0 and SHIFTED.tolist() == [[-1.0, 0.0, -2.0]] and (SHIFTED <= 0).all()
assert f"   shifted logits   {SHIFTED[0]}\n" in OUT
assert f"   exp of those     {np.round(EXP_STABLE[0], 3)}\n" in OUT
assert f"   stable softmax   {np.round(P_STABLE[0], 3)}\n" in OUT
assert np.allclose(P_STABLE, EXP_STABLE / EXP_STABLE.sum()) and abs(P_STABLE.sum() - 1) < 1e-12
EXP3 = [f"{v:.3f}" for v in EXP_STABLE[0]]
P3 = [f"{v:.3f}" for v in P_STABLE[0]]
assert EXP3 == ["0.368", "1.000", "0.135"] and P3 == ["0.245", "0.665", "0.090"]

# -- the two overflow limits of item 4
L64, L32 = np.log(np.finfo(np.float64).max), np.log(np.finfo(np.float32).max)
assert f"4. largest float64 = exp({L64:.2f})   largest float32 = exp({L32:.2f})" in OUT
assert (f"{L64:.2f}", f"{L32:.2f}") == ("709.78", "88.72")


def say(vals):
    return ", ".join(str(v).replace("-", "minus ") for v in vals)


def ints(row):
    """Whole numbers as the post writes them: no thousands comma (1000, as NumPy prints it), the true minus."""
    return [str(int(v)).replace("-", MINUS) for v in row]


fig = Figure(
    "03-softmax-stability", "Subtracting the row's maximum keeps np.exp from overflowing",
    "Two columns of three steps for the logits 1000, 1001, 999 in float64, where np.exp returns inf above "
    f"{L64:.2f} ({L32:.2f} in float32). Naive column: the exponents are the logits themselves; np.exp gives inf, "
    "inf, inf; dividing by the row sum gives nan, nan, nan, with the two RuntimeWarnings NumPy prints, overflow "
    "encountered in exp and invalid value encountered in divide. An arrow labelled minus 1001, the row's maximum, "
    f"leads to the stable column: the exponents are {say(ints(SHIFTED[0]))}, every one at most 0; np.exp gives "
    f"{say(EXP3)}; dividing by the row sum gives {say(P3)}.",
    subtitle=rich("The logits [1000, 1001, 999] in float64: ", span("np.exp", mono=True), " returns inf above ",
                  f"{L64:.2f}", " (", f"{L32:.2f}", " in float32)."))

CW, CH = 80, 48                                   # cells
GN, GS = 248, 600                                 # the naive and the stable column
ROWS = [160, 256, 352]                            # row tops; 48 between rows for the arrows
LABELS = ["exponent", rich(span("np.exp", mono=True)), rich("÷ row sum")]

fig.text(GN, 136, "Naive", "head")
fig.text(GS, 136, "Stable", "head")
for y, lab in zip(ROWS, LABELS):
    fig.text(40, y + 28, lab, "label")

cols = []
for x, rows in [(GN, [(ints(BIG[0]), "input-soft"), (["inf"] * 3, "error-soft"), (["nan"] * 3, "error-soft")]),
                (GS, [(ints(SHIFTED[0]), "input-soft"), (EXP3, None), (P3, "output-soft")])]:
    strips = []
    for y, (vals, fill) in zip(ROWS, rows):
        strips.append(fig.strip(x, y, 3, width=3 * CW, height=CH, values=vals, fill=(lambda k, f=fill: f), font=16))
    for a, b in zip(strips, strips[1:]):
        fig.arrow((a.box.cx, a.box.bottom + 4), (b.box.cx, b.box.y - 4))
    cols.append(strips)

# the row's maximum: the cell it comes from, and the shift that leads to the stable column
naive_a, stable_a = cols[0][0], cols[1][0]
k_max = int(np.argmax(BIG[0]))
naive_a.window(k_max, k_max, color="input")         # a strip takes window(i0, i1)
stable_a.window(k_max, k_max, color="input")
fig.arrow((naive_a.box.right + 8, naive_a.box.cy), (stable_a.box.x - 8, stable_a.box.cy),
          label=rich(MINUS, " ", str(int(MAX))))
fig.text((naive_a.box.right + stable_a.box.x) / 2, naive_a.box.cy + 24, "the row's max", "note", anchor="middle")

# what NumPy says on the naive path, and what holds on the stable one
yb = ROWS[-1] + CH
for k, m in enumerate(w_naive):
    fig.text(40, yb + 36 + 20 * k, m, "code13", color="error")
fig.text(GS, yb + 36, rich("Every exponent ≤ 0, every exponential ≤ 1"), "note")
fig.text(GS, yb + 56, rich("The row sums to 1"), "note")

fig.caption("The shift cancels in the ratio, so the probabilities are those of the plain formula.")
fig.write()
