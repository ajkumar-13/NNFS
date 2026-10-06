"""Post 12 sections 6.1 and 9.2: the loss of the 200-iteration loop on a log axis, against 36 * 0.49^t.

Run from anywhere:  python posts/12-backprop-through-a-single-neuron/diagrams/src/03-loss-decay.py
Writes posts/12-backprop-through-a-single-neuron/diagrams/03-loss-decay.svg.
The loop of snippets/training_loop.py is repeated here statement for statement, so that every iteration's loss
is available; every line it prints is formatted the same way and asserted against the snippet's own output,
which this script captures, together with the first iteration with a loss of exactly 0.0 (113). The display
threshold is the four-decimal format of that print: 0.0001 at iteration 18, 0.0000 from iteration 19.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, Raw, rich, var, sup, num, span  # noqa: E402

T_SUP = Raw('<tspan class="sub i" dy="-6">t</tspan><tspan dy="6">​</tspan>')   # an italic superscript t

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "training_loop.py"
with contextlib.redirect_stdout(io.StringIO()) as out:
    runpy.run_path(str(SNIP), run_name="snippet")
OUT = out.getvalue().splitlines()

# -- the loop of training_loop.py, keeping every loss
inputs = np.array([1.0, -2.0, 3.0])
weights = np.array([-3.0, -1.0, 2.0])
bias, target, lr = 1.0, 0.0, 0.01
LOSS, ZS, printed = [], [], []
first_zero = None
for i in range(200):
    z = float(np.dot(inputs, weights)) + bias
    yhat = max(0.0, z)
    loss = (yhat - target) ** 2
    dloss_dyhat = 2.0 * (yhat - target)
    dyhat_dz = 1.0 if z > 0 else 0.0
    upstream = dloss_dyhat * dyhat_dz
    dweights = upstream * inputs
    dbias = upstream * 1.0
    weights -= lr * dweights
    bias -= lr * dbias
    LOSS.append(loss)
    ZS.append(z)
    if first_zero is None and loss == 0.0:
        first_zero = i
    if i <= 20 and i % 4 == 0:
        printed.append(f"iter {i:3d}  loss = {loss:.4f}   (exactly {loss:.3e})")
    elif (i % 20 == 0 and i <= 120) or i == 199:
        printed.append(f"iter {i:3d}  loss = {loss:.4f}   (exactly {loss:.3e})   z = {z:.3e}")
assert printed == OUT[:len(printed)], "the repeated loop must print what the snippet prints"
assert first_zero == 113 and OUT[len(printed)] == "first iteration with a loss of exactly 0.0: 113"
LAST = first_zero - 1                                   # the last iteration with a positive loss
assert all(v > 0 for v in LOSS[:first_zero]) and all(v == 0.0 for v in LOSS[first_zero:])
FIRST_DOT = next(i for i, v in enumerate(LOSS) if f"{v:.4f}" == "0.0000")
assert FIRST_DOT == 19 and f"{LOSS[18]:.4f}" == "0.0001"                 # section 6.1: from iteration 19 onward
for i in range(61):                                      # section 9.2: 36 * 0.49^i to the digits shown up to 60
    assert f"{LOSS[i]:.3e}" == f"{36 * 0.49 ** i:.3e}"
assert f"{LOSS[80]:.3e}" == "5.916e-24" and f"{36 * 0.49 ** 80:.3e}" == "5.915e-24"
assert f"{ZS[100]:.1e}" == "1.8e-15"


def sci(s):
    """'2.292e-05' as 2.292 x 10 with a superscript -5; '3.600e+01' as 36; '0.000e+00' as 0."""
    m, e = s.split("e")
    m, e = float(m), int(e)
    if m == 0:
        return "0"
    if -1 <= e <= 2:
        return num(m * 10 ** e)
    return rich(f"{m:.3f} \u00d7 ", sup("10", num(e), italic=False))


ROWS_AT = (0, 16, 20, 100, 120)
table = [["iteration", "printed", "value"]]
for line in printed:
    i = int(line[5:8])
    if i in ROWS_AT:
        shown = line.split("loss = ")[1].split()[0]
        exact = line.split("exactly ")[1].split(")")[0]
        table.append([num(i), span(shown, mono=True), sci(exact)])
assert len(table) == 1 + len(ROWS_AT)

fig = Figure(
    "03-loss-decay", "Every step multiplies the loss by 0.49, until rounding",
    "A chart of the loss of the 200-iteration loop against the iteration, 0 to 120, on a log axis from 10 to "
    "the minus 35 up to 100. The run falls on the dashed straight line 36 times 0.49 to the power t, from 36 at "
    "iteration 0. A dotted horizontal line at 5 times 10 to the minus 5 marks where the loss rounds to 0.0000 in "
    "four decimals; the run crosses it at iteration 19. From about iteration 100 the run leaves the straight "
    "line in steps near 10 to the minus 30, and a dotted vertical line marks iteration 113, from which the loss "
    "is exactly 0.0, which a log axis cannot show. A table lists what the loop prints at iterations 0, 16, 20, "
    "100 and 120: 36.0000, 0.0004, 0.0000, 0.0000 and 0.0000, against the values 36, 3.976 times 10 to the "
    "minus 4, 2.292 times 10 to the minus 5, 3.155 times 10 to the minus 30, and 0.",
    subtitle=rich("The loop of section 9.2, ", var("y"), " = 0, learning rate 0.01; the loss on a log axis."))

XS = list(range(0, LAST + 1))
TS = list(range(0, 121))
ax = fig.line_chart(
    Box(40, 104, 576, 372),
    [dict(xs=TS, ys=[36 * 0.49 ** t for t in TS], color="ink-muted", dash="proj", points=False, label=None),
     dict(xs=XS, ys=[LOSS[i] for i in XS], color="error", points=False, label=None)],
    x=(0, 120, [0, 20, 40, 60, 80, 100, 120]), y=(1e-35, 1e2, [1e-30, 1e-20, 1e-10, 1]), y_log=True,
    x_label="iteration", y_label=rich("loss ", var("L")), label_w=8,
    ref_lines=[dict(y=5e-5),
               dict(x=first_zero, label=f"exactly 0.0 from {first_zero}")])
ax.point(FIRST_DOT, LOSS[FIRST_DOT], "circle", "error", size=8)
ax.text(FIRST_DOT, LOSS[FIRST_DOT], f"iteration {FIRST_DOT}", style="note", dx=12, dy=-12)
ax.text(44, 5e-5, "rounds to 0.0000 below this line", style="note", dx=0, dy=24)

# -- right: what the loop prints
XT = 648
fig.text(XT, 128, "What the loop prints", "head")
fig.table(XT, 144, table, [80, 88, 104], row_h=32, col_align=["end", "end", "end"])
fig.legend(XT, 376, [dict(color="error", label="the float64 run", mark="line"),
                     dict(color="ink-muted", label=rich("36 ", "\u00b7", " ", sup("0.49", "t", italic=False)),
                          mark="line", dash="proj")])

fig.caption("A straight line on a log axis is a constant factor per step; 0.0000 is rounding, not zero.")
fig.write()
