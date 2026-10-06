"""Post 10 section 6.1: the error of the forward and the central difference of x^3 at x = 2 over the sweep of h.

Run from anywhere:  python posts/10-derivatives-partial-derivatives-and-gradients/diagrams/src/04-step-size-sweep.py
Writes posts/10-derivatives-partial-derivatives-and-gradients/diagrams/04-step-size-sweep.svg.
Every point is a row of the float64 sweep of snippets/finite_differences.py, which this script runs: the
errors are recomputed with the snippet's own functions and step list, and every row, both minima and the
ratio are asserted against the lines the snippet prints. The dotted line is the rounding estimate 1e-15 / h
of section 6.1, evaluated here.
"""
import contextlib
import io
import math
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num, TIMES  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "finite_differences.py"
ns = runpy.run_path(str(SNIP), run_name="snippet")
with contextlib.redirect_stdout(io.StringIO()) as out:
    ns["main"]()
OUT = out.getvalue()

fwd_d, cen_d, cube, STEPS = ns["forward_difference"], ns["central_difference"], ns["cube"], ns["STEPS"]
X, EXACT = 2.0, 12.0
ROWS = []
for h in STEPS:
    fwd, cen = fwd_d(cube, X, h), cen_d(cube, X, h)
    line = f"{h:.0e}   {fwd:16.12f}   {abs(fwd - EXACT):.1e}   {cen:16.12f}   {abs(cen - EXACT):.1e}"
    assert line in OUT, line
    ROWS.append((h, abs(fwd - EXACT), abs(cen - EXACT), fwd, cen))
assert len(ROWS) == 16 and ROWS[0][0] == 1e-1 and ROWS[-1][0] == 1e-16
assert ROWS[-1][3] == 0.0 and ROWS[-1][4] == 0.0                     # at 1e-16 both estimates are exactly 0
BEST_F = min(ROWS, key=lambda r: r[1])
BEST_C = min(ROWS, key=lambda r: r[2])
assert f"smallest forward error on this grid: {BEST_F[1]:.1e} at h = {BEST_F[0]:.0e}" in OUT
assert f"smallest central error on this grid: {BEST_C[2]:.1e} at h = {BEST_C[0]:.0e}" in OUT
assert (f"{BEST_F[1]:.1e}", BEST_F[0], f"{BEST_C[2]:.1e}", BEST_C[0]) == ("7.3e-08", 1e-08, "2.1e-10", 1e-05)
RATIO = f"{BEST_F[1] / BEST_C[2]:.0f}"
assert f"the smallest forward error is {RATIO} times the smallest central error" in OUT and RATIO == "344"


def sci(v):
    """2.1e-10 as 2.1 x 10 to the -10, with a true superscript, as the snippet rounds it (.1e)."""
    m, e = f"{v:.1e}".split("e")
    return rich(f"{m} {TIMES} ", sup("10", num(int(e)), italic=False))


def p10(e):
    return sup("10", num(e), italic=False)


def say(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m} times 10 to the minus {-int(e)}"


fig = Figure(
    "04-step-size-sweep", "A smaller step helps only until rounding takes over",
    "A log-log chart of the error of two estimates of the slope of x cubed at x = 2, whose exact value is 12, "
    "against the step h from 10 to the minus 16 up to 10 to the minus 1, in float64. From the right, the forward "
    "difference's error falls as 6h and the central difference's as h squared. Then both turn and rise along "
    "a dotted line, the rounding estimate 10 to the minus 15 over h. The smallest central error is "
    f"{say(BEST_C[2])}, at h = 10 to the minus 5; the smallest forward error is {say(BEST_F[1])}, at h = 10 to "
    f"the minus 8, {RATIO} times larger. From h = 10 to the minus 8 to 10 to the minus 13 the two estimates "
    "agree, and at h = 10 to the minus 16 both return 0, an error of 12.",
    subtitle=rich("Error of each estimate of the slope of ", sup("x", "3"), " at ", var("x"),
                  " = 2 (exact 12) in float64, against the step ", var("h"), "."))

HS = [r[0] for r in ROWS][::-1]                      # left to right: h from 1e-16 up to 1e-1
EF = [r[1] for r in ROWS][::-1]
EC = [r[2] for r in ROWS][::-1]
XLO, XHI, YLO, YHI = 1e-16, 1e-1, 1e-11, 1e2
ROUND = dict(xs=[1e-16, 1e-6], ys=[1e-15 / 1e-16, 1e-15 / 1e-6], color="ink-muted", dash="ref", label="")
SER = [ROUND,
       dict(xs=HS, ys=EF, color="orange", label="forward difference"),
       dict(xs=HS, ys=EC, color="blue", label="central difference")]
ax = fig.line_chart(Box(40, 104, 880, 372), SER, x=(XLO, XHI, [1e-16, 1e-13, 1e-10, 1e-7, 1e-4, 1e-1]),
                    y=(YLO, YHI, [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2]),
                    fmt_y=lambda t: "1" if t == 1 else p10(round(math.log10(t))),
                    x_label=rich("step ", var("h"), ", log scale"),
                    y_label=rich("error, distance from the exact 12, log scale"), label_w=160, points=False,
                    x_log=True, y_log=True)

with fig.data():
    # squares first and circles smaller on top, so a forward point that equals a central one still shows
    for h, e in zip(HS, EC):
        fig.marker(*ax.to_px(h, e), "square", "blue", size=10)
    for h, e in zip(HS, EF):
        fig.marker(*ax.to_px(h, e), "circle", "orange", size=8)
    # the two minima, labelled under their points
    bx, by = ax.to_px(BEST_C[0], BEST_C[2])
    fig.text(bx + 16, by + 20, rich("best central: ", sci(BEST_C[2])), "value", color="blue", snap=False)
    fx, fy = ax.to_px(BEST_F[0], BEST_F[1])
    fig.text(fx - 16, fy + 24, rich("best forward: ", sci(BEST_F[1])), "value", color="orange", anchor="end",
             snap=False)
    # the three regimes, named on the chart
    tx, ty = ax.to_px(1e-3, 1e-1)
    fig.text(tx, ty, rich("truncation: 6", var("h")), "note", color="orange", anchor="middle", snap=False)
    tx, ty = ax.to_px(2e-3, 1e-8)
    fig.text(tx, ty, rich("truncation: ", sup("h", "2")), "note", color="blue", snap=False)
    # just above the dotted line, in the clear wedge right of h = 1e-10 between the line and the forward series
    rx, ry = ax.to_px(1.2e-10, 3e-5)
    fig.text(rx, ry, rich("rounding: about ", p10(-15), " / ", var("h")), "note", anchor="start", snap=False)
    # tied by a short leader to the two coinciding marks at h = 1e-16
    zx, zy = ax.to_px(1e-16, 12.0)
    fig.leader((zx + 6, zy - 3), (zx + 22, zy - 8))
    fig.text(zx + 28, zy - 4, "both return 0", "note", anchor="start", snap=False)

fig.caption(rich("Below ", var("h"), " = ", p10(-5), " the central difference gets worse, not better."))
fig.write()
