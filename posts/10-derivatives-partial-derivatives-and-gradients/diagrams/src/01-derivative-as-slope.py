"""Post 10 hero: f(x) = x^2 with its tangent at x = -1.5, 0.5 and 2, whose slopes are f'(x) = 2x.

Run from anywhere:  python posts/10-derivatives-partial-derivatives-and-gradients/diagrams/src/01-derivative-as-slope.py
Writes posts/10-derivatives-partial-derivatives-and-gradients/diagrams/01-derivative-as-slope.svg.
The curve and the tangents are evaluated here from f(x) = x^2 and f'(x) = 2x (section 2). The measured slopes
are the central differences of snippets/finite_differences.py, which this script runs: its own
central_difference function is used, and each value is asserted against the "Hero figure" line it prints.
"""
import contextlib
import io
import math
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num  # noqa: E402

PRIME, MINUS = "\u2032", "\u2212"
SNIP = Path(__file__).resolve().parents[2] / "snippets" / "finite_differences.py"
ns = runpy.run_path(str(SNIP), run_name="snippet")
with contextlib.redirect_stdout(io.StringIO()) as out:
    ns["main"]()
OUT = out.getvalue()


def f(x):
    return x * x


def df(x):
    return 2 * x


POINTS = (-1.5, 0.5, 2.0)                       # section 2: "the tangent at x = -1.5 has slope -3, ..."
SLOPES = [df(p) for p in POINTS]
assert SLOPES == [-3.0, 1.0, 4.0]
H = 1e-5
MEASURED = [ns["central_difference"](f, p, H) for p in POINTS]
for p, m in zip(POINTS, MEASURED):
    line = f"x = {p:4.1f}   2x = {2 * p:4.1f}   central difference = {m:9.6f}"
    assert line in OUT, line                    # the snippet prints exactly this line
    assert abs(m - df(p)) < 1e-9
PRINTED = [f"{m:.6f}" for m in MEASURED]
assert PRINTED == ["-3.000000", "1.000000", "4.000000"]
MEAS_TXT = [s.replace("-", MINUS) for s in PRINTED]


def signed(v):
    """A whole-number slope with its sign: +4, -3 (a true minus)."""
    return ("+" if v > 0 else "") + num(int(v))


FPRIME = rich(var("f"), PRIME, "(", var("x"), ")")

fig = Figure(
    "01-derivative-as-slope", "The derivative is the slope of the tangent",
    "A plot of the parabola f(x) = x squared for x from minus 3 to 3, with its tangent line drawn at three "
    "points and each tangent labelled with its slope: minus 3 at x = minus 1.5, plus 1 at x = 0.5 and plus 4 at "
    "x = 2. A table beside the plot lists, for the same three points, x, the rule f prime of x = 2x, which gives "
    "minus 3, 1 and 4, and the central difference with h = 10 to the minus 5, which gives minus 3.000000, "
    "1.000000 and 4.000000.",
    subtitle=rich(var("f"), "(", var("x"), ") = ", sup("x", "2"), ", and its tangent at three points; each slope is ",
                  FPRIME, " = 2", var("x"), " there."))

# -- the chart: f on x from -3 to 3; the y range starts at -1 so the tangent at 0.5 has room below its point
XL, XH, YL, YH = -3, 3, -1, 9
xs = [XL + (XH - XL) * k / 120 for k in range(121)]
CURVE = dict(xs=xs, ys=[f(x) for x in xs], color="output", label=rich(var("f"), "(", var("x"), ") = ", sup("x", "2")),
             points=False)
ax = fig.line_chart(Box(40, 104, 504, 372), [CURVE],
                    x=(XL, XH, [-3, -2, -1, 0, 1, 2, 3]), y=(YL, YH, [0, 3, 6, 9]),
                    x_label=var("x"), y_label=rich(var("f"), "(", var("x"), ")"), label_w=96, points=False)

# Each tangent is drawn 64 units either side of its point, measured along the line on the page, so the three
# look the same length although the two axes have different scales.
HALF = 64
LABEL_AT = {-1.5: ("end", -12, 40), 0.5: ("start", 16, 32), 2.0: ("start", 16, 40)}
with fig.data():
    for p, m in zip(POINTS, SLOPES):
        px, py = ax.to_px(p, f(p))
        qx, qy = ax.to_px(p + 1, f(p) + m)          # one unit along the tangent, in page units
        L = math.hypot(qx - px, qy - py)
        ux, uy = (qx - px) / L, (qy - py) / L
        fig.edge((px - HALF * ux, py - HALF * uy), (px + HALF * ux, py + HALF * uy), color="gradient", width=1.5)
    for p, m in zip(POINTS, SLOPES):
        px, py = ax.to_px(p, f(p))
        fig.marker(px, py, "circle", "gradient", size=10)
        anchor, dx, dy = LABEL_AT[p]
        fig.text(px + dx, py + dy, f"slope {signed(m)}", "value", color="gradient", anchor=anchor, snap=False)

# -- the table: rule against measurement at the three points, centred on the plot's height
TX, TY = 552, 200
fig.text(TX, TY - 16, "Rule against measurement", "head")
rows = [[var("x"), rich(FPRIME, " = 2", var("x")), "central difference"]]
for p, m, t in zip(POINTS, SLOPES, MEAS_TXT):
    rows.append([num(p, 1), num(int(m)), t])
fig.table(TX, TY, rows, [64, 128, 176], row_h=40, align=["end", "end", "end"])
fig.text(TX, TY + 4 * 40 + 36, rich("central difference: (", var("f"), "(", var("x"), " + ", var("h"), ") ", MINUS, " ",
                                    var("f"), "(", var("x"), " ", MINUS, " ", var("h"), ")) / 2", var("h")), "note")
fig.text(TX, TY + 4 * 40 + 60, rich("with ", var("h"), " = ", sup("10", num(-5), italic=False),
                                    ", from finite_differences.py"), "note")

fig.caption(rich("The derivative is a function: it returns the slope at every ", var("x"), "."))
fig.write()
