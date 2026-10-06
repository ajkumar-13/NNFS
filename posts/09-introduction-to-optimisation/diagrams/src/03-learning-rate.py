"""Post 09 section 5.1: gradient descent on f(w) = w ** 2 from w = 5, for four of the five learning rates.

Run from anywhere:  python posts/09-introduction-to-optimisation/diagrams/src/03-learning-rate.py
Writes posts/09-introduction-to-optimisation/diagrams/03-learning-rate.svg.

The curve is f evaluated here; the steps are the update rule of snippets/learning_rate_1d.py, run here with the
snippet's own f and slope, and the w after 1, 2, 3 and 100 steps is asserted against the table the snippet
prints (and the post quotes).
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num, TIMES, MINUS, CDOT  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "learning_rate_1d.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(SNIP), run_name="snippet")
OUT = buf.getvalue()
f, slope = ns["f"], ns["slope"]
assert f(3.0) == 9.0 and slope(3.0) == 6.0

RATES = [0.001, 0.1, 1.0, 10.0]
PATHS = {}
for lr in RATES:
    w, path = 5.0, [5.0]
    for _ in range(100):
        w = w - lr * slope(w)
        path.append(w)
    PATHS[lr] = path
    row = f"{lr:<15} {path[1]:<11.4g} {path[2]:<11.4g} {path[3]:<11.4g} {path[100]:.4g}"
    assert row in OUT, row                                        # the snippet's own table row
P100 = {lr: f"{PATHS[lr][100]:.4g}" for lr in RATES}
assert P100 == {0.001: "4.093", 0.1: "1.019e-09", 1.0: "5", 10.0: "3.753e+128"}


def g(v):
    """A value as the snippet prints it (4 significant figures), in house style: 1.019e-09 as 1.019 × 10 to the
    −9, -3.43e+04 as −3.43 × 10 to the 4, 1805 as 1,805, -5 with a true minus."""
    s = f"{v:.4g}"
    if "e" in s:
        m, e = s.split("e")
        return rich(num(m), f" {TIMES} ", sup("10", num(int(e)), italic=False))
    return num(int(s)) if abs(float(s)) >= 1000 else num(s)


W = var("w")
def steps(lr):
    """The note under a panel: w at the start and after 1, 2 and 3 steps (2 for the run that leaves the chart at
    once), then after 100, as the snippet's table prints them."""
    first = ", ".join(g(v) for v in PATHS[lr][:3 if lr == 10.0 else 4])
    return rich(W, ": ", first, "; ", g(PATHS[lr][100]), " after 100 steps")


PANELS = [(0.001, "crawls"), (0.1, "converges"), (1.0, "oscillates"), (10.0, "diverges")]
assert [f"{v:.4g}" for v in PATHS[0.1][:4]] == ["5", "4", "3.2", "2.56"]
assert [f"{v:.4g}" for v in PATHS[1.0][:4]] == ["5", "-5", "5", "-5"]
assert [f"{v:.4g}" for v in PATHS[10.0][:3]] == ["5", "-95", "1805"]

fig = Figure(
    "03-learning-rate", "Too small crawls, too large bounces or diverges",
    "Four small charts of the loss f(w) = w squared for w from minus 6 to 6, each with the start w = 5 and the "
    "first three steps of gradient descent drawn as dots on the curve joined by straight jumps, and w after "
    "100 steps as a ring. Learning rate 0.001: the dots 5, 4.99, 4.98, 4.97 sit on top of one another, and "
    "after 100 steps w is 4.093. "
    "Learning rate 0.1: w goes 5, 4, 3.2, 2.56 down the right side of the valley and is 1.019 times 10 to the "
    "minus 9 after 100 steps. Learning rate 1: w jumps between 5 and minus 5 at loss 25, and is 5 again after "
    "100 steps. Learning rate 10: the first jump leaves the chart, to minus 95 and then 1,805, and w is 3.753 "
    "times 10 to the 128 after 100 steps.",
    subtitle=rich(var("f"), "(", W, ") = ", sup("w", "2"), ", start at ", W, " = 5, update ", W, " ← ", W,
                  " ", MINUS, " ", var("α"), " ", CDOT, " 2", W, "; dots: the start and 3 steps; ring: step 100."),
    height=720)

# the axis name w is printed once per column, under the bottom row; the top row's boxes are 24 shorter so all
# four plots keep one height
top_row = fig.row(2, y=104, h=252)
bottom_row = fig.row(2, y=380, h=276)
X_LO, X_HI, Y_LO, Y_HI = -6, 6, 0, 36
ws = np.linspace(X_LO, X_HI, 121)
curve = dict(xs=ws.tolist(), ys=[f(v) for v in ws], color="ink-muted", points=False, label=None)

plots = []
for k, (box, (lr, verb)) in enumerate(zip(top_row + bottom_row, PANELS)):
    body = fig.panel(box, rich(var("α"), f" = {lr:g}: {verb}"))
    chart = Box(body.x, body.y, body.w, body.h - 32)
    ax = fig.line_chart(chart, [curve], x=(X_LO, X_HI, [-5, 0, 5]), y=(Y_LO, Y_HI, [0, 25]),
                        x_label=W if k >= 2 else None, y_label=rich(var("f"), "(", W, ")"),
                        label_w=8, fmt_x=lambda v: num(v), fmt_y=lambda v: str(v))
    plots.append((ax.y, ax.h))
    path = PATHS[lr]
    shown = [v for v in path[:4]]
    with fig.data():
        pts = [ax.to_px(v, f(v)) for v in shown]
        for (x0, y0), (x1, y1), v0, v1 in zip(pts, pts[1:], shown, shown[1:]):
            if f(v1) <= Y_HI:
                fig.edge((x0, y0), (x1, y1), color="gradient", width=1.5)
            else:
                # the jump leaves the chart: the segment is clipped at the plot's top edge, where it exits
                ax.segment(v0, f(v0), v1, f(v1), color="gradient", width=1.5)
                ax.text(v0, Y_HI, rich("to ", W, " = ", num(v1)), "note", anchor="end", dx=-12, dy=20)
                break
        for v, (x, y) in zip(shown, pts):
            if f(v) <= Y_HI:
                fig.marker(x, y, "circle", "gradient", size=8)
        if lr == 0.001:
            # 5, 4.99, 4.98, 4.97 are a third of a pixel apart at this scale: say so beside the one dot they make
            assert abs(ax.to_px(shown[0], 0)[0] - ax.to_px(shown[3], 0)[0]) < 1.5
            ax.text(shown[0], f(shown[0]), "four dots on top of one another", "note", anchor="end", dx=-12,
                    dy=-8)
        end = path[100]
        if abs(end) <= X_HI and f(end) <= Y_HI:
            fig.marker(*ax.to_px(end, f(end)), "circle", "gradient", size=10, hollow=True)
    fig.note(Box(body.x, chart.bottom, body.w, 0), steps(lr))

FACTORS = [1 - 2 * lr for lr in RATES]
assert [f"{v:g}" for v in FACTORS] == ["0.998", "0.8", "-1", "-19"]          # section 5.1's four factors
assert len(set(plots)) == 2 and plots[0][1] == plots[2][1]   # one plot height in both rows
fig.caption(rich("Each step multiplies ", W, " by 1 ", MINUS, " 2", var("α"), ": ",
                 ", ".join(num(v) if v < 0 else f"{v:g}" for v in FACTORS[:3]), " and ", num(FACTORS[3]), "."))
fig.write()
