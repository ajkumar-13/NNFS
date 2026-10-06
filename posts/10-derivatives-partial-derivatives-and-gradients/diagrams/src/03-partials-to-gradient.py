"""Post 10 section 4: each partial derivative of example 2 is the slope of a one-variable slice, and the
three slopes at (1, 2, 3) are the gradient (27, 2, 12).

Run from anywhere:  python posts/10-derivatives-partial-derivatives-and-gradients/diagrams/src/03-partials-to-gradient.py
Writes posts/10-derivatives-partial-derivatives-and-gradients/diagrams/03-partials-to-gradient.svg.
The function, its freezing-rule gradient and its numerical gradient are those of
snippets/partials_and_gradient.py, which this script runs. Each slice is the snippet's f evaluated with two
coordinates held at (1, 2, 3), asserted against the one-variable polynomial printed in the figure; each slope
and each central difference is asserted against the line the snippet prints.
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
from figkit import Figure, Box, rich, var, sup, coef, num  # noqa: E402

PARTIAL, NABLA, MINUS, SQRT, APPROX = "\u2202", "\u2207", "\u2212", "\u221a", "\u2248"
# After a raised script the empty space under it already reads as part of the gap, so a full space before the
# next operator looks doubled; a thin space (TS) there matches the gaps between the other terms.
TS = "\u2009"
SNIP = Path(__file__).resolve().parents[2] / "snippets" / "partials_and_gradient.py"
ns = runpy.run_path(str(SNIP), run_name="snippet")
with contextlib.redirect_stdout(io.StringIO()) as out:
    ns["main"]()
OUT = out.getvalue()

f, grad_f, numerical_gradient = ns["f"], ns["grad_f"], ns["numerical_gradient"]
P = np.array([1.0, 2.0, 3.0])
F0 = f(P)
assert F0 == 32 and "f(1, 2, 3) = 32" in OUT
G = grad_f(P)
NG = numerical_gradient(f, P)
assert list(G) == [27.0, 2.0, 12.0]
for name, formula, a, n in zip(("df/dx", "df/dy", "df/dz"), ("9x^2 z", "-2y + 2z", "3x^3 + 5 + 2y"), G, NG):
    line = f"{name} = {formula:<14} freezing rule {a:5.1f}   central difference {n:12.8f}"
    assert line in OUT, line
NG_TXT = [f"{n:.8f}" for n in NG]
assert NG_TXT == ["27.00000000", "2.00000000", "12.00000000"]
LENGTH = float(np.sqrt(np.sum(G * G)))
assert f"sqrt(877) = {LENGTH:.6f}" in OUT and f"{LENGTH:.2f}" == "29.61" and int(np.sum(G * G)) == 877

# The three slices through (1, 2, 3): the snippet's f with two coordinates frozen, and the polynomial each is.
NAMES = ("x", "y", "z")
SLICES = [
    dict(k=0, poly=lambda t: 9 * t ** 3 + 23, shown=rich(coef(9, sup("x", "3")), TS, "+ 23"),
         rule=rich(coef(9, sup("x", "2")), var("z"))),
    dict(k=1, poly=lambda t: -t * t + 6 * t + 24, shown=rich(MINUS, sup("y", "2"), TS, "+ ", coef(6, "y"), " + 24"),
         rule=rich(MINUS, "2", var("y"), " + 2", var("z"))),
    dict(k=2, poly=lambda t: 12 * t - 4, shown=rich(coef(12, "z"), " ", MINUS, " 4"),
         rule=rich(coef(3, sup("x", "3")), TS, "+ 5 + ", coef(2, "y"))),
]
for s in SLICES:
    def along(t, k=s["k"]):
        q = P.copy()
        q[k] = t
        return float(f(q))
    s["along"] = along
    for t in np.linspace(P[s["k"]] - 0.6, P[s["k"]] + 0.6, 13):
        assert abs(along(t) - s["poly"](t)) < 1e-9


def frozen(k):
    """'y = 2, z = 3 frozen' for the slice along axis k."""
    parts = []
    for j in range(3):
        if j != k:
            parts += [", " if parts else "", var(NAMES[j]), f" = {int(P[j])}"]
    return rich(*parts)


fig = Figure(
    "03-partials-to-gradient", "Three slopes along three axes make the gradient",
    "Three panels slice f(x, y, z) = 3 x cubed z minus y squared plus 5z plus 2yz through the point (1, 2, 3), "
    "where f = 32, each along one axis with the other two variables frozen, all on the same f axis from 20 to "
    "60. Along x, with y = 2 and z = 3, the slice is 9 x cubed plus 23 and its tangent at x = 1 has slope 9 x "
    "squared z = 27. Along y, with x = 1 and z = 3, the slice is minus y squared plus 6y plus 24, slope minus 2y "
    "plus 2z = 2. Along z, with x = 1 and y = 2, the slice is 12z minus 4, slope 3 x cubed plus 5 plus 2y = 12. "
    "Arrows carry the three slopes into one vector, the gradient of f at (1, 2, 3), (27, 2, 12). Its central "
    "differences are 27.00000000, 2.00000000 and 12.00000000, and its length is the square root of 877, about "
    "29.61.",
    subtitle=rich(var("f"), "(", var("x"), ", ", var("y"), ", ", var("z"), ") = ", coef(3, sup("x", "3")), var("z"), " ",
                  MINUS, " ", sup("y", "2"), TS, "+ 5", var("z"), " + 2", var("y"), var("z"),
                  ", sliced through the point (1, 2, 3), where ", var("f"), " = 32."),
    height=720)

panels = fig.row(3, y=104, h=368)
YL, YH = 20, 60
tops = []
for s, box in zip(SLICES, panels):
    k = s["k"]
    v, p, m = NAMES[k], float(P[k]), float(G[k])
    body = fig.panel(box, rich("Along ", var(v), "; ", frozen(k), " frozen"))
    ts = [p - 0.6 + 1.2 * i / 60 for i in range(61)]
    cur = dict(xs=ts, ys=[s["along"](t) for t in ts], color="output", points=False, label=None)
    chart = Box(body.x, body.y, body.w, body.h - 56)
    ax = fig.line_chart(chart, [cur], x=(p - 0.6, p + 0.6, [p - 0.5, p, p + 0.5]), y=(YL, YH, [20, 40, 60]),
                        x_label=var(v), y_label=var("f"), label_w=16, points=False, fmt_x=lambda t: num(t, 1))
    with fig.data():
        px, py = ax.to_px(p, F0)
        qx, qy = ax.to_px(p + 1, F0 + m)
        n = math.hypot(qx - px, qy - py)
        ux, uy = (qx - px) / n, (qy - py) / n
        fig.edge((px - 56 * ux, py - 56 * uy), (px + 56 * ux, py + 56 * uy), color="gradient", width=1.5)
        fig.marker(px, py, "circle", "gradient", size=10)
    # under the chart: the slice as a polynomial, then its slope by the freezing rule
    fig.text(ax.cx, chart.bottom + 28, rich("slice: ", s["shown"]), "note", anchor="middle")
    fig.text(ax.cx, chart.bottom + 56, rich(PARTIAL, var("f"), "/", PARTIAL, var(v), " = ", s["rule"], " = ",
                                           num(int(m))), "value", color="gradient", anchor="middle")
    tops.append(ax.cx)

# -- the gradient: the three slopes as one vector, centred under the middle panel so its arrow runs straight
# down; the outer two arrows drop, run level at y = 520, and drop again onto their cells
SY, CW = 552, 72
SX = tops[1] - 3 * CW // 2
assert SX % 8 == 0 and tops[1] == SX + 3 * CW / 2
g = fig.strip(SX, SY, 3, cell=CW, height=48, values=[num(int(v)) for v in G], font=18,
              fill=lambda i: "gradient-soft")
fig.text(SX - 16, SY + 30, rich(NABLA, var("f"), "(1, 2, 3) ="), "math", anchor="end")
for i, cx in enumerate(tops):
    c = g.cell(0, i)
    fig.arrow((cx, 488), (c.cx, c.y), via=[(cx, 520), (c.cx, 520)] if cx != c.cx else ())
fig.text(g.box.right + 24, SY + 18, rich("length ", SQRT, "877 ", APPROX, " ", f"{LENGTH:.2f}"), "note")
fig.text(g.box.right + 24, SY + 38, "points uphill", "note")
fig.note(g.box, rich("central differences: ", ", ".join(NG_TXT)), anchor="middle")

fig.caption(rich("Each component is the slope of ", var("f"), " along one axis, the other two held fixed."))
fig.write()
