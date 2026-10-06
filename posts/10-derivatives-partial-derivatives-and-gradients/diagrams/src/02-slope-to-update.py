"""Post 10 section 2.3: the sign of dL/dw decides which way the update of section 1 moves a weight.

Run from anywhere:  python posts/10-derivatives-partial-derivatives-and-gradients/diagrams/src/02-slope-to-update.py
Writes posts/10-derivatives-partial-derivatives-and-gradients/diagrams/02-slope-to-update.svg.
The loss is L(w) = (w - 1)^2, the one-weight loss of section 4.3 and of snippets/what_can_go_wrong.py; the
curve, the three tangents and their slopes 2(w - 1) are evaluated here. The slope at w = 2 is also measured
with that snippet's central_difference and asserted against the line it prints ("dL/dw at w = 2: 2.000000").
The post fixes no learning rate for this loss, so the update is written with the symbol alpha of section 1
and no step is drawn to scale: the arrows show only the direction of each move.
"""
import contextlib
import io
import math
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num, CDOT  # noqa: E402

ALPHA, MINUS, ARROW, PARTIAL = "\u03b1", "\u2212", "\u2190", "\u2202"
SNIP = Path(__file__).resolve().parents[2] / "snippets" / "what_can_go_wrong.py"
ns = runpy.run_path(str(SNIP), run_name="snippet")
with contextlib.redirect_stdout(io.StringIO()) as out:
    ns["main"]()
OUT = out.getvalue()


def L(w):
    return (w - 1.0) * (w - 1.0)


def dL(w):
    return 2.0 * (w - 1.0)


H = 1e-5
measured = ns["central_difference"](L, 2.0, H)
assert f"dL/dw at w = 2: {measured:.6f}" in OUT and f"{measured:.6f}" == "2.000000"
POINTS = (2.0, 1.0, 0.0)
SLOPES = [dL(w) for w in POINTS]
assert SLOPES == [2.0, 0.0, -2.0] and L(2.0) == 1.0 and L(1.0) == 0.0 and L(0.0) == 1.0
for w, m in zip(POINTS, SLOPES):
    assert abs(ns["central_difference"](L, w, H) - m) < 1e-9


def signed(v):
    return ("+" if v > 0 else "") + num(int(v))


DLDW = rich(PARTIAL, var("L"), "/", PARTIAL, var("w"))

fig = Figure(
    "02-slope-to-update", "The sign of the slope decides where the weight moves",
    "On the left, the loss L(w) = (w minus 1) squared for w from minus 1 to 3, with its tangent at w = 2, "
    "w = 1 and w = 0, labelled slope plus 2, slope 0 and slope minus 2. A short arrow at w = 2 points to the left "
    "and one at w = 0 points to the right, both towards w = 1, where the loss is lowest. On the right, three cards "
    "apply the update w gets w minus alpha times dL/dw: at w = 2, w gets 2 minus 2 alpha, so w falls; at w = 1, "
    "w gets 1 minus alpha times 0, which is 1, so w stays; at w = 0, w gets 0 minus alpha times minus 2, which "
    "is 2 alpha, so w rises.",
    subtitle=rich(var("L"), "(", var("w"), ") = ", sup(rich("(", var("w"), " ", MINUS, " 1)"), "2", italic=False),
                  ", the loss of section 4.3; ", ALPHA, " > 0 is the learning rate."))

# -- the chart
XL, XH, YL, YH = -1, 3, -0.5, 2.5
xs = [-0.5 + 3.0 * k / 120 for k in range(121)]      # w from -0.5 to 2.5, where L stays under 2.25
CURVE = dict(xs=xs, ys=[L(w) for w in xs], color="error", points=False,
             label=rich(var("L"), "(", var("w"), ")"))
ax = fig.line_chart(Box(40, 104, 520, 372), [CURVE], x=(XL, XH, [-1, 0, 1, 2, 3]), y=(YL, YH, [0, 1, 2]),
                    x_label=var("w"), y_label=rich("loss ", var("L")), label_w=64, points=False)

HALF = 56
LABEL_AT = {2.0: ("start", 16, 36), 1.0: ("middle", 0, 32), 0.0: ("end", -16, 36)}
with fig.data():
    for w, m in zip(POINTS, SLOPES):
        px, py = ax.to_px(w, L(w))
        qx, qy = ax.to_px(w + 1, L(w) + m)
        n = math.hypot(qx - px, qy - py)
        ux, uy = (qx - px) / n, (qy - py) / n
        fig.edge((px - HALF * ux, py - HALF * uy), (px + HALF * ux, py + HALF * uy), color="gradient", width=1.5)
    for w, m in zip(POINTS, SLOPES):
        px, py = ax.to_px(w, L(w))
        fig.marker(px, py, "circle", "gradient", size=10)
        anchor, dx, dy = LABEL_AT[w]
        fig.text(px + dx, py + dy, f"slope {signed(m)}", "value", color="gradient", anchor=anchor, snap=False)
# the direction of each move, inside the bowl at the height of its point
for w, m in zip(POINTS, SLOPES):
    if m == 0:
        continue
    px, py = ax.to_px(w, L(w))
    d = -1 if m > 0 else 1                           # the rule subtracts: a positive slope moves w left
    fig.arrow((px + d * 16, py), (px + d * 72, py))

# -- the three cards
CX, CW, CH, GAP = 592, 328, 88, 16
fig.text(CX, 128, rich("Update: ", var("w"), " ", ARROW, " ", var("w"), " ", MINUS, " ", ALPHA, " ", CDOT, " ",
                       DLDW), "head")
CARDS = [
    (2.0, rich(var("w"), " ", ARROW, " 2 ", MINUS, " 2", ALPHA), "w falls"),
    (1.0, rich(var("w"), " ", ARROW, " 1 ", MINUS, " ", ALPHA, " ", CDOT, " 0 = 1"), "w stays"),
    (0.0, rich(var("w"), " ", ARROW, " 0 ", MINUS, " ", ALPHA, " ", CDOT, " (", MINUS, "2) = 2", ALPHA), "w rises"),
]
for k, (w, line, verdict) in enumerate(CARDS):
    y = 144 + k * (CH + GAP)
    m = dL(w)
    body = fig.card(CX, y, CW, CH, heading=rich("At ", var("w"), f" = {int(w)}, ", DLDW, f" = {signed(m)}"))
    fig.text(body.x, body.y + 16, rich(line, ", so ", var("w"), verdict[1:]), "math")   # the verdict right after its update

fig.caption(rich("Every move heads for ", var("w"), " = 1, where the loss is lowest."))
fig.write()
