"""Post 24 hero (sections 1 and 2): the 100-step paths on the ravine, without momentum and with momentum 0.9.

Run from anywhere:  python posts/24-momentum/diagrams/src/01-ravine-paths.py
Writes posts/24-momentum/diagrams/01-ravine-paths.svg.
snippets/ravine.py is run here (runpy, about a second): its run() and Optimizer_SGD give both paths, from (10, 1)
with learning rate 0.019, and every number in the right-hand columns is asserted against the line ravine.py
prints for that coefficient (x after 100 steps, the loss, the travel across and along, the reversals). The contour
ellipses are the levels of the snippet's own loss(), L(x, y) = (x^2 + 100 y^2) / 2.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num, coef  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets"
sys.path.insert(0, str(SNIP))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIP / "ravine.py"), run_name="__main__")
OUT = buf.getvalue()
run, loss, Opt, START = s["run"], s["loss"], s["Optimizer_SGD"], s["START"]
ALPHA = 0.019
assert f"learning rate {ALPHA}" in OUT and loss(START) == 100.0
assert loss(np.array([[3.0, 0.5]])) == (9 + 100 * 0.25) / 2            # the post's L(x, y) = (x^2 + 100 y^2)/2


def stats(beta):
    """The row ravine.py prints for this coefficient, recomputed from the same path, and asserted."""
    path = run(Opt(learning_rate=ALPHA, momentum=beta), 100)
    steps = np.diff(path, axis=0)
    across = steps[np.abs(steps[:, 1]) > 1e-9, 1]
    rev = int(np.sum(across[1:] * across[:-1] < 0))
    row = (f"{beta:4.2f}  {path[-1, 0]:7.4f}  {abs(path[-1, 1]):.2e}  {loss(path[-1]):9.4f}"
           f"  {np.abs(steps[:, 1]).sum():13.3f}  {np.abs(steps[:, 0]).sum():12.3f}  {rev:25d}")
    assert row in OUT, row
    return path, dict(x=path[-1, 0], loss=loss(path[-1]), across=np.abs(steps[:, 1]).sum(),
                      along=np.abs(steps[:, 0]).sum(), rev=rev)


PLAIN, SP = stats(0.0)
HEAVY, SH = stats(0.9)
assert (f"{SP['x']:.4f}", f"{SP['loss']:.4f}", f"{SP['across']:.3f}", f"{SP['along']:.3f}", SP["rev"]) == \
    ("1.4686", "1.0784", "18.999", "8.531", 99)
assert (f"{SH['x']:.4f}", f"{SH['loss']:.4f}", f"{SH['across']:.3f}", f"{SH['along']:.3f}", SH["rev"]) == \
    ("0.0523", "0.0027", "18.902", "17.834", 49)
XMIN = HEAVY[:, 0].min()
KMIN = int(HEAVY[:, 0].argmin())
assert f"{XMIN:.2f}" == "-2.84" and PLAIN[:, 0].min() > 1.4
LEVELS = [2, 8, 18, 32, 50, 72, 100]                                     # semi-axes 2, 4, ..., 12 and 14.14 along x

fig = Figure(
    "01-ravine-paths", "Same bounce across, twice the travel along the floor",
    "Two charts of the ravine L(x, y) = (x squared + 100 y squared)/2, x from minus 4 to 11 along the floor, y "
    "from minus 1.2 to 1.2 across it, drawn at twice the scale of x, with the elliptical contours L = 2, 8, 18, 32, "
    "50, 72 and 100 and the minimum at (0, 0). Both paths start at (10, 1), where L is 100, and take 100 steps with "
    "learning rate 0.019. Top, no momentum: the path zig-zags from wall to wall while it creeps along the floor and "
    "ends at x = 1.4686 with loss 1.0784; it travels 18.999 across and 8.531 along, and the across step reverses "
    "99 times. Bottom, momentum 0.9: the zig-zag holds still on every second step, the path runs past the minimum "
    "to x = minus 2.84 and comes back, and ends at x = 0.0523 with loss 0.0027; it travels 18.902 across and "
    "17.834 along, and the across step reverses 49 times. A key above the charts names the start, the "
    "point after 100 steps, the minimum and the contours.",
    subtitle=rich(var("L"), " = (", sup("x", "2"), " + ", coef(100, sup("y", "2")), ")/2 from (10, 1), ",
                  "learning rate 0.019, 100 steps. ", var("y"), " is drawn at twice the scale of ", var("x"), "."),
    height=720, data_w=True)

X_AX = (-4, 11, [-4, -2, 0, 2, 4, 6, 8, 10])
Y_AX = (-1.2, 1.2, [-1, 0, 1])
LW = 232                                                   # the right-hand column: heading and the run's numbers
T = np.linspace(0, 2 * np.pi, 241)


def panel(box, path, st, heading, x_label):
    ax = fig.line_chart(box, [dict(xs=list(path[:, 0]), ys=list(path[:, 1]), color="weight", points=False,
                                   label=None)],
                        x=X_AX, y=Y_AX, y_label=rich(var("y"), ", across"), x_label=x_label, label_w=LW,
                        fmt_y=lambda v: num(v))
    for L in LEVELS:
        a, b = np.sqrt(2 * L), np.sqrt(2 * L / 100)
        ax.polyline(list(zip(a * np.cos(T), b * np.sin(T))), color="border", width=1)
    ax.point(0, 0, "circle", "ink-muted", size=8, hollow=True)
    ax.point(10, 1, "circle", "weight", size=10)
    ax.point(path[-1, 0], path[-1, 1], "diamond", "weight", size=10)
    # the right-hand column
    cx = ax.right + 32
    fig.text(cx, ax.y + 16, heading, "head")
    fig.text(cx, ax.y + 44, "after 100 steps", "note")
    rows = [rich(var("x"), " = ", num(round(st["x"], 4), 4)), rich("loss ", num(round(st["loss"], 4), 4)),
            f"travel across {st['across']:.3f}", f"travel along {st['along']:.3f}", f"reversals across {st['rev']}"]
    for k, t in enumerate(rows):
        fig.text(cx, ax.y + 68 + 20 * k, t, "label")
    return ax


A = panel(Box(40, 112, 880, 248), PLAIN, SP, "No momentum", None)
B = panel(Box(40, 384, 880, 272), HEAVY, SH, "Momentum 0.9", rich(var("x"), ", along the floor"))
B.segment(XMIN, 0.88, XMIN, HEAVY[KMIN, 1] + 0.1, color="arrow", width=1, dash="lead")
B.text(XMIN, 0.88, rich("runs past the minimum to ", var("x"), " = ", num(round(XMIN, 2), 2)), "note",
       dx=-8, dy=-8)
fig.legend(296, A.y - 12, [dict(color="weight", label="start, (10, 1)", mark="circle"),
                                      dict(color="weight", label="after 100 steps", mark="diamond"),
                                      dict(color="ink-muted", label="minimum, (0, 0)", mark="circle", hollow=True),
                                      dict(color="border", label=rich("contours of ", var("L")), mark="line")],
           direction="row")

fig.caption("Momentum keeps the bounce across about the same and doubles the distance covered along the floor.")
fig.write()
