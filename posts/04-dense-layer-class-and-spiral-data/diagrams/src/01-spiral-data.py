"""Post 04 hero: the 300 spiral points, shaded by the regions of a trained linear layer.

Run from anywhere:  python posts/04-dense-layer-class-and-spiral-data/diagrams/src/01-spiral-data.py
Writes posts/04-dense-layer-class-and-spiral-data/diagrams/01-spiral-data.svg.
The points are the X, y of snippets/spiral_dataset.py (nnfs.init(), then spiral_data(samples=100, classes=3)).
The shaded regions are those of the trained Layer_Dense(2, 3) of snippets/linear_baseline.py, and the counts
118 and 100 are asserted against what that script prints. The kit has no scatter plot, so the marks are drawn
inside fig.data().
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, num  # noqa: E402

SNIPPETS = Path(__file__).resolve().parents[2] / "snippets"


def run(name):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g = runpy.run_path(str(SNIPPETS / name), run_name="snippet")
    return g, out.getvalue()


data, data_out = run("spiral_dataset.py")
base, base_out = run("linear_baseline.py")
X, y = data["X"], data["y"]
assert X.shape == (300, 2) and y.shape == (300,) and list(np.bincount(y)) == [100, 100, 100]
assert "X: (300, 2) float32" in data_out and "y: (300,) uint8" in data_out
assert np.array_equal(X, base["X"]) and np.array_equal(y, base["y"])

# The trained linear layer: its scores are X W + b, and it answers the class with the largest score.
Wt = np.asarray(base["dense"].weights, dtype=float)
bt = np.asarray(base["dense"].biases, dtype=float)[0]
pred = np.argmax(X.astype(float) @ Wt + bt, axis=1)
CORRECT = int(np.sum(pred == y))
FIXED = int(np.max(np.bincount(y)))
assert CORRECT == 118 and FIXED == 100
assert "step 1000: loss 1.0830, 118 of 300 correct, 39.3%" in base_out
assert "always answering one class: 100 of 300 correct, 33.3%" in base_out
PCT, PCT_FIXED = f"{CORRECT / 3:.1f} percent", f"{FIXED / 3:.1f} percent"   # as the post writes it
assert (PCT, PCT_FIXED) == ("39.3 percent", "33.3 percent")

# -- geometry: a square plot of the plane from -1.1 to 1.1 on both axes
LO, HI = -1.1, 1.1
PX, PY, PS = 104, 128, 464                       # plot box x, y, side
sx = lambda v: PX + PS * (v - LO) / (HI - LO)    # noqa: E731
sy = lambda v: PY + PS - PS * (v - LO) / (HI - LO)  # noqa: E731
TICKS = [-1, -0.5, 0, 0.5, 1]
COLORS = ["blue", "orange", "green"]
SHAPES = ["circle", "square", "triangle"]


def region(k):
    """The part of the plot square where class k has the largest score: the square clipped by two half-planes."""
    poly = [(LO, LO), (HI, LO), (HI, HI), (LO, HI)]
    for j in range(3):
        if j == k:
            continue
        a = Wt[:, k] - Wt[:, j]
        c = bt[k] - bt[j]
        f = lambda p: a[0] * p[0] + a[1] * p[1] + c  # noqa: E731
        out = []
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            fp, fq = f(p), f(q)
            if fp >= 0:
                out.append(p)
            if (fp >= 0) != (fq >= 0):
                t = fp / (fp - fq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        poly = out
    return poly


def marker(fig, shape, cx, cy, cls):
    if shape == "circle":
        fig.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" class="{cls}"/>')
    elif shape == "square":
        fig.add(f'<rect x="{cx - 3.5:.2f}" y="{cy - 3.5:.2f}" width="7" height="7" class="{cls}"/>')
    else:
        fig.add(f'<path d="M{cx:.2f},{cy - 5:.2f} L{cx + 4.5:.2f},{cy + 3.5:.2f} L{cx - 4.5:.2f},{cy + 3.5:.2f} Z" '
                f'class="{cls}"/>')


fig = Figure(
    "01-spiral-data", "Three spiral arms that no straight cut separates",
    "A scatter plot of all 300 points that spiral_data(samples=100, classes=3) returns after nnfs.init(), the "
    "first feature X[:, 0] across and the second X[:, 1] up, both from minus 1 to 1. Class 0 is drawn as blue "
    "circles, class 1 as orange squares and class 2 as green triangles, 100 points each, each class an arm "
    "that winds out from the origin. The background is split by straight edges into three tinted regions, "
    "where a trained Layer_Dense(2, 3) answers each class: blue for class 0, orange for class 1, green for "
    "class 2, as the key beside the plot shows. The arms cross every edge: the layer gets "
    f"{CORRECT} of 300 points right, {PCT}, against {FIXED} of 300, {PCT_FIXED}, for always giving one answer. "
    "A side panel lists X: (300, 2) float32 and y: (300,) uint8.",
    subtitle="spiral_data(samples=100, classes=3) after nnfs.init(): all 300 points.",
    height=720)

with fig.data():
    # tinted regions of the trained linear layer, then grid lines, then the points
    for k in range(3):
        pts = region(k)
        d = "M" + " L".join(f"{sx(px):.2f},{sy(py):.2f}" for px, py in pts) + " Z"
        fig.add(f'<path d="{d}" class="{fig._cls("f", COLORS[k] + "-soft")} {fig._cls("s", "ink-muted")} w1"/>')
    d = "".join(f"M{sx(t):.2f},{PY}V{PY + PS}" for t in TICKS if t != 0)
    d += "".join(f"M{PX},{sy(t):.2f}H{PX + PS}" for t in TICKS if t != 0)
    fig.add(f'<path d="{d}" class="gridline"/>')
    for k in range(3):
        cls = f'{fig._cls("f", COLORS[k])} {fig._cls("s", "surface")} w1'
        for (px, py) in X[y == k]:
            marker(fig, SHAPES[k], sx(float(px)), sy(float(py)), cls)
    for t in TICKS:
        lab = num(t) if t != int(t) else num(int(t))
        fig.text(sx(t), PY + PS + 20, lab, "tick", anchor="middle", snap=False)
        fig.text(PX - 8, sy(t) + 4, lab, "tick", anchor="end", snap=False)
fig.add(f'<rect x="{PX}" y="{PY}" width="{PS}" height="{PS}" class="frame" data-fit="skip"/>')
fig.text(PX, PY - 12, "X[:, 1], second feature", "note")
fig.text(PX + PS // 2, PY + PS + 44, "X[:, 0], first feature", "note", anchor="middle")

# -- side panel
RX = 624
fig.text(RX, 152, "Key: 100 points per class", "head")
for k in range(3):
    by = 192 + 32 * k
    # the key: each class's mark on its class's tint, so the tint of a region names its class
    fig.add(f'<rect x="{RX}" y="{by - 16}" width="24" height="24" '
            f'class="{fig._cls("f", COLORS[k] + "-soft")} {fig._cls("s", COLORS[k] + "-line")} w1"/>')
    with fig.data():
        marker(fig, SHAPES[k], RX + 12, by - 4, f'{fig._cls("f", COLORS[k])} {fig._cls("s", "surface")} w1')
    fig.text(RX + 36, by, f"class {k}, rows {100 * k} to {100 * k + 99}", "label")
fig.text(RX, 320, "X: (300, 2) float32", "code")
fig.text(RX, 344, "y: (300,) uint8", "code")

fig.text(RX, 424, "Tints: a trained Layer_Dense(2, 3)", "head")
fig.text(RX, 456, "Each tint, as in the key, is where", "note")
fig.text(RX, 476, "the layer answers that class.", "note")
fig.text(RX, 528, f"{CORRECT} of 300 correct, {PCT}", "value", color="ink", bold=True)
fig.text(RX, 556, f"one fixed answer: {FIXED} of 300, {PCT_FIXED}", "note")

fig.caption("Straight edges cut every arm into pieces; the layer does only a little better than chance.")
fig.write()
