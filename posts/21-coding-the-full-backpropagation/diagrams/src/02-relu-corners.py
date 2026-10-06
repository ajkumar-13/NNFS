"""Post 21, section 11: the ten samples whose hidden pre-activations lie within h of zero on the 0.01 initialisation.

Run from anywhere:  python posts/21-coding-the-full-backpropagation/diagrams/src/02-relu-corners.py
Writes posts/21-coding-the-full-backpropagation/diagrams/02-relu-corners.svg.
The data and the network are those of snippets/gradient_check.py, built here with its own build(seed=0,
redraw=False): seed 0, float64, the 0.01 weights and zero biases of the script. With zero biases, each hidden
neuron's pre-activation is 0 on a line through the origin, w1 x1 + w2 x2 = 0, drawn from dense1.weights. The
samples with an entry of Z1 within h = 1e-5 of 0 are computed here, and their counts and distances are asserted
against what snippets/what_can_go_wrong.py prints (run here as a subprocess, under a second; it calls nnfs.init(),
which must not touch this process).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure, Box, rich, arr, var, sub, sup, num  # noqa: E402
from gradient_check import build  # noqa: E402

OUT = subprocess.run([sys.executable, str(POST / "snippets" / "what_can_go_wrong.py")], capture_output=True,
                     text=True, check=True, cwd=str(POST / "snippets")).stdout
BODY = (POST / "index.md").read_text(encoding="utf-8")

H = 1e-5
dense1, activation1, dense2, loss_activation, X, y = build(seed=0, redraw=False)
dense1.forward(X)
Z1 = dense1.output
assert X.dtype == np.float64 and X.shape == (300, 2) and not np.any(dense1.biases)
near = np.any(np.abs(Z1) < H, axis=1)
origin = np.all(X == 0, axis=1)
others = near & ~origin
dist = np.linalg.norm(X[others], axis=1)
line = (f"largest |Z1| {np.max(np.abs(Z1)):.1e}; samples with an entry of Z1 within h of 0: "
        f"{int(near.sum())}, of which at the origin: {int(origin.sum())}")
assert line in OUT and line in BODY, line
assert (int(near.sum()), int(origin.sum()), int(others.sum())) == (10, 3, 7)
assert sorted(y[origin].tolist()) == [0, 1, 2]                          # three different labels
assert f"near-corner samples away from the origin: 7, at distances {dist.min():.2f} to {dist.max():.2f}" in OUT
assert (f"{dist.min():.2f}", f"{dist.max():.2f}") == ("0.02", "0.71") and "between 0.02 and 0.71" in BODY
assert int(np.sum(Z1 == 0)) == 9                                       # "entries of Z1 exactly 0: 9"
assert "entries of Z1 exactly 0: 9; within h of 0: 16" in OUT
NEURON = [int(np.argmin(np.abs(Z1[i]))) for i in np.where(others)[0]]   # the neuron each of the seven is near
PER = [NEURON.count(k) for k in range(3)]
W = dense1.weights

fig = Figure(
    "02-relu-corners", "Ten samples sit on a corner of a hidden ReLU",
    "Two scatter plots of the 300 spiral points of seed 0 in float64, class 0 as blue circles, class 1 as orange "
    "squares and class 2 as green triangles: on the left the whole plane from minus 1.1 to 1.1, on the right the "
    "square from minus 0.2 to 0.2 around the origin, enlarged. Three grey lines through the origin mark where each "
    "of the three hidden neurons has a pre-activation of 0, from the script's 0.01 weights and zero biases. Red "
    "rings mark the ten samples with an entry of Z1 within h = 10 to the minus 5 of 0: three at the origin, one of "
    f"each class, and seven on the lines ({PER[0]} on neuron 1, {PER[1]} on neuron 2, {PER[2]} on neuron 3), "
    "between 0.02 and 0.71 from the origin. Only the one at 0.71 lies outside the enlarged square.",
    subtitle="The first check of section 11: seed 0, float64, the script's 0.01 weights and zero biases.",
    height=720, data_w=True)

CLASS = [("class 0", "blue", "circle"), ("class 1", "orange", "square"), ("class 2", "green", "triangle")]
RING = 20


def tick(t):
    return num(int(t)) if t == int(t) else num(t)


def panel(box, lo, hi, ticks, heading):
    fig.text(box.x + 56, 128, heading, "head")
    ax = fig.scatter(box, [], x=(lo, hi, ticks), y=(lo, hi, ticks), x_label="X[:, 0]", y_label="X[:, 1]",
                     fmt_x=tick, fmt_y=tick)
    assert abs(ax.w - ax.h) < 1e-9
    with fig.data():
        for k in range(3):                                             # z_k = w1k x1 + w2k x2 = 0
            d = np.array([-W[1, k], W[0, k]]) / np.hypot(W[0, k], W[1, k])
            ax.segment(-3 * d[0], -3 * d[1], 3 * d[0], 3 * d[1], "ink-muted", width=1)
        for k, (_, color, shape) in enumerate(CLASS):
            for px, py in X[y == k]:
                ax.point(float(px), float(py), shape, color, size=8)
        for px, py in X[near]:
            if lo <= px <= hi and lo <= py <= hi:
                ax.point(float(px), float(py), "circle", "error", size=RING, hollow=True)
    return ax


BOX_L = Box(40, 136, 424, 416)
BOX_R = Box(496, 136, 424, 416)
axL = panel(BOX_L, -1.1, 1.1, [-1, -0.5, 0, 0.5, 1], "All 300 points")
axR = panel(BOX_R, -0.2, 0.2, [-0.2, -0.1, 0, 0.1, 0.2], "The dashed square, enlarged")
assert int(np.sum(np.all(np.abs(X[near]) <= 0.2, axis=1))) == 9          # all but the one at 0.71

# the enlarged square, dashed, on the left plot
with fig.data():
    x0, y0 = axL.to_px(-0.2, 0.2)
    x1, y1 = axL.to_px(0.2, -0.2)
    fig.outline(Box(x0, y0, x1 - x0, y1 - y0), "ink-muted", width=1, dash="lead")


def end(k, lo, hi, upper):
    """Where neuron k's line leaves the square [lo, hi]^2: its end in the upper half, or the right end."""
    d = np.array([-W[1, k], W[0, k]]) / np.hypot(W[0, k], W[1, k])
    if (upper and d[1] < 0) or (not upper and d[0] < 0):
        d = -d
    t = min(hi / abs(d[0]), hi / abs(d[1]))
    return t * d


# each line's number just outside the frame where it leaves: 1 and 2 over the top, 3 right of the right side
with fig.data():
    for ax, lo, hi in ((axL, -1.1, 1.1), (axR, -0.2, 0.2)):
        for k in range(3):
            ex, ey = end(k, lo, hi, upper=k < 2)
            px, py = ax.to_px(ex, ey)
            if k < 2:
                assert abs(ey - hi) < 1e-12
                fig.text(px, ax.y - 8, str(k + 1), "tick", anchor="middle")
            else:
                assert abs(ex - hi) < 1e-12
                fig.text(ax.right + 8, py + 5, str(k + 1), "tick")
        # the two labels over the top must stay apart
        a, b = (ax.to_px(*end(k, lo, hi, True))[0] for k in (0, 1))
        assert abs(a - b) >= 16, (a, b)

# -- the key, two rows under the plots
YK = 604
items = [dict(color=c, label=lab, mark=shp) for lab, c, shp in CLASS]
items.append(dict(color="ink-muted", label=rich("line ", var("k"), ": neuron ", var("k"), " has ", var("z"), " = 0"),
                  mark="line"))
fig.legend(axL.x, YK, items, direction="row", gap=24)
with fig.data():
    fig.marker(axL.x + 8, YK + 36 - 4, "circle", "error", size=RING, hollow=True)
fig.text(axL.x + 32, YK + 36, rich("an entry of ", arr("Z", sub="1"), " within ", var("h"), " = ",
                                   sup("10", num(-5), italic=False), " of 0: 3 samples at the origin, one per class, "
                                   "and 7 on the lines"), "label")
fig.caption("The three at the origin cancel; the seven on the lines make the large gaps of dense1.")
fig.write()
