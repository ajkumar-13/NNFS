"""Post 25, sections 3 and 4: under a constant gradient the cache grows as t and the effective rate falls as 1/sqrt(t).

Run from anywhere:  python posts/25-adagrad/diagrams/src/02-constant-gradient.py
Writes posts/25-adagrad/diagrams/02-constant-gradient.svg.
snippets/adagrad.py is imported here (runpy, without its main block). Its Optimizer_Adagrad and run_toy follow a
parameter with the constant gradient 0.5 and alpha = 1, as section 3 does, and every value of section 3's table is
asserted against the listing in index.md (which the series lint checks against the snippet's output). The same
class is run for 1,000 steps with the gradients 2.0 and 0.2, the two parameters section 3 names. The curves are
the formulas of sections 3 and 4: rate alpha / (|g| sqrt(t)), AdaGrad distance sum of 1/sqrt(k) for |g| = 0.5,
gradient descent distance 0.5 t; each is asserted against the class at the drawn points.

Layout: two log-log charts side by side, the effective rate left and the distance moved right.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path(str(POST / "snippets" / "adagrad.py"), run_name="snippet")
INDEX = (POST / "index.md").read_text(encoding="utf-8")
Adagrad, run_toy = s["Optimizer_Adagrad"], s["run_toy"]
ALPHA, G0 = 1.0, 0.5
TS = (1, 2, 10, 100, 1000, 10000)

# -- section 3's table, from the class: cache, root, rate and distance after t updates of the gradient 0.5
TABLE = {}
for t in TS:
    layer = run_toy(Adagrad(learning_rate=ALPHA), t, lambda w: np.array([[G0]]), [0.0])
    cache = layer.weight_cache[0, 0]
    TABLE[t] = (cache, 1 / np.sqrt(cache), abs(layer.weights[0, 0]))
    line = (f"{t:7d}  {cache:7.2f}  {np.sqrt(cache):8.3f}  {1 / np.sqrt(cache):11.4f}  "
            f"{abs(layer.weights[0, 0]):13.2f}")
    assert line in INDEX, line

# -- the formulas the figure draws, checked against the class
T = np.unique(np.round(np.logspace(0, 4, 241)).astype(int))
rate = lambda g, t: ALPHA / (g * np.sqrt(t))                      # noqa: E731  alpha / sqrt(t g^2)
DIST = np.cumsum(ALPHA * G0 / np.sqrt(G0 ** 2 * np.arange(1, 10001)))   # sum of the steps 1/sqrt(k)
for t in TS:
    cache, r, d = TABLE[t]
    assert abs(cache - 0.25 * t) < 1e-6 * t                       # G_t = 0.25 t
    assert abs(r - rate(G0, t)) < 1e-9 and abs(r - 2 / np.sqrt(t)) < 1e-9
    assert abs(d - DIST[t - 1]) < 1e-4 * d                        # the class moves exactly the sum of the steps
assert f"{0.5 * 10000:,.0f}" == "5,000" and "moves 0.5 at every step, 5,000 in all" in INDEX
assert abs(DIST[-1] - 2 * np.sqrt(10000)) < 2                     # "approaches 2 sqrt(t)"

# -- the two parameters of section 3: |g| = 2.0 and 0.2 after 1,000 updates
SNAP = {}
for g in (2.0, 0.2):
    layer = run_toy(Adagrad(learning_rate=ALPHA), 1000, lambda w, g=g: np.array([[g]]), [0.0])
    cache = layer.weight_cache[0, 0]
    assert abs(cache - 1000 * g ** 2) < 1e-6 * cache and abs(1 / np.sqrt(cache) - rate(g, 1000)) < 1e-9
    SNAP[g] = (cache, 1 / np.sqrt(cache))
assert (f"{SNAP[2.0][0]:,.0f}", f"{SNAP[0.2][0]:,.0f}", f"{SNAP[2.0][1]:.3g}", f"{SNAP[0.2][1]:.3g}") == \
       ("4,000", "40", "0.0158", "0.158")
assert "hold caches of 4,000 and 40 after 1,000 updates and effective rates of 0.0158 and 0.158" in INDEX


def absg(v):
    return rich("|", var("g"), "| = ", v)


fig = Figure(
    "02-constant-gradient", rich("Every effective rate falls as 1/√", var("t")),
    "Two log-log charts over 1 to 10,000 updates with a learning rate of 1. Left, the effective learning rate "
    "alpha over the root of the cache for three constant gradients: for 0.2 it falls from 5 to 0.05, for 0.5 from 2 "
    "to 0.02, for 2.0 from 0.5 to 0.005, three parallel lines. Dots on the 0.5 line at the updates of section 3's "
    "table, 2, 1.4142, 0.6325, 0.2, 0.0632 and 0.02; at update 1,000 the parameter with gradient 0.2 has a cache "
    "of 40 and a rate of 0.158, the one with 2.0 a cache of 4,000 and a rate of 0.0158. Right, for the gradient "
    "0.5, the distance moved: AdaGrad 1.00, 1.71, 5.02, 18.59, 61.80 and 198.54, gradient descent 0.5 per step, "
    "5,000 after 10,000 updates.",
    subtitle=rich(var("α"), " = 1. Left: the effective rate for three gradient sizes. Right: the distance moved for ",
                  absg("0.5"), "."),
    data_w=True)

left, right = fig.row(2, y=104, h=360)
SQ = "√"


def lg(lo, hi):
    return [10.0 ** e for e in range(lo, hi + 1)]


body = fig.panel(left, rich("Effective learning rate ", var("α"), "/", SQ, var("G")))
GS = [(0.2, "ink-muted", "triangle"), (0.5, "ink", "circle"), (2.0, "ink-muted", "square")]
ax = fig.line_chart(body, [dict(xs=list(T), ys=list(rate(g, T)), color=c, points=False) for g, c, _ in GS],
                    x=(1, 1e4, lg(0, 4)), y=(1e-3, 1e1, lg(-3, 1)), x_log=True, y_log=True, minor=True,
                    labels=False, label_w=88, x_label=rich("update ", var("t"), ", log scale"),
                    y_label="rate, log scale")
for g, c, shape in GS:
    ax.text(1e4, rate(g, 1e4), absg(f"{g}"), "note", color=c, dx=12, dy=5)
for t in TS:
    ax.point(t, TABLE[t][1], "circle", "ink", size=8)
for g, (cache, r) in SNAP.items():
    ax.point(1000, r, "triangle" if g < 1 else "square", "ink-muted", size=10)
ax.text(1000, SNAP[0.2][1], rich(var("G"), " = 40, rate 0.158"), "note", dx=8, dy=-14)
ax.text(1000, SNAP[2.0][1], rich(var("G"), " = 4,000, rate 0.0158"), "note", dx=-12, dy=26, anchor="end")

body = fig.panel(right, rich("Distance moved, ", absg("0.5")))
ax = fig.line_chart(body, [dict(xs=list(T), ys=list(0.5 * T), color="ink-muted", points=False),
                           dict(xs=list(T), ys=list(DIST[T - 1]), color="ink", points=False)],
                    x=(1, 1e4, lg(0, 4)), y=(1e-1, 1e4, lg(-1, 4)), x_log=True, y_log=True, minor=True,
                    labels=False, label_w=128, x_label=rich("update ", var("t"), ", log scale"),
                    y_label="distance, log scale")
for t in TS:
    ax.point(t, TABLE[t][2], "circle", "ink", size=8)
ax.text(1e4, 0.5e4, "gradient descent,", "note", color="ink-muted", dx=12, dy=-11)
ax.text(1e4, 0.5e4, num(5000), "note", color="ink-muted", dx=12, dy=9)
ax.text(1e4, DIST[-1], "AdaGrad,", "note", dx=12, dy=-11)
ax.text(1e4, DIST[-1], f"{TABLE[10000][2]:.2f}", "note", dx=12, dy=9)

fig.caption("Dots: section 3's table; triangle and square: update 1,000. A hundred times the updates buy ten times the distance.")
fig.write()
