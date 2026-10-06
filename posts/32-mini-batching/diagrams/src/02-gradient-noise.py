"""Post 32, section 2: the noise of a batch gradient against the batch size, measured and in closed form.

Run from anywhere:  python posts/32-mini-batching/diagrams/src/02-gradient-noise.py   (about 5 seconds)
Writes posts/32-mini-batching/diagrams/02-gradient-noise.svg.

snippets/gradient_noise.py is run here (float64, fixed weights, 2,000 random batches per size). Its printed table is
parsed and asserted against the listing of section 2 in index.md. The two curves are the formulas of section 2,
sigma / sqrt(B) and sigma / sqrt(B) * sqrt((N - B) / (N - 1)), with the printed sigma, 3.0631, and N = 300; each
printed closed-form column is recomputed from them and asserted. The diamonds are the measured column.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")

OUT = subprocess.run([sys.executable, str(POST / "snippets" / "gradient_noise.py")], cwd=str(ROOT),
                     capture_output=True, text=True, check=True).stdout
FULL = re.search(r"^length of the full-batch gradient: (\d\.\d{4})$", OUT, re.M).group(1)
SIGMA_S = re.search(r"^spread of the 300 single-row gradients, sigma: (\d\.\d{4})$", OUT, re.M).group(1)
rows = re.findall(r"^\s+(\d+)\s+(\d\.\d{4})\s+(\d\.\d{4})\s+(\d\.\d{4})\s+(\d+\.\d{2})$", OUT, re.M)
assert [int(r[0]) for r in rows] == [1, 8, 32, 100, 128, 300]
for line in OUT.splitlines():                                  # every printed line of the table is in the listing
    if re.match(r"^\s+\d+\s+\d\.\d{4}\s", line) or line.startswith(("length of", "spread of")):
        assert line in INDEX, line
assert (FULL, SIGMA_S) == ("0.4570", "3.0631")

N, SIGMA = 300, float(SIGMA_S)
simple = lambda b: SIGMA / math.sqrt(b)                                       # noqa: E731
exact = lambda b: simple(b) * math.sqrt((N - b) / (N - 1))                    # noqa: E731
assert r"\frac{\sigma^2}{B} \cdot \frac{N - B}{N - 1}" in INDEX
B = [int(r[0]) for r in rows]
MEAS = {int(r[0]): float(r[1]) for r in rows}
for b, _, s, e, _ in rows:                    # the printed closed forms, from the printed sigma (4 decimals)
    assert abs(simple(int(b)) - float(s)) < 2e-4 and abs(exact(int(b)) - float(e)) < 2e-4, b
assert "within 2 percent at every size" in INDEX
assert all(abs(MEAS[b] - exact(b)) <= 0.02 * exact(b) + 1e-9 for b in B)

F4 = lambda v: f"{v:.4f}"                                                     # noqa: E731
fig = Figure(
    "02-gradient-noise", "Batch noise falls with the batch size and is zero at 300",
    "A chart on logarithmic axes of the root-mean-square distance between a batch gradient and the full-batch "
    "gradient against the batch size B from 1 to 300, on the spiral network at fixed weights in float64. Diamonds: "
    "measured over 2,000 random batches per size, " + ", ".join(f"{F4(MEAS[b])} at {b}" for b in B) + "; the value "
    "0 at 300 is marked on the bottom edge. A solid line: the closed form sigma over root B times the root of "
    "(N minus B) over (N minus 1), with sigma 3.0631 and N = 300, which bends down to 0 at B = 300. A dashed line: "
    "sigma over root B alone, a straight line on these axes, 0.1768 at 300. A dotted line marks the length of the "
    f"full gradient, {FULL}. A table at the right lists the measured value and both closed forms for each size.",
    subtitle=rich("The spiral network at fixed weights, float64, ", var("N"), " = 300: a batch gradient's distance "
                  "from the full one."),
    data_w=True)

CHART = Box(40, 120, 544, 320)
YLO = 0.1
lo, hi = 250.0, 299.9                       # where the closed form reaches the bottom of the axis, by bisection
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if exact(mid) > YLO else (lo, mid)
X_END = lo
xs = sorted({round(10 ** (i / 200 * math.log10(X_END)), 6) for i in range(201)} |
            {280 + 0.05 * k for k in range(int((X_END - 280) / 0.05))} | {X_END})
xs = [v for v in xs if v <= X_END]
SIMPLE_LABEL = rich("σ / √", var("B"))
series = [dict(xs=xs + [N], ys=[simple(v) for v in xs + [N]], color="ink-muted", dash="proj", points=False,
               label=None),
          dict(xs=xs, ys=[exact(v) for v in xs], color="gradient", points=False, label=None)]
ax = fig.line_chart(CHART, series, x=(1, 300, [1, 10, 100]), y=(YLO, 4, [0.1, 0.2, 0.5, 1, 2]),
                    x_label=rich("batch size ", var("B"), ", rows"), y_label="distance from the full gradient",
                    x_log=True, y_log=True, minor=False, label_w=88, fmt_x=lambda v: num(int(v)),
                    fmt_y=lambda v: num(v),
                    ref_lines=[dict(y=float(FULL), label=rich("length of the full gradient, ", FULL), at="left")])
for b in B:
    if MEAS[b] > 0:
        ax.point(b, MEAS[b], "diamond", "gradient", size=10)
assert MEAS[N] == 0.0
X300, YB = ax.to_px(N, YLO)
with fig.data():
    fig.marker(X300, YB, "diamond", "gradient", size=10)
    fig.text(X300 + 12, YB + 5, rich("0 at ", var("B"), " = ", var("N")), "note", color="gradient", snap=False)

# -- the table: every number of the chart
TX = 600
cells = [[var("B"), "measured", "closed form", SIMPLE_LABEL]]
for b, m, s, e, _ in rows:
    cells.append([num(int(b)), m, e, s])
fig.table(TX, 136, cells, [48, 96, 104, 72], row_h=40, style="label", col_align=["end", "end", "end", "end"])
fig.legend(CHART.x + 56, 476, [dict(color="gradient", label="measured", mark="diamond"),
                               dict(color="gradient", label="closed form, rows drawn without replacement",
                                    mark="line"),
                               dict(color="ink-muted", label=SIMPLE_LABEL, mark="dash")], direction="row")
RATIO32 = next(r[4] for r in rows if r[0] == "32")
assert RATIO32 == "1.12" and "a batch of 32 by 1.12 times" in INDEX
fig.caption(rich("At ", var("B"), " = 32 the noise is still ", RATIO32,
                 " times the length of the gradient it estimates."))
fig.write()
