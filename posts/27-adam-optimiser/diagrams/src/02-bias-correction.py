"""Post 27, sections 3.1 and 3.2: the two correction factors over t, and the step they hold at one learning rate.

Run from anywhere:  python posts/27-adam-optimiser/diagrams/src/02-bias-correction.py   (about a second)
Writes posts/27-adam-optimiser/diagrams/02-bias-correction.svg.
The curves are the formulas of section 3: the factor 1 / (1 - beta^t) for beta_1 = 0.9 and beta_2 = 0.999, and the
uncorrected step (1 - beta_1^t) / sqrt(1 - beta_2^t) in learning rates (constant gradient, epsilon left out). The
dots are the rows of the two listings of section 3 in index.md, parsed here and asserted against the formulas;
snippets/bias_correction.py is run here for the largest step (6.569 at t = 12) and the last t above 1.1 (1,750).

Layout: two charts side by side on one log axis of t, the factors left (log scale) and the step right.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sub, isup  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
B1, B2 = 0.9, 0.999

# -- section 3.1's listing: t, beta_1^t, factor for m, beta_2^t, factor for v
FAC = {int(t): (float(fm), float(fv)) for t, fm, fv in
       re.findall(r"^\s+(\d+)\s+\d\.\d{5}\s+(\d+\.\d{3})\s+\d\.\d{5}\s+(\d+\.\d{3})$", INDEX, re.M)}
assert sorted(FAC) == [1, 10, 100, 1000, 5000], FAC
for t, (fm, fv) in FAC.items():
    assert f"{1 / (1 - B1 ** t):.3f}" == f"{fm:.3f}" and f"{1 / (1 - B2 ** t):.3f}" == f"{fv:.3f}", t

# -- section 3.2's listing: t, corrected, uncorrected
UNC = {int(t): float(u) for t, u in re.findall(r"^\s+(\d+)\s+1\.000\s+(\d\.\d{3})$", INDEX, re.M)}
assert sorted(UNC) == [1, 10, 100, 1000], UNC
step = lambda t: (1 - B1 ** t) / np.sqrt(1 - B2 ** t)            # noqa: E731
for t, u in UNC.items():
    assert f"{step(t):.3f}" == f"{u:.3f}", t

out = subprocess.run([sys.executable, str(POST / "snippets" / "bias_correction.py")], cwd=ROOT,
                     capture_output=True, text=True, check=True).stdout
peak = re.search(r"largest uncorrected step: (\d\.\d{3}) learning rates at t = (\d+)", out).groups()
last = int(re.search(r"above 1\.1 learning rates until t = (\d+)", out).group(1))
TS = np.arange(1, 10001)
assert peak == (f"{step(TS).max():.3f}", f"{TS[step(TS).argmax()]}") == ("6.569", "12")
assert last == TS[step(TS) > 1.1].max() == 1750
for s in ("largest uncorrected step: 6.569 learning rates at t = 12",
          "uncorrected step still above 1.1 learning rates until t = 1750"):
    assert s in INDEX, s
assert "it is still 10.5 at step 100 and 1.58 at step 1,000" in INDEX

T = np.unique(np.round(np.logspace(0, 4, 321)).astype(int))
b1, b2 = sub(var("β"), "1"), sub(var("β"), "2")


def lg(lo, hi):
    return [10.0 ** e for e in range(lo, hi + 1)]


fig = Figure(
    "02-bias-correction", "The corrections fade and hold the early steps at one rate",
    "Two charts against the update t on a log axis from 1 to 10,000, for beta 1 = 0.9 and beta 2 = 0.999. Left, "
    "the correction factor 1 over (1 minus beta to the t) on a log scale: for m it falls from 10 at t = 1 to 1.535 "
    "at t = 10 and 1.000 at t = 100; for v from 1,000 to 100.451 at t = 10, 10.503 at t = 100, 1.582 at t = 1,000 "
    "and 1.007 at t = 5,000. Right, the step for a constant gradient in learning rates: with the correction it is 1 "
    "at every t; without it, (1 minus beta 1 to the t) over the root of (1 minus beta 2 to the t), it is 3.162 at "
    "t = 1, 6.528 at t = 10, 3.241 at t = 100 and 1.258 at t = 1,000, largest at 6.569 at t = 12 and above 1.1 "
    "until t = 1,750.",
    subtitle=rich(b1, " = 0.9, ", b2, " = 0.999, a constant gradient, ", var("ε"),
                  " left out. Dots: the rows of the two tables of section 3."),
    data_w=True)

left, right = fig.row(2, y=104, h=360)
xlab = rich("update ", var("t"), ", log scale")

body = fig.panel(left, rich("Correction factor 1 / (1 − ", isup("β", "t"), ")"))
fm, fv = 1 / (1 - B1 ** T), 1 / (1 - B2 ** T)
ax = fig.line_chart(body, [dict(xs=list(T), ys=list(fm), color="ink-muted", points=False),
                           dict(xs=list(T), ys=list(fv), color="ink", points=False)],
                    x=(1, 1e4, lg(0, 4)), y=(0.5, 2e3, [1, 10, 100, 1000]), x_log=True, y_log=True, minor=True,
                    labels=False, label_w=24, x_label=xlab, y_label="factor, log scale")
for t, (a, b) in FAC.items():
    ax.point(t, a, "circle", "ink-muted", size=8)
    ax.point(t, b, "square", "ink", size=8)
ax.text(10, FAC[10][1], rich("for ", var("v"), ", ", b2, " = 0.999"), "note", color="ink", dx=10, dy=-10)
ax.text(10, FAC[10][0], rich("for ", var("m"), ", ", b1, " = 0.9"), "note", color="ink-muted", dx=10, dy=-10)
ax.text(100, FAC[100][1], "10.5 at 100", "note", color="ink", dx=10, dy=-8)
ax.text(1000, FAC[1000][1], "1.58 at 1,000", "note", color="ink", dx=10, dy=-10)

body = fig.panel(right, "Step for a constant gradient, in learning rates")
ax = fig.line_chart(body, [dict(xs=list(T), ys=list(step(T)), color="ink", points=False)],
                    x=(1, 1e4, lg(0, 4)), y=(0, 7, [0, 1, 2, 3, 4, 5, 6, 7]), x_log=True, labels=False,
                    label_w=24, x_label=xlab, y_label="step, learning rates", minor=True,
                    ref_lines=[dict(y=1, color="blue", dash=None, width=1.5)])
for t, u in UNC.items():
    ax.point(t, u, "circle", "ink", size=8)
ax.text(12, step(12), rich("largest, 6.569 at ", var("t"), " = 12"), "note", color="ink", dx=8, dy=-10)
ax.point(1750, step(1750), "triangle", "ink", size=10)
ax.text(1e4, 1, rich("above 1.1 until ", var("t"), " = 1,750"), "note", color="ink", dy=28, anchor="end")
ax.text(1.4, 1, "with the correction", "note", color="blue", dy=28)
ax.text(30, step(30), "without the correction", "note", color="ink", dx=14, dy=-4)

fig.caption("Both factors reach 1 by themselves; no schedule switches either of them off.")
fig.write()
