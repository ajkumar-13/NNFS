"""Post 23, sections 2 and 3: the rate alpha_t = alpha_0 / (1 + d t) for the four decay rates of section 6, and the
sum of the 10,001 rates of a run.

Run from anywhere:  python posts/23-learning-rate-decay/diagrams/src/02-schedule-and-budget.py
Writes posts/23-learning-rate-decay/diagrams/02-schedule-and-budget.svg.

snippets/decay_schedule.py is run here (runpy, under a second, nothing trained): every curve is the rate its class
Optimizer_SGD hands to update_params at each of the 10,001 updates (its rates_of_a_run), the halving points are the
updates the snippet prints, and every sum is the snippet's sum. The printed table rows, the halving lines and the sum
lines are asserted against the snippet's output and against the listings of index.md.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, isub, sup, num, CDOT, span  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIP = POST / "snippets" / "decay_schedule.py"
INDEX = (POST / "index.md").read_text(encoding="utf-8")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(SNIP), run_name="snippet")
OUT = buf.getvalue()
runs, UPDATES = ns["runs"], ns["UPDATES"]
assert UPDATES == 10001

DECAYS = [0.0, 1e-4, 1e-3, 1e-2]                      # the four settings trained in section 6
T = np.arange(UPDATES)
for d in DECAYS:
    assert np.max(np.abs(runs[d] - 1.0 / (1.0 + d * T))) == 0.0          # the formula of section 2, exactly

# the table of section 2.1, as printed and as listed in index.md
for t in (0, 1, 100, 1000, 5000, 10000):
    row = f"{t:6d}   " + "   ".join(f"{runs[d][t]:8.4f}" for d in ns["decays"])
    assert row in OUT and row in INDEX, row

# the halvings of d = 1e-3 that the snippet prints (section 2: at 1/d, 3/d and 7/d)
HALVINGS = []
for label, k in (("1/2", 2), ("1/4", 4), ("1/8", 8)):
    t = int(np.argmax(runs[1e-3] <= 1.0 / k + 1e-12))
    assert f"alpha_0 * {label:<4}  first at t = {t:5d}" in OUT
    assert t == (k - 1) * 1000
    HALVINGS.append((t, 1.0 / k))

# the sums of section 3
SUMS = {}
for d in DECAYS:
    s, share = runs[d].sum(), runs[d].sum() / UPDATES
    line = f"d={d:<6g}  sum {s:9.1f}   share of the constant rate {share:6.1%}"
    assert line in OUT and line in INDEX, line
    SUMS[d] = (float(f"{s:.1f}"), f"{share * 100:.1f}")
assert SUMS == {0.0: (10001.0, "100.0"), 1e-4: (6932.2, "69.3"), 1e-3: (2398.4, "24.0"), 1e-2: (462.0, "4.6")}


def dname(d):
    """d = 0, d = 10 to the minus 4, ..."""
    if d == 0:
        return rich(var("d"), " = 0")
    return rich(var("d"), " = ", sup("10", num(int(round(np.log10(d)))), italic=False))


ACCENT = "blue"                                          # the post's decay, d = 10^-3; the others in ink-muted
COLOR = {d: (ACCENT if d == 1e-3 else "ink-muted") for d in DECAYS}
ALPHA_T, ALPHA_0 = isub("α", "t"), sub("α", "0")

fig = Figure(
    "02-schedule-and-budget", "The rate falls, and so does the distance it can cover",
    "Top: the learning rate alpha_t = alpha_0 / (1 + d t) with alpha_0 = 1 over the updates t = 0 to 10,000, for "
    "four decay rates. d = 0 stays at 1. d = 10 to the minus 4 ends at 0.5. d = 10 to the minus 3, the decay of the "
    "post, drawn in blue, is at a half at update 1,000, a quarter at 3,000 and an eighth at 7,000, and ends at 0.0909. "
    "d = 10 to the minus 2 is at 0.5 after 100 updates and ends at 0.0099. Bottom: the sum of the 10,001 rates of a "
    "run, as bars: 10,001.0 for d = 0 (100.0 percent), 6,932.2 for 10 to the minus 4 (69.3 percent), 2,398.4 for "
    "10 to the minus 3 (24.0 percent) and 462.0 for 10 to the minus 2 (4.6 percent).",
    subtitle=rich(ALPHA_T, " = ", ALPHA_0, " / (1 + ", var("d"), " ", CDOT, " ", var("t"), "), with ", ALPHA_0,
                  " = 1 and the four decay rates trained in section 6, over updates 0 to 10,000."),
    height=720, data_w=True)

# -- top: the schedule
top = fig.panel(Box(40, 104, 880, 352), rich("The rate ", ALPHA_T, " at update ", var("t")))
XS = sorted(set(range(0, 400, 2)) | set(range(400, UPDATES, 25)) | {UPDATES - 1})
# d = 10^-2 ends 0.08 below d = 10^-3, too close for two end labels: it is labelled on its steep start instead
series = [dict(xs=XS, ys=[float(runs[d][t]) for t in XS], color=COLOR[d], points=False,
               label=None if d == 1e-2 else dname(d)) for d in DECAYS]
ax = fig.line_chart(top, series, x=(0, 10000, [0, 2500, 5000, 7500, 10000]), y=(0, 1, [0, 0.25, 0.5, 0.75, 1]),
                    x_label=rich("update ", var("t")), y_label=ALPHA_T, label_w=96,
                    fmt_x=lambda v: num(int(v)), fmt_y=lambda v: f"{v:.2f}")
FRAC = {2: "a half", 4: "a quarter", 8: "an eighth"}
for (t, v), k in zip(HALVINGS, (2, 4, 8)):
    ax.point(t, v, "diamond", ACCENT, size=10)
    ax.text(t, v, rich(FRAC[k], " at ", num(t)), "note", color=ACCENT, dx=10, dy=-10)
T2 = 300
ax.text(T2, float(runs[1e-2][T2]), dname(1e-2), "note", color=COLOR[1e-2], dx=10, dy=-6)

# -- bottom: the budget, with each sum's share of the constant rate's in a column at the right
bot = fig.panel(Box(40, 480, 880, 176), "The sum of all 10,001 rates")
rows = [(dname(d), SUMS[d][0], COLOR[d]) for d in DECAYS]
plot = fig.bar_chart(bot, rows, 0, 10001, [0, 2500, 5000, 7500, 10000], label_w=96, value_w=200,
                     fmt_tick=lambda v: num(int(v)), fmt_value=lambda v: num(f"{v:,.1f}"))
step = plot.h / len(rows)
fig.text(920, bot.y - 16, rich("share of ", var("d"), " = 0"), "note", anchor="end")
with fig.data():
    for i, d in enumerate(DECAYS):
        fig.text(920, plot.y + step * (i + 0.5) + 5, f"{SUMS[d][1]} percent", "value", anchor="end",
                 color=COLOR[d], snap=False)

fig.caption("Each sum is how far a parameter moves if its gradient has length 1 throughout.")
fig.write()
