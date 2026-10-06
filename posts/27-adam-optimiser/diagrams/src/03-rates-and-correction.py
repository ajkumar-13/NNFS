"""Post 27, sections 3.2 and 7: Adam's final accuracy over seeds 0 to 4 at three learning rates, and without the
bias correction at 0.02 and at 0.1.

Run from anywhere:  python posts/27-adam-optimiser/diagrams/src/03-rates-and-correction.py   (about 90 seconds)
Writes posts/27-adam-optimiser/diagrams/03-rates-and-correction.svg.
The five seed scripts of the post, snippets/rate_low.py, seeds_adam.py, rate_high.py, no_bias_correction.py and
no_bias_correction_high.py, are run here as they are (five processes side by side, each 40 to 95 seconds), and
their printed rows give every value drawn. The rows are asserted against index.md: the per-seed table of section
3.2 (corrected and uncorrected at 0.02), the learning-rate table of section 7 (ranges, means, seeds past 90
percent), and the range 41.7 to 52.0 percent of the uncorrected optimiser at 0.1.

Layout: one chart, a row per setting, each a band over the five seeds with a tick per seed and a dot for seed 0,
the mean and the number of seeds past 90 percent printed at the right.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
SEEDS = range(5)
SETTINGS = [  # snippet, row label, note under it, colour
    ("rate_low", "0.001", "the class default", "blue"),
    ("seeds_adam", "0.02", "the documented setting", "blue"),
    ("rate_high", "0.1", "five times the documented rate", "blue"),
    ("no_bias_correction", "0.02", "no bias correction", None),
    ("no_bias_correction_high", "0.1", "no bias correction", None),
]
procs = {
    n: subprocess.Popen([sys.executable, str(POST / "snippets" / f"{n}.py")], cwd=str(ROOT), stdout=subprocess.PIPE,
                        text=True) for n, *_ in SETTINGS}
RUNS = {}
for n, *_ in SETTINGS:
    out, _ = procs[n].communicate()
    assert procs[n].returncode == 0, n
    rows = re.findall(r"^Adam.*?\s+(\d)\s+\d\.\d{4}\s+\d\.\d{4}\s+(\d\.\d{4})\s+(\d\.\d{4})\s+\d\.\d{4}\s+(\S+)$",
                      out, re.M)
    assert [int(r[0]) for r in rows] == list(SEEDS), (n, out)
    RUNS[n] = dict(loss=[float(r[1]) for r in rows], acc=[100 * float(r[2]) for r in rows],
                   past90=sum(r[3] != "never" for r in rows))

P = lambda v: f"{v:.1f}"                                                    # noqa: E731
MEAN = {n: sum(r["acc"]) / 5 for n, r in RUNS.items()}

# -- section 3.2's table: Adam and Adam, uncorrected, seed by seed, with the mean
for n, head in (("seeds_adam", "Adam (`seeds_adam.py`)"), ("no_bias_correction", "Adam, uncorrected")):
    cells = " | ".join(P(a) for a in RUNS[n]["acc"])
    assert f"| {head} | {cells} | {P(MEAN[n])} |" in INDEX, (n, cells)
# -- section 7's learning-rate table
for n, rate in (("rate_low", "0.001"), ("seeds_adam", "0.02 (documented)"), ("rate_high", "0.1")):
    r = RUNS[n]
    row = (f"| {rate} | {P(min(r['acc']))} to {P(max(r['acc']))} | {P(MEAN[n])} | "
           f"{min(r['loss']):.4f} to {max(r['loss']):.4f} | {r['past90']} |")
    assert row in INDEX, row
r = RUNS["no_bias_correction_high"]
assert (P(min(r["acc"])), P(max(r["acc"]))) == ("41.7", "52.0")
assert "it ends between 41.7 and 52.0 percent, behind on all five seeds" in INDEX
assert all(a < b for a, b in zip(r["acc"], RUNS["rate_high"]["acc"]))
assert all(a > b for a, b in zip(RUNS["no_bias_correction"]["acc"], RUNS["seeds_adam"]["acc"]))
assert RUNS["no_bias_correction"]["past90"] == 5 and r["past90"] == 0

ALPHA = var("α")
desc_rows = []
for n, rate, note, _ in SETTINGS:
    a = RUNS[n]["acc"]
    who = "Adam without the correction" if n.startswith("no_") else "Adam"
    desc_rows.append(f"{who} at {rate}: {', '.join(P(v) for v in a)} for seeds 0 to 4, mean {P(MEAN[n])}, "
                     f"{RUNS[n]['past90']} of 5 past 90 percent")
fig = Figure(
    "03-rates-and-correction", "Larger early steps help at 0.02 and wreck training at 0.1",
    "A chart of the final training accuracy on the spiral after 10,001 epochs, one row per setting, each a band "
    "from the lowest to the highest of seeds 0 to 4 with a tick per seed and a dot for seed 0, on an axis from 40 "
    "to 100 percent, with the mean and the number of seeds that pass 90 percent at the right. "
    + "; ".join(desc_rows) + ".",
    subtitle=rich("Final training accuracy on the spiral after 10,001 epochs, seeds 0 to 4, decay ", sup("10", "−5", italic=False),
                  " throughout."),
    data_w=True)


def band(x0, x1, y, ticks, dot, color, h=24):
    """The band-with-ticks mark: range_mark in a hue, or the same form in neutrals (soft card fill, rule outline,
    muted ticks and dot) when color is None. Inside fig.data()."""
    if color:
        fig.range_mark(x0, x1, y, ticks=ticks, color=color, dot=dot, h=h)
        return
    b = Box(min(x0, x1), y - h / 2, abs(x1 - x0), h)
    fig.fill(b, "neutral-soft", fit=False)
    fig.outline(b, "rule")
    for t in ticks:
        fig.edge((t, y - h / 3), (t, y + h / 3), color="ink-muted")
    if dot is not None:
        fig.marker(dot, y, "circle", "ink-muted", size=10)


H = 312
ax = fig.dot_plot(Box(40, 128, 720, H), [("", []) for _ in SETTINGS], 40, 100, [40, 50, 60, 70, 80, 90, 100],
                  label_w=208, pad_right=16, fmt=lambda v: num(v), axis_label="final accuracy, percent")
VX, CX = 840, 920                                                          # value columns (right edges)
fig.text(VX, 112, "mean", "note", anchor="end")
fig.text(CX, 112, "past 90%", "note", anchor="end")
with fig.data():
    for i, (n, rate, note, color) in enumerate(SETTINGS):
        a = RUNS[n]["acc"]
        y = ax.sy(i)
        fig.text(40, y - 2, rich(ALPHA, " = ", rate), "label", snap=False)
        fig.text(40, y + 18, note, "note", snap=False)
        band(ax.sx(min(a)), ax.sx(max(a)), y, [ax.sx(v) for v in a], ax.sx(a[0]), color)
        fig.text(VX, y + 5, P(MEAN[n]), "value", anchor="end", color=color, snap=False)
        fig.text(CX, y + 5, f"{RUNS[n]['past90']} of 5", "label", anchor="end", snap=False)
# legend, in the forms of the marks
with fig.data():
    LY = 468
    band(248, 272, LY - 4, [260], None, "blue", h=16)
    fig.text(284, LY, "Adam, one tick per seed", "label", snap=False)
    band(500, 524, LY - 4, [512], None, None, h=16)
    fig.text(536, LY, "Adam without the correction", "label", snap=False)
    fig.marker(780, LY - 4, "circle", "ink-muted", size=10)
    fig.text(796, LY, "seed 0", "label", snap=False)
fig.caption(rich("Removing the correction multiplies the early steps by up to 6.6; whether that helps depends on ", ALPHA, "."))
fig.write()
