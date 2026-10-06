"""Post 23 hero (section 6.1): the loss in the last 1,000 epochs, constant rate against decay=1e-3, seeds 0 to 4.

Run from anywhere:  python posts/23-learning-rate-decay/diagrams/src/01-last-epochs-per-seed.py
Writes posts/23-learning-rate-decay/diagrams/01-last-epochs-per-seed.svg. Takes about 75 seconds.

Every number is printed by snippets/seed_spread.py, which is run here (about 73 seconds, ten runs of 10,001 epochs,
nothing added to it): per run the final loss, the largest one-step rise, the lowest and highest loss of the last
1,000 epochs and their mean. Each parsed row is asserted to be a line of the listing in index.md, and the summary
lines (band widths, the seeds on which decay has the lower mean loss) are recomputed from the rows and asserted
against the snippet's own summary.
"""
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, sup, num, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
CACHE = os.environ.get("FIG23_SEED_SPREAD_OUT")      # layout work only: a saved stdout of the same snippet
if CACHE:
    OUT = Path(CACHE).read_text(encoding="utf-8")
else:
    OUT = subprocess.run([sys.executable, str(POST / "snippets" / "seed_spread.py")], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout

ROW = re.compile(r"^ +(\d)  (constant|decay 1e-3)  +(\d\.\d{4})  (\d\.\d{4}) +(\d+) +(\d\.\d{4})  (\d\.\d{3}) to "
                 r"(\d\.\d{3}) +(\d\.\d{4}) +(\d\.\d{4}) +(\d\.\d{4})$", re.M)
RUNS = {}
for m in ROW.finditer(OUT):
    assert m.group(0) in INDEX, m.group(0)                       # the listing of section 6.1
    seed, setting = int(m.group(1)), m.group(2)
    RUNS[(seed, setting)] = dict(final=float(m.group(3)), rises=int(m.group(5)), rise=m.group(6),
                                 lo=float(m.group(7)), hi=float(m.group(8)), mean=float(m.group(10)))
SEEDS, SETTINGS = [0, 1, 2, 3, 4], ["constant", "decay 1e-3"]
assert sorted(RUNS) == sorted((s, k) for s in SEEDS for k in SETTINGS)

# the snippet's summary, recomputed from the rows
# band width per row: highest minus lowest of the printed ends, which are rounded to 3 decimals, so the difference is
# within 0.001 of the true width; the narrowest and the widest band of each setting take the snippet's own printed
# value (seed 3 without decay: 2.636 - 0.352 = 2.284 from the ends, 2.283 as printed)
WIDTH = {}
for k, (a, b) in zip(SETTINGS, (("0.098", "2.283"), ("0.021", "0.035"))):
    widths = {s: round(RUNS[(s, k)]["hi"] - RUNS[(s, k)]["lo"], 3) for s in SEEDS}
    assert abs(min(widths.values()) - float(a)) <= 0.0011 and abs(max(widths.values()) - float(b)) <= 0.0011
    assert f"width of the last band {a} to {b}" in OUT
    s_min, s_max = min(widths, key=widths.get), max(widths, key=widths.get)
    widths[s_min], widths[s_max] = float(a), float(b)
    for s in SEEDS:
        WIDTH[(s, k)] = f"{widths[s]:.3f}"
assert [WIDTH[(s, "constant")] for s in SEEDS] == ["0.098", "0.748", "0.533", "2.283", "0.978"]
assert [WIDTH[(s, "decay 1e-3")] for s in SEEDS] == ["0.021", "0.030", "0.023", "0.035", "0.022"]
LOWER_MEAN = [s for s in SEEDS if RUNS[(s, "decay 1e-3")]["mean"] < RUNS[(s, "constant")]["mean"]]
assert LOWER_MEAN == [0] and "decay has the lower mean loss on seeds [0]" in OUT
assert RUNS[(3, "constant")]["hi"] == 2.636 and RUNS[(0, "decay 1e-3")]["final"] == 0.7612

D0 = rich(var("d"), " = 0")
D3 = rich(var("d"), " = ", sup("10", num(-3), italic=False))
STYLE = {"constant": ("neutral-soft", "rule", "ink-muted"),         # d = 0 in ink-muted, as in figure 02
         "decay 1e-3": ("blue-soft", "blue-line", "blue")}          # the post's decay, the one accent

fig = Figure(
    "01-last-epochs-per-seed", "Decay removes the jumps but mostly ends higher",
    "For seeds 0 to 4, two rows each: the constant rate, d = 0, in grey and decay d = 10 to the minus 3 in blue, on "
    "a loss axis from 0.3 to 2.7. Each row is a band from the lowest to the highest loss of the last 1,000 of the "
    "10,001 epochs, a tick at their mean and a dot at the loss of epoch 10,000. Constant rate: seed 0 0.835 to "
    "0.933, mean 0.8703, final 0.8737; seed 1 0.483 to 1.231, mean 0.5880, final 0.5091; seed 2 0.793 to 1.326, "
    "mean 0.8905, final 0.9906; seed 3 0.352 to 2.636, mean 0.4159, final 0.3943; seed 4 0.483 to 1.461, mean "
    "0.5842, final 0.4843. Decay: seed 0 0.761 to 0.782, mean 0.7714, final 0.7612; seed 1 0.733 to 0.763, mean "
    "0.7486, final 0.7335; seed 2 0.911 to 0.934, mean 0.9222, final 0.9107; seed 3 0.731 to 0.766, mean 0.7500, "
    "final 0.7310; seed 4 0.775 to 0.797, mean 0.7867, final 0.7750. A column gives the largest one-step rise of "
    "the loss: 0.0553, 0.2879, 0.2551, 1.1043 and 0.3507 with the constant rate, 0.0152, 0.0002, 0.0000, 0.0099 "
    "and 0.0017 with decay, a column repeats the mean loss, and a column gives the band width: 0.098, 0.748, 0.533, "
    "2.283 and 0.978 with the constant rate, 0.021, 0.030, 0.023, 0.035 and 0.022 with decay. A key explains the "
    "band, the tick and the dot, and that grey is d = 0 and blue is d = 10 to the minus 3.",
    subtitle=rich("Spiral classifier, ", sub("α", "0"), " = 1, 10,001 epochs: the last 1,000 epochs of each run."),
    height=720, data_w=True)

LO, HI, TICKS = 0.3, 2.7, [0.5, 1.0, 1.5, 2.0, 2.5]
PX0, PX1 = 176, 632
SET_X = 80                                              # the setting labels; the plot starts 24 past the widest
RISE_X, MEAN_X, WIDTH_X = 736, 832, 920
sx = lambda v: PX0 + (PX1 - PX0) * (v - LO) / (HI - LO)  # noqa: E731
TOP, AXIS = 152, 576
cy = lambda s, k: 180 + 84 * s + 32 * k  # noqa: E731
assert cy(4, 1) + 16 < AXIS


def band(x0, x1, y, color, ticks=(), dot=None, h=20):
    """range_mark's form for any colour, the neutral too: a soft box outlined in the line form, thin ticks, a dot."""
    soft, line, base = STYLE[color]
    fig.fill(Box(x0, y - h / 2, x1 - x0, h), soft, fit=False)
    fig.outline(Box(x0, y - h / 2, x1 - x0, h), line)
    for t in ticks:
        fig.edge((t, y - h / 3), (t, y + h / 3), color=base, width=1)
    if dot is not None:                  # size 6, so that a band 4 units wide still shows above and below it
        fig.marker(dot, y, "circle", base, size=6)


assert SET_X + text_width(D3, 14, weight=400) + 24 <= PX0
# -- column headings
fig.text(40, 140, "seed", "note")
fig.text(SET_X, 140, "setting", "note")
fig.text(WIDTH_X, 120, "band", "note", anchor="end")
fig.text(WIDTH_X, 140, "width", "note", anchor="end")
fig.text(RISE_X, 120, "largest rise", "note", anchor="end")
fig.text(RISE_X, 140, "in one step", "note", anchor="end")
fig.text(MEAN_X, 120, "mean loss,", "note", anchor="end")
fig.text(MEAN_X, 140, "last 1,000", "note", anchor="end")

with fig.data():
    for t in TICKS:
        fig.edge((sx(t), TOP), (sx(t), AXIS), color="grid", width=0.75)
    fig.edge((PX0, AXIS), (PX1, AXIS), color="ink-muted", width=1)
    for t in TICKS:
        fig.text(sx(t), AXIS + 20, f"{t:.1f}", "tick", anchor="middle", snap=False)
    for s in SEEDS:
        for k, setting in enumerate(SETTINGS):
            r, y = RUNS[(s, setting)], cy(s, k)
            band(sx(r["lo"]), sx(r["hi"]), y, setting, ticks=[sx(r["mean"])], dot=sx(r["final"]))
            ink = STYLE[setting][2]
            fig.text(RISE_X, y + 5, r["rise"], "value", anchor="end", color=ink, snap=False)
            fig.text(MEAN_X, y + 5, f"{r['mean']:.4f}", "value", anchor="end", color=ink, snap=False)
            fig.text(WIDTH_X, y + 5, WIDTH[(s, setting)], "value", anchor="end", color=ink, snap=False)
    for s in SEEDS:                                    # on the values' baseline, 5 under the row's centre
        fig.text(40, cy(s, 0) + 5, str(s), "label", snap=False)
        fig.text(SET_X, cy(s, 0) + 5, D0, "label", snap=False)
        fig.text(SET_X, cy(s, 1) + 5, D3, "label", snap=False)
fig.text(40, AXIS + 20, "loss", "note")

# -- the key, one row under the axis: the three marks in their neutral form, then the two settings' colours
KY = 640
GAP = 32


def key_text(x, s):
    fig.text(x, KY, s, "label", snap=False)
    return x + math.ceil(text_width(s, 14, weight=400)) + GAP


with fig.data():
    x = PX0
    band(x, x + 32, KY - 5, "constant", ticks=[x + 20], h=16)
    x = key_text(x + 44, "lowest to highest")
    fig.edge((x, KY - 12), (x, KY + 2), color="ink-muted", width=1)
    x = key_text(x + 12, "mean")
    fig.marker(x + 3, KY - 5, "circle", "ink-muted", size=6)
    x = key_text(x + 16, "epoch 10,000")
    for setting, lab in (("constant", D0), ("decay 1e-3", D3)):
        band(x, x + 24, KY - 5, setting, h=16)
        x = key_text(x + 36, lab)
assert x - GAP <= 920

fig.caption("By the mean of the last 1,000 epochs, decay is the lower run on seed 0 only.")
fig.write()
