"""Post 22, section 5: six learning rates, five seeds each, after 10,001 epochs.

Run from anywhere:  python posts/22-gradient-descent-optimiser/diagrams/src/03-learning-rates.py
Writes posts/22-gradient-descent-optimiser/diagrams/03-learning-rates.svg.
Source: `snippets/learning_rate.py full` takes about 185 s, so it is not rerun here. Its printed table is parsed
from the listing in section 5 of index.md, which the series lint (--run) keeps equal to the snippet's output.
The rate-1.0 row is cross-checked against the seed_spread.py table of section 4 (seeds 0 to 4), and ln 3 is
computed here.
"""
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
MD = (POST / "index.md").read_text(encoding="utf-8")
SNIP = (POST / "snippets" / "learning_rate.py").read_text(encoding="utf-8")
assert "RATES = (0.001, 0.01, 0.1, 1.0, 3.0, 10.0)" in SNIP and "SEEDS = range(5)" in SNIP

HEAD = "rate    final loss        final accuracy    loss rose on      largest rise    highest loss      dead of 64"
block = MD[MD.index(HEAD):].split("```")[0].splitlines()
RANGE = r"(\S+) to (\S+)"
ROW = re.compile(rf"^(\S+)\s+{RANGE}\s+{RANGE}\s+{RANGE}\s+(\S+)\s+{RANGE}\s+{RANGE}$")
rows = {}
for line in block[1:7]:
    m = ROW.match(line)
    assert m, line
    g = m.groups()
    rows[float(g[0])] = dict(loss=(float(g[1]), float(g[2])), acc=(float(g[3]), float(g[4])),
                             rose=(int(g[5].replace(",", "")), int(g[6].replace(",", ""))), jump=g[7],
                             peak=(float(g[8]), float(g[9])), dead=(int(g[10]), int(g[11])))
RATES = [0.001, 0.01, 0.1, 1.0, 3.0, 10.0]
assert list(rows) == RATES
assert "seed 0 to 4: [64, 38, 36, 79, 38]" in block[8]
# the numbers the post's prose reads from this table
assert rows[0.1]["rose"] == (0, 0) == rows[0.01]["rose"] and rows[0.1]["loss"] == (1.0014, 1.0583)
assert rows[10.0]["dead"] == (64, 64) and rows[10.0]["acc"] == (0.3333, 0.3333)
assert rows[3.0]["dead"] == (2, 13) and rows[1.0]["rose"] == (4154, 4765)
# the rate-1.0 row against seeds 0 to 4 of section 4's table
SEED_HEAD = "seed  final loss  final acc  loss rose on  dead"
seed_rows = MD[MD.index(SEED_HEAD):].split("```")[0].splitlines()[1:6]
finals = [float(r.split()[1]) for r in seed_rows]
assert [r.split()[0] for r in seed_rows] == ["0", "1", "2", "3", "4"]
assert (min(finals), max(finals)) == rows[1.0]["loss"]
LN3 = math.log(3)


def rng(a, b, fmt=lambda v: num(v)):
    return fmt(a) if a == b else rich(fmt(a), " to ", fmt(b))


VERDICT = {0.001: "crawls; its rises are rounding", 0.01: "crawls", 0.1: "steady and slow",
           1.0: "fast and unsteady", 3.0: "fast and unsteady", 10.0: "kills all 64 hidden neurons"}
A = var("α")
fig = Figure(
    "03-learning-rates", "No learning rate is both steady and quick",
    "Six rows, one per learning rate, each with a band from the smallest to the largest final loss over seeds "
    "0 to 4 after 10,001 epochs, on an axis from 0.3 to 2.0 with a dotted line at ln 3 = 1.0986, the loss of "
    "the untrained network. 0.001: 1.0984 to 1.0985, crawls; the loss rose on 292 to 462 updates, float32 "
    "rounding; 0 to 1 dead of 64. 0.01: 1.0702 to 1.0867, never rose, 2 to 9 dead. 0.1: 1.0014 to 1.0583, "
    "never rose, 2 to 10 dead, steady and slow. 1: 0.3943 to 0.9906, rose on 4,154 to 4,765, 2 to 11 dead. "
    "3: 0.3746 to 0.5482, rose on 4,690 to 4,907, 2 to 13 dead; both fast and unsteady. 10: 1.1854 to 1.9295, "
    "rose on 6,425 to 6,468, all 64 hidden neurons dead. The 0.001 row is a dot, all five within 0.0001.",
    subtitle="10,001 epochs from each of the seeds 0 to 4; a band spans the five final losses.",
    data_w=True)

LABEL_W, PAD_R = 224, 256
ax = fig.dot_plot(Box(40, 104, 880, 348), [(rich(A, " = ", f"{r:g}"), []) for r in RATES], 0.3, 2.0,
                  [0.5, 1.0, 1.5, 2.0], fmt=lambda v: f"{v:.1f}", label_w=LABEL_W, pad_right=PAD_R,
                  axis_label="final loss", ref_lines=[dict(x=LN3, label=rich("ln 3 = ", f"{LN3:.4f}"))])
XR, XD = 800, 912                                  # right ends of the two columns
with fig.data():
    for i, r in enumerate(RATES):
        lo, hi = rows[r]["loss"]
        y = ax.sy(i)
        fig.text(40, y + 25, VERDICT[r], "note", snap=False)
        if ax.sx(hi) - ax.sx(lo) >= 2:
            fig.range_mark(ax.sx(lo), ax.sx(hi), y, ticks=[ax.sx(lo), ax.sx(hi)], color="error")
        else:                                       # 0.001: the five losses lie within a tenth of a unit
            assert hi - lo <= 0.0001
            fig.marker(ax.sx((lo + hi) / 2), y, "circle", "error", size=10)
            fig.text(ax.sx(hi) + 12, y + 5, "all five within 0.0001", "note", snap=False)
        rose = rows[r]["rose"]
        fig.text(XR, y + 5, "never" if rose == (0, 0) else rng(*rose), "label", anchor="end", snap=False)
        dead = rows[r]["dead"]
        fig.text(XD, y + 5, rng(*dead), "label", anchor="end", color="error" if dead[0] == 64 else None,
                 snap=False)
fig.text(XR, 120, "loss rose on", "note", anchor="end")
fig.text(XD, 120, "dead of 64", "note", anchor="end")

fig.caption("The rates that never raise the loss run out of budget; the rates that use it bounce.")
fig.write()
