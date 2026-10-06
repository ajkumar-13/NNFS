"""Post 29, sections 5 and 8: the 5-fold means of five learning rates and of two narrow grids, seeds 0 to 4.

Run from anywhere:  python posts/29-validation-and-hyperparameter-tuning/diagrams/src/04-learning-rates-over-seeds.py
Writes posts/29-validation-and-hyperparameter-tuning/diagrams/04-learning-rates-over-seeds.svg (about a minute).
Width 64: the "seed s" summary lines of search.py and of seeds_1_2.py and seeds_3_4.py, parsed from the listings
of section 5 in index.md (the series lint keeps them equal to the snippets' output). Width 16: seed 0 from the
grid.py listing of section 8; seeds 1 to 4 are printed only by seeds_1_2.py and seeds_3_4.py, which are run here
side by side (each 35 to 60 seconds). Their learning-rate lines are asserted equal to the index.md listing, and the
readings of sections 5 and 8 are asserted on the values.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var  # noqa: E402

POST = Path(__file__).resolve().parents[2]
MD = (POST / "index.md").read_text(encoding="utf-8")
procs = [subprocess.Popen([sys.executable, "-B", str(POST / "snippets" / f"seeds_{a}.py")], cwd=str(POST.parents[1]),
                          stdout=subprocess.PIPE, text=True) for a in ("1_2", "3_4")]
OUT = ""
for p in procs:
    out, _ = p.communicate()
    assert p.returncode == 0
    OUT += out

SEEDS = range(5)
RATES = (0.01, 0.05, 0.1, 0.5, 1.0)
LR_LINE = re.compile(r"seed (\d)  0\.01: (\S+)  0\.05: (\S+)  0\.1: (\S+)  0\.5: (\S+)  1\.0: (\S+)  \| best (\S+),")
W64 = {}
for m in LR_LINE.finditer(MD):
    W64[int(m.group(1))] = [float(v) for v in m.groups()[1:6]]
assert sorted(W64) == list(SEEDS)
for line in OUT.splitlines():                      # the run agrees with the listing of section 5
    if line.startswith("learning rates"):
        assert line in MD, line

W16 = {0: {}}
for lr, n, mean in re.findall(r"^lr=(0\.05|0\.1)\s+n_neurons=(16|64)\s+mean_acc=(\d\.\d{3})", MD, re.M):
    if n == "16":
        W16[0][float(lr)] = float(mean)
    else:
        assert float(mean) == W64[0][RATES.index(float(lr))]
for s, a, b in re.findall(r"grid \(rate/width\) seed (\d)  0\.05/16: (\S+)  0\.05/64: \S+  0\.1/16: (\S+)", OUT):
    W16[int(s)] = {0.05: float(a), 0.1: float(b)}
assert sorted(W16) == list(SEEDS) and all(len(W16[s]) == 2 for s in SEEDS)

# -- the readings of sections 5 and 8
big = [W64[s][3] for s in SEEDS], [W64[s][4] for s in SEEDS]
assert (min(big[0]), max(big[0]), min(big[1]), max(big[1])) == (0.350, 0.453, 0.343, 0.403)
assert min(W64[s][k] for s in SEEDS for k in range(3)) == 0.603
best = [RATES[max(range(5), key=lambda k: W64[s][k])] for s in SEEDS]
assert best == [0.1, 0.1, 0.05, 0.1, 0.1]
assert all(W64[s][2] > W64[s][0] for s in SEEDS)
narrow = [W16[s][r] for s in SEEDS for r in (0.05, 0.1)]
wide = [W64[s][k] for s in SEEDS for k in (1, 2)]
assert (min(narrow), max(narrow), min(wide), max(wide)) == (0.423, 0.537, 0.603, 0.797)
assert all(max(W16[s].values()) < min(W64[s][1], W64[s][2]) for s in SEEDS)

ROWS = [(rich(var("α"), " = ", str(r)), [W64[s][k] for s in SEEDS]) for k, r in enumerate(RATES)]
ROWS += [(rich(var("α"), " = ", str(r), ", 16 neurons"), [W16[s][r] for s in SEEDS]) for r in (0.05, 0.1)]
WINS = {1: best.count(0.05), 2: best.count(0.1)}
CHANCE = 1 / 3

F3 = lambda v: f"{v:.3f}"  # noqa: E731
names = [str(r) for r in RATES] + ["0.05 with 16 neurons", "0.1 with 16 neurons"]
fig = Figure(
    "04-learning-rates-over-seeds", "Large gaps hold in every seed; 0.05 against 0.1 does not",
    "Seven rows, each a band over seeds 0 to 4 of the 5-fold mean accuracy on the spiral, with a tick per seed and a "
    "dot for seed 0, on an axis from 0.3 to 0.85 with a dotted line at 0.333, guessing among three classes. "
    + "; ".join(f"learning rate {n}: {F3(min(v))} to {F3(max(v))}, seed 0 {F3(v[0])}" for n, (_, v) in
                zip(names, ROWS)).capitalize()
    + ". 64 hidden neurons unless the row says 16. The highest mean of a seed is 0.1 in four seeds and 0.05 in one.",
    subtitle="5-fold mean accuracy on the spiral, seeds 0 to 4; 64 hidden neurons unless a row says 16.",
    data_w=True)

XB = 920
ax = fig.dot_plot(Box(40, 112, 880, 336), [(lab, []) for lab, _ in ROWS], 0.3, 0.85,
                  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fmt=lambda v: f"{v:.1f}", label_w=192, pad_right=104,
                  axis_label="5-fold mean accuracy", ref_lines=[dict(x=CHANCE, label="guessing, 0.333")])
fig.text(XB, 128, "best in", "note", anchor="end")
with fig.data():
    for i, (_, vals) in enumerate(ROWS):
        y = ax.sy(i)
        fig.range_mark(ax.sx(min(vals)), ax.sx(max(vals)), y, ticks=[ax.sx(v) for v in vals], color="output",
                       dot=ax.sx(vals[0]))
        if i in WINS:
            fig.text(XB, y + 5, f"{WINS[i]} of 5 seeds", "value", anchor="end",
                     color="output", snap=False)
    yd = (ax.sy(4) + ax.sy(5)) / 2
    fig.edge((40, yd), (XB, yd), color="border")
fig.legend(232, 476, [dict(color="output", label="five seeds, one tick each", mark="band"),
                      dict(color="output", label="seed 0", mark="circle")], direction="row", gap=40)
fig.caption("Every seed rules out 0.5, 1.0 and 16 neurons; the bands of 0.05 and 0.1 overlap.")
fig.write()
