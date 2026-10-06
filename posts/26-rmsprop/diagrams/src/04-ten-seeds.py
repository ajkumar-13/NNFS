"""Post 26, section 7.2: AdaGrad against RMSProp on seeds 0 to 9, the final epoch and the last 1,000 epochs.

Run from anywhere:  python posts/26-rmsprop/diagrams/src/04-ten-seeds.py   (about a minute)
Writes posts/26-rmsprop/diagrams/04-ten-seeds.svg.
The four seed scripts of section 7.2, snippets/seeds_adagrad.py, seeds_adagrad_more.py, seeds_rmsprop.py and
seeds_rmsprop_more.py, are run here as they are (four processes side by side, about a minute together), and their
printed rows give every value drawn. Each row is asserted against its line of the section 7.2 table in index.md, and
the counts the section states are asserted on the printed values. For layout work only, FIG26_SEEDS_DIR may name a
folder holding saved stdouts of the same four scripts (seeds_<name>.out).

Layout: three dot plots side by side, one row per seed: the final accuracy, the lowest accuracy of the last 1,000
epochs and the highest loss of the last 1,000 epochs, AdaGrad and RMSProp on each row.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
NAMES = ("adagrad", "adagrad_more", "rmsprop", "rmsprop_more")
CACHE = os.environ.get("FIG26_SEEDS_DIR")
if CACHE:
    OUTS = {n: (Path(CACHE) / f"seeds_{n}.out").read_text(encoding="utf-8") for n in NAMES}
else:
    procs = {n: subprocess.Popen([sys.executable, str(POST / "snippets" / f"seeds_{n}.py")], cwd=str(ROOT),
                                 stdout=subprocess.PIPE, text=True) for n in NAMES}
    OUTS = {}
    for n, p in procs.items():
        OUTS[n], _ = p.communicate()
        assert p.returncode == 0, n

ROW = re.compile(r"^ +(\d)   (\d\.\d{4}) +(\d\.\d{4}) +(\d\.\d{4}) +(\d\.\d{4}) +(\d\.\d{4})$", re.M)
RUNS = {"adagrad": {}, "rmsprop": {}}
for n, out in OUTS.items():
    setting = {"adagrad": "Optimizer_Adagrad(learning_rate=1.0, decay=1e-4)",
               "rmsprop": "Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)"}[n.split("_")[0]]
    assert setting in out and setting in INDEX, n
    for m in ROW.finditer(out):
        loss, acc, mean, low, high = (float(v) for v in m.groups()[1:])
        RUNS[n.split("_")[0]][int(m.group(1))] = dict(loss=loss, acc=100 * acc, mean=100 * mean, low=100 * low,
                                                      high=high)
SEEDS = range(10)
assert all(sorted(RUNS[k]) == list(SEEDS) for k in RUNS), RUNS

# -- every row is a line of the section 7.2 table
for seed in SEEDS:
    cells = " | ".join(f"{r['loss']:.4f} | {r['acc']:.2f} | {r['low']:.2f} | {r['high']:.4f}"
                       for r in (RUNS["adagrad"][seed], RUNS["rmsprop"][seed]))
    assert f"| {seed} | {cells} |" in INDEX, (seed, cells)

# -- the counts of section 7.2
A, R = RUNS["adagrad"], RUNS["rmsprop"]
AHEAD = [s for s in SEEDS if R[s]["acc"] > A[s]["acc"]]
assert len(AHEAD) == 7 and [s for s in SEEDS if s not in AHEAD] == [3, 5, 7]
assert [s for s in SEEDS if R[s]["mean"] > A[s]["mean"]] == AHEAD
assert "RMSProp is ahead in seven of the ten seeds and behind in seeds 3, 5 and 7" in INDEX
assert (min(R[s]["acc"] for s in SEEDS), max(R[s]["acc"] for s in SEEDS)) == (77.0, 95.0)
assert (round(min(A[s]["acc"] for s in SEEDS), 2), round(max(A[s]["acc"] for s in SEEDS), 2)) == (75.67, 92.67)
LOW = [s for s in SEEDS if R[s]["low"] < 70]
HIGH = [s for s in SEEDS if R[s]["high"] > 1.5]
assert len(LOW) == 8 and len(HIGH) == 8 and LOW != HIGH
assert "In eight of the ten seeds its loss exceeds 1.5" in INDEX and "in eight, not the same eight" in INDEX
assert all(A[s]["low"] >= 75 and A[s]["high"] < 0.6 for s in SEEDS)
assert max(R[s]["high"] for s in SEEDS) == 4.7996 and min(R[s]["low"] for s in SEEDS) == 45.67

ADA_C, RMS_C = "ink-muted", "blue"
COLORS, SHAPES = (ADA_C, RMS_C), ("circle", "diamond")


def pcts(key, k):
    return ", ".join(f"{RUNS[k][s][key]:.2f}" for s in SEEDS)


fig = Figure(
    "04-ten-seeds", "RMSProp mostly ends higher, and spikes late on most seeds",
    "Three dot plots, one row per seed from 0 to 9, AdaGrad (learning rate 1, decay 10 to the minus 4) as a grey "
    "circle and RMSProp (learning rate 0.02, decay 10 to the minus 5, rho 0.999) as a blue diamond on each row, "
    "10,001 epochs per run. Final accuracy in percent, seeds 0 to 9: AdaGrad " + pcts("acc", "adagrad") + "; RMSProp "
    + pcts("acc", "rmsprop") + "; RMSProp is ahead on seven seeds and behind on seeds 3, 5 and 7. Lowest accuracy "
    "of the last 1,000 epochs, with a dotted line at 70 percent: AdaGrad " + pcts("low", "adagrad") + "; RMSProp "
    + pcts("low", "rmsprop") + ", below 70 on eight seeds. Highest loss of the last 1,000 epochs, with a dotted line "
    "at 1.5: AdaGrad " + ", ".join(f"{A[s]['high']:.4f}" for s in SEEDS) + "; RMSProp "
    + ", ".join(f"{R[s]['high']:.4f}" for s in SEEDS) + ", above 1.5 on eight seeds.",
    subtitle="Seeds 0 to 9, 10,001 epochs each, at the documented settings. Late: the last 1,000 epochs.",
    height=720, data_w=True)

Y, H = 136, 464
LABEL_W, GUT = 64, 32
W = (880 - LABEL_W - 2 * GUT) // 3                      # 248: one panel each, 16 of it kept free on the right
X0 = 40
ACC_TICKS = [40, 60, 80, 100]
rows = lambda key, label: [(label(s), [A[s][key], R[s][key]]) for s in SEEDS]  # noqa: E731

la = fig.dot_plot(Box(X0, Y + 24, LABEL_W + W, H - 24), rows("acc", lambda s: f"seed {s}"), 40, 100, ACC_TICKS,
                  colors=COLORS, shapes=SHAPES, label_w=LABEL_W, pad_right=16, fmt=lambda v: num(v))
ma = fig.dot_plot(Box(X0 + LABEL_W + W + GUT, Y, W, H), rows("low", lambda s: ""), 40, 100, ACC_TICKS,
                  colors=COLORS, shapes=SHAPES, label_w=0, pad_right=16, fmt=lambda v: num(v),
                  ref_lines=[dict(x=70, label="70")])
ra = fig.dot_plot(Box(X0 + LABEL_W + 2 * (W + GUT), Y, W, H), rows("high", lambda s: ""), 0, 5,
                  [0, 1, 2, 3, 4, 5], colors=COLORS, shapes=SHAPES, label_w=0, pad_right=16,
                  fmt=lambda v: num(v), ref_lines=[dict(x=1.5, label="1.5")])
assert la.y == ma.y == ra.y and la.bottom == ma.bottom == ra.bottom
fig.text(la.x, 120, "Final accuracy, percent", "head")
fig.text(ma.x, 120, "Lowest late accuracy, percent", "head")
fig.text(ra.x, 120, "Highest late loss", "head")

fig.legend(la.x, 648, [dict(color=ADA_C, label="AdaGrad, post 25", mark="circle"),
                                 dict(color=RMS_C, label="RMSProp", mark="diamond")], direction="row", gap=40)
fig.caption("RMSProp ends higher on seven seeds; on eight its late loss passes 1.5 at least once.")
fig.write()
