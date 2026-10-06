"""Post 22, sections 4 and 6: the documented run from ten seeds, and five of them run five times as long.

Run from anywhere:  python posts/22-gradient-descent-optimiser/diagrams/src/04-seeds-and-budget.py
Writes posts/22-gradient-descent-optimiser/diagrams/04-seeds-and-budget.svg.
Sources: seed_spread.py's train() is run here for seeds 0 to 9 (about 70 s, what `seed_spread.py 10` runs), and
each seed's printed row is formatted as the snippet formats it and asserted to be in the section 4 listing of
index.md. `seed_spread.py long` takes about 160 s, so the accuracy at epoch 50,000 is parsed from the section 6
listing of index.md, which the series lint (--run) keeps equal to the snippet's output; its epoch-10,000 columns
are asserted against the runs made here.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIPPETS = POST / "snippets"
sys.path.insert(0, str(SNIPPETS))
MD = (POST / "index.md").read_text(encoding="utf-8")

with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path(str(SNIPPETS / "seed_spread.py"), run_name="snippet")
import nnfs  # noqa: E402

nnfs.init()                                    # as seed_spread.py's main does before its runs
SEEDS = range(10)
RUNS = {}
for seed in SEEDS:
    losses, accuracies, dead = s["train"](seed)
    rises = int(np.sum(np.diff(losses) > 0))
    line = (f"{seed:<4}  {losses[-1]:<10.4f}  {accuracies[-1]:<9.4f}  {rises:<12,}  {dead:<5}  "
            f"{losses[-1000:].min():<11.4f}  {losses[-1000:].max():<12.4f}  "
            f"{accuracies[-1000:].min():<10.4f}  {accuracies[-1000:].max():.4f}")
    assert line in MD, line                    # the row the section 4 listing prints for this seed
    RUNS[seed] = dict(final=accuracies[-1], lo=accuracies[-1000:].min(), hi=accuracies[-1000:].max(),
                      loss=losses[-1])
FINALS = [RUNS[k]["final"] for k in SEEDS]
assert (f"{min(FINALS):.4f}", f"{max(FINALS):.4f}") == ("0.4300", "0.8700")
assert "over the 10 seeds: final loss 0.3943 to 2.2696, final accuracy 0.4300 to 0.8700" in MD

LONG_HEAD = "seed         10,000         20,000         30,000         40,000         50,000   mean accuracy, last 1,000"
long_rows = MD[MD.index(LONG_HEAD):].split("```")[0].splitlines()[1:6]
LONG = {}
for row in long_rows:
    v = row.split()
    seed, nums = int(v[0]), [float(t) for t in v[1:]]
    assert len(nums) == 11
    assert f"{nums[0]:.4f}" == f"{RUNS[seed]['loss']:.4f}" and f"{nums[1]:.4f}" == f"{RUNS[seed]['final']:.4f}"
    LONG[seed] = nums[9]                        # accuracy at epoch 50,000
assert sorted(LONG) == [0, 1, 2, 3, 4]
assert all(LONG[k] > RUNS[k]["final"] for k in LONG)              # section 6: every seed higher
assert sum(LONG[k] > 0.9 for k in LONG) == 4                      # four of the five above 90 percent
assert "accuracy at epoch 50,000: 0.7433 to 0.9633" in MD

GREEN, BLUE = "output", "positive"
fig = Figure(
    "04-seeds-and-budget", "One seed is one draw; 50,000 epochs lift every seed tried",
    "Ten rows, seeds 0 to 9, on an accuracy axis from 0.35 to 1.0, learning rate 1. For each seed a band spans "
    "the lowest to the highest accuracy over epochs 9,001 to 10,000 and a circle marks the accuracy at epoch "
    "10,000; for seeds 0 to 4 a diamond marks the accuracy at epoch 50,000. At epoch 10,000: "
    + ", ".join(f"seed {k} {RUNS[k]['final']:.4f}" for k in SEEDS)
    + ", from 0.4300 to 0.8700. Bands: "
    + ", ".join(f"{RUNS[k]['lo']:.4f} to {RUNS[k]['hi']:.4f}" for k in SEEDS)
    + ". At epoch 50,000: " + ", ".join(f"seed {k} {LONG[k]:.4f}" for k in sorted(LONG))
    + ", higher than at 10,000 for all five, four of them above 0.9. Seed 0 is the documented run.",
    subtitle="Accuracy of the documented setup from ten seeds, learning rate 1.", height=720, data_w=True)

ROWS = [(f"seed {k}", []) for k in SEEDS]
LABEL_W, PAD_R = 144, 176
ax = fig.dot_plot(Box(40, 136, 880, 464), ROWS, 0.35, 1.0, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  fmt=lambda v: f"{v:.1f}", label_w=LABEL_W, pad_right=PAD_R, axis_label="accuracy")
X10, X50 = 832, 920                            # right ends of the two value columns
with fig.data():
    for k in SEEDS:
        y, r = ax.sy(k), RUNS[k]
        fig.range_mark(ax.sx(r["lo"]), ax.sx(r["hi"]), y, color=GREEN, h=20)
        fig.marker(ax.sx(r["final"]), y, "circle", GREEN, size=10)
        fig.text(X10, y + 5, f"{r['final']:.4f}", "label", anchor="end", snap=False)
        if k in LONG:
            fig.marker(ax.sx(LONG[k]), y, "diamond", BLUE, size=10)
            fig.text(X50, y + 5, f"{LONG[k]:.4f}", "value", anchor="end", color=BLUE, snap=False)
    fig.text(ax.sx(RUNS[0]["lo"]) - 12, ax.sy(0) + 5, "the documented run", "note", anchor="end", snap=False)
fig.text(X10, 120, "epoch 10,000", "note", anchor="end")
fig.text(X50, 120, "50,000", "note", anchor="end")

# the key: the band drawn as the rows draw it (no ticks, which the legend's band mark has), then the two marks
KX, KY = 224, 640
fig.range_mark(KX, KX + 24, KY - 5, color=GREEN, h=16)
fig.text(KX + 32, KY, "range over epochs 9,001 to 10,000", "label")
fig.legend(KX + 296, KY, [dict(color=GREEN, label="epoch 10,000", mark="circle"),
                          dict(color=BLUE, label="epoch 50,000", mark="diamond")], direction="row")
fig.caption("Ten seeds end between 0.43 and 0.87 at epoch 10,000; each of seeds 0 to 4 is higher at 50,000.")
fig.write()
