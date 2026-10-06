"""Post 23, section 7.1: the plateau test on the five stalled runs of decay=1e-2.

Run from anywhere:  python posts/23-learning-rate-decay/diagrams/src/04-plateau-test.py
Writes posts/23-learning-rate-decay/diagrams/04-plateau-test.svg. Takes about a minute.

Every number is printed by snippets/decay_too_large.py, which is run here (about 58 seconds, nothing added to it):
per seed the loss and accuracy after 10,001 epochs with decay=1e-2, the length of the gradient there, and the loss
and accuracy after 3,000 more epochs at a constant rate of 1.0. Each parsed row is asserted to be a line of the
listing in index.md. ln 3 is computed here and asserted to be the loss of epoch 0 that train_with_decay.py prints.
"""
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
CACHE = os.environ.get("FIG23_DECAY_TOO_LARGE_OUT")  # layout work only: a saved stdout of the same snippet
if CACHE:
    OUT = Path(CACHE).read_text(encoding="utf-8")
else:
    OUT = subprocess.run([sys.executable, str(POST / "snippets" / "decay_too_large.py")], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout

ROW = re.compile(r"^ +(\d)  (\d\.\d{4})  (\d\.\d{4})  (0\.0099) +(\d\.\d{4}) +(\d\.\d{5}) +(\d\.\d{4}) +(\d\.\d{4})$",
                 re.M)
RUNS = []
for m in ROW.finditer(OUT):
    assert m.group(0) in INDEX, m.group(0)                       # the listing of section 7.1
    RUNS.append(dict(seed=int(m.group(1)), loss=float(m.group(2)), acc=float(m.group(3)), grad=m.group(5),
                     after=float(m.group(7)), acc_after=float(m.group(8))))
assert [r["seed"] for r in RUNS] == [0, 1, 2, 3, 4]
assert all(r["after"] < r["loss"] and r["acc_after"] > r["acc"] for r in RUNS)     # the loss falls on every seed
GRADS = sorted(float(r["grad"]) for r in RUNS)
assert (f"{GRADS[0]:.4f}", f"{GRADS[-1]:.4f}") == ("0.0030", "0.0094")
LN3 = math.log(3)
assert "epoch     0  loss 1.0986" in INDEX and f"{LN3:.4f}" == "1.0986"

D2 = rich(var("d"), " = ", sup("10", num(-2), italic=False))
A1 = rich(var("α"), " = 1")
PLATEAU, AFTER = dict(color="ink-muted", shape="circle", hollow=True), dict(color="ink", shape="diamond")
pct = lambda a: f"{100 * a:.1f}"  # noqa: E731

fig = Figure(
    "04-plateau-test", "Restoring the rate ends the plateau on every seed",
    "A dot chart, one row per seed 0 to 4, on a loss axis from 0.7 to 1.15 with a dotted line at ln 3 = 1.0986. A "
    "hollow circle marks the loss after 10,001 epochs with decay 10 to the minus 2, a diamond the loss of the same "
    "network after 3,000 more epochs at a constant rate of 1, joined by a line. Seed 0: 1.0725 to 0.9548, accuracy "
    "39.7 to 53.3 percent. Seed 1: 1.0539 to 0.8449, 44.0 to 54.7. Seed 2: 1.0636 to 0.9019, 46.3 to 56.7. Seed 3: "
    "1.0636 to 0.7486, 42.0 to 59.7. Seed 4: 1.0457 to 0.7384, 46.3 to 63.7. The caption gives the gradient length "
    "at the plateaus, 0.0030 to 0.0094.",
    subtitle=rich("Five runs of 10,001 epochs with ", D2, ", then 3,000 more epochs on the same network at ", A1, "."),
    data_w=True)

LO, HI, TICKS = 0.7, 1.15, [0.7, 0.8, 0.9, 1.0, 1.1]
LOSS_X, ACC_X = 792, 920
rows = [(f"seed {r['seed']}", [r["loss"], r["after"]]) for r in RUNS]
ax = fig.dot_plot(Box(40, 128, 880, 288), rows, LO, HI, TICKS, colors=(PLATEAU["color"], AFTER["color"]),
                  shapes=(PLATEAU["shape"], AFTER["shape"]), hollow=False, label_w=96, pad_right=264,
                  axis_label="loss", fmt_tick=lambda v: f"{v:.1f}",
                  ref_lines=[dict(x=LN3, label=rich("ln 3 = ", f"{LN3:.4f}"))])
with fig.data():
    for i, r in enumerate(RUNS):        # the hollow plateau mark over the solid one dot_plot drew
        fig.marker(ax.sx(r["loss"]), ax.sy(i), "circle", "surface", size=10)
        fig.marker(ax.sx(r["loss"]), ax.sy(i), "circle", PLATEAU["color"], size=10, hollow=True)
        fig.text(LOSS_X, ax.sy(i) + 5, rich(f"{r['loss']:.4f}", " → ", f"{r['after']:.4f}"), "value",
                 anchor="end", snap=False)
        fig.text(ACC_X, ax.sy(i) + 5, rich(pct(r["acc"]), " → ", pct(r["acc_after"])), "value", anchor="end",
                 snap=False)
fig.text(LOSS_X, 120, "loss", "note", anchor="end")
fig.text(ACC_X, 120, "accuracy, percent", "note", anchor="end")

fig.legend(136, 456, [dict(color=PLATEAU["color"], label=rich("after 10,001 epochs at ", D2), mark="circle",
                           hollow=True),
                      dict(color=AFTER["color"], label=rich("after 3,000 more at ", A1), mark="diamond")],
           direction="row")
fig.caption(rich("At the plateaus the gradient had length ", f"{GRADS[0]:.4f}", " to ",
                 f"{GRADS[-1]:.4f}", ", which looked like a minimum."))
fig.write()
