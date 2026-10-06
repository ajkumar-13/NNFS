"""Post 09 hero (section 4): the final loss of each strategy on the spiral after 10,000 iterations.

Run from anywhere:  python posts/09-introduction-to-optimisation/diagrams/src/01-strategies-compared.py
Writes posts/09-introduction-to-optimisation/diagrams/01-strategies-compared.svg (about two minutes).

Every number comes from the post's snippets, run here:
  random_selection.py           the best loss after 10,000 draws (21 parameters)
  random_perturbation.py        the printed walk on 21 and on 387 parameters; then the 20 other streams of
                                nudges of its seeds mode, recomputed with the snippet's own function, data
                                and seeds (500 to 519), and asserted against the lowest, median and highest
                                that section 4 quotes
  gradient_descent_preview.py   the final loss at learning rates 1 and 0.1 on both networks
"""
import contextlib
import io
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
SNIP = POST / "snippets"


def run(name):
    return subprocess.run([sys.executable, str(SNIP / name)], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


# -- random selection: the best of 10,000 draws
sel_out = run("random_selection.py")
SEL = float(re.search(r"after  10,000 draws: best loss (\d\.\d{4})", sel_out).group(1))
assert SEL == 1.0981

# -- gradient descent: the final loss of each of the four runs, in the order the script prints them
gd_out = run("gradient_descent_preview.py")
blocks = re.split(r"(?m)^spiral, ", gd_out)[1:]
GD = {}
for b in blocks:
    n = int(re.match(r"(\d+) hidden", b).group(1))
    lr = re.search(r"learning rate (\S+)", b).group(1)
    GD[(n, lr)] = float(re.search(r"after  10,000 steps: loss (\d\.\d{4})", b).group(1))
assert GD == {(3, "1.0"): 1.0776, (3, "0.1"): 1.0797, (64, "1.0"): 0.8737, (64, "0.1"): 1.0226}, GD

# -- random perturbation: the printed runs, then the 20 other streams of the seeds mode
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(SNIP / "random_perturbation.py"), run_name="snippet")
rp_out = buf.getvalue()
PRINTED = {}
for b in re.split(r"(?m)^spiral, ", rp_out)[1:]:
    n = int(re.match(r"(\d+) hidden", b).group(1))
    PRINTED[n] = float(re.search(r"after  10,000 iterations: loss (\d\.\d{4})", b).group(1))
assert PRINTED == {3: 1.0435, 64: 1.0730}, PRINTED

STREAMS = {}
for n_hidden in (3, 64):                      # the loop of the snippet's seeds mode, unchanged
    finals = []
    for nudge_seed in range(500, 520):
        np.random.seed(0)
        X, y = ns["spiral_data"](samples=100, classes=3)
        with contextlib.redirect_stdout(io.StringIO()):
            finals.append(float(ns["random_perturbation"](X, y, n_hidden=n_hidden, nudge_seed=nudge_seed)[0]))
    STREAMS[n_hidden] = finals
STATS = {n: (f"{min(v):.4f}", f"{np.median(v):.4f}", f"{max(v):.4f}") for n, v in STREAMS.items()}
assert STATS == {3: ("1.0321", "1.0684", "1.0793"), 64: ("1.0442", "1.0595", "1.0666")}, STATS
assert sum(v < 1.0776 for v in STREAMS[3]) == 16 and sum(v < 1.0797 for v in STREAMS[3]) == 20
assert sum(v < 0.8737 for v in STREAMS[64]) == 0 and sum(v < 1.0226 for v in STREAMS[64]) == 0
LN3 = float(np.log(3))
assert f"{LN3:.4f}" == "1.0986"

f4 = lambda v: f"{v:.4f}"  # noqa: E731

fig = Figure(
    "01-strategies-compared", "Only gradient descent uses the 387 parameters",
    "A dot chart of the final loss on the spiral data after 10,000 iterations from the same start, on a loss "
    "axis from 0.85 to 1.10, with a dotted line at ln 3, 1.0986, the loss of a uniform guess. With 21 "
    f"parameters: random selection's best draw {f4(SEL)}; random perturbation, the printed run {f4(PRINTED[3])} "
    f"and 20 other streams of nudges, one tick each in a band, from {STATS[3][0]} to {STATS[3][2]}, median "
    f"{STATS[3][1]}; gradient descent "
    f"{f4(GD[(3, '1.0')])} at learning rate 1 and {f4(GD[(3, '0.1')])} at 0.1. With 387 parameters: random "
    f"perturbation, the printed run {f4(PRINTED[64])} and 20 streams from {STATS[64][0]} to {STATS[64][2]}, "
    f"median {STATS[64][1]}; gradient descent {f4(GD[(64, '1.0')])} at learning rate 1 and "
    f"{f4(GD[(64, '0.1')])} at 0.1.",
    subtitle="Spiral data, 10,000 iterations from one start; the walk repeated with 20 streams of nudges.")

LO, HI, TICKS = 0.85, 1.10, [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
PX0, PX1 = 312, 784                       # the plot's left and right edges (clear of the longest heading)
VX = 920                                  # the value column, right-aligned
sx = lambda v: PX0 + (PX1 - PX0) * (v - LO) / (HI - LO)  # noqa: E731
ALPHA = "α"
GROUPS = [
    ("21 parameters, 3 hidden neurons", 120, [
        ("Random selection, best draw", "sel", SEL),
        ("Random perturbation", "walk", 3),
        (rich("Gradient descent, ", ALPHA, " = 1"), "gd", GD[(3, "1.0")]),
        (rich("Gradient descent, ", ALPHA, " = 0.1"), "gd", GD[(3, "0.1")]),
    ]),
    ("387 parameters, 64 hidden neurons", 300, [
        ("Random perturbation", "walk", 64),
        (rich("Gradient descent, ", ALPHA, " = 1"), "gd", GD[(64, "1.0")]),
        (rich("Gradient descent, ", ALPHA, " = 0.1"), "gd", GD[(64, "0.1")]),
    ]),
]
STEP = 36
TOP, AXIS = 132, 428                      # the plot's vertical extent

# -- grid lines, the ln 3 reference, the axis and its ticks
with fig.data():
    for t in TICKS:
        fig.edge((sx(t), TOP), (sx(t), AXIS), color="grid", width=0.75)
    fig.edge((sx(LN3), TOP), (sx(LN3), AXIS), color="ink-muted", width=1, dash="ref")
    fig.edge((PX0, AXIS), (PX1, AXIS), color="ink-muted", width=1)
    for t in TICKS:
        fig.text(sx(t), AXIS + 20, f"{t:.2f}", "tick", anchor="middle", snap=False)
    fig.text(sx(LN3), 120, "ln 3", "note", anchor="middle", snap=False)
fig.text(40, AXIS + 20, "final loss", "note")
fig.text(VX, 120, "final loss", "note", anchor="end")


for heading, base, rows in GROUPS:
    fig.text(40, base, heading, "head")
    for k, (label, kind, val) in enumerate(rows):
        cy = base + 32 + k * STEP
        fig.text(40, cy + 5, label, "label")
        with fig.data():
            if kind == "sel":
                fig.marker(sx(val), cy, "square", "ink-muted", size=10)
                fig.text(VX, cy + 5, f4(val), "value", anchor="end", color="ink-muted", snap=False)
            elif kind == "gd":
                fig.marker(sx(val), cy, "diamond", "gradient", size=10)
                fig.text(VX, cy + 5, f4(val), "value", anchor="end", color="gradient", snap=False,
                         bold=(val == GD[(64, "1.0")]))
            else:
                lo, hi = min(STREAMS[val]), max(STREAMS[val])
                # the band with ticks: one tick per stream, the printed run as the solid dot
                fig.range_mark(sx(lo) - 4, sx(hi) + 4, cy, ticks=[sx(v) for v in STREAMS[val]], color="blue",
                               dot=sx(PRINTED[val]), h=24)
                # two short lines in the value column: the printed run (its mark's colour) over the median (muted)
                fig.text(VX, cy - 3, rich("run ", f4(PRINTED[val])), "value", anchor="end", color="blue",
                         snap=False)
                fig.text(VX, cy + 15, rich("median ", STATS[val][1]), "value", anchor="end", color="ink-muted",
                         snap=False)

# -- legend for the walk's two marks, under the axis
fig.legend(PX0, 472, [dict(color="blue", label="20 other streams, one tick each", mark="band"),
                      dict(color="blue", label="printed run", mark="circle")], direction="row")
fig.caption("Dotted line: ln 3 = 1.0986, the loss of a uniform guess.")
fig.write()
