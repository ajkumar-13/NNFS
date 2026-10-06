"""Post 29, section 3: the spread of one validation split against the spread of a 5-fold mean.

Run from anywhere:  python posts/29-validation-and-hyperparameter-tuning/diagrams/src/03-steadiness.py
Writes posts/29-validation-and-hyperparameter-tuning/diagrams/03-steadiness.svg (about half a minute).
snippets/steadiness.py is run here as it is (25 to 60 seconds); its eight "shuffle r" lines give every value drawn.
The three summary lines that index.md quotes are asserted to be in that output and are recomputed from the
printed fold accuracies; shuffle 0 is asserted equal to the learning-rate-0.1 row of search.py in section 5.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich  # noqa: E402

POST = Path(__file__).resolve().parents[2]
MD = (POST / "index.md").read_text(encoding="utf-8")
OUT = subprocess.run([sys.executable, "-B", str(POST / "snippets" / "steadiness.py")], cwd=str(POST.parents[1]),
                     capture_output=True, text=True, check=True).stdout

listing = MD[MD.index("```text\none split of 60 (40 values)"):].split("```")[1].splitlines()[1:]
assert len(listing) == 3 and all(line in OUT.splitlines() for line in listing), listing

ROWS = re.findall(r"^shuffle (\d)  folds \[([^\]]+)\]  mean (\d\.\d{3})$", OUT, re.M)
assert [int(r[0]) for r in ROWS] == list(range(8))
FOLDS = [[float(v) for v in r[1].split()] for r in ROWS]
MEANS = [float(r[2]) for r in ROWS]
assert all(len(f) == 5 for f in FOLDS)
assert all(abs(np.mean(f) - m) <= 0.0006 for f, m in zip(FOLDS, MEANS))   # fold values printed to 3 decimals
single = [v for f in FOLDS for v in f]
assert (f"{min(single):.3f}", f"{max(single):.3f}", f"{min(MEANS):.3f}", f"{max(MEANS):.3f}") == \
       ("0.600", "0.883", "0.723", "0.800")
SD1, SD5 = re.search(r"one split of 60 \(40 values\):  .*standard deviation (\d\.\d{3})", OUT).group(1), \
    re.search(r"5-fold mean \(8 values\):  .*standard deviation (\d\.\d{3})", OUT).group(1)
RATIO = re.search(r"ratio of the standard deviations (\d\.\d\d)", OUT).group(1)
assert (SD1, SD5, RATIO) == ("0.064", "0.025", "2.59")
assert f"{np.std(single):.3f}" == SD1           # the printed folds reproduce the spread
# shuffle 0 is the k-fold run of search.py: same data, rate 0.1, weights after seed 100, folds from default_rng(0)
m = re.search(r"^lr=0\.1\s+mean_acc=(\d\.\d{3}) std=\d\.\d{3}  folds \[([^\]]+)\]$", MD, re.M)
assert [float(v) for v in m.group(2).split()] == FOLDS[0] and float(m.group(1)) == MEANS[0]

F3 = lambda v: f"{v:.3f}"  # noqa: E731
fig = Figure(
    "03-steadiness", "One split of 60 swings; the mean of five swings less",
    "Eight rows, shuffles 0 to 7 of the same 300 points of seed 0, learning rate 0.1 and the same initial weights. "
    "Each row is a band over its five fold accuracies, with a tick per fold and a dot at their mean: "
    + "; ".join(f"shuffle {r}: {F3(min(f))} to {F3(max(f))}, mean {F3(m)}" for r, (f, m) in
                enumerate(zip(FOLDS, MEANS)))
    + f". A last row is a band over the eight means, 0.723 to 0.800. The 40 single folds have a standard deviation "
    f"of {SD1}, the eight means one of {SD5}, a ratio of {RATIO}.",
    subtitle="Seed 0, learning rate 0.1, the same initial weights; only the shuffle that cuts the folds changes.",
    height=720, data_w=True)

XS = 920                                            # right end of the mean column
labels = [f"shuffle {r}" for r in range(8)] + ["eight means"]
ax = fig.dot_plot(Box(40, 144, 880, 456), [(lab, []) for lab in labels], 0.55, 0.9,
                  [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], fmt=lambda v: f"{v:.2f}", label_w=128,
                  pad_right=96, axis_label="validation accuracy")
fig.text(XS, 136, "mean", "note", anchor="end")
with fig.data():
    for r in range(8):
        y = ax.sy(r)
        fig.range_mark(ax.sx(min(FOLDS[r])), ax.sx(max(FOLDS[r])), y, ticks=[ax.sx(v) for v in FOLDS[r]],
                       color="output")
        fig.marker(ax.sx(MEANS[r]), y, "circle", "blue", size=10)
        fig.text(XS, y + 5, F3(MEANS[r]), "value", anchor="end", color="blue", snap=False)
    y = ax.sy(8)
    fig.edge((ax.x, (ax.sy(7) + y) / 2), (XS, (ax.sy(7) + y) / 2), color="border")
    fig.range_mark(ax.sx(min(MEANS)), ax.sx(max(MEANS)), y, ticks=[ax.sx(v) for v in MEANS], color="blue")
fig.legend(168, 632, [dict(color="output", label=rich("the five folds of one shuffle; standard deviation of all 40: ", SD1),
                                mark="band"),
                           dict(color="blue", label=rich("their mean, the 5-fold score; standard deviation of the eight: ", SD5),
                                mark="circle")])
fig.caption(rich("The 40 single folds spread ", RATIO, " times as widely as the eight means."))
fig.write()
