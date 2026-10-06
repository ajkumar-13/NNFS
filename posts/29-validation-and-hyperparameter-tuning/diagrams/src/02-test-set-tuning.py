"""Post 29, section 2: what choosing a learning rate by its test accuracy does to the reported number.

Run from anywhere:  python posts/29-validation-and-hyperparameter-tuning/diagrams/src/02-test-set-tuning.py
Writes posts/29-validation-and-hyperparameter-tuning/diagrams/02-test-set-tuning.svg (about a minute).
snippets/test_set_tuning.py is run here as it is (35 to 60 seconds); its per-seed lines give every value drawn.
The lines index.md quotes (seed 3 and the three summary lines) are asserted to be in that output, and the readings
of the section's prose (every seed positive, the ranges, 31 to 96 negative sets) are asserted on the values.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
MD = (POST / "index.md").read_text(encoding="utf-8")
OUT = subprocess.run([sys.executable, "-B", str(POST / "snippets" / "test_set_tuning.py")], cwd=str(POST.parents[1]),
                     capture_output=True, text=True, check=True).stdout

# -- the listing of section 2 is part of the output
listing = MD[MD.index("```text\nseed 3  accuracy on the 30,000"):].split("```")[1].splitlines()[1:]
assert len(listing) == 6 and all(line in OUT.splitlines() for line in listing), listing

SEEDS = range(5)
T300 = [float(v) for v in re.findall(r"200 test sets of 300: chosen by test accuracy, mean \(test - further\) "
                                     r"([+-]\d\.\d{4})", OUT)]
NEG = [int(v) for v in re.findall(r"negative on (\d+)$", OUT, re.M)]
CLEAN = [float(v) for v in re.findall(r"chosen on one set, reported on the next: mean \(test - further\) "
                                      r"([+-]\d\.\d{4})", OUT)]
T60 = [float(v) for v in re.findall(r"test sets of 60: chosen by test accuracy, mean \(test - further\) "
                                    r"([+-]\d\.\d{4})", OUT)]
assert len(T300) == len(CLEAN) == len(T60) == len(NEG) == 5
assert re.findall(r"^seed (\d)  ", OUT, re.M) == [str(s) for s in SEEDS]
assert all(v > 0 for v in T300 + T60)                                   # too high on average in every seed
assert (min(T300), max(T300), min(T60), max(T60)) == (0.0005, 0.0182, 0.0286, 0.0502)
assert max(abs(v) for v in CLEAN) == 0.0021 and (min(NEG), max(NEG)) == (31, 96)
assert "by 0.05 to 1.8 percentage points when the test set has 300 points, and by 2.9 to 5.0 points" in MD
assert "that report is off by 0.21 points at most, in either direction" in MD

PP = lambda v: 100 * v                                                   # noqa: E731  percentage points
F = lambda v: num(round(PP(v), 2), 2)                                    # noqa: E731
COLORS = ("error", "error", "blue")
SHAPES = ("diamond", "circle", "square")
HOLLOW = (False, True, False)
LABELS = ("chosen by test accuracy, 60 test points", "chosen by test accuracy, 300 test points",
          "chosen on one set, reported on the next")


def listed(vals):
    return ", ".join(F(v) for v in vals)


fig = Figure(
    "02-test-set-tuning", "Picking the best test score reports too high a number",
    "A dot plot, one row per seed 0 to 4, of the mean over 200 test sets of the reported test accuracy minus the "
    "accuracy on 30,000 further points, in percentage points, for eight learning rates trained on 300 points. "
    f"Chosen by test accuracy on test sets of 60 points: {listed(T60)}; on test sets of 300 points: {listed(T300)}; "
    f"chosen on one set and reported on the next: {listed(CLEAN)}. A dotted line marks zero.",
    subtitle="Mean of (reported test accuracy − accuracy on 30,000 further points) over 200 test sets; eight learning rates.",
    data_w=True)

rows = [(f"seed {s}", [PP(T60[s]), PP(T300[s]), PP(CLEAN[s])]) for s in SEEDS]
XC = (712, 816, 920)                                       # right ends of the three value columns
ax = fig.dot_plot(Box(40, 128, 880, 296), [(lab, []) for lab, _ in rows], -1, 6, [-1, 0, 1, 2, 3, 4, 5, 6],
                  fmt=lambda v: num(v), label_w=80, pad_right=264, axis_label="percentage points",
                  ref_lines=[dict(x=0, label="no bias")])
with fig.data():
    for s, (lab, vals) in enumerate(rows):
        y = ax.sy(s)
        fig.edge((ax.sx(min(vals)), y), (ax.sx(max(vals)), y), color="rule", width=1.5)
        for k, v in enumerate(vals):
            fig.marker(ax.sx(v), y, SHAPES[k], COLORS[k], size=10, hollow=HOLLOW[k])
        for k, v in enumerate((T60[s], T300[s], CLEAN[s])):
            fig.text(XC[k], y + 5, F(v), "value", anchor="end", color=COLORS[k], snap=False)
for k, head in enumerate(("60 points", "300 points", "next set")):
    fig.text(XC[k], 144, head, "note", anchor="end")

fig.legend(40, 468, [dict(color=c, label=l, mark=m, hollow=h) for c, l, m, h in
                     zip(COLORS, ("best of eight on 60 points", "best of eight on 300", "chosen on one, reported on the next"),
                         SHAPES, HOLLOW)], direction="row", gap=32)
fig.caption("The fewer the test points, the more the luckiest of eight scores lifts the report.")
fig.write()
