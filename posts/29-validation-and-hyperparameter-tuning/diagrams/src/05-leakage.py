"""Post 29, section 6: feature selection fitted before the split against the same step fitted inside each fold.

Run from anywhere:  python posts/29-validation-and-hyperparameter-tuning/diagrams/src/05-leakage.py
Writes posts/29-validation-and-hyperparameter-tuning/diagrams/05-leakage.svg.
snippets/leakage.py is run here as it is (a few seconds, no training loop); its three rows of ten accuracies give
every value drawn, and the three lines are asserted equal to the listing of section 6 in index.md. The fold sizes
(80 training rows, 20 validation rows) are computed with the post's k_fold_indices on 100 rows.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich  # noqa: E402

POST = Path(__file__).resolve().parents[2]
MD = (POST / "index.md").read_text(encoding="utf-8")
OUT = subprocess.run([sys.executable, "-B", str(POST / "snippets" / "leakage.py")], cwd=str(POST.parents[1]),
                     capture_output=True, text=True, check=True).stdout
sys.path.insert(0, str(POST / "snippets"))
from kfold import k_fold_indices  # noqa: E402

NAMES = ("selected before the split ", "selected inside each fold ", "leaky pipeline on new rows")
ROWS = {}
for name in NAMES:
    m = re.search(re.escape(name) + r"  \[([^\]]+)\]  mean (\d\.\d{3})$", OUT, re.M)
    assert m, name
    line = m.group(0)
    assert line in MD.splitlines(), line                  # the listing of section 6
    ROWS[name] = ([float(v) for v in m.group(1).split()], m.group(2))
assert all(len(v) == 10 for v, _ in ROWS.values())
LEAKY, CLEAN, NEW = (ROWS[n] for n in NAMES)
assert (min(LEAKY[0]), max(LEAKY[0]), LEAKY[1], CLEAN[1], NEW[1]) == (0.82, 0.92, "0.887", "0.523", "0.503")
assert "100 rows, 2,000 random features" in MD and "keeps the 20 features whose class means differ most" in MD
folds = list(k_fold_indices(100, 5))
assert {(len(t), len(v)) for t, v in folds} == {(80, 20)}

COLORS = ("error", "blue", "output")
LABELS = ("selected before the split", "selected inside each fold", "leaky pipeline on new rows")
NOTES = ("20 of 2,000 features picked on all 100 rows, then 5-fold", "picked again on the 80 training rows of each fold",
         "its final model, scored on 2,000 new rows")
F2 = lambda v: f"{v:.2f}"  # noqa: E731
fig = Figure(
    "05-leakage", "Selecting features before the split finds signal in noise",
    "Three rows on an accuracy axis from 0.3 to 1.0 with a dotted line at 0.5, chance on these random labels; each "
    "row is a band over ten seeds with a tick per seed and a dot at the mean. 5-fold cross-validation with the 20 "
    f"features selected on all 100 rows: {F2(min(LEAKY[0]))} to {F2(max(LEAKY[0]))}, mean {LEAKY[1]}. Selected "
    f"inside each fold on its 80 training rows: {F2(min(CLEAN[0]))} to {F2(max(CLEAN[0]))}, mean {CLEAN[1]}. The "
    f"leaky pipeline's model on 2,000 new rows: {F2(min(NEW[0]))} to {F2(max(NEW[0]))}, mean {NEW[1]}.",
    subtitle="Pure noise: 100 rows, 2,000 random features, random labels. A band spans ten seeds; its dot is their mean.",
    data_w=True)

XM = 920
ax = fig.dot_plot(Box(40, 112, 880, 360), [(lab, []) for lab in LABELS], 0.3, 1.0,
                  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fmt=lambda v: f"{v:.1f}", label_w=360, pad_right=88,
                  axis_label="accuracy", ref_lines=[dict(x=0.5, label="chance, 0.5")])
fig.text(XM, 128, "mean", "note", anchor="end")
with fig.data():
    for i, (vals, mean) in enumerate((LEAKY, CLEAN, NEW)):
        y = ax.sy(i)
        fig.text(40, y + 25, NOTES[i], "note", snap=False)
        fig.range_mark(ax.sx(min(vals)), ax.sx(max(vals)), y, ticks=[ax.sx(v) for v in vals], color=COLORS[i],
                       dot=ax.sx(float(mean)))
        fig.text(XM, y + 5, mean, "value", anchor="end", color=COLORS[i], snap=False)
fig.caption("The split is the same in both pipelines; in the first, the selection has already seen the validation labels.")
fig.write()
