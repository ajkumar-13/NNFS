"""Post 24, sections 7 and 8: the final loss and accuracy on the spiral for momentum 0, 0.5, 0.9 and 0.99, five seeds.

Run from anywhere:  python posts/24-momentum/diagrams/src/04-beta-over-seeds.py
Writes posts/24-momentum/diagrams/04-beta-over-seeds.svg (about a minute).
The rows for no momentum and 0.9 are parsed from the seed_spread.py listing printed in section 7 of index.md (the
series lint checks that listing against the snippet's output; the snippet itself takes about 70 seconds).
snippets/beta_sweep.py is run here (about a minute) for the rows 0.5 and 0.99, which index.md gives only as
ranges. The ranges and mean accuracies of the section 8 table are asserted against all four rows.
"""
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
SEEDS = range(5)

# -- no momentum and 0.9: the seed_spread.py listing of section 7
ROWS = {0.0: {}, 0.9: {}}
pat = re.compile(r"^\s+(\d)  (decay only|momentum 0\.9)\s+(\d\.\d{4})\s+(\d\.\d{4})\s+(\d\.\d{4})\s", re.M)
for m in pat.finditer(INDEX):
    beta = 0.0 if m.group(2) == "decay only" else 0.9
    ROWS[beta][int(m.group(1))] = (float(m.group(4)), float(m.group(5)))
assert all(sorted(ROWS[b]) == list(SEEDS) for b in ROWS), ROWS

# -- 0.5 and 0.99: beta_sweep.py, run here
out = subprocess.run([sys.executable, str(POST / "snippets" / "beta_sweep.py")], cwd=ROOT, capture_output=True,
                     text=True, check=True).stdout
for m in re.finditer(r"^\s+(\d)\s+(0\.50|0\.99)\s+(\d\.\d{4})\s+(\d\.\d{4})\s", out, re.M):
    ROWS.setdefault(float(m.group(2)), {})[int(m.group(1))] = (float(m.group(3)), float(m.group(4)))
BETAS = (0.0, 0.5, 0.9, 0.99)
assert all(sorted(ROWS[b]) == list(SEEDS) for b in BETAS), ROWS

# -- the section 8 table: ranges of loss and accuracy, mean accuracy
for b, label in ((0.0, r"0 \(decay only\)"), (0.5, r"0\.5"), (0.9, r"0\.9"), (0.99, r"0\.99")):
    loss = [ROWS[b][s][0] for s in SEEDS]
    acc = [ROWS[b][s][1] for s in SEEDS]
    row = (rf"\| {label} \| \d+ \| {min(loss):.4f} to {max(loss):.4f} \| {min(acc):.4f} to {max(acc):.4f} \| "
           rf"{sum(acc) / 5:.4f} \|")
    assert re.search(row, INDEX), (b, row)
assert ROWS[0.9][0] == (0.1209, 0.9567) and ROWS[0.0][0] == (0.7612, 0.6467) and ROWS[0.5][0][1] == 0.7800
# the three readings of section 8
assert all(ROWS[0.5][s][0] < ROWS[0.0][s][0] and ROWS[0.5][s][1] > ROWS[0.0][s][1] for s in SEEDS)
assert all(ROWS[0.9][s][0] < ROWS[0.0][s][0] for s in SEEDS)
assert all(ROWS[0.99][s][1] < ROWS[0.0][s][1] for s in SEEDS)

BETA = "β"
P = lambda v: f"{100 * v:.1f}"                                              # noqa: E731  accuracy in percent
MEANS = {b: sum(ROWS[b][s][1] for s in SEEDS) / 5 for b in BETAS}
assert [P(MEANS[b]) for b in BETAS] == ["64.5", "80.2", "83.4", "54.5"]

desc_rows = []
for b in BETAS:
    loss = [ROWS[b][s][0] for s in SEEDS]
    acc = [ROWS[b][s][1] for s in SEEDS]
    desc_rows.append(f"beta {num(b)}: loss {min(loss):.4f} to {max(loss):.4f} (seed 0 {ROWS[b][0][0]:.4f}), "
                     f"accuracy {P(min(acc))} to {P(max(acc))} percent (seed 0 {P(ROWS[b][0][1])}), mean "
                     f"{P(MEANS[b])}")
fig = Figure(
    "04-beta-over-seeds", "On the spiral, 0.5 and 0.9 help and 0.99 does not",
    "Two dot charts, one row per momentum coefficient, each row a band from the lowest to the highest of five "
    "seeds with one tick per seed and a dot for seed 0. Left the final loss on an axis from 0 to 1.1, right the "
    "accuracy in percent from 30 to 100, with the mean accuracy printed at the right. " + "; ".join(desc_rows) + ".",
    subtitle="The spiral, 10,001 epochs, learning rate 1.0 with decay 0.001; seeds 0 to 4 for each coefficient.",
    data_w=True)

LABELS = [rich(BETA, " = ", num(b)) for b in BETAS]
H = 304
la = fig.dot_plot(Box(40, 128, 440, H), [(lab, []) for lab in LABELS], 0, 1.1, [0, 0.25, 0.5, 0.75, 1.0],
                  label_w=88, pad_right=16, fmt=lambda v: f"{v:.2f}")
ra = fig.dot_plot(Box(512, 128, 408, H), [("", []) for _ in BETAS], 30, 100, [30, 40, 50, 60, 70, 80, 90, 100],
                  label_w=0, pad_right=80, fmt=lambda v: num(v))
fig.text(la.x, 112, "Final loss", "head")
fig.text(ra.x, 112, "Accuracy, percent", "head")
fig.text(ra.right + 80, 112, "mean", "note", anchor="end")
with fig.data():
    for i, b in enumerate(BETAS):
        loss = [ROWS[b][s][0] for s in SEEDS]
        acc = [100 * ROWS[b][s][1] for s in SEEDS]
        fig.range_mark(la.sx(min(loss)), la.sx(max(loss)), la.sy(i), ticks=[la.sx(v) for v in loss],
                       color="output", dot=la.sx(ROWS[b][0][0]))
        fig.range_mark(ra.sx(min(acc)), ra.sx(max(acc)), ra.sy(i), ticks=[ra.sx(v) for v in acc],
                       color="output", dot=ra.sx(100 * ROWS[b][0][1]))
        fig.text(ra.right + 80, ra.sy(i) + 5, P(MEANS[b]), "value", anchor="end", color="output", snap=False)
fig.legend(la.x, 468, [dict(color="output", label="five seeds, one tick each", mark="band"),
                            dict(color="output", label="seed 0, the documented run", mark="circle")],
           direction="row")
fig.caption("Every coefficient is spread over its five seeds; only 0.99 is below no momentum in accuracy every time.")
fig.write()
