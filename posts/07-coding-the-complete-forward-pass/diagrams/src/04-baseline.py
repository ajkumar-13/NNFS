"""Post 07, section 5: what the untrained network predicts for each true class, and the 0.34 that comes of it.

Run from anywhere:  python posts/07-coding-the-complete-forward-pass/diagrams/src/04-baseline.py
Writes posts/07-coding-the-complete-forward-pass/diagrams/04-baseline.svg.
snippets/uniform_baseline.py is run here (runpy). The counts are tallied from its `predictions` and `y` arrays and
asserted against its printout (block 4 and 5) and against section 5 and 8 of the text (3 of the 8 ties are class 0,
99 class-2 points are predicted as class 2, 102 correct).
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, span  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "uniform_baseline.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
pred, y = s["predictions"], s["y"]

COUNT = np.zeros((3, 3), dtype=int)                 # COUNT[true, predicted]
for t, p in zip(y, pred):
    COUNT[t, p] += 1
CORRECT = int(np.trace(COUNT))
assert "predictions per class: [  8   0 292]" in OUT and list(COUNT.sum(axis=0)) == [8, 0, 292]
assert list(COUNT.sum(axis=1)) == [100, 100, 100]
assert f"correct: {CORRECT} of 300, accuracy {CORRECT / 300:.2f}" in OUT and CORRECT == 102
assert "true classes: [0 0 0 1 1 1 1 2]" in OUT and COUNT[0, 0] == 3 and COUNT[2, 2] == 99
assert "rows whose three logits are all exactly 0: 8" in OUT
for k in range(3):
    assert f"always answering class {k}: 100 of 300 correct" in OUT
ROWS = [(f"true class {t}", [int(COUNT[t, p]) for p in range(3)]) for t in range(3)]
assert ROWS == [("true class 0", [3, 0, 97]), ("true class 1", [4, 0, 96]), ("true class 2", [1, 0, 99])]
COLORS = ["blue", "orange", "green"]                # the class colours of post 04's spiral figure

fig = Figure(
    "04-baseline", "The untrained network answers class 2 for 292 of 300 points",
    "Stacked horizontal bars, one per true class of the 300 spiral points, 100 points each, split by the class the "
    "untrained network predicts: true class 0, 3 predicted as class 0 and 97 as class 2; true class 1, 4 as class 0 "
    "and 96 as class 2; true class 2, 1 as class 0 and 99 as class 2. No point is predicted as "
    "class 1. After each bar its split is written out, 3 + 97, 4 + 96 and 1 + 99, in the colours of the two "
    "predicted classes. A column on the right counts the correct answers, 3, 0 and 99, 102 of 300 in all, an accuracy of 0.34; "
    "answering one class for every point scores 100 of 300. A note says that the 8 points predicted as class 0 are "
    "the rows whose logits are all exactly 0, a three-way tie that np.argmax gives to the first class.",
    subtitle="snippets/uniform_baseline.py: the 300 predictions of section 5, counted by true class.")

chart = Box(40, 136, 712, 232)
ax = fig.stacked_bars(chart, ROWS, COLORS, 0, 100, [0, 25, 50, 75, 100], label_w=128, value_w=104, totals=False,
                      bar_h=40, axis_label="points of the true class, by predicted class")
# the split of each bar, written after its end: predicted class 0 + predicted class 2 (no point is predicted as 1)
SPLITS = [(int(COUNT[t, 0]), int(COUNT[t, 2])) for t in range(3)]
assert SPLITS == [(3, 97), (4, 96), (1, 99)] and all(COUNT[t, 1] == 0 for t in range(3))
assert all(a + b == 100 for a, b in SPLITS)
CX = 864
fig.text(CX, 128, "correct", "head", anchor="end")
with fig.data():
    for i, (a, b) in enumerate(SPLITS):
        fig.text(ax.right + 12, ax.sy(i) + 5, rich(span(str(a), color=COLORS[0], bold=True), " + ",
                                                   span(str(b), color=COLORS[2], bold=True)), "note", snap=False)
    for i in range(3):
        fig.text(CX, ax.sy(i) + 5, str(int(COUNT[i, i])), "label", anchor="end", snap=False)
fig.text(CX, 412, rich(span(f"{CORRECT} of 300", bold=True)), "label", anchor="end")
fig.text(CX, 432, f"accuracy {CORRECT / 300:.2f}", "note", anchor="end")
fig.legend(168, 412, [dict(color=c, label=f"predicted class {k}" + (" (none)" if COUNT[:, k].sum() == 0 else "")) for k, c in enumerate(COLORS)], direction="row")
fig.note(Box(168, 432, 0, 0), "The 8 points predicted as class 0 have three logits of exactly 0: np.argmax gives the tie "
         "to class 0.")
fig.caption("One fixed answer for every point also scores 100 of 300.")
fig.write()
