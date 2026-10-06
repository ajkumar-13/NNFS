"""Post 29, the hero (sections 3, 5 and 8): 5-fold cross-validation of the winner on seed 0, then one test.

Run from anywhere:  python posts/29-validation-and-hyperparameter-tuning/diagrams/src/01-kfold-and-test-set.py
Writes posts/29-validation-and-hyperparameter-tuning/diagrams/01-kfold-and-test-set.svg.
Sources, all parsed from the listings in index.md, which the series lint (--run) keeps equal to the snippets' output:
the fold sizes from kfold.py (section 4), the five fold accuracies of learning rate 0.1 from search.py (section 5),
and the winner, its k-fold mean and the training and test accuracies of the retrained winner from grid.py
(section 8). search.py and grid.py train 25 and 20 networks; they are not rerun here.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var  # noqa: E402

POST = Path(__file__).resolve().parents[2]
MD = (POST / "index.md").read_text(encoding="utf-8")

# -- kfold.py: five validation folds of 60 out of 300
assert "n=300 k=5  validation sizes [60, 60, 60, 60, 60]  every sample validated once: True" in MD
N, K, FOLD = 300, 5, 60

# -- search.py, seed 0: the fold accuracies of learning rate 0.1 (width 64)
m = re.search(r"^lr=0\.1\s+mean_acc=(\d\.\d{3}) std=(\d\.\d{3})  folds \[([^\]]+)\]$", MD, re.M)
assert m, "search.py listing"
MEAN, STD = m.group(1), m.group(2)
FOLDS = [float(v) for v in m.group(3).split()]
assert len(FOLDS) == K and f"{sum(FOLDS) / K:.3f}" == MEAN == "0.773" and STD == "0.040"

# -- grid.py, seed 0: the winner, its k-fold mean, and the one test
assert "winner (learning rate, width): (0.1, 64)" in MD
assert f"k-fold mean of the winner {MEAN}" in MD
m = re.search(r"retrained on all 300 points: training accuracy (\d\.\d{3}), test accuracy (\d\.\d{3}) "
              r"\((\d+) of 300\)", MD)
assert m
TRAIN_ACC, TEST_ACC, TEST_RIGHT = m.group(1), m.group(2), int(m.group(3))
assert f"{TEST_RIGHT / N:.3f}" == TEST_ACC == "0.793" and TRAIN_ACC == "0.927"

F3 = lambda v: f"{v:.3f}"  # noqa: E731
PARTS = "ABCDE"
fig = Figure(
    "01-kfold-and-test-set", "Every point validates once; the test set is read once",
    "Five rows, folds 1 to 5, each a strip of the five parts A to E of the 300 shuffled spiral points of seed 0, "
    "60 points each. In fold i part i, in blue, is the validation part and the other four, 240 points, train a new "
    "network with learning rate 0.1 and 64 hidden neurons. The validation accuracies are "
    + ", ".join(F3(v) for v in FOLDS) + f", mean {MEAN}. A sixth row, the final model, trains on all 300 points "
    f"(training accuracy {TRAIN_ACC}) and is then scored once on the test set, 300 further points that no step "
    f"of the search used: {TEST_ACC}, {TEST_RIGHT} of 300.",
    subtitle="Seed 0: the winner of the search, learning rate 0.1 with 64 hidden neurons; 300 points in five parts of 60.",
    height=720, data_w=True)

X0, CW, CH = 136, 88, 40            # the strips: five cells of 88 x 40
XV = 696                            # right end of the accuracy column
XT, WT = 744, 176                   # the test-set column
Y0, STEP = 176, 56                  # first strip, row pitch
fig.text(40, 128, "5-fold cross-validation", "head")
fig.text(XV, 128, "validation accuracy", "note", anchor="end")
fig.text(XT, 128, "Test set", "head")
fig.text(X0 + 2.5 * CW, 160, "the 300 search points, five parts of 60", "note", anchor="middle")

for i in range(K):
    y = Y0 + i * STEP
    g = fig.strip(X0, y, K, cell_w=CW, cell_h=CH, values=list(PARTS), font=14,
                  fill=lambda j, i=i: "blue-soft" if j == i else "neutral-soft", strong={i: "blue"})
    g.outline(i, i, color="blue", width=1.5)
    fig.text(40, y + 25, f"fold {i + 1}", "label")
    fig.text(XV, y + 25, F3(FOLDS[i]), "value", anchor="end", color="output")

YM = Y0 + K * STEP + 8                                 # the mean, under the accuracy column
with fig.data():
    fig.edge((XV - 88, YM - 20), (XV, YM - 20), color="rule")
fig.text(XV, YM + 4, rich("mean ", MEAN), "value", anchor="end", color="output")
fig.text(XV - 104, YM + 4, rich("score of the candidate"), "note", anchor="end")

# the test set: sealed while the folds rotate
sealed = Box(XT, Y0, WT, (K - 1) * STEP + CH)
fig.outline(sealed, "ink-muted", width=1.5, dash="lead", radius=8)
for k, line in enumerate(("300 further points,", "drawn after the 300,", "unused by the search")):
    fig.text(sealed.cx, sealed.cy - 16 + 20 * k, line, "note", anchor="middle")

# the final model and the one test
YF = YM + 48
final = Box(X0, YF, K * CW, CH)
fig.fill(final, "neutral-soft", fit=False)
fig.outline(final, "ink-muted", width=1.5)
fig.text(final.cx, YF + 25, "all five parts, 300 points", "label", anchor="middle")
fig.text(40, YF + 25, "final", "label")
fig.text(X0, YF + CH + 28, rich("training accuracy ", TRAIN_ACC), "note")
test = Box(XT, YF, WT, CH)
fig.fill(test, "output-soft", fit=False)
fig.outline(test, "output", width=1.5)
fig.text(test.cx, YF + 25, rich(TEST_ACC, ", ", f"{TEST_RIGHT} of 300"), "value", anchor="middle",
         color="output")
fig.arrow((final.right + 8, YF + 20), (XT, YF + 20), label="scored once")
fig.text(XT + WT, YF + CH + 28, "after every choice is made", "note", anchor="end")

YL = 636                                              # legend baseline; marks drawn as the strip cells are
for x, color, line, label in ((40, "blue-soft", "blue", "validation part, 60 points"),
                              (336, "neutral-soft", "ink-muted", "training parts, 240 points")):
    fig.fill(Box(x, YL - 16, 32, 20), color, fit=False)
    fig.outline(Box(x, YL - 16, 32, 20), line, width=1.5)
    fig.text(x + 44, YL, label, "label")
fig.caption("Cross-validation takes the place of the validation set only; the test set stays outside the rotation.")
fig.write()
