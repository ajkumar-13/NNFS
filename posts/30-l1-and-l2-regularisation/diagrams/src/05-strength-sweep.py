"""Post 30, section 8: training and test accuracy over seeds 0 to 4 for five L2 strengths on the first layer.

Run from anywhere:  python posts/30-l1-and-l2-regularisation/diagrams/src/05-strength-sweep.py
Writes posts/30-l1-and-l2-regularisation/diagrams/05-strength-sweep.svg. Takes about a minute and a half.

The five scripts of section 8's table, snippets/seeds_none.py, l2_weak.py, seeds_l2.py, l2_strong.py and
l2_too_strong.py, are run as they are, side by side (_runs.py), and their printed rows give every value drawn: one
tick per seed, a band from the lowest to the highest, and a dot at the mean of the five. The printed mean test
accuracy is the value column. Every row of section 8's table is recomputed from the printed values and asserted
against index.md, and so are the statements of the text drawn here (the three middle means within 1.2 points,
5.0 to 6.1 points above no penalty; at 1e-2 test accuracy below the unpenalised run on four seeds).

Layout: post 24's form, two dot charts side by side, training accuracy left and test accuracy right; no penalty in
grey, the four L2 strengths in blue.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figkit import Figure, Box, rich, var, sup, num  # noqa: E402
from _runs import INDEX, runs, rng  # noqa: E402

SWEEP = [("seeds_none", "0", None), ("l2_weak", "$10^{-4}$", (1, -4)), ("seeds_l2", "$5 \\times 10^{-4}$", (5, -4)),
         ("l2_strong", "$10^{-3}$", (1, -3)), ("l2_too_strong", "$10^{-2}$", (1, -2))]
NAMES = [n for n, _, _ in SWEEP]
R = runs(NAMES)
TRAIN = {n: [r["train_pct"] for r in R[n]["rows"]] for n in NAMES}
TEST = {n: [r["test_pct"] for r in R[n]["rows"]] for n in NAMES}
MEAN = {n: R[n]["test_mean"] for n in NAMES}

for n, md, _ in SWEEP:
    rows = R[n]["rows"]
    gaps = [a - b for a, b in zip(TRAIN[n], TEST[n])]
    cells = (f"| {md} | `{n}.py` | {rng(TRAIN[n])} | {rng(TEST[n])} ({MEAN[n]}) | {rng(gaps)} | "
             f"{rng([float(r['sum_sq']) for r in rows], '{:,.2f}')} |")
    assert cells in INDEX, cells
    assert f"{sum(TEST[n]) / 5:.2f}" == MEAN[n], n                 # the dot is the printed mean
MID = [float(MEAN[n]) for n in NAMES[1:4]]
assert max(MID) - min(MID) <= 1.2 and "The three means lie within 1.2 points of each other" in INDEX
LIFT = [m - float(MEAN["seeds_none"]) for m in MID]
assert (f"{min(LIFT):.1f}", f"{max(LIFT):.1f}") == ("5.0", "6.1") and "by 5.0 to 6.1 points in the mean" in INDEX
below = sum(a < b for a, b in zip(TEST["l2_too_strong"], TEST["seeds_none"]))
assert below == 4 and "test accuracy is below that of the unpenalised run on four of the five seeds" in INDEX
assert rng(TRAIN["l2_too_strong"]) == "81.00 to 89.00" and "Training accuracy falls to 81.00 to 89.00 percent" in INDEX

lam = var("λ")


def strength(p):
    if p is None:
        return rich(lam, " = 0")
    c, e = p
    power = sup("10", num(e), italic=False)
    return rich(lam, " = ", power) if c == 1 else rich(lam, " = ", str(c), " × ", power)


def say(n, p):
    word = "no penalty" if p is None else (f"{p[0]} times 10 to the minus {-p[1]}" if p[0] != 1 else
                                           f"10 to the minus {-p[1]}")
    return (f"{word}: training {', '.join(f'{v:.2f}' for v in TRAIN[n])}; test "
            f"{', '.join(f'{v:.2f}' for v in TEST[n])}, mean {MEAN[n]}")


fig = Figure(
    "05-strength-sweep", "Three strengths tie in test accuracy; the largest underfits",
    "Two dot charts, one row per L2 strength on the first layer, each row a band from the lowest to the highest "
    "of seeds 0 to 4 with one tick per seed and a dot at the mean of the five. Left the training accuracy in "
    "percent on an axis from 80 to 100, right the test accuracy from 70 to 95, with the mean test accuracy "
    "printed at the right. In percent, " + "; ".join(say(n, p) for n, _, p in SWEEP) + ". No penalty in grey, "
    "the four strengths in blue.",
    subtitle="L2 on the weights and biases of the first layer, seeds 0 to 4 for each strength, 10,001 epochs.",
    data_w=True)

STYLE = {False: ("neutral-soft", "rule", "ink-muted"), True: ("blue-soft", "blue-line", "blue")}


def band(x0, x1, y, penalised, ticks, dot, h=24):
    """range_mark's form in either colour, the neutral too: a soft box outlined in the line form, ticks, a dot."""
    soft, line, base = STYLE[penalised]
    fig.fill(Box(x0, y - h / 2, x1 - x0, h), soft, fit=False)
    fig.outline(Box(x0, y - h / 2, x1 - x0, h), line)
    for t in ticks:
        fig.edge((t, y - h / 3), (t, y + h / 3), color=base, width=1)
    fig.marker(dot, y, "circle", base, size=10)


LABELS = [strength(p) for _, _, p in SWEEP]
H = 304
la = fig.dot_plot(Box(40, 128, 456, H), [(lab, []) for lab in LABELS], 80, 100, [80, 85, 90, 95, 100],
                  label_w=136, pad_right=16, fmt=lambda v: num(v))
ra = fig.dot_plot(Box(528, 128, 392, H), [("", []) for _ in SWEEP], 70, 95, [70, 75, 80, 85, 90, 95],
                  label_w=0, pad_right=80, fmt=lambda v: num(v))
fig.text(la.x, 112, "Training accuracy, percent", "head")
fig.text(ra.x, 112, "Test accuracy, percent", "head")
fig.text(ra.right + 80, 112, "mean", "note", anchor="end")
with fig.data():
    for i, (n, _, p) in enumerate(SWEEP):
        for ax, vals in ((la, TRAIN[n]), (ra, TEST[n])):
            band(ax.sx(min(vals)), ax.sx(max(vals)), ax.sy(i), p is not None, [ax.sx(v) for v in vals],
                 ax.sx(sum(vals) / 5))
        fig.text(ra.right + 80, ra.sy(i) + 5, MEAN[n], "value", anchor="end",
                 color="blue" if p is not None else "ink-muted", snap=False)
fig.legend(la.x, 468, [dict(color="blue", label="five seeds, one tick each", mark="band"),
                       dict(color="blue", label="mean of the five", mark="circle")], direction="row")
fig.caption("The spread between seeds is wider than the difference between the three middle strengths.")
fig.write()
