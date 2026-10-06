"""Post 27, section 7: the six optimisers of Part VI over seeds 0 to 4, final training accuracy.

Run from anywhere:  python posts/27-adam-optimiser/diagrams/src/04-six-optimisers.py   (about 90 seconds)
Writes posts/27-adam-optimiser/diagrams/04-six-optimisers.svg.
The values are the cells of section 7's five-seed table in index.md, parsed here. The rows for RMSProp and Adam
are asserted against snippets/seeds_rmsprop.py and seeds_adam.py, run here side by side (40 to 95 seconds each).
The first four rows are the five-seed runs of posts 22 to 25 and are asserted against those posts' printed
outputs: gradient descent, momentum and AdaGrad against the per-seed table of post 25's section 7 (built from
post 22's, 24's and 25's seed scripts), decay against post 23's seed_spread.py listing. The best of each seed,
bold in the table, is recomputed and asserted.

Layout: one chart, a row per optimiser, each a band over the five seeds with a tick per seed and a dot for seed
0; seed 0, the mean and the seeds on which the optimiser is the best of the six printed at the right.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, span, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
POSTS = POST.parent
INDEX = (POST / "index.md").read_text(encoding="utf-8")
SEEDS = range(5)
ROWS = [  # table name, label, note, colour (None: neutral)
    ("SGD", "gradient descent", "post 22", None),
    ("SGD, decay", "learning-rate decay", "post 23", None),
    ("SGD, momentum", "momentum", "post 24", None),
    ("AdaGrad", "AdaGrad", "post 25", None),
    ("RMSProp", "RMSProp", "post 26", None),
    ("Adam", "Adam", "this post", "blue"),
]

# -- section 7's table (one decimal; the exact values come from the printed outputs below)
TAB, TMEAN, BOLD = {}, {}, {}
for name, *_ in ROWS:
    m = re.search(rf"^\| {re.escape(name)} \|((?: \**\d+\.\d\** \|){{5}}) (\d+\.\d) \|$", INDEX, re.M)
    assert m, name
    cells = [c.strip() for c in m.group(1).split("|")[:-1]]
    TAB[name] = [c.strip("*") for c in cells]
    BOLD[name] = [c.startswith("**") for c in cells]
    TMEAN[name] = m.group(2)
ACC = {}                                                                   # percent, from the printed fractions

# -- RMSProp and Adam: the post's own seed scripts
OWN = {"RMSProp": "seeds_rmsprop", "Adam": "seeds_adam"}
procs = {
    n: subprocess.Popen([sys.executable, str(POST / "snippets" / f"{s}.py")], cwd=str(ROOT), stdout=subprocess.PIPE,
                        text=True) for n, s in OWN.items()}
for n, s in OWN.items():
    out, _ = procs[n].communicate()
    assert procs[n].returncode == 0, n
    rows = re.findall(r"^\S+ \(post 2[67]\)\s+(\d)\s+\d\.\d{4}\s+\d\.\d{4}\s+\d\.\d{4}\s+(\d\.\d{4})\s", out, re.M)
    assert [int(r[0]) for r in rows] == list(SEEDS), (n, out)
    ACC[n] = [100 * float(r[1]) for r in rows]

# -- gradient descent, momentum, AdaGrad: post 25's per-seed table; decay: post 23's listing
P25 = (POSTS / "25-adagrad" / "index.md").read_text(encoding="utf-8")
for seed in SEEDS:
    m = re.search(rf"^\| {seed} \| \d\.\d{{4}}, (\d\.\d{{4}}), \d+ \| \d\.\d{{4}}, (\d\.\d{{4}}), \d+ \| "
                  rf"\d\.\d{{4}}, (\d\.\d{{4}}), \d+ \|$", P25, re.M)
    for name, v in zip(("SGD", "SGD, momentum", "AdaGrad"), m.groups()):
        ACC.setdefault(name, []).append(100 * float(v))
P23 = next(POSTS.glob("23-*")) / "index.md"
dec = dict(re.findall(r"^\s+(\d)  decay 1e-3  \d\.\d{4}  (\d\.\d{4})\s", P23.read_text(encoding="utf-8"), re.M))
assert sorted(dec) == [str(s) for s in SEEDS]
ACC["SGD, decay"] = [100 * float(dec[str(s)]) for s in SEEDS]
MEAN = {n: sum(a) / 5 for n, a in ACC.items()}
for n in TAB:                                                              # every cell and mean of the table
    assert [f"{v:.1f}" for v in ACC[n]] == TAB[n] and f"{MEAN[n]:.1f}" == TMEAN[n], n

# -- the best of each seed: bold in the table, and the readings of section 7
BEST = {s: max(ACC, key=lambda n: ACC[n][s]) for s in SEEDS}
assert all(BOLD[n][s] == (BEST[s] == n) for n in ACC for s in SEEDS)
assert BEST == {0: "Adam", 1: "RMSProp", 2: "RMSProp", 3: "AdaGrad", 4: "SGD, momentum"}
assert "The best figure of each seed is in bold, and four different optimisers hold one." in INDEX
assert "**Adam is the best of the six on seed 0 and on no other seed.**" in INDEX


def best_on(n):
    s = [k for k in SEEDS if BEST[k] == n]
    if not s:
        return "none"
    return f"seed {s[0]}" if len(s) == 1 else "seeds " + ", ".join(map(str, s[:-1])) + f" and {s[-1]}"


P = lambda v: f"{v:.1f}"                                                    # noqa: E731
desc_rows = [f"{label}: {', '.join(P(v) for v in ACC[n])}, mean {P(MEAN[n])}, best on {best_on(n)}"
             for n, label, *_ in ROWS]
fig = Figure(
    "04-six-optimisers", "Adam is the best of the six on seed 0 and on no other",
    "A chart of the final training accuracy on the spiral after 10,001 epochs for the six optimisers of Part VI, "
    "one row each, a band from the lowest to the highest of seeds 0 to 4 with a tick per seed and a dot for seed "
    "0, on an axis from 50 to 100 percent, with seed 0, the mean and the seeds on which the row is the best of the "
    "six at the right. Seeds 0 to 4: " + "; ".join(desc_rows) + ".",
    subtitle="Final training accuracy on the spiral after 10,001 epochs, seeds 0 to 4, each at its documented setting.",
    data_w=True)


def band(x0, x1, y, ticks, dot, color, h=24):
    """The band-with-ticks mark: range_mark in a hue, or the same form in neutrals (soft card fill, rule outline,
    muted ticks and dot) when color is None. Inside fig.data()."""
    if color:
        fig.range_mark(x0, x1, y, ticks=ticks, color=color, dot=dot, h=h)
        return
    b = Box(min(x0, x1), y - h / 2, abs(x1 - x0), h)
    fig.fill(b, "neutral-soft", fit=False)
    fig.outline(b, "rule")
    for t in ticks:
        fig.edge((t, y - h / 3), (t, y + h / 3), color="ink-muted")
    if dot is not None:
        fig.marker(dot, y, "circle", "ink-muted", size=10)


ax = fig.dot_plot(Box(40, 128, 640, 312), [("", []) for _ in ROWS], 50, 100, [50, 60, 70, 80, 90, 100],
                  label_w=200, pad_right=16, fmt=lambda v: num(v), axis_label="final accuracy, percent")
C0, CM, CB = 752, 816, 920                                                 # value columns (right edges)
for x, head in ((C0, "seed 0"), (CM, "mean"), (CB, "best on")):
    fig.text(x, 112, head, "note", anchor="end")
with fig.data():
    for i, (n, label, note, color) in enumerate(ROWS):
        a = ACC[n]
        y = ax.sy(i)
        fig.text(40, y + 5, rich(label, ", ", span(note, color="ink-muted")), "label", snap=False)
        band(ax.sx(min(a)), ax.sx(max(a)), y, [ax.sx(v) for v in a], ax.sx(a[0]), color)
        fig.text(C0, y + 5, P(a[0]), "label", anchor="end", snap=False)
        fig.text(CM, y + 5, P(MEAN[n]), "value", anchor="end", color=color, snap=False)
        fig.text(CB, y + 5, best_on(n), "label", anchor="end", snap=False)
    LY = 468
    band(240, 264, LY - 4, [252], None, None, h=16)
    fig.text(276, LY, "five seeds, one tick each", "label", snap=False)
    fig.marker(512, LY - 4, "circle", "ink-muted", size=10)
    fig.text(528, LY, "seed 0, the documented run", "label", snap=False)
fig.caption("Four different optimisers are the best of some seed; the order on seed 0 is a property of that seed.")
fig.write()
