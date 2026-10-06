"""Post 28, sections 4 and 6: training and test accuracy over seeds 0 to 4 with 8, 64 and 128 hidden neurons.

Run from anywhere:  python posts/28-generalization-and-testing/diagrams/src/05-three-widths.py   (about 1 to 2 minutes)
Writes posts/28-generalization-and-testing/diagrams/05-three-widths.svg.
snippets/narrower.py (about 25 s) and snippets/wider.py (about 70 s) are run here as they are, side by side, and
their printed per-seed rows give the 8- and 128-neuron values; each script's printed range line is asserted to be
in index.md. The 64-neuron rows are parsed from the section 3 listing of index.md (seed_spread.py, about 45 s; the
series lint keeps the listing equal to the snippet's output). The comparisons section 6 states (8 neurons: smaller
gap and 28.33 to 41.67 points lower test accuracy on every seed; 128 neurons: higher test accuracy on every seed,
by 0.33 to 19.33 points, gap wider on three seeds) are asserted on the counts.

Layout: three dot plots side by side, one per width, a row per seed, the training and the test accuracy joined.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
ROW = re.compile(r"^ +(\d)  +\d\.\d{4}  +(\d\.\d{4})  +\d\.\d{4}  +(\d\.\d{4})  +(\d+\.\d{2})$", re.M)
RANGE = re.compile(r"^train accuracy .* points$", re.M)


def parse(text):
    """Seed -> (training count, test count) out of 300, with the printed gap checked."""
    out = {}
    for m in ROW.finditer(text):
        tr, te = round(300 * float(m.group(2))), round(300 * float(m.group(3)))
        assert f"{tr / 300:.4f}" == m.group(2) and f"{te / 300:.4f}" == m.group(3)
        assert f"{(tr - te) / 3:.2f}" == m.group(4), m.group(0)
        out[int(m.group(1))] = (tr, te)
    assert sorted(out) == [0, 1, 2, 3, 4], text
    return out


procs = {n: subprocess.Popen([sys.executable, str(POST / "snippets" / f"{n}.py")], cwd=str(ROOT),
                             stdout=subprocess.PIPE, text=True) for n in ("narrower", "wider")}
OUT = {}
for n, p in procs.items():
    OUT[n], _ = p.communicate()
    assert p.returncode == 0, n
assert "Layer_Dense(2, 8) and Layer_Dense(8, 3), 51 parameters" in OUT["narrower"]
assert "Layer_Dense(2, 128) and Layer_Dense(128, 3), 771 parameters" in OUT["wider"]
for n in OUT:
    line = RANGE.search(OUT[n]).group(0)
    assert line in INDEX, line

HEAD = "seed  train loss  train acc  test loss  test acc  gap in points"
listing = INDEX[INDEX.index(HEAD):].split("```")[0]
RUNS = {8: parse(OUT["narrower"]), 64: parse(listing), 128: parse(OUT["wider"])}
assert all(2 * w + w + 3 * w + 3 == p for w, p in ((8, 51), (64, 387), (128, 771)))

gap = lambda w, s: (RUNS[w][s][0] - RUNS[w][s][1]) / 3       # noqa: E731
test = lambda w, s: RUNS[w][s][1] / 3                         # noqa: E731
SEEDS = range(5)
assert all(gap(8, s) < gap(64, s) for s in SEEDS)
d8 = [test(64, s) - test(8, s) for s in SEEDS]
assert (f"{min(d8):.2f}", f"{max(d8):.2f}") == ("28.33", "41.67")
d128 = [test(128, s) - test(64, s) for s in SEEDS]
assert min(d128) > 0 and (f"{min(d128):.2f}", f"{max(d128):.2f}") == ("0.33", "19.33")
assert sum(gap(128, s) > gap(64, s) for s in SEEDS) == 3
assert (f"{test(128, 2):.2f}", f"{test(64, 2):.2f}") == ("86.67", "67.33")
for frag in ("a test accuracy that is 28.33 to 41.67 points lower",
             "by 0.33 to 19.33 points (86.67 against 67.33 percent on seed 2)",
             "its gap is wider in three runs and narrower in two"):
    assert frag in INDEX, frag


def rng(w, f):
    vals = [f(w, s) for s in SEEDS]
    return f"{min(vals):.2f} to {max(vals):.2f}"


train = lambda w, s: RUNS[w][s][0] / 3                        # noqa: E731
GREEN, BLUE = "output", "blue"
PARAMS = {8: 51, 64: 387, 128: 771}

fig = Figure(
    "05-three-widths", "8 neurons shrink the gap; 128 raise the test accuracy",
    "Three dot plots side by side for 8, 64 and 128 hidden neurons (51, 387 and 771 parameters), one row per "
    "seed, 0 to 4, on an accuracy axis from 30 to 100 percent; a green circle marks the training accuracy and a "
    "hollow blue diamond the test accuracy, joined by a line. "
    + " ".join(f"{w} neurons: training " + ", ".join(f"{train(w, s):.2f}" for s in SEEDS) + "; test "
               + ", ".join(f"{test(w, s):.2f}" for s in SEEDS) + f"; gap {rng(w, gap)} points." for w in (8, 64, 128))
    + " The 8-neuron network has the smallest gaps and the lowest test accuracy; the 128-neuron network has the "
    "highest test accuracy on every seed.",
    subtitle="Seeds 0 to 4, 10,001 epochs each; only the width of the hidden layer changes.", data_w=True)

PANELS = ((40, 312, 64, 8), (384, 248, 0, 64), (664, 248, 0, 128))
for x, w, label_w, width in PANELS:
    fig.text(x + label_w, 120, f"{width} neurons, {PARAMS[width]} parameters", "head")
    rows = [(f"seed {s}" if label_w else "", []) for s in SEEDS]
    ax = fig.dot_plot(Box(x, 136, w, 256), rows, 30, 100, [40, 60, 80, 100], fmt=lambda v: num(v), unit="%",
                      label_w=label_w, pad_right=16)
    with fig.data():                           # the dumbbells by hand, the hollow test mark last (as figure 01)
        for s in SEEDS:
            a, b = train(width, s), test(width, s)
            fig.edge((ax.sx(b), ax.sy(s)), (ax.sx(a), ax.sy(s)), color="rule", width=1.5)
            fig.marker(ax.sx(a), ax.sy(s), "circle", GREEN, size=10)
            fig.marker(ax.sx(b), ax.sy(s), "diamond", BLUE, size=13, hollow=True)
    fig.text(x + label_w, 424, f"test {rng(width, test)}%", "label", color=BLUE + "-ink")
    fig.text(x + label_w, 444, f"gap {rng(width, gap)} points", "note")

fig.legend(240, 472, [dict(color=GREEN, label="training points", mark="circle"),
                      dict(color=BLUE, label="300 new points, forward-only", mark="diamond", hollow=True)],
           direction="row", gap=40)
fig.caption("Each width is tested on its own 300 points, whose standard error is about 2.2 points.")
fig.write()
