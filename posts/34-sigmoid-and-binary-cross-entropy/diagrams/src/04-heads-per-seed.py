"""Post 34, section 7: test points correct and test data loss of the sigmoid head and the softmax head, seeds 0 to 9.

Run from anywhere:  python posts/34-sigmoid-and-binary-cross-entropy/diagrams/src/04-heads-per-seed.py
Writes posts/34-sigmoid-and-binary-cross-entropy/diagrams/04-heads-per-seed.svg. Takes about 40 seconds.

snippets/seeds.py is run here as it is (a subprocess from the series root, about 35 seconds) and its printed rows
give every value drawn: the test points correct of 200 and the test data loss, printed to 4 decimals, of each head
on each seed. The rows of the section 7 table in index.md are recomputed from them and asserted, and so are the
seed-by-seed statements of the text (equal counts on nine seeds, one point more for the sigmoid head on seed 3, a
lower test loss for each head on five seeds).

Layout: one row per seed. Left, test points correct on an axis from 196 to 200; right, the test data loss on a log
axis from 10^-4 to 10^-1, the two heads joined by a line. The sigmoid head (this post's) in the blue accent as
solid diamonds, the softmax head in ink as hollow circles, drawn larger so that a tie shows both marks.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, sup, num, pow10, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
OUT = subprocess.run([sys.executable, str(POST / "snippets" / "seeds.py")], cwd=str(ROOT),
                     capture_output=True, text=True, check=True).stdout

RE_ROW = re.compile(r"^ +(\d) +(\d+) +(\d+)/(\d+) +(\d+)/(\d+) +(\d\.\d{4}) +(\d\.\d{4})$", re.M)
blocks = OUT.split("head: ")[1:]
assert [b.split("\n", 1)[0] for b in blocks] == ["sigmoid", "softmax"]
R = {}
for b in blocks:
    name = b.split("\n", 1)[0]
    rows = [m.groups() for m in RE_ROW.finditer(b.split("test points correct")[0])]
    assert [int(r[0]) for r in rows] == list(range(10)), name
    R[name] = dict(params=int(rows[0][1]), train=[int(r[2]) for r in rows], test=[int(r[4]) for r in rows],
                   test_n=int(rows[0][5]), train_loss=[r[6] for r in rows], test_loss=[r[7] for r in rows])
    assert all(int(r[3]) == 800 and int(r[5]) == 200 for r in rows)
S, M = R["sigmoid"], R["softmax"]

# -- the section 7 table, row by row
for name, label in (("sigmoid", "sigmoid, 1 output"), ("softmax", "softmax, 2 outputs")):
    r = R[name]
    lo, hi = min(r["test_loss"], key=float), max(r["test_loss"], key=float)
    cells = (f"| {label} | {r['params']} | 800 on all ten seeds | {min(r['test'])} to {max(r['test'])} | "
             f"{sum(t == 200 for t in r['test'])} of 10 | {lo} to {hi} |")
    assert all(t == 800 for t in r["train"]) and cells in INDEX, cells
diff = [a - b for a, b in zip(S["test"], M["test"])]
assert diff == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
assert "the test counts are equal on nine seeds and the sigmoid head has one point more on seed 3" in INDEX
assert "differences: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]" in OUT
assert "lower test data loss: sigmoid on 5 seeds, softmax on 5" in OUT
assert "the test loss is lower with the sigmoid head on five seeds and with the softmax head on five" in INDEX
SL, ML = [float(v) for v in S["test_loss"]], [float(v) for v in M["test_loss"]]
assert sum(a < b for a, b in zip(SL, ML)) == 5 and sum(a > b for a, b in zip(SL, ML)) == 5   # printed values agree


def listing(r):
    return "; ".join(f"seed {k} {t}, {v}" for k, (t, v) in enumerate(zip(r["test"], r["test_loss"])))


fig = Figure(
    "04-heads-per-seed", "The two heads tie on nine seeds of ten",
    "One row per seed, 0 to 9, for the two-moons network trained 2,000 epochs with each head. Left, test points "
    "correct of 200 on an axis from 196 to 200; right, the test data loss on a log axis from 10 to the minus 4 to "
    "10 to the minus 1, the two heads joined by a line. Sigmoid head, 1 output, blue diamonds, test points correct "
    f"and test data loss: {listing(S)}. Softmax head, 2 outputs, hollow ink circles: {listing(M)}. The counts are "
    "equal on every seed but seed 3, where the sigmoid head has 200 and the softmax head 199; the test loss is "
    "lower with the sigmoid head on five seeds and with the softmax head on five.",
    subtitle=rich("Two moons, 800 training and 200 test points, 2,000 epochs of Adam. One row per seed."),
    height=720, data_w=True)

TOP, PITCH = 200, 40
cy = lambda k: TOP + PITCH * k  # noqa: E731
AXIS = cy(9) + 32
# left: test points correct
LX0, LX1, LLO, LHI = 104, 424, 196, 200
lx = lambda v: LX0 + (LX1 - LX0) * (v - LLO) / (LHI - LLO)  # noqa: E731
# right: test data loss, log
RX0, RX1, RLO, RHI = 536, 896, -4, -1
rx = lambda v: RX0 + (RX1 - RX0) * (math.log10(v) - RLO) / (RHI - RLO)  # noqa: E731
assert all(10 ** RLO < v < 10 ** RHI for v in SL + ML)

HY = 136
fig.text(40, HY, "Test points correct, of 200", "head")
fig.text(RX0, HY, "Test data loss", "head")
fig.text(40, TOP - 32, "seed", "note")

with fig.data():
    for t in range(LLO, LHI + 1):
        fig.edge((lx(t), TOP - 16), (lx(t), AXIS), color="grid", width=0.75)
        fig.text(lx(t), AXIS + 20, str(t), "tick", anchor="middle", snap=False)
    fig.edge((LX0, AXIS), (LX1, AXIS), color="ink-muted", width=1)
    for e in range(RLO, RHI + 1):
        fig.edge((rx(10 ** e), TOP - 16), (rx(10 ** e), AXIS), color="grid", width=0.75)
        fig.text(rx(10 ** e), AXIS + 24, pow10(10 ** e), "tick", anchor="middle", snap=False)
    for e in range(RLO, RHI):
        for m in range(2, 10):
            x = rx(m * 10 ** e)
            fig.edge((x, TOP - 16), (x, AXIS), color="grid", width=0.75, dash="ref")
    fig.edge((RX0, AXIS), (RX1, AXIS), color="ink-muted", width=1)
    for k in range(10):
        y = cy(k)
        fig.text(40, y + 5, str(k), "label", snap=False)
        # left: the softmax circle drawn larger and first, the sigmoid diamond over it
        fig.marker(lx(M["test"][k]), y, "circle", "ink", size=16, hollow=True)
        fig.marker(lx(S["test"][k]), y, "diamond", "blue", size=10)
        # right: the two losses joined
        a, b = rx(SL[k]), rx(ML[k])
        fig.edge((a, y), (b, y), color="rule", width=1.5)
        fig.marker(b, y, "circle", "ink", size=12, hollow=True)
        fig.marker(a, y, "diamond", "blue", size=10)
fig.text((LX0 + LX1) / 2, AXIS + 48, "points correct", "note", anchor="middle")
fig.text((RX0 + RX1) / 2, AXIS + 52, "data loss, log scale", "note", anchor="middle")

# the key, on the seed line over the plots
KY = TOP - 32
x = LX0
with fig.data():
    for shape, color, size, hollow, text in (("diamond", "blue", 10, False, "sigmoid, 1 output"),
                                             ("circle", "ink", 12, True, "softmax, 2 outputs")):
        fig.marker(x + 8, KY - 5, shape, color, size=size, hollow=hollow)
        fig.text(x + 24, KY, text, "label", snap=False)
        x += 24 + math.ceil(text_width(text, 14, weight=400)) + 40
fig.caption("Seed 3 is the one difference in test counts: 200 for the sigmoid head, 199 for the softmax head.")
fig.write()
