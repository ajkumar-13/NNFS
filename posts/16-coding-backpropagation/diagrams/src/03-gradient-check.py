"""Post 16, section 7: the largest relative error of the gradient check on ten seeds, with and without the mask.

Run from anywhere:  python posts/16-coding-backpropagation/diagrams/src/03-gradient-check.py
Writes posts/16-coding-backpropagation/diagrams/03-gradient-check.svg.
snippets/gradient_check.py is run here (runpy; it takes about a second): every point is a row of the seed table it
prints, read back from that printout, and each row is asserted to be the line the post shows in section 7. The
values are plotted as printed (two significant digits), which is the precision the post states them in.
"""
import contextlib
import io
import math
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, sup, num, span, TIMES  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIPPET = POST / "snippets" / "gradient_check.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue().splitlines()
BODY = (POST / "index.md").read_text(encoding="utf-8")

head = OUT.index("seed   without the mask   as written   at an entry of     analytic value   absolute gap")
ROWS = []
for line in OUT[head + 1:head + 11]:
    assert line in BODY, line                                  # the table of section 7, verbatim
    seed, bad, good, where, value, gap = line.split()
    ROWS.append((int(seed), float(bad), float(good), where, float(value), float(gap)))
assert [r[0] for r in ROWS] == list(range(10))
BAD = [r[1] for r in ROWS]
GOOD = [r[2] for r in ROWS]
assert all(1.0 < b < 2.0 for b in BAD) and (min(BAD), max(BAD)) == (1.1, 1.9)         # "between 1.1 and 1.9"
FAILS = [r for r in ROWS if r[2] >= 1e-7]
assert [r[0] for r in FAILS] == [2] and FAILS[0][2] == 4.8e-6                         # nine pass, seed 2 does not
S2 = FAILS[0]
assert (S2[3], S2[4], S2[5]) == ("dense1.dinputs", -2.58e-6, 1.2e-11)
assert ROWS[0][2] == 1.2e-10                                                          # the TL;DR's seed 0 value


def p10(e):
    return sup("10", num(e), italic=False)


def sci(v, digits=1):
    m, e = f"{v:.{digits}e}".split("e")
    return rich(num(float(m)) if float(m) < 0 else m, f" {TIMES} ", p10(int(e)))


def say(v, digits=1):
    m, e = f"{v:.{digits}e}".split("e")
    return f"{m.replace('-', 'minus ')} times 10 to the minus {-int(e)}"


fig = Figure(
    "03-gradient-check", "Without the mask every seed fails; with it nine of ten pass",
    "A chart of the largest relative error over the five gradients of the Dense, ReLU, Dense chain, on a log scale "
    "from 10 to the minus 12 to 10, for seeds 0 to 9. Red triangles, the ReLU backward without the mask: "
    f"{', '.join(f'{b}' for b in BAD)}, every one between 1.1 and 1.9. Blue circles, the classes as written: "
    f"{', '.join(say(g) for g in GOOD)}. A dotted line marks the pass threshold 10 to the minus 7, with the region "
    f"below it shaded. Nine seeds pass; seed 2 reaches {say(S2[2])}, at an entry of dense1.dinputs whose analytic "
    f"value is {say(S2[4], 2)}, with an absolute gap of only {say(S2[5])}.",
    subtitle="The check of section 7 on seeds 0 to 9: the largest relative error over the five gradients.",
    data_w=True)

ax = fig.scatter(Box(40, 104, 880, 372),
                 [dict(points=list(zip(range(10), BAD)), color="error", shape="triangle", size=10,
                       label="ReLU backward without the mask"),
                  dict(points=list(zip(range(10), GOOD)), color="positive", shape="circle", size=10,
                       label="the classes as written")],
                 x=(-0.5, 9.5, list(range(10))), y=(1e-12, 10, [1e-12, 1e-9, 1e-6, 1e-3, 1]),
                 x_label="seed", y_label="largest relative error, log scale",
                 fmt_y=lambda t: "1" if t == 1 else p10(round(math.log10(t))),
                 y_log=True, vgrid=False, label_w=24,
                 regions=[([(-0.5, 1e-12), (9.5, 1e-12), (9.5, 1e-7), (-0.5, 1e-7)], "output")])

with fig.data():
    ax.segment(-0.5, 1e-7, 9.5, 1e-7, color="ink-muted", width=1, dash="ref")
    ax.text(9.5, 1e-7, rich("pass below ", p10(-7)), "note", anchor="end", dx=-8, dy=-8)
    # seed 2, the one that does not pass, explained beside its point
    ax.text(2, S2[2], rich("seed 2: an entry of ", sci(S2[4], 2), ","), "note", dx=16, dy=-20)
    ax.text(2, S2[2], rich("off by only ", sci(S2[5])), "note", dx=16, dy=4)

fig.caption(rich("A relative error near 1 is a bug; ", sci(S2[2]), " on a near-zero gradient is rounding."))
fig.write()
