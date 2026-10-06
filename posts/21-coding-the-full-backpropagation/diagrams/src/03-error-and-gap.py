"""Post 21, sections 9 and 11: each checked array as a point, its largest relative error against its largest absolute gap.

Run from anywhere:  python posts/21-coding-the-full-backpropagation/diagrams/src/03-error-and-gap.py
Writes posts/21-coding-the-full-backpropagation/diagrams/03-error-and-gap.svg.
snippets/gradient_check.py and snippets/what_can_go_wrong.py are run here as subprocesses (each under a second):
every point is a row of the check tables they print, read back from that printout, and each row the post shows
is asserted to be the line in index.md. Four checks, four arrays each: the passing check of section 9 (seed 0,
parameters redrawn at scale 1); the check on the script's 0.01 initialisation; the same without the ten samples
near a ReLU corner; and the backward pass without the division by the number of samples. The values are plotted
as printed (two significant digits). The horizontal line is the largest absolute gap the correct code shows at
scale 1 over seeds 0 to 49, from the two seed summaries of gradient_check.py.
"""
import math
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, sup, num, var, TIMES  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SN = POST / "snippets"


def run(name):
    return subprocess.run([sys.executable, str(SN / name)], capture_output=True, text=True, check=True,
                          cwd=str(SN)).stdout.splitlines()


GC, WR = run("gradient_check.py"), run("what_can_go_wrong.py")
BODY = (POST / "index.md").read_text(encoding="utf-8")
HEAD = "gradient         shape    largest |entry|   largest relative error   largest absolute gap"
NAMES = ["dense2.dweights", "dense2.dbiases", "dense1.dweights", "dense1.dbiases"]


def rows(lines, start, shown=4):
    """The four rows after a header at or after line start: name -> (relative error, gap, verdict)."""
    h = lines.index(HEAD, start)
    out = {}
    for k, line in enumerate(lines[h + 1:h + 5]):
        name, _, _, _, error, gap, verdict = line.split()
        if k < shown:
            assert line in BODY, line
        out[name] = (float(error), float(gap), verdict)
    assert list(out) == NAMES
    return out, h + 5


SCALE1, _ = rows(GC, 0)
INIT, i = rows(WR, 0)
LEFT_OUT, i = rows(WR, i)
NO_DIV, _ = rows(WR, i, shown=1)                    # the post shows the first of the four rows

assert all(v[2] == "pass" for v in SCALE1.values()) and all(v[2] == "FAIL" for v in INIT.values())
assert all(v[2] == "FAIL" for v in NO_DIV.values()) and all(v[0] == 1.0 for v in NO_DIV.values())
assert "relative error of dense2.dweights to four decimals: 0.9967; 1 - 1/N = 0.9967" in WR
assert (INIT["dense1.dweights"][1], INIT["dense1.dbiases"][1]) == (6.8e-8, 1.1e-5)
assert (LEFT_OUT["dense1.dweights"][1], LEFT_OUT["dense1.dbiases"][1]) == (9.6e-12, 1.2e-11)

# the largest absolute gap of the correct code at scale 1: seeds 0 to 9 (table) and 10 to 49 (summary line)
t = GC.index("seed   closed gates   smallest |Z1|   largest relative error   in                largest absolute gap")
GAPS = [float(l.split()[-1]) for l in GC[t + 1:t + 11]]
summary = next(l for l in GC if l.startswith("largest relative error") and "on seed" in l)
GAP_REST = float(summary.split()[-1])
GAP_MAX = max(max(GAPS), GAP_REST)
assert (max(GAPS), GAP_REST, GAP_MAX) == (6.0e-11, 5.6e-11, 6.0e-11)
assert "no absolute gap exceeds $6.0 \\times 10^{-11}$" in BODY and "exceeds $5.6 \\times 10^{-11}$" in BODY
PASS = 1e-7


def p10(e):
    return sup("10", num(e), italic=False)


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return rich(m, f" {TIMES} ", p10(int(e)))


def say(v):
    m, e = f"{v:.1e}".split("e")
    e = int(e)
    return m if e == 0 else f"{m} times 10 to the {'minus ' if e < 0 else ''}{abs(e)}"


def pts(d):
    return [(d[n][0], d[n][1]) for n in NAMES]


def listing(d):
    return "; ".join(f"{n} {say(d[n][0])} and {say(d[n][1])}" for n in NAMES)


fig = Figure(
    "03-error-and-gap", "A failed line is read together with its gap",
    "A scatter plot on two logarithmic axes: across, the largest relative error of a gradient array, from 10 to the "
    "minus 12 to 10; up, its largest absolute gap, from 10 to the minus 14 to 10 to the 4. Points are given as "
    "relative error and gap. Blue circles, the passing check of section 9 at scale 1: "
    f"{listing(SCALE1)}. Orange triangles, the check on the script's 0.01 weights: {listing(INIT)}. Hollow orange "
    f"squares, the same without the ten samples near a ReLU corner: {listing(LEFT_OUT)}. Red diamonds, a backward pass "
    f"without the division by N: all four at a relative error of 1.0, with gaps {', '.join(say(NO_DIV[n][1]) for n in NAMES)}. "
    "A dotted vertical line marks the pass mark, 10 to the minus 7, with the passing side shaded; a dotted "
    "horizontal line marks 6.0 times 10 to the minus 11, the largest gap of the correct code at scale 1 on seeds 0 "
    "to 49, labelled above: the two slopes disagree, and below: rounding. Dashed arrows take the two dense1 points of the 0.01 check down to the rounding level once the ten "
    "samples are left out, labelled without the 10 samples.",
    subtitle="Each point is one gradient array of one check, as the check tables of sections 9 and 11 print it.",
    height=720, data_w=True)

XR = (1e-12, 10, [1e-12, 1e-9, 1e-6, 1e-3, 1])
YR = (1e-14, 1e4, [10.0 ** e for e in range(-14, 5, 3)])
fmt = lambda v: "1" if v == 1 else ("10" if v == 10 else p10(round(math.log10(v))))  # noqa: E731
ax = fig.scatter(
    Box(40, 104, 880, 552),
    [dict(points=pts(SCALE1), color="positive", shape="circle", size=10, label="correct, scale 1"),
     dict(points=pts(INIT), color="weight", shape="triangle", size=10, label="correct, 0.01 weights"),
     dict(points=pts(LEFT_OUT), color="weight", shape="square", size=10, hollow=True,
          label="the same, 10 samples left out"),
     dict(points=pts(NO_DIV), color="error", shape="diamond", size=10, label=rich("no division by ", var("N")))],
    x=XR, y=YR, x_label="largest relative error, log scale", y_label="largest absolute gap, log scale",
    fmt_x=fmt, fmt_y=fmt, x_log=True, y_log=True, label_w=24,
    regions=[([(1e-12, 1e-14), (PASS, 1e-14), (PASS, 1e4), (1e-12, 1e4)], "output")])

with fig.data():
    ax.segment(PASS, 1e-14, PASS, 1e4, color="ink-muted", width=1, dash="ref")
    ax.text(PASS, 1e4, rich("pass below ", p10(-7)), "note", anchor="end", dx=-8, dy=20)
    ax.segment(1e-12, GAP_MAX, 10, GAP_MAX, color="ink-muted", width=1, dash="ref")
    ax.text(1e-12, GAP_MAX, rich("largest gap at scale 1, 50 seeds: ", sci(GAP_MAX)), "note", dx=8, dy=-8)
    ax.text(10, GAP_MAX, "above: the two slopes disagree", "note", anchor="end", dx=-8, dy=-8)
    ax.text(10, GAP_MAX, "below: rounding", "note", anchor="end", dx=-8, dy=20)
    # the two dense1 points of the 0.01 check, and where they go without the ten samples
    for n in ("dense1.dweights", "dense1.dbiases"):
        p0 = ax.to_px(*INIT[n][:2])
        p1 = ax.to_px(*LEFT_OUT[n][:2])
        fig.arrow((p0[0] - 6, p0[1] + 8), (p1[0] + 6, p1[1] - 8), width=1, dash="proj")
        ax.text(*INIT[n][:2], n, "code", dx=12, dy=5)
    ax.text(*INIT["dense2.dbiases"][:2], "dense2", "code", dx=12, dy=20)
    ax.text(1e-6, 10 ** -6.8, "without the 10 samples", "note")
    ax.text(1.0, NO_DIV["dense1.dweights"][1], "every gradient 300 times too large", "note", anchor="end",
            dx=-16, dy=5)

fig.caption("Only the missing division is a bug; the dense1 points of the 0.01 check are ReLU corners.")
fig.write()
