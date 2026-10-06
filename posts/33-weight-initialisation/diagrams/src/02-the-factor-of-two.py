"""Post 33, section 4: ReLU sets the negative half of a zero-centred pre-activation to zero and keeps half of its
mean square, but less than half of its variance.

Run from anywhere:  python posts/33-weight-initialisation/diagrams/src/02-the-factor-of-two.py
Writes posts/33-weight-initialisation/diagrams/02-the-factor-of-two.svg.
Sources: the curves are the standard normal density, the distribution section 4 states for z (zero-centred,
symmetric; the ratios do not depend on its standard deviation). The closed forms are those of section 4 (1/2 for
the mean square, 1/2 - 1/(2 pi) for the variance), recomputed here by integrating the density; the measured values
come from snippets/variance_factors.py, run here (about a second), and each printed line drawn is asserted to be
in the section 4 listing of index.md.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, sub, span, MINUS  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
OUT = subprocess.run([sys.executable, str(POST / "snippets" / "variance_factors.py")], cwd=ROOT, capture_output=True,
                     text=True, check=True).stdout

LINES = [l for l in OUT.splitlines() if l.startswith("ReLU: ") and "mean of a" not in l]
assert len(LINES) == 3, LINES
for l in LINES:
    assert l in INDEX, l                                     # the section 4 listing
ZEROS = re.search(r"fraction of zeros (\d\.\d{4})", OUT).group(1)
MSQ = re.search(r"mean square of a / mean square of z = (\d\.\d{4})", OUT).group(1)
VAR = re.search(r"variance of a / variance of z\s+= (\d\.\d{4})", OUT).group(1)
assert (ZEROS, MSQ, VAR) == ("0.4995", "0.5003", "0.3408")


def pdf(z):
    return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


# the closed forms, by integrating the standard normal density (midpoint rule on -10..10)
N, LIM = 200000, 10.0
dz = 2 * LIM / N
zs = [-LIM + (i + 0.5) * dz for i in range(N)]
e_z2 = sum(z * z * pdf(z) for z in zs) * dz
e_a = sum(z * pdf(z) for z in zs if z > 0) * dz
e_a2 = sum(z * z * pdf(z) for z in zs if z > 0) * dz
p0 = sum(pdf(z) for z in zs if z <= 0) * dz
assert abs(p0 - 0.5) < 1e-6 and abs(e_a2 / e_z2 - 0.5) < 1e-6
var_ratio = (e_a2 - e_a ** 2) / e_z2
assert abs(var_ratio - (0.5 - 1 / (2 * math.pi))) < 1e-6 and f"{var_ratio:.4f}" == "0.3408"
assert "the positive half carries half of the total of $z_k^2$" in INDEX

LO, HI = -3.2, 3.2
XT = [-3, -2, -1, 0, 1, 2, 3]
YT = [0, 0.1, 0.2, 0.3, 0.4]
CURVE = [(LO + (HI - LO) * i / 160, pdf(LO + (HI - LO) * i / 160)) for i in range(161)]
NEG = [(LO, 0)] + [p for p in CURVE if p[0] <= 0] + [(0, pdf(0)), (0, 0)]
POS = [(0, 0), (0, pdf(0))] + [p for p in CURVE if p[0] >= 0] + [(HI, 0)]
fmt_x = lambda v: f"{MINUS}{-v:g}" if v < 0 else f"{v:g}"  # noqa: E731
fmt_y = lambda v: f"{v:.1f}"  # noqa: E731

fig = Figure(
    "02-the-factor-of-two", "ReLU keeps half of the mean square, not half of the variance",
    "Left, the density of a zero-centred normal pre-activation z, its negative half shaded red and its positive half "
    "blue. An arrow labelled ReLU leads to the right chart, the output a = max(0, z): the blue positive half is kept "
    "as it was, and the negative half becomes a spike at a = 0 that holds half of the draws. A table gives the "
    "closed form and the value measured on 1,000,000 draws: fraction of a at 0, 1/2 and 0.4995; mean square of a "
    "over mean square of z, 1/2 and 0.5003; variance of a over variance of z, 1/2 minus 1/(2 pi) = 0.3408 and "
    "0.3408.",
    subtitle=rich("A zero-centred normal pre-activation ", var("z"), " and the ReLU output ", var("a"), " = max(0, ",
                  var("z"), ")."),
    height=720, data_w=True)

LEFT, RIGHT = Box(40, 128, 384, 296), Box(536, 128, 384, 296)
fig.text(LEFT.x + 56, 120, rich("Pre-activation ", var("z")), "head")
fig.text(RIGHT.x + 56, 120, rich("ReLU output ", var("a")), "head")
la = fig.scatter(LEFT, [], x=(LO, HI, XT), y=(0, 0.45, YT), fmt_x=fmt_x, fmt_y=fmt_y, x_label=var("z"),
                 y_label="density", regions=[(NEG, "negative"), (POS, "positive")], label_w=8)
ra = fig.scatter(RIGHT, [], x=(LO, HI, XT), y=(0, 0.45, YT), fmt_x=fmt_x, fmt_y=fmt_y, x_label=var("a"),
                 y_label="density", regions=[(POS, "positive")], label_w=8)
la.polyline(CURVE, color="ink", width=1.5)
ra.polyline([p for p in CURVE if p[0] >= 0], color="ink", width=1.5)
ra.segment(LO, 0, 0, 0, color="ink", width=1.5)
la.text(-0.9, 0.04, "set to 0", "label", anchor="middle", color="negative", dy=0)
la.text(0.9, 0.04, "kept", "label", anchor="middle", color="positive", dy=0)
ra.text(0.9, 0.04, "kept", "label", anchor="middle", color="positive", dy=0)

# the point mass at a = 0: a spike, drawn as an arrow, for the half of the draws that ReLU sets to zero
x0 = ra.sx(0)
fig.arrow((x0, ra.sy(0)), (x0, ra.sy(0.44)), color="negative", width=2.5)
ra.text(0, 0.42, "half of the draws", "label", anchor="end", color="negative", dx=-12, dy=5)
ra.text(0, 0.42, "at exactly 0", "label", anchor="end", color="negative", dx=-12, dy=25)

# ReLU between the two charts
yA = (la.y + la.bottom) / 2
fig.arrow((la.right + 16, yA), (ra.x - 16, yA), label="ReLU")

# the table: closed form against the measurement
TX, TY = 96, 496
fig.text(TX, TY - 16, "Closed form and measurement on 1,000,000 draws", "head")
rows = [["quantity", "closed form", "measured"],
        ["fraction of the outputs at 0", "1/2", ZEROS],
        [rich("mean square of ", var("a"), " / mean square of ", var("z")), "1/2", span(MSQ, color="positive")],
        [rich("variance of ", var("a"), " / variance of ", var("z")), rich("1/2 ", MINUS, " 1/(2π) = 0.3408"), VAR]]
fig.table(TX, TY, rows, [376, 248, 144], col_align=["start", "end", "end"], highlight_row={2: "positive"})
fig.caption(rich("The next layer scales the mean square, so ReLU's factor is 1/2; the 2 in Var(", var("W"),
                 ") = 2/", sub("n", "in"), " undoes it."))
fig.write()
