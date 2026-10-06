"""Post 04 section 5.1: the spread of each layer's output through six 64-neuron layers, for three weight scales.

Run from anywhere:  python posts/04-dense-layer-class-and-spiral-data/diagrams/src/04-init-scale.py
Writes posts/04-dense-layer-class-and-spiral-data/diagrams/04-init-scale.svg.
Every number is one that snippets/init_scale.py prints: the script is run (runpy), its table is plotted, and
every label is asserted to appear, as printed, in the script's output.
"""
import contextlib
import io
import math
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, sup, num, TIMES  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "init_scale.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
TABLE, SCALES, N = s["table"], s["SCALES"], s["N_LAYERS"]
assert SCALES == [0.01, 1.0, 0.125] and N == 6 and s["WIDTH"] == 64

PRINTED = {sc: [f"{v:.3e}" for v in TABLE[sc]] for sc in SCALES}       # the table's own format
for i in range(N):
    row = f"{i + 1:<8}" + "   ".join(f"{TABLE[sc][i]:<13.3e}" for sc in SCALES).rstrip()
    assert row in OUT, row
FACTOR = {sc: f"{(TABLE[sc][-1] / TABLE[sc][0]) ** (1 / (N - 1)):.3f}" for sc in SCALES}
RULE = {sc: f"{sc * math.sqrt(64):.2f}" for sc in SCALES}
for sc in SCALES:
    assert f"  scale {sc}: x {FACTOR[sc]}   (scale * sqrt(64) = {RULE[sc]})" in OUT
assert FACTOR == {0.01: "0.082", 1.0: "8.243", 0.125: "1.030"}


def sci(p):
    """A printed value such as 2.151e+04 as 2.151 x 10 to the 4, with a true superscript."""
    m, e = p.split("e")
    return rich(f"{m} {TIMES} ", sup("10", num(int(e)), italic=False))


def say(p):
    m, e = p.split("e")
    return f"{m} times 10 to the {int(e)}"


SER = [
    dict(sc=1.0, color="red", label="scale 1.0"),
    dict(sc=0.125, color="green", label="scale 0.125"),
    dict(sc=0.01, color="blue", label="scale 0.01"),
]
for d in SER:
    d["xs"] = list(range(1, N + 1))
    d["ys"] = [math.log10(float(p)) for p in PRINTED[d["sc"]]]
    d["label"] = rich(d["label"] + ", ", sci(PRINTED[d["sc"]][-1]))

fig = Figure(
    "04-init-scale", "With 0.01 the spread fades; with 1.0 it explodes",
    "A line chart of the standard deviation of each layer's output, layers 1 to 6 across, on a log scale from "
    "10 to the minus 9 to 10 to the 5, for six stacked 64-neuron Layer_Dense layers on the spiral data with "
    "weights scale times randn. Scale 1.0 climbs from " + say(PRINTED[1.0][0]) + " to " + say(PRINTED[1.0][-1])
    + ", times " + FACTOR[1.0] + " per layer. Scale 0.125 stays level, from " + say(PRINTED[0.125][0]) + " to "
    + say(PRINTED[0.125][-1]) + ", times " + FACTOR[0.125] + " per layer. Scale 0.01 falls from "
    + say(PRINTED[0.01][0]) + " to " + say(PRINTED[0.01][-1]) + ", times " + FACTOR[0.01] + " per layer.",
    subtitle="Six 64-neuron layers on the spiral data, nothing between them; the same draws, only the scale differs.")

YT = [-8, -6, -4, -2, 0, 2, 4]
X_LO, X_HI, Y_LO, Y_HI = 0.6, 6.4, -9, 5
box = Box(40, 104, 880, 372)
plot = fig.line_chart(box, SER, x=(X_LO, X_HI, list(range(1, N + 1))), y=(Y_LO, Y_HI, YT), x_label="layer",
                      fmt_y=lambda t: "1" if t == 0 else sup("10", num(t), italic=False),
                      y_label="standard deviation of the layer's output, log scale", label_w=216)

sx = lambda v: plot.x + plot.w * (v - X_LO) / (X_HI - X_LO)       # noqa: E731
sy = lambda v: plot.bottom - plot.h * (v - Y_LO) / (Y_HI - Y_LO)  # noqa: E731
with fig.data():
    for d, dy in [(SER[0], -28), (SER[1], -16), (SER[2], 32)]:
        mid = (d["ys"][2] + d["ys"][3]) / 2
        fig.text(sx(3.5), sy(mid) + dy, rich(TIMES, " ", FACTOR[d["sc"]], " per layer"), "label",
                 color=d["color"], anchor="middle", snap=False)
fig.caption(rich("Per layer, about scale ", TIMES, " √64: ", RULE[1.0], " for 1.0, ", RULE[0.125],
                 " for 0.125, ", RULE[0.01], " for 0.01."))
fig.write()
