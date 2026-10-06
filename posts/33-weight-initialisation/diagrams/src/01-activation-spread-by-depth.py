"""Post 33, section 5: the spread of the activations through ten 64-neuron ReLU layers at three weight scales.

Run from anywhere:  python posts/33-weight-initialisation/diagrams/src/01-activation-spread-by-depth.py
Writes posts/33-weight-initialisation/diagrams/01-activation-spread-by-depth.svg.
Source: snippets/ten_layers.py is run here (about ten seconds). Its section 2 rows (the standard deviation of each
layer's activations on seed 0) give the chart, one point per layer; its section 3 rows (the factor per layer on the
mean square, mean over seeds 0 to 99) give the table. Every printed row drawn is asserted to be in the listings of
section 5 of index.md, and the predicted factors are recomputed from the formula n_in * Var(W) / 2 of section 4.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, span, sub, var, pow10  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
OUT = subprocess.run([sys.executable, str(POST / "snippets" / "ten_layers.py")], cwd=ROOT, capture_output=True,
                     text=True, check=True).stdout

# -- section 2 of the output: one row per layer, seed 0
INITS = ("small", "xavier", "he")
STD = {k: [] for k in INITS}
rows = re.findall(r"^(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(0\.\d{3})$", OUT, re.M)
assert [int(r[0]) for r in rows] == list(range(1, 11)), rows
for r in rows:
    line = next(l for l in OUT.splitlines() if l.startswith(r[0] + " ") and r[1] in l)
    assert line in INDEX, line                                   # the section 5 listing
    for k, v in zip(INITS, r[1:4]):
        STD[k].append(float(v))
assert (STD["small"][0], STD["small"][9]) == (3.565e-03, 1.337e-14)
assert (STD["xavier"][0], STD["xavier"][9]) == (6.205e-02, 1.734e-03)
assert min(STD["he"]) == 2.254e-01 and max(STD["he"]) == 3.565e-01
assert "with He it stays between 0.22 and 0.36" in INDEX
assert f"{STD['xavier'][0] / STD['xavier'][9]:.0f}" == "36" and "with Glorot by a factor of 36" in INDEX
assert math.floor(math.log10(STD["small"][0] / STD["small"][9])) == 11 and "eleven orders of magnitude" in INDEX

# -- section 3 of the output: the factor per layer, layers 2 to 10, mean over seeds 0 to 99
FACT = {}
for k in INITS:
    line = re.search(rf"^{k}\s+.*$", OUT, re.M).group(0)
    assert line in INDEX, line
    v = line.split()
    FACT[k] = dict(pred=v[3], fwd=f"{v[4]} to {v[6]}", bwd=f"{v[8]} to {v[10]}")
# predicted: a dense layer with 64 inputs followed by ReLU, n_in * Var(W) / 2, at each scheme's scale
SCALE = {"small": 0.01, "xavier": math.sqrt(2 / (64 + 64)), "he": math.sqrt(2 / 64)}
PRED = {k: 64 * SCALE[k] ** 2 / 2 for k in INITS}
assert [f"{PRED[k]:.4g}" for k in INITS] == [FACT[k]["pred"] for k in INITS] == ["0.0032", "0.5", "1"]
assert [f"{SCALE[k]:.3f}" for k in INITS] == ["0.010", "0.125", "0.177"]

NAME = {"small": "0.01", "xavier": "Glorot", "he": "He"}
COLOR = {"small": "ink-muted", "xavier": "ink", "he": "blue"}
SHAPE = {"small": "circle", "xavier": "square", "he": "diamond"}
LAYERS = list(range(1, 11))


def words(v):
    m, e = f"{v:.3e}".split("e")
    return f"{m} times 10 to the minus {-int(e)}" if int(e) < 0 else m


def listing(k):
    return ", ".join(words(v) for v in STD[k])


fig = Figure(
    "01-activation-spread-by-depth", "At 0.01 the signal vanishes; He keeps it through ten layers",
    "A chart on a log axis of the standard deviation of each layer's activations, layers 1 to 10, for ten "
    "64-neuron layers with ReLU on the spiral, seed 0, under three weight scales on the same draws. Scale 0.01: "
    f"{listing('small')}. Glorot: {listing('xavier')}. He: {listing('he')}. A table below gives the factor per "
    "layer on the mean square for layers 2 to 10, mean over seeds 0 to 99, with the weights' standard deviation: "
    "0.01, predicted 0.0032, measured 0.0031 to 0.0032 forward and 0.0032 to 0.0032 backward; Glorot, 0.125, "
    "predicted 0.5, measured 0.4846 to 0.5077 forward and 0.4922 to 0.5024 backward; He, 0.177, predicted 1, "
    "measured 0.9692 to 1.0153 forward and 0.9896 to 1.0091 backward.",
    subtitle="Ten 64-neuron layers, each followed by ReLU, on the spiral; one set of draws at three scales, seed 0.",
    height=720, data_w=True)

series = [dict(xs=LAYERS, ys=STD[k], color=COLOR[k], shape=SHAPE[k], size=10, label=NAME[k]) for k in INITS]
YT = [10.0 ** e for e in range(-15, 1, 3)]
ax = fig.line_chart(Box(40, 104, 880, 360), series, x=(1, 10, LAYERS), y=(1e-15, 1, YT), y_log=True,
                    fmt_y=pow10, label_w=96, x_label="layer",
                    y_label="standard deviation of the layer's activations, log scale")

# -- the table: the factor per layer on the mean square
TX, TY = 96, 512
fig.text(TX, TY - 16, "Factor per layer on the mean square, layers 2 to 10, mean over seeds 0 to 99", "head")
cells = [["scheme", "weight std", "predicted", "measured forward", "measured backward"]]
for k in INITS:
    c = COLOR[k] if k == "he" else None
    cells.append([span(NAME[k], color=c) if c else NAME[k], f"{SCALE[k]:.3f}" if k != "small" else "0.01",
                  FACT[k]["pred"], FACT[k]["fwd"], FACT[k]["bwd"]])
fig.table(TX, TY, cells, [136, 136, 120, 192, 184], col_align=["start", "end", "end", "end", "end"],
          highlight_row={3: "blue"})
fig.caption(rich("A dense layer and its ReLU multiply the mean square by ", sub("n", "in"), " · Var(", var("W"),
                 ") / 2; He sets it to 1."))
fig.write()
