"""Post 31, section 2: the 256 thinned networks of a 2-8-3 network, against the full network.

Run from anywhere:  python posts/31-dropout/diagrams/src/02-thinned-networks.py
Writes posts/31-dropout/diagrams/02-thinned-networks.svg. Takes about a second.

snippets/ensemble.py is run as it is; its printout gives every number drawn: the largest gap between the mean
logits over the 256 masks and the full network's, the mean disagreement over the masks, and the table of
disagreement by the number of neurons a mask keeps (lowest, mean, highest). The whole printout is asserted
against the listing of section 2 in index.md, and the table against the summary line above it (the overall
range, and the mean over the masks recomputed from the per-row means and counts).

The network drawn on the left is one of the masks that keep 5 neurons; which one is not a measured choice, and
no number is attached to it.
"""
import os
import re
import subprocess
import sys
from math import comb
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIP = POST / "snippets"
INDEX = (POST / "index.md").read_text(encoding="utf-8")
OUT = subprocess.run([sys.executable, str(SNIP / "ensemble.py")], cwd=str(SNIP), capture_output=True, text=True,
                     check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).stdout
assert f"```text\n{OUT}```" in INDEX                     # the listing of section 2 is this printout, whole

N, P = 8, 0.5
assert f"masks: {2 ** N}, their probabilities sum to 1.000000; points: 300; p = {P}" in OUT
GAP = re.search(r"mean of the logits over the masks, against the full network: +largest gap (\S+)", OUT).group(1)
assert GAP == "8.4e-15"
m = re.search(r"another class on (\S+) to (\S+) percent of the points \(mean over the masks (\S+)\)", OUT)
LO_ALL, HI_ALL, MEAN_ALL = (float(v) for v in m.groups())
rows = []
for line in OUT.split("kept   masks   lowest    mean   highest\n")[1].splitlines():
    k, n, lo, mean, hi = line.split()
    rows.append(dict(k=int(k), n=int(n), lo=float(lo), mean=float(mean), hi=float(hi)))
assert [r["k"] for r in rows] == list(range(N + 1)) and all(r["n"] == comb(N, r["k"]) for r in rows)
assert min(r["lo"] for r in rows) == LO_ALL and max(r["hi"] for r in rows) == HI_ALL
assert abs(sum(r["n"] * r["mean"] for r in rows) / 2 ** N - MEAN_ALL) < 0.05     # p = 0.5: every mask 1/256
assert "the mean disagreement falls from 64.6 percent with one neuron kept to 15.4 with seven" in INDEX
assert "picks another class on 8.0 percent of the points" in INDEX
assert (rows[1]["mean"], rows[7]["mean"], rows[8]["mean"]) == (64.6, 15.4, 8.0)

KEPT = [1, 0, 1, 1, 0, 1, 1, 0]                          # drawn: one of the masks that keep 5
K = sum(KEPT)


def say(v):
    return f"{v:.1f}"


fig = Figure(
    "02-thinned-networks", "Each thinned network disagrees; their average does not",
    "Left: a network of 2 inputs, 8 ReLU neurons and 3 classes with one dropout mask that keeps 5 of the 8 "
    f"neurons, one of {comb(N, K)} such masks; the 3 dropped neurons are dashed and have no connections. Right: "
    "for each number of kept neurons from 0 to 8, a band from the lowest to the highest share of the 300 spiral "
    "points on which a thinned network picks another class than the full network, with a dot at the mean over "
    "its masks, and the number of masks and the mean printed beside it: "
    + "; ".join(f"{r['k']} kept, {r['n']} masks, {say(r['lo'])} to {say(r['hi'])}, mean {say(r['mean'])} percent"
                for r in rows)
    + f". A dotted line marks the mean over all 256 masks, {say(MEAN_ALL)} percent. The caption states that the "
    f"logits averaged over the 256 masks equal the full network's to within {GAP}.",
    subtitle=rich("A network of 2 inputs, 8 ReLU neurons and 3 classes, ", var("p"),
                  " = 0.5, every one of its 256 masks, 300 spiral points."),
    data_w=True)

# -- left: one thinned network
LX = [80, 192, 304]
HY = [164 + 36 * i for i in range(N)]                    # 164 to 416
IY = [272, 308]
OY = [254, 290, 326]
fig.text(40, 120, rich("One of the ", str(comb(N, K)), " masks that keep ", str(K)), "head")
with fig.data():
    for iy in IY:
        for h, hy in zip(KEPT, HY):
            if h:
                fig.edge((LX[0], iy), (LX[1], hy))
    for h, hy in zip(KEPT, HY):
        if h:
            for oy in OY:
                fig.edge((LX[1], hy), (LX[2], oy))
    for iy in IY:
        fig.node(LX[0], iy, r=12, color="input")
    for h, hy in zip(KEPT, HY):
        if h:
            fig.node(LX[1], hy, r=12)
        else:
            fig.outline(Box(LX[1] - 12, hy - 12, 24, 24), "ink-muted", width=1.5, dash="lead", radius=12)
    for oy in OY:
        fig.node(LX[2], oy, r=12, color="output")
for x, s in zip(LX, ["inputs", "ReLU", "classes"]):
    fig.text(x, 452, s, "note", anchor="middle")
fig.text(LX[1] + 24, HY[7] + 5, "dropped", "note")       # below the edges of the last kept neuron

# -- right: disagreement by the number of kept neurons
LABEL_W, PAD_R = 72, 136
ax = fig.dot_plot(Box(360, 104, 560, 372), [(str(r["k"]), []) for r in rows], 0, 100, [0, 25, 50, 75, 100],
                  label_w=LABEL_W, pad_right=PAD_R, axis_label="points on another class than the full network, percent",
                  ref_lines=[dict(x=MEAN_ALL, label=rich("all 256 masks, mean ", say(MEAN_ALL)))])
XM, XV = 848, 920                                        # right ends of the two value columns
with fig.data():
    for i, r in enumerate(rows):
        y = ax.sy(i)
        x0, x1 = ax.sx(r["lo"]), ax.sx(r["hi"])
        if x1 - x0 >= 2:
            fig.range_mark(x0, x1, y, ticks=[x0, x1], color="blue", dot=ax.sx(r["mean"]), h=20)
        else:                                            # one mask: the band is a point
            fig.marker(ax.sx(r["mean"]), y, "circle", "blue", size=10)
        fig.text(XM, y + 5, num(r["n"]), "label", anchor="end", snap=False)
        fig.text(XV, y + 5, say(r["mean"]), "value", anchor="end", color="blue", snap=False)
fig.text(360, 120, "neurons kept", "note")
fig.text(XM, 120, "masks", "note", anchor="end")
fig.text(XV, 120, "mean", "note", anchor="end")

fig.caption(rich("Averaged over all 256 masks, the logits equal the full network's to within 8.4 × ",
                 sup("10", num(-15), italic=False), "."))
fig.write()
