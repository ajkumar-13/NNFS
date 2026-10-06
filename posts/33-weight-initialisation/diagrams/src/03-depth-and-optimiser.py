"""Post 33, section 7: the loss after 501 epochs from the 0.01 scale and from He, by number of hidden layers, under
plain gradient descent and under Adam, seeds 0 to 4.

Run from anywhere:  python posts/33-weight-initialisation/diagrams/src/03-depth-and-optimiser.py
Writes posts/33-weight-initialisation/diagrams/03-depth-and-optimiser.svg. Takes about two minutes.
Sources: snippets/depth_sgd.py (about 30 s), depth_small.py and depth_he.py (about 40 s each) are run here as they
are. Each prints one line per depth with, per seed, the loss at epoch 500 and the first epoch below 1.0, and the
number of runs never below 1.0; every value drawn comes from those lines. The two tables of section 7 and the epoch
counts the text quotes are recomputed from them and asserted against index.md.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, span, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")


def run(name):
    return subprocess.run([sys.executable, str(POST / "snippets" / f"{name}.py")], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def parse(out):
    """{init: {hidden: dict(loss=[...], first=[...], stuck=k)}} from depth_sweep's lines."""
    res, init = {}, None
    for line in out.splitlines():
        m = re.match(r"init = '(\w+)';", line)
        if m:
            init = m.group(1)
            res[init] = {}
            continue
        m = re.match(r"hidden layers (\d+): (.*?)   never below 1\.0: (\d) of 5", line)
        if m:
            cells = re.findall(r"(\d\.\d{4}) \((\d+|None)\)", m.group(2))
            assert len(cells) == 5, line
            res[init][int(m.group(1))] = dict(loss=[float(a) for a, _ in cells],
                                              first=[None if b == "None" else int(b) for _, b in cells],
                                              stuck=int(m.group(3)))
    return res


SGD = parse(run("depth_sgd"))
ADAM = {**parse(run("depth_small")), **parse(run("depth_he"))}
assert sorted(SGD) == ["he", "small"] and all(sorted(SGD[k]) == [1, 2, 3] for k in SGD)
assert sorted(ADAM) == ["he", "small"] and all(sorted(ADAM[k]) == [2, 4, 5, 6] for k in ADAM)
for d in (SGD, ADAM):
    for k in d:
        for h, r in d[k].items():
            assert r["stuck"] == sum(f is None for f in r["first"]), (k, h)


def rng(v):
    return f"{min(v):.4f} to {max(v):.4f}"


# -- the plain gradient descent table of section 7
for h in (1, 2, 3):
    a, b = SGD["small"][h]["loss"], SGD["he"][h]["loss"]
    cell = "1.0986 on all five seeds" if len(set(a)) == 1 else rng(a)
    assert f"| {h} | {cell} | {rng(b)} |" in INDEX, h
assert SGD["small"][3]["loss"] == [1.0986] * 5
he2 = SGD["he"][2]["first"]
assert f"after {min(he2)} to {max(he2)} epochs on every seed" in INDEX
# -- the Adam table of section 7 (losses and the runs that never go below 1.0)
for h in (2, 4, 5, 6):
    a, b = ADAM["small"][h], ADAM["he"][h]
    assert re.search(rf"\| {h} \| .*? \| {a['stuck']} of 5 \| {rng(a['loss'])} \| {rng(b['loss'])} \|", INDEX), h
esc = sorted(f for f in ADAM["small"][6]["first"] if f is not None)
he6 = ADAM["he"][6]["first"]
assert f"need {esc[0]} and {esc[1]} epochs to pass a loss of 1.0, against {min(he6)} to {max(he6)} for He" in INDEX
assert max(f for h in ADAM["he"] for f in ADAM["he"][h]["first"]) == 37 and "below 1.0 within 37 epochs" in INDEX
LN3 = math.log(3)
assert f"{LN3:.4f}" == "1.0986"

# -- layout
LO, HI, TICKS = 0, 1.15, [0, 0.25, 0.5, 0.75, 1.0]
PX0, PX1 = 192, 712
sx = lambda v: PX0 + (PX1 - PX0) * (v - LO) / (HI - LO)  # noqa: E731
C1, C2 = 800, 920                                   # right ends of the two value columns
KEYS = (("small", "0.01"), ("he", "He"))
PANELS = [("Plain gradient descent, learning rate 1.0", SGD, (1, 2, 3), 168),
          ("Adam, learning rate 0.02", ADAM, (2, 4, 5, 6), 408)]
GROUP, PITCH = 56, 24


def epochs(firsts):
    got = [f for f in firsts if f is not None]
    if not got:
        return "none"
    return f"{min(got)}" if len(got) == 1 or min(got) == max(got) else f"{min(got)} to {max(got)}"


def desc_panel(d, depths):
    parts = []
    for h in depths:
        for k, name in KEYS:
            r = d[k][h]
            parts.append(f"{h} hidden, {name}: " + ", ".join(f"{v:.4f}" for v in r["loss"])
                         + f", {5 - r['stuck']} of 5 below 1.0, first epoch below 1.0 {epochs(r['first'])}")
    return "; ".join(parts)


fig = Figure(
    "03-depth-and-optimiser", "Adam carries the 0.01 scale deeper than plain descent",
    "Two panels of bands, one row per number of hidden layers and initial scale, 0.01 in grey and He in blue, on an "
    "axis of the loss at epoch 500 from 0 to 1.15 with a dashed line at ln 3 = 1.0986; each band spans five seeds "
    "with one tick per seed, and two columns give the runs that went below a loss of 1.0 and the first epoch "
    "below 1.0. Plain gradient descent at learning rate 1.0: " + desc_panel(SGD, (1, 2, 3)) + ". Adam at learning "
    "rate 0.02: " + desc_panel(ADAM, (2, 4, 5, 6)) + ".",
    subtitle="Loss at epoch 500 on the spiral, 64 neurons per hidden layer with ReLU; seeds 0 to 4 for each row.",
    height=720, data_w=True)

fig.text(C1, 128, "runs", "note", anchor="end")
fig.text(C1, 152, "below 1.0", "note", anchor="end")
fig.text(C2, 128, "first epoch", "note", anchor="end")
fig.text(C2, 152, "below 1.0", "note", anchor="end")
for heading, d, depths, top in PANELS:
    fig.text(40, top - 40, heading, "head")
    fig.text(40, top - 16, "hidden layers", "note")
    fig.text(144, top - 16, "init", "note")
    fig.text(PX0, top - 16, "loss at epoch 500", "note")
    y_end = top + GROUP * (len(depths) - 1) + PITCH + 16
    with fig.data():
        for t in TICKS:
            fig.edge((sx(t), top - 12), (sx(t), y_end), color="grid", width=0.75)
        fig.edge((PX0, y_end), (PX1, y_end), color="ink-muted", width=1)
        for t in TICKS:
            fig.text(sx(t), y_end + 20, f"{t:.2f}", "tick", anchor="middle", snap=False)
        fig.edge((sx(LN3), top - 12), (sx(LN3), y_end), color="ink-muted", width=1, dash="ref")
        fig.text(sx(LN3) - 6, top - 16, "ln 3", "note", anchor="end", snap=False)
        for g, h in enumerate(depths):
            fig.text(40, top + GROUP * g + 5, str(h), "label", snap=False)
            for i, (k, name) in enumerate(KEYS):
                y = top + GROUP * g + PITCH * i
                r = d[k][h]
                a, b = sx(min(r["loss"])), sx(max(r["loss"]))
                if b - a < 4:                       # five equal values: a band 4 wide, centred on them
                    a, b = (a + b) / 2 - 2, (a + b) / 2 + 2
                ticks = [sx(v) for v in r["loss"]]
                if k == "he":
                    fig.range_mark(a, b, y, ticks=ticks, color="blue", h=16)
                else:
                    fig.fill(Box(a, y - 8, b - a, 16), "neutral-soft", fit=False)
                    fig.outline(Box(a, y - 8, b - a, 16), "rule", width=1)
                    for t in ticks:
                        fig.edge((t, y - 5), (t, y + 5), color="ink-muted", width=1)
                color = "blue" if k == "he" else None
                fig.text(144, y + 5, name, "label", color=color, snap=False)
                fig.text(C1, y + 5, f"{5 - r['stuck']} of 5", "label", anchor="end", color=color, snap=False)
                fig.text(C2, y + 5, epochs(r["first"]), "label", anchor="end", color=color, snap=False)

# -- the key, on the second heading's line, right of the heading
KX, KY = 432, PANELS[1][3] - 40
with fig.data():
    fig.fill(Box(KX, KY - 13, 24, 16), "neutral-soft", fit=False)
    fig.outline(Box(KX, KY - 13, 24, 16), "rule", width=1)
    fig.edge((KX + 12, KY - 10), (KX + 12, KY), color="ink-muted", width=1)
    fig.text(KX + 32, KY, "0.01", "label", snap=False)
    x2 = KX + 32 + math.ceil(text_width("0.01", 14, weight=400)) + 24
    fig.range_mark(x2, x2 + 24, KY - 5, ticks=[x2 + 12], color="blue", h=16)
    fig.text(x2 + 32, KY, "He", "label", color="blue", snap=False)
    x3 = x2 + 32 + math.ceil(text_width("He", 14, weight=400)) + 24
    fig.text(x3, KY, "one tick per seed", "note", snap=False)
fig.caption("Under plain descent 0.01 has not left ln 3 at three hidden layers; under Adam it loses runs from five.")
fig.write()
