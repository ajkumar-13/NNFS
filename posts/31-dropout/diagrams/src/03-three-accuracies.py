"""Post 31, section 8: the three accuracies of every run, without dropout and at p = 0.1, 0.2 and 0.5, seeds 0 to 4.

Run from anywhere:  python posts/31-dropout/diagrams/src/03-three-accuracies.py
Writes posts/31-dropout/diagrams/03-three-accuracies.svg. Takes one and a half to two and a half minutes.

The four scripts of section 8's table, snippets/seeds_none.py, seeds_rate_10.py, seeds_rate_20.py and
seeds_rate_50.py (30 to 60 seconds each), are run here as they are, side by side, and their printed rows give
every value drawn: per seed the training accuracy through the mask (the mean over 100 masks), the training
accuracy with the mask off, the test accuracy, and the two printed differences. Nothing is added to the scripts and
no other seed is run; snippets/jobs/ is not used. Each row of section 8's table is recomputed from the rows and
asserted against index.md.

Layout work only: with FIG31_RUNS_DIR set to a folder holding <name>.txt, a saved stdout of the same script, that
text is parsed instead of a new run.
"""
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, num, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
NAMES = ("seeds_none", "seeds_rate_10", "seeds_rate_20", "seeds_rate_50")
ROW = re.compile(r"^ +(\d) +(\d\.\d{4}) \((\d\.\d{4}) to (\d\.\d{4})\) +(\d\.\d{4}) +(\d\.\d{4}) +(-?\d+\.\d\d) +"
                 r"(-?\d+\.\d\d) +(\d\.\d{4}) +(\d+)$", re.M)


def outputs():
    cache = os.environ.get("FIG31_RUNS_DIR")
    if cache:
        return {n: (Path(cache) / f"{n}.txt").read_text(encoding="utf-8") for n in NAMES}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    procs = {n: subprocess.Popen([sys.executable, str(POST / "snippets" / f"{n}.py")], cwd=str(ROOT),
                                 stdout=subprocess.PIPE, text=True, env=env) for n in NAMES}
    outs = {}
    for n, p in procs.items():
        outs[n], _ = p.communicate()
        assert p.returncode == 0, n
    return outs


def parse(out):
    rows = []
    for m in ROW.finditer(out):
        g = m.groups()
        r = dict(seed=int(g[0]), on=100 * float(g[1]), off_test=g[6], on_test=g[7])
        for k, v in (("off", g[4]), ("test", g[5])):          # 300 points: an accuracy is a count over 300
            count = round(300 * float(v))
            assert f"{count / 300:.4f}" == v, r
            r[k] = count / 3
        assert f"{r['off'] - r['test']:.2f}" == r["off_test"], r
        rows.append(r)
    assert [r["seed"] for r in rows] == [0, 1, 2, 3, 4], out
    return rows


OUT = outputs()
R = {n: parse(o) for n, o in OUT.items()}


def rng(vals):
    return f"{min(vals):.2f} to {max(vals):.2f}"


def tex(s):                                                 # the table writes a negative bound as $-2.93$
    return re.sub(r"^(-\d+\.\d\d)", r"$\1$", s)


LABEL = {"seeds_none": "none", "seeds_rate_10": "$p = 0.1$", "seeds_rate_20": "$p = 0.2$",
         "seeds_rate_50": "$p = 0.5$"}
for n in NAMES:
    rows = R[n]
    on_test = [r["on"] - r["test"] for r in rows]
    for r, d in zip(rows, on_test):                          # the printed difference uses the unrounded mean
        assert abs(d - float(r["on_test"])) < 0.01, (n, r)        # the mean is printed to 4 decimals
    cells = (f"| {LABEL[n]} | `{n}.py` | {rng([r['on'] for r in rows])} | {rng([r['off'] for r in rows])} | "
             f"{rng([r['test'] for r in rows])} | {rng([r['off'] - r['test'] for r in rows])} | "
             f"{tex(rng([float(r['on_test']) for r in rows]))} |")
    assert cells in INDEX, cells
ALL = [r for n in NAMES for r in R[n]]
assert all(r["off"] > r["test"] for r in ALL)               # mask off: training above test in every run
assert all(abs(r["on"] - r["off"]) < 0.005 for r in R["seeds_none"])    # no layer: mask on is mask off
assert all(r["on"] < r["off"] for n in NAMES[1:] for r in R[n])
ABOVE = {n: sum(r["test"] > r["on"] for r in R[n]) for n in NAMES}
assert [ABOVE[n] for n in NAMES] == [0, 1, 1, 4], ABOVE

GROUP = {"seeds_none": "none", "seeds_rate_10": rich(var("p"), " = 0.1"),
         "seeds_rate_20": rich(var("p"), " = 0.2"), "seeds_rate_50": rich(var("p"), " = 0.5")}
SAY = {"seeds_none": "No dropout layer", "seeds_rate_10": "p = 0.1", "seeds_rate_20": "p = 0.2",
       "seeds_rate_50": "p = 0.5"}


def listing(n):
    return "; ".join(f"seed {r['seed']} {r['on']:.2f}, {r['off']:.2f}, {r['test']:.2f}" for r in R[n])


fig = Figure(
    "03-three-accuracies", "Test beats training only when training runs through the mask",
    "Twenty rows, seeds 0 to 4 in four groups: no dropout layer and Layer_Dropout at p = 0.1, 0.2 and 0.5, on an "
    "accuracy axis from 50 to 100 percent. Each row has three marks: the training accuracy through the mask (mean "
    "of 100 masks, hollow green circle), the training accuracy with the mask off (solid green circle) and the test "
    "accuracy (blue diamond), in percent. "
    + " ".join(f"{SAY[n]}, mask on, mask off, test: {listing(n)}." for n in NAMES)
    + " Without a dropout layer the mask-on and mask-off marks coincide. Two columns print mask off minus test and mask on minus test in points; the second is negative, test above "
    "the masked training figure, on 0, 1, 1 and 4 of the five seeds. Mask off minus test is positive in all 20 runs.",
    subtitle="Seeds 0 to 4, 10,001 epochs, all measured forward-only on the final weights; differences in points.",
    height=720, data_w=True)

LO, HI, TICKS = 50, 100, [50, 60, 70, 80, 90, 100]
PX0, PX1 = 192, 688
C1, C2 = 808, 920                                            # right ends of the two value columns
sx = lambda v: PX0 + (PX1 - PX0) * (v - LO) / (HI - LO)  # noqa: E731
TOP, PITCH, GAP = 160, 20, 16
cy = lambda g, s: TOP + (5 * PITCH + GAP) * g + PITCH * s  # noqa: E731
AXIS = cy(3, 4) + 20
assert AXIS + 44 <= 656, AXIS

HY = 128
fig.text(40, HY, "dropout", "note")
fig.text(136, HY, "seed", "note")
fig.text(C1, HY, "mask off − test", "note", anchor="end")
fig.text(C2, HY, "mask on − test", "note", anchor="end")
with fig.data():
    for t in TICKS:
        fig.edge((sx(t), TOP - 16), (sx(t), AXIS), color="grid", width=0.75)
    fig.edge((PX0, AXIS), (PX1, AXIS), color="ink-muted", width=1)
    for t in TICKS:
        fig.text(sx(t), AXIS + 20, f"{t}", "tick", anchor="middle", snap=False)
    for g, n in enumerate(NAMES):
        fig.text(40, cy(g, 0) + 5, GROUP[n], "label", snap=False)
        if n == "seeds_none":                                # no layer: the hollow mark sits on the solid one
            fig.text(40, cy(g, 1) + 5, "on = off", "note", snap=False)
        for r in R[n]:
            y = cy(g, r["seed"])
            vals = (r["on"], r["off"], r["test"])
            fig.edge((sx(min(vals)), y), (sx(max(vals)), y), color="rule", width=1)
            fig.marker(sx(r["test"]), y, "diamond", "blue", size=10, hollow=True)
            fig.marker(sx(r["off"]), y, "circle", "output", size=10)
            fig.marker(sx(r["on"]), y, "circle", "output", size=10, hollow=True)
            fig.text(136, y + 5, str(r["seed"]), "tick", snap=False)
            fig.text(C1, y + 5, num(r["off_test"]), "label", anchor="end", snap=False)
            neg = float(r["on_test"]) < 0
            fig.text(C2, y + 5, num(r["on_test"]), "value" if neg else "label", anchor="end",
                     color="blue" if neg else None, snap=False)
fig.text((PX0 + PX1) / 2, AXIS + 44, "accuracy, percent", "note", anchor="middle")

# -- the key on the headings' line, over the plot
x = PX0
with fig.data():
    for shape, color, hollow, text in (("circle", "output", True, "training, mask on"),
                                       ("circle", "output", False, "mask off"),
                                       ("diamond", "blue", True, "test")):
        fig.marker(x + 8, HY - 4, shape, color, size=10, hollow=hollow)
        fig.text(x + 24, HY, text, "label", snap=False)
        x += 24 + math.ceil(text_width(text, 14, weight=400)) + 32
fig.caption("With the mask off, training accuracy is above test accuracy in all 20 runs.")
fig.write()
