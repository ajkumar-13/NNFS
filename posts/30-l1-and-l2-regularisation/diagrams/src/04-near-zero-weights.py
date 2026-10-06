"""Post 30, section 7: the weights of the first layer below 0.001 after training, split by dead and live neurons.

Run from anywhere:  python posts/30-l1-and-l2-regularisation/diagrams/src/04-near-zero-weights.py
Writes posts/30-l1-and-l2-regularisation/diagrams/04-near-zero-weights.svg. Takes about a minute.

snippets/seeds_none.py, seeds_l2.py and seeds_l1.py are run as they are, side by side (_runs.py). Each prints, per
seed, the count of the 128 weights of dense1 with |w| < 0.001 in its row, and the same count split into weights of
dead neurons and of live neurons in its summary line; those splits are drawn. The ranges of section 7's second
table are recomputed from the printed values and asserted against index.md, as is "under L2 every near-zero weight
belongs to a dead neuron", and no run has a weight exactly 0.

Layout: two stacked bar charts, L2 left and L1 right, one bar per seed: weights of dead neurons in grey, of live
neurons in the weight colour.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figkit import Figure, rich, var, sup, num  # noqa: E402
from _runs import INDEX, runs, rng  # noqa: E402

NAMES = ("seeds_none", "seeds_l2", "seeds_l1")
R = runs(NAMES)


def irange(vals):
    return f"{min(vals)}" if min(vals) == max(vals) else f"{min(vals)} to {max(vals)}"


ROWS = {n: R[n]["rows"] for n in NAMES}
for n, label in (("seeds_none", "none"), ("seeds_l2", "L2, $5 \\times 10^{-4}$"),
                 ("seeds_l1", "L1, $5 \\times 10^{-4}$")):
    rows = ROWS[n]
    cells = (f"| {label} | {rng([float(r['sum_abs']) for r in rows])} | "
             f"{rng([float(r['sum_sq']) for r in rows], '{:,.2f}')} | {irange([r['dead'] for r in rows])} | "
             f"{irange([int(r['near']) for r in rows])} | {irange([r['near_live'] for r in rows])} | "
             f"{irange([int(r['exact']) for r in rows])} |")
    assert cells in INDEX, cells
    assert all(int(r["exact"]) == 0 for r in rows)
assert all(int(r["near"]) == 0 for r in ROWS["seeds_none"])
assert all(r["near_live"] == 0 for r in ROWS["seeds_l2"])
assert "Under L2 every near-zero weight belongs to a dead neuron, on all five seeds" in INDEX
assert "among live neurons L2 leaves no weight below the threshold and L1 leaves 4 to 16" in INDEX

DEAD = {n: [r["near_dead"] for r in ROWS[n]] for n in NAMES}
LIVE = {n: [r["near_live"] for r in ROWS[n]] for n in NAMES}
NEURONS = {n: [r["dead"] for r in ROWS[n]] for n in NAMES}
assert (DEAD["seeds_l2"], LIVE["seeds_l1"], DEAD["seeds_l1"]) == ([18, 0, 4, 12, 6], [9, 4, 11, 16, 8], [7, 2, 3, 6, 3])

STRENGTH = rich("5 × ", sup("10", num(-4), italic=False))


def say(n):
    return "; ".join(f"seed {s} {d} and {v}" for s, (d, v) in enumerate(zip(DEAD[n], LIVE[n])))


fig = Figure(
    "04-near-zero-weights", "Only L1 leaves near-zero weights in neurons that fire",
    "Two stacked bar charts of the number of the 128 weights of the first layer with |w| below 0.001 after "
    "training, one bar per seed, 0 to 4, on an axis from 0 to 24. Each bar splits into weights of dead neurons, "
    "grey, and of live neurons, in the weight colour. L2 at 5 times 10 to the minus 4, dead and live: "
    f"{say('seeds_l2')}. L1 at the same strength: {say('seeds_l1')}. Under L2 every near-zero weight belongs "
    "to a dead neuron. Without a penalty no weight is below 0.001, and no run has a weight exactly 0.",
    subtitle=rich("Weights of the first layer with |", var("w"), "| < 0.001 after training, by whether their neuron is dead."),
    data_w=True)

COLORS = ("ink-muted", "weight")
left, right = fig.row(2, y=104, h=312)
for box, n, name in ((left, "seeds_l2", "L2"), (right, "seeds_l1", "L1")):
    body = fig.panel(box, rich(name, " at ", STRENGTH))
    fig.stacked_bars(body, [(f"seed {s}", [DEAD[n][s], LIVE[n][s]]) for s in range(5)], COLORS, 0, 24,
                     [0, 6, 12, 18, 24], label_w=72, value_w=104, segment_labels=name == "L1",
                     axis_label="weights below 0.001, of 128")
fig.legend(left.x + 72, 456, [dict(color="ink-muted", label="weights of dead neurons"),
                              dict(color="weight", label="weights of live neurons")], direction="row", gap=40)
fig.caption("No run has a weight exactly 0; without a penalty no weight of the layer is below 0.001.")
fig.write()
