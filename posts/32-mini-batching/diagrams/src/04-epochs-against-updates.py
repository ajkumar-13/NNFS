"""Post 32, section 6: batch sizes compared at equal epochs and at equal updates, seeds 0 to 4.

Run from anywhere:  python posts/32-mini-batching/diagrams/src/04-epochs-against-updates.py   (about 70 seconds)
Writes posts/32-mini-batching/diagrams/04-epochs-against-updates.svg.

The three scripts of section 6's table are run here side by side: snippets/head_to_head.py (about 65 seconds),
batch_100.py (about 30) and batch_8.py (about 25; its first table, the learning rate of the other runs). Each
row of the figure is one of their five-seed tables, measured forward-only after the last update: the training loss
and the test accuracy of every seed. Every row's updates, mean training loss and accuracy ranges with means are
asserted against the table of section 6 in index.md. The rows are grouped as the post reads the table: 1,000 epochs
each, about 10,000 updates each, and 1,000 updates each (two runs appear in two groups).
"""
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, num, sup  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
SNIP = POST / "snippets"
INDEX = (POST / "index.md").read_text(encoding="utf-8")

SCRIPTS = ("head_to_head", "batch_100", "batch_8")
procs = {s: subprocess.Popen([sys.executable, str(SNIP / f"{s}.py")], cwd=str(ROOT), stdout=subprocess.PIPE,
                             text=True) for s in SCRIPTS}
OUT = {}
for s, p in procs.items():
    OUT[s], _ = p.communicate()
    assert p.returncode == 0, s
SUMMARY = re.compile(r"^train loss (\d\.\d{4}) to (\d\.\d{4}) \(mean (\d\.\d{4})\), train accuracy ([\d.]+) to ([\d.]+) "
                     r"\(mean ([\d.]+)\) percent, test accuracy ([\d.]+) to ([\d.]+) \(mean ([\d.]+)\) percent$", re.M)

ROW = re.compile(r"^\s+(\d)\s+(\d+)\s+\d\.\d{6}\s+(\d\.\d{4})\s+(\d\.\d{4})\s+\d\.\d{4}\s+(\d\.\d{4})\s+\d+$", re.M)
RUNS = {}                                      # (batch size, epochs) -> dict(updates, loss, train, test)


def take(block, b, e, script):
    rows = ROW.findall(block)
    assert [int(r[0]) for r in rows] == [0, 1, 2, 3, 4], (b, e)
    updates = {int(r[1]) for r in rows}
    assert len(updates) == 1
    RUNS[(b, e)] = dict(updates=updates.pop(), loss=[float(r[2]) for r in rows],
                        train=[100 * float(r[3]) for r in rows], test=[100 * float(r[4]) for r in rows],
                        script=script, summary=SUMMARY.search(block).groups())
    r = RUNS[(b, e)]                                         # the printed ranges are the per-seed extremes
    sm = r["summary"]
    assert (sm[0], sm[1]) == (f"{min(r['loss']):.4f}", f"{max(r['loss']):.4f}")
    assert (sm[3], sm[4], sm[6], sm[7]) == tuple(f"{v:.2f}" for v in (min(r["train"]), max(r["train"]),
                                                                       min(r["test"]), max(r["test"])))


for block in re.split(r"^== ", OUT["head_to_head"], flags=re.M)[1:5]:
    m = re.match(r"Batch size (\d+), ([\d,]+) epochs", block)
    take(block, int(m.group(1)), int(m.group(2).replace(",", "")), "head_to_head")
for block in re.split(r"^== ", OUT["batch_100"], flags=re.M)[1:3]:
    m = re.match(r"Batch size (\d+), ([\d,]+) epochs", block)
    take(block, int(m.group(1)), int(m.group(2).replace(",", "")), "batch_100")
first = re.split(r"^== ", OUT["batch_8"], flags=re.M)[1]
assert "batch size 8, 264 epochs, learning_rate 0.02," in first
take(first, 8, 264, "batch_8")
assert sorted(RUNS) == sorted([(300, 1000), (300, 10000), (32, 100), (32, 1000), (100, 1000), (100, 3334), (8, 264)])

# -- section 6's table, row by row
P2 = lambda v: f"{v:.2f}"                                                    # noqa: E731
for (b, e), r in RUNS.items():
    sm = r["summary"]                                        # the means as the scripts print them
    row = (f"| {b} | {num(e)} | {num(r['updates'])} | `{r['script']}.py` | {sm[2]} | "
           f"{sm[3]} to {sm[4]} ({sm[5]}) | {sm[6]} to {sm[7]} ({sm[8]}) |")
    assert row in INDEX, row

GROUPS = [
    ("1,000 epochs each", [(300, 1000), (100, 1000), (32, 1000)], "updates"),
    ("About 10,000 updates each", [(300, 10000), (100, 3334), (32, 1000), (8, 264)], "epochs"),
    ("1,000 updates each", [(300, 1000), (32, 100)], "epochs"),
]
assert all(abs(RUNS[k]["updates"] - 10000) <= 32 for k in GROUPS[1][1])
assert all(RUNS[k]["updates"] == 1000 for k in GROUPS[2][1])
COLOR = lambda k: "blue" if k[0] == 32 else None                              # noqa: E731

LAYOUT = []                                                                    # (kind, payload) per chart row
for head, keys, what in GROUPS:
    LAYOUT.append(("head", head))
    for k in keys:
        LAYOUT.append(("run", (k, what)))


def row_label(k, what):
    b, e = k
    n = RUNS[k]["updates"] if what == "updates" else e
    return rich(var("B"), f" = {b}, ", num(n), f" {what}")


desc = []
for head, keys, what in GROUPS:
    parts = []
    for k in keys:
        r = RUNS[k]
        parts.append(f"B = {k[0]} ({num(k[1])} epochs, {num(r['updates'])} updates): training loss "
                     + ", ".join(f"{v:.4f}" for v in r["loss"]) + "; test accuracy "
                     + ", ".join(P2(v) for v in r["test"]))
    desc.append(head + ": " + "; ".join(parts))
fig = Figure(
    "04-epochs-against-updates", "Equal epochs favour batches of 32; equal updates do not",
    "Two dot charts with one row per run of section 6, grouped as 1,000 epochs each, about 10,000 updates each and "
    "1,000 updates each; each row a band over seeds 0 to 4 with a tick per seed and a dot for seed 0, batches of 32 "
    "in blue. Left the training loss on an axis from 0 to 1, right the test accuracy in percent from 50 to 90. "
    + ". ".join(desc) + ".",
    subtitle=rich("The spiral, Adam at 0.02 with decay ", sup("10", "−5", italic=False),
                  ", seeds 0 to 4, measured forward-only after the last update."),
    height=720, data_w=True)

H = 488
rows = [("", []) for _ in LAYOUT]
la = fig.dot_plot(Box(40, 128, 496, H), rows, 0, 1.0, [0, 0.2, 0.4, 0.6, 0.8, 1.0], label_w=216, pad_right=16,
                  fmt=lambda v: f"{v:.1f}")
ra = fig.dot_plot(Box(568, 128, 352, H), rows, 50, 90, [50, 60, 70, 80, 90], label_w=0, pad_right=16,
                  fmt=lambda v: num(v))
fig.text(la.x, 112, "Training loss", "head")
fig.text(ra.x, 112, "Test accuracy, percent", "head")


def band(ax, lo, hi, y, ticks, dot, color, h=20):
    """The band-with-ticks mark: range_mark in blue, or the same form in neutrals."""
    if color:
        fig.range_mark(ax.sx(lo), ax.sx(hi), y, ticks=[ax.sx(v) for v in ticks], color=color, dot=ax.sx(dot), h=h)
        return
    b = Box(ax.sx(lo), y - h / 2, ax.sx(hi) - ax.sx(lo), h)
    fig.fill(b, "neutral-soft", fit=False)
    fig.outline(b, "rule")
    for v in ticks:
        fig.edge((ax.sx(v), y - h / 3), (ax.sx(v), y + h / 3), color="ink-muted")
    fig.marker(ax.sx(dot), y, "circle", "ink-muted", size=10)


with fig.data():
    for i, (kind, payload) in enumerate(LAYOUT):
        y = la.sy(i)
        if kind == "head":
            fig.text(40, y + 5, payload, "label", bold=True, snap=False)
            continue
        k, what = payload
        r = RUNS[k]
        fig.text(56, y + 5, row_label(k, what), "label", snap=False)
        band(la, min(r["loss"]), max(r["loss"]), y, r["loss"], r["loss"][0], COLOR(k))
        band(ra, min(r["test"]), max(r["test"]), y, r["test"], r["test"][0], COLOR(k))
    LY = 668 - 16
    lb = Box(la.x, LY - 12, 24, 16)
    fig.fill(lb, "neutral-soft", fit=False)
    fig.outline(lb, "rule")
    fig.edge((la.x + 12, LY - 9), (la.x + 12, LY + 1), color="ink-muted")
    fig.text(la.x + 36, LY, "five seeds, one tick each", "label", snap=False)
    fig.marker(la.x + 264, LY - 4, "circle", "ink-muted", size=10)
    fig.text(la.x + 280, LY, "seed 0", "label", snap=False)
    fig.range_mark(la.x + 392, la.x + 416, LY - 4, ticks=[la.x + 404], color="blue", h=16)
    fig.text(la.x + 428, LY, rich("batches of 32"), "label", snap=False)
fig.caption("Equal epochs: batches of 32 lead. About 10,000 updates: no order. 1,000 updates: the full batch leads.")
fig.write()
