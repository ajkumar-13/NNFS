"""Post 32, section 5: a decay chosen for full-batch epochs, kept for batches of 32 and chosen again.

Run from anywhere:  python posts/32-mini-batching/diagrams/src/03-decay-counts-updates.py   (about 35 seconds)
Writes posts/32-mini-batching/diagrams/03-decay-counts-updates.svg.

Top: the rate alpha_t = alpha_0 / (1 + decay * t) of section 5, computed here with alpha_0 = 0.02 (the setup of
section 4) for the three runs of snippets/decay.py, t counting updates (1 per epoch at B = 300, 10 at B = 32, from
counts.py's batches()). The last rate of each curve, at t = 1,000 * updates per epoch - 1, is asserted against the
last-rate column decay.py prints and against the table of section 5 in index.md.
Bottom: the training loss of each run on seeds 0 to 4, forward-only after the last update, from decay.py, which is
run here (about 35 seconds); the ranges and means are asserted against the same table.
"""
import contextlib
import io
import re
import runpy
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, isub, sub, num, CDOT  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
SNIP = POST / "snippets"
INDEX = (POST / "index.md").read_text(encoding="utf-8")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    counts = runpy.run_path(str(SNIP / "counts.py"), run_name="snippet")
batches, decayed = counts["batches"], counts["decayed"]

OUT = subprocess.run([sys.executable, str(SNIP / "decay.py")], cwd=str(ROOT), capture_output=True, text=True,
                     check=True).stdout
assert "Optimizer_Adam(learning_rate=0.02" in OUT
ALPHA0, EPOCHS = 0.02, 1000
RUNS = [(300, 1e-2), (32, 1e-2), (32, 1e-3)]            # the three tables of decay.py, in order
blocks = re.split(r"^== ", OUT, flags=re.M)[1:4]
DATA = {}
for (b, d), block in zip(RUNS, blocks):
    assert f"batch size {b}, 1000 epochs, learning_rate 0.02, decay {d:g}," in block, (b, d)
    rows = re.findall(r"^\s+(\d)\s+(\d+)\s+(\d\.\d{6})\s+(\d\.\d{4})\s+(\d\.\d{4})\s", block, re.M)
    assert [int(r[0]) for r in rows] == [0, 1, 2, 3, 4]
    per_epoch = batches(300, b)[0]
    updates = EPOCHS * per_epoch
    assert all(int(r[1]) == updates for r in rows)
    last = decayed(ALPHA0, d, updates - 1)
    assert all(r[2] == f"{last:.6f}" for r in rows), (b, d)
    loss = [float(r[3]) for r in rows]
    mean = sum(loss) / 5
    m = re.search(r"train loss (\d\.\d{4}) to (\d\.\d{4}) \(mean (\d\.\d{4})\)", block)
    assert m.groups() == (f"{min(loss):.4f}", f"{max(loss):.4f}", f"{mean:.4f}")
    DATA[(b, d)] = dict(per_epoch=per_epoch, updates=updates, last=f"{last:.6f}", loss=loss, mean=f"{mean:.4f}")

# -- the table of section 5
EXP = {1e-2: "$10^{-2}$", 1e-3: "$10^{-3}$"}
for (b, d), r in DATA.items():
    row = (f"| {b} | {EXP[d]} | {num(r['updates'])} | {r['last']} | {min(r['loss']):.4f} to "
           f"{max(r['loss']):.4f} ({r['mean']}) |")
    assert row in INDEX, row
assert r"\alpha_t = \frac{\alpha_0}{1 + \text{decay} \cdot t}" in INDEX


def dec(d):
    return sup("10", num(round(__import__("math").log10(d))), italic=False)


def name(b, d):
    return rich(var("B"), f" = {b}, decay ", dec(d))


ACCENT = {(32, 1e-3): "blue"}                             # the decay chosen again; the other two in neutrals
COLOR = {k: ACCENT.get(k, "ink-muted") for k in RUNS}
P4 = lambda v: f"{v:.4f}"                                 # noqa: E731
desc_runs = "; ".join(
    f"B = {b}, decay 10 to the minus {round(-__import__('math').log10(d))}: {num(DATA[(b, d)]['updates'])} updates, "
    f"last rate {DATA[(b, d)]['last']}, training loss {', '.join(P4(v) for v in DATA[(b, d)]['loss'])}, mean "
    f"{DATA[(b, d)]['mean']}" for b, d in RUNS)
fig = Figure(
    "03-decay-counts-updates", "Kept for batches of 32, the decay ends at a hundredth",
    "Top: the learning rate alpha_t = 0.02 / (1 + decay times t) against the epoch from 0 to 1,000 on a logarithmic "
    "rate axis, t counting updates. The full batch with decay 10 to the minus 2 and batches of 32 with decay 10 to "
    "the minus 3 fall along the same curve to 0.00182; batches of 32 with the decay kept at 10 to the minus 2 count "
    "ten updates an epoch and end at 0.000198. Bottom: the training loss after 1,000 epochs on seeds 0 to 4, one "
    "band per run with a tick per seed and a dot for seed 0, the mean at the right. " + desc_runs + ".",
    subtitle=rich("Section 4's setup for 1,000 epochs: ", isub("α", "t"), " = 0.02 / (1 + decay ", CDOT, " ",
                  var("t"), ") with ", var("t"), " counting updates."),
    height=720, data_w=True)

# -- top: the rate per epoch
top = fig.panel(Box(40, 104, 880, 320), rich("The rate ", isub("α", "t"), " against the epoch"))
series = []
for b, d in RUNS:
    u = DATA[(b, d)]["per_epoch"]
    ts = sorted(set(range(0, 40 * u, max(1, u // 10))) | set(range(0, EPOCHS * u, u * 5)) | {EPOCHS * u - 1})
    series.append(dict(xs=[t / u for t in ts], ys=[decayed(ALPHA0, d, t) for t in ts], color=COLOR[(b, d)],
                       points=False, label=None))
# the full batch is drawn first and marked with hollow squares every 100 epochs, so it shows under the blue curve
ax = fig.line_chart(top, [series[1], series[0], series[2]], x=(0, 1000, [0, 250, 500, 750, 1000]),
                    y=(1e-4, 0.04, [1e-4, 1e-3, 1e-2]), y_log=True, minor=True,
                    x_label="epoch", y_label=isub("α", "t"), label_w=200, fmt_x=lambda v: num(int(v)))
for e in range(0, 1001, 100):
    t = e if e < 1000 else 999
    ax.point(e, decayed(ALPHA0, 1e-2, t), "square", "ink-muted", size=8, hollow=True)
END = {k: decayed(ALPHA0, k[1], DATA[k]["updates"] - 1) for k in RUNS}
with fig.data():
    xe, ye = ax.to_px(1000, END[(32, 1e-3)])
    fig.text(xe + 16, ye - 8, name(32, 1e-3), "note", color="blue", snap=False)
    fig.marker(xe + 22, ye + 14, "square", "ink-muted", size=8, hollow=True)
    fig.text(xe + 36, ye + 19, name(300, 1e-2), "note", snap=False)
    xe, ye = ax.to_px(1000, END[(32, 1e-2)])
    fig.text(xe + 16, ye + 5, name(32, 1e-2), "note", snap=False)

# -- bottom: the training loss over seeds
bot_y = 472
ax2 = fig.dot_plot(Box(40, bot_y, 784, 152), [("", []) for _ in RUNS], 0.3, 1.0,
                   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], label_w=208, pad_right=16, fmt=lambda v: f"{v:.1f}")
fig.text(40, bot_y - 16, "Training loss after 1,000 epochs, seeds 0 to 4", "head")
fig.text(920, bot_y - 16, "mean", "note", anchor="end")
with fig.data():
    for i, k in enumerate(RUNS):
        y = ax2.sy(i)
        loss = DATA[k]["loss"]
        fig.text(40, y + 5, name(*k), "label", snap=False)
        if COLOR[k] == "blue":
            fig.range_mark(ax2.sx(min(loss)), ax2.sx(max(loss)), y, ticks=[ax2.sx(v) for v in loss], color="blue",
                           dot=ax2.sx(loss[0]), h=24)
        else:
            bx = Box(ax2.sx(min(loss)), y - 12, ax2.sx(max(loss)) - ax2.sx(min(loss)), 24)
            fig.fill(bx, "neutral-soft", fit=False)
            fig.outline(bx, "rule")
            for v in loss:
                fig.edge((ax2.sx(v), y - 8), (ax2.sx(v), y + 8), color="ink-muted")
            fig.marker(ax2.sx(loss[0]), y, "circle", "ink-muted", size=10)
        fig.text(920, y + 5, DATA[k]["mean"], "value", anchor="end", color=COLOR[k] if COLOR[k] == "blue" else None,
                 snap=False)
with fig.data():                                          # the neutral band drawn small, as in the rows
    LY = 652
    lb = Box(ax2.x, LY - 12, 24, 16)
    fig.fill(lb, "neutral-soft", fit=False)
    fig.outline(lb, "rule")
    fig.edge((ax2.x + 12, LY - 9), (ax2.x + 12, LY + 1), color="ink-muted")
    fig.text(ax2.x + 36, LY, "five seeds, one tick each", "label", snap=False)
    fig.marker(ax2.x + 264, LY - 4, "circle", "ink-muted", size=10)
    fig.text(ax2.x + 280, LY, "seed 0", "label", snap=False)
fig.caption("Divided by the 10 updates of an epoch, the decay gives back the full batch's schedule.")
fig.write()
