"""Post 08 section 8.2: two batches with the same accuracy and different losses.

Run from anywhere:  python posts/08-loss-categorical-cross-entropy/diagrams/src/03-loss-vs-accuracy.py
Writes posts/08-loss-categorical-cross-entropy/diagrams/03-loss-vs-accuracy.svg.
The two batches and the true classes are the ones snippets/loss_vs_accuracy.py defines (it is run here); the
accuracies, per-sample losses and means are computed with NumPy and checked against what it prints.

Layout: two panels, one per batch. Each shows the softmax rows with the true-class cell outlined, a bar per
sample for its loss on one shared scale, a solid bar for the mean loss, and the accuracy underneath.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Box, Figure, rich  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "loss_vs_accuracy.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

Y = np.asarray(s["y_true"])
assert Y.tolist() == [0, 1, 1]
BATCHES = []
for name, key, head in [("barely right", "barely_right", "Barely right: the worked batch of section 4"),
                        ("confidently right", "confidently_right", "Confidently right")]:
    P = np.asarray(s[key])
    pred = np.argmax(P, axis=1)
    acc = np.mean(pred == Y)
    losses = -np.log(P[range(3), Y])
    mean = np.mean(losses)
    block = (f"{name}\n   predictions: {pred}  true classes: {Y}\n"
             f"   accuracy: {acc:.3f} ({np.sum(pred == Y)} of {len(Y)})\n"
             f"   per-sample losses: {np.round(losses, 3)}\n   mean loss: {mean:.3f}\n")
    assert block in OUT, block
    BATCHES.append(dict(head=head, P=P, acc=acc, losses=losses, mean=mean))
assert [b["acc"] for b in BATCHES] == [1.0, 1.0]
assert [f"{v:.3f}" for v in BATCHES[0]["losses"]] == ["0.357", "0.693", "0.105"]
assert [f"{v:.3f}" for v in BATCHES[1]["losses"]] == ["0.051", "0.041", "0.030"]
assert [f"{b['mean']:.3f}" for b in BATCHES] == ["0.385", "0.041"]
RATIO = BATCHES[0]["mean"] / BATCHES[1]["mean"]
assert f"ratio of the two mean losses: {RATIO:.1f}" in OUT and f"{RATIO:.1f}" == "9.4"


def say(b):
    rows = "; ".join(", ".join(f"{v:.2f}" for v in r) for r in b["P"])
    return (f"softmax rows {rows}, on the true classes {', '.join(f'{v:.2f}' for v in b['P'][range(3), Y])}, "
            f"per-sample losses {', '.join(f'{v:.3f}' for v in b['losses'])}, accuracy {b['acc']:.3f} (3 of 3) "
            f"and mean loss {b['mean']:.3f}")


fig = Figure(
    "03-loss-vs-accuracy", "Same accuracy, different loss",
    "Two panels, each a batch of three softmax rows with true classes 0, 1 and 1, the true-class cell outlined "
    "and a bar per sample for its loss on one shared scale, with a solid bar for the mean and a row of three "
    "solid squares, one per correctly predicted sample. Left, the worked batch of section 4: " + say(BATCHES[0])
    + ". Right, a confident batch: " + say(BATCHES[1]) + f". The caption gives the ratio of the means, {RATIO:.1f}.",
    subtitle="True classes 0, 1, 1, outlined; in both batches every row's largest probability is on its true class.")

C = 48
PANEL_W = fig.row(2)[0].w
LANE = PANEL_W - 3 * C - 24                     # the loss lane, the same width in both panels
BAR_MAX = LANE - 56                             # the bar of the largest loss on the page, room left for its value
SCALE = BAR_MAX / max(float(np.max(b["losses"])) for b in BATCHES)
for box, b in zip(fig.row(2), BATCHES):
    assert box.w == PANEL_W
    body = fig.panel(box, b["head"])
    gy = body.y + 32
    g = fig.grid(body.x, gy, 3, 3, C, values=b["P"].tolist(), decimals=2,
                 fill=lambda i, j: "output-soft" if j == Y[i] else None)
    for i in range(3):
        g.outline(i, int(Y[i]), color="output", width=1.5)
    g.col_labels(["class 0", "class 1", "class 2"])
    x0 = g.box.right + 24
    fig.text(x0, gy - 8, "per-sample loss", "tick")
    my = g.box.bottom + 44                       # the centre of the mean bar
    ay = my + 48                                 # the centre of the accuracy row
    right = np.argmax(b["P"], axis=1) == Y
    with fig.data():
        fig.edge((x0, gy), (x0, my + 16), color="ink-muted", width=1)
        for i, v in enumerate(b["losses"]):
            cy = g.cell(i, 0).cy
            w = float(v) * SCALE
            fig.fill(Box(x0, cy - 12, w, 24), "error-soft", fit=False)
            fig.outline(Box(x0, cy - 12, w, 24), "error-line")
            fig.text(x0 + w + 8, cy + 5, f"{v:.3f}", "value", color="error", snap=False)
        w = float(b["mean"]) * SCALE
        fig.fill(Box(x0, my - 12, w, 24), "error", fit=False)
        fig.text(x0 + w + 8, my + 5, f"{b['mean']:.3f}", "value", color="error", bold=True, snap=False)
        fig.text(x0 - 8, my + 5, "mean", "label", anchor="end", snap=False)
        # accuracy as a drawn row: one solid mark per sample whose top prediction is right
        assert right.all()
        for i in range(3):
            fig.marker(x0 + 12 + 24 * i, ay, "square", color="output")
        fig.text(x0 - 8, ay + 5, "right", "label", anchor="end", snap=False)
        fig.text(x0 + 76, ay + 5, f"3 of 3, accuracy {b['acc']:.3f}", "label", color="output", snap=False)
fig.caption(f"The same accuracy for both; the mean losses differ by a factor of {RATIO:.1f}.")
fig.write()
