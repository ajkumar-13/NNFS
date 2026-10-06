"""Post 32, the hero (sections 3 and 5): the two loops as a worked schedule on the spiral, N = 300 and B = 32.

Run from anywhere:  python posts/32-mini-batching/diagrams/src/01-two-loops.py
Writes posts/32-mini-batching/diagrams/01-two-loops.svg.

snippets/counts.py is run here (runpy, well under a second, nothing trained): the batches per epoch and the rows in
the last batch are its batches() and its printed table rows, asserted. The counter values are what the loop of
snippets/network.py makes them: Optimizer_Adam.post_update_params raises iterations by one after every batch, and
the optimiser is created once, before both loops (asserted in the snippet's source), so after the k-th batch of
epoch e it reads e * 10 + k.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIP = POST / "snippets"
INDEX = (POST / "index.md").read_text(encoding="utf-8")

# -- counts.py: batches per epoch and the last batch
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(SNIP / "counts.py"), run_name="snippet")
OUT = buf.getvalue()
N = 300
N_BATCH, LAST = ns["batches"](N, 32)
assert (N_BATCH, LAST) == (10, 12)
assert ns["batches"](N, 300) == (1, 300)
assert "   300    32       10                12" in OUT
assert "   300   300        1               300" in OUT
SIZES = [32] * (N_BATCH - 1) + [LAST]
assert sum(SIZES) == N
assert "For the 300 spiral points and $B = 32$ it is 10," in INDEX
assert "with 300 rows and $B = 32$ the tenth batch holds 12" in INDEX

# -- the loop: one update per batch, the counter raised after each, the optimiser made once
NET = (SNIP / "network.py").read_text(encoding="utf-8")
assert "    def post_update_params(self):\n        self.iterations += 1" in NET
for line in ("    for epoch in range(epochs):", "        idx = order(n_samples)",
             "        for start in range(0, n_samples, batch_size):",
             "            optimizer.post_update_params()"):
    assert line in NET and line in INDEX, line
assert NET.index("    for epoch in range(epochs):") < NET.index("        for start in range(0, n_samples, batch_size):")
EPOCHS = 1000
assert "| 32 | 1,000 | 10,000 | `head_to_head.py` |" in INDEX         # section 6: 1,000 epochs are 10,000 updates
assert "| 300 | 1,000 | 1,000 | `head_to_head.py` |" in INDEX


def counters(epoch, sizes):
    """iterations after each batch of the given epoch (the optimiser is made once, before both loops)."""
    return [epoch * len(sizes) + k + 1 for k in range(len(sizes))]


assert counters(EPOCHS - 1, SIZES)[-1] == EPOCHS * N_BATCH == 10000
assert counters(EPOCHS - 1, [N])[-1] == EPOCHS

fig = Figure(
    "01-two-loops", "Batches of 32 turn one epoch into ten updates",
    "A worked schedule on the 300 rows of the spiral, each strip as wide as the rows it holds. Epoch 0 with the "
    "full batch: one cell of 300 rows and one update, after which the counter iterations reads 1. Epoch 0 with "
    "batches of 32: nine cells of 32 rows and a last of 12, each passed forward and backward and used for one "
    "update, the counter reading 1 to 10. Later epochs shuffle the rows again and the counter runs on: epoch 1 "
    "reads 11 to 20, epoch 999 reads 9,991 to 10,000.",
    subtitle=rich(var("N"), " = 300 rows, ", var("B"), " = 32: ⌈", var("N"), " / ", var("B"),
                  "⌉ = 10 batches an epoch, the last of 12 rows."),
    height=720, data_w=True)

X0, PX = 160, 2.5                      # strip start; viewBox units per row (32 rows = 80, 12 rows = 30)
SH = 48                                # strip height
LABEL_X = 40


def strip(y, sizes, color, epoch, first_label):
    """One epoch as cells as wide as their batches, rows printed inside, the counter under each cell."""
    x = X0
    cells = []
    with fig.data():
        for k, (n, t) in enumerate(zip(sizes, counters(epoch, sizes))):
            b = Box(x, y, n * PX, SH)
            fig.fill(b, f"{color}-soft", fit=False)
            fig.outline(b, f"{color}-line" if color != "neutral" else "rule", width=1)
            txt = f"{n} rows" if (k == 0 and first_label) else str(n)
            fig.text(b.cx, y + SH / 2 + 5, txt, "label", anchor="middle", snap=False)
            fig.text(b.cx, y + SH + 24, num(t), "label", anchor="middle", snap=False)
            cells.append(b)
            x += n * PX
    fig.text(LABEL_X, y + SH + 24, "iterations", "code", color="ink-muted")
    return cells


# -- epoch 0, cut two ways
fig.text(LABEL_X, 136, "Epoch 0: the same 300 rows, cut two ways", "head")
fig.text(LABEL_X, 160 + SH / 2 + 5, rich(var("B"), " = 300"), "label")
full = strip(160, [N], "neutral", 0, True)

Y32 = 288
fig.text(LABEL_X, Y32 + SH / 2 + 5, rich(var("B"), " = 32"), "label")
mini = strip(Y32, SIZES, "input", 0, True)
fig.brace(mini[0].x, mini[0].right, Y32 - 8, side="above", kind="bracket", color="input")
fig.text(mini[0].x, Y32 - 24, "forward, backward and one update per batch", "note", color="input")
fig.text(mini[-1].right, Y32 - 24, "the last batch: 12 rows", "note", anchor="end")
with fig.data():
    fig.edge((mini[-1].cx, Y32 - 20), (mini[-1].cx, Y32 - 2), color="arrow")

# -- later epochs
fig.text(LABEL_X, 424, rich("Later epochs at ", var("B"), " = 32: a new shuffle, the counter runs on"), "head")
Y1, Y999 = 448, 568
fig.text(LABEL_X, Y1 + SH / 2 + 5, "epoch 1", "label")
strip(Y1, SIZES, "input", 1, False)
fig.text(LABEL_X, Y999 + SH / 2 + 5, "epoch 999", "label")
strip(Y999, SIZES, "input", EPOCHS - 1, False)
fig.text(X0 + N * PX / 2, 548, "⋮", "head", anchor="middle", color="ink-muted")
fig.text(LABEL_X, 548, "epochs 2 to 998", "note")

fig.caption(rich("After 1,000 epochs the counter reads 10,000 at ", var("B"), " = 32 and 1,000 with the full batch."))
fig.write()
