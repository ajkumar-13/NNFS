"""Post 07, section 4: the shape of every array of the forward pass, for the 300-row batch and for its first 7 rows.

Run from anywhere:  python posts/07-coding-the-complete-forward-pass/diagrams/src/02-shape-audit.py
Writes posts/07-coding-the-complete-forward-pass/diagrams/02-shape-audit.svg.
snippets/shape_audit.py is run here (runpy). Every shape is read from the arrays it builds: the 300-row arrays from
its `steps` list, the 7-row arrays from the layers after its second pass, the weight shapes from the layers; each
is asserted against the snippet's printout.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, span  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "shape_audit.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

NAMES = [name for name, _, _ in s["steps"]]
BIG = [a.shape for _, a, _ in s["steps"]]
SMALL = [s["X"][:7].shape] + [s[k].output.shape for k in ("dense1", "activation1", "dense2", "activation2")]
W1, W2 = s["dense1"].weights.shape, s["dense2"].weights.shape
assert NAMES == ["X", "dense1.output", "activation1.output", "dense2.output", "activation2.output"]
for k, (name, shape) in enumerate(zip(NAMES, BIG)):
    assert f"step {k}  {name:<19} {str(shape):<9}" in OUT, name
assert "a batch of 7 rows: " + " -> ".join(str(t) for t in SMALL) in OUT
assert f"dense1: weights {W1}, biases (1, 3), 9 numbers" in OUT and f"dense2: weights {W2}, biases (1, 3)" in OUT
assert BIG == [(300, 2)] + [(300, 3)] * 4 and SMALL == [(7, 2)] + [(7, 3)] * 4
SAME = [k for k, t in enumerate(BIG) if t == BIG[-1]]
assert SAME == [1, 2, 3, 4]


def shape(t):
    """A shape with its row count in the input colour and its column count in the weight colour."""
    return rich("(", span(str(t[0]), color="input", bold=True), ", ", span(str(t[1]), color="weight", bold=True), ")")


def wshape(t):
    return rich("weights (", str(t[0]), ", ", span(str(t[1]), color="weight", bold=True), ")")


fig = Figure(
    "02-shape-audit", "Rows come from the batch, columns from the last dense layer",
    "The five arrays of the forward pass drawn twice, as bands one column per feature. Top, the batch of 300 "
    "spiral points: X is (300, 2), dense1.output (300, 3), activation1.output (300, 3), dense2.output (300, 3), "
    "activation2.output (300, 3). Between the arrays, the four objects: dense1 with weights (2, 3), activation1 "
    "with no weights, dense2 with weights (3, 3), activation2 with no weights. A bracket under the last four arrays "
    "says that they share the shape (300, 3). Bottom, the same pass on the first 7 rows, drawn row by row: (7, 2), "
    "then (7, 3) four times. In every shape the row count is printed in blue and the column count in orange, and "
    "the orange column count of each dense layer's weights is the column count of its output.",
    subtitle="snippets/shape_audit.py, run on all 300 spiral points and again on the first 7.",
    height=720)

X0, GAP, CW = 56, 144, 16
COLS = [t[1] for t in BIG]
xs = [X0]
for c in COLS[:-1]:
    xs.append(xs[-1] + c * CW + GAP)
assert xs == [56, 232, 424, 616, 808]          # the columns of the hero, figure 1
OBJ = [("dense1", "weight", wshape(W1)), ("activation1", None, "no weights"),
       ("dense2", "weight", wshape(W2)), ("activation2", None, "no weights")]

# -- the batch of 300
fig.text(40, 128, "All 300 rows", "head")
T1, H1 = 176, 144
big = []
for x, c, name, k in zip(xs, COLS, NAMES, range(5)):
    g = fig.grid(x, T1, 1, c, cell_w=CW, cell_h=H1, fill=lambda i, j, k=k: ("input" if k == 0 else "output") + "-soft")
    fig.text(g.box.cx, T1 - 16, name, "code", anchor="middle", color="input" if k == 0 else "output")
    fig.text(g.box.cx, g.box.bottom + 28, shape(BIG[k]), "label", anchor="middle")
    big.append(g.box)
M1 = T1 + H1 // 2
for (name, color, w), a, b in zip(OBJ, big, big[1:]):
    fig.arrow((a.right + 8, M1), (b.x - 8, M1))
    cx = (a.right + b.x) / 2
    fig.text(cx, M1 - 12, name, "code", anchor="middle", color=color)
    fig.text(cx, M1 + 28, w, "note", anchor="middle")
fig.brace(big[SAME[0]].x, big[SAME[-1]].right, big[1].bottom + 44,
          rich("four arrays of the same shape, ", shape(BIG[-1]), ": a mix-up between them raises no error"),
          side="below")

# -- the first 7 rows, cell by cell
fig.text(40, 440, rich("The first 7 rows, ", span("X[:7]", mono=True), ", through the same four objects"), "head")
T2, C2 = 456, 16
small = []
for x, c, k in zip(xs, COLS, range(5)):
    g = fig.grid(x, T2, SMALL[k][0], c, cell=C2, fill=lambda i, j, k=k: ("input" if k == 0 else "output") + "-soft")
    fig.text(g.box.cx, g.box.bottom + 28, shape(SMALL[k]), "label", anchor="middle")
    small.append(g.box)
M2 = T2 + SMALL[0][0] * C2 // 2
for a, b in zip(small, small[1:]):
    fig.arrow((a.right + 8, M2), (b.x - 8, M2))

fig.legend(40, 632, [dict(color="input", label="rows: one per point of the batch"),
                     dict(color="weight", label="columns: the features; a dense layer sets them to its neuron count")], direction="row")
fig.caption("Section 9 hands dense2 the wrong one of the four; only a check on the values catches it.")
fig.write()
