"""Post 03, section 5: every array of the two-layer forward pass for the post's batch, laid out so rows meet columns.

Run from anywhere:  python posts/03-stacking-layers-and-the-forward-pass/diagrams/src/02-batch-through-two-layers.py
Writes posts/03-stacking-layers-and-the-forward-pass/diagrams/02-batch-through-two-layers.svg.
Every value is computed by snippets/two_layer_forward.py (run here) and checked against the printout in section 8.

Layout: each product is drawn with the left factor beside the result and the right factor above it, so row i of the
left factor and column k of the right factor meet at entry (i, k). Z1 is the result of layer 1 and, without being
redrawn, the left factor of layer 2: the batch rows run straight through.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, Raw, rich, var, num, CDOT, MINUS  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "two_layer_forward.py"
with contextlib.redirect_stdout(io.StringIO()):
    S = runpy.run_path(str(SNIP))
X, W1, B1, W2, B2 = S["inputs"], S["weights1"], S["biases1"], S["weights2"], S["biases2"]
Z1, Z2 = S["layer1_outputs"], S["layer2_outputs"]


def fmt(v):
    """The value as the post prints it: shortest form, at most six decimals, true minus."""
    return num(repr(round(float(v), 6)))


# section 8's printout, entry by entry
assert [fmt(v) for v in Z1.ravel()] == [num(s) for s in
                                       "4.8 1.21 2.385 8.9 -1.81 0.2 1.41 1.051 0.026".split()]
assert [fmt(v) for v in Z2.ravel()] == [num(s) for s in
                                       "0.5031 -1.04185 -2.03875 0.2434 -2.7332 -5.7633 -0.99314 1.41254 -0.35655".split()]
# the hand check of section 8: neuron 1 of layer 2 on sample 1
HAND = [(W2[0, j], Z1[0, j]) for j in range(3)]
assert B2[0] < 0                                     # the caption writes it as a subtraction
assert abs(sum(w * z for w, z in HAND) + B2[0] - 0.5031) < 1e-12
assert X.shape == (3, 4) and W1.shape == (3, 4) and W2.shape == (3, 3) and Z1.shape == Z2.shape == (3, 3)


def bold(base, s=None, t=False):
    out = f'<tspan class="b">{base}</tspan>'
    if s is not None:
        out += f'<tspan class="sub" dy="4">{s}</tspan><tspan dy="-4">\u200b</tspan>'
    if t:
        out += '<tspan class="sub" dy="-7">T</tspan><tspan dy="7">\u200b</tspan>'
    return Raw(out)


def shape(t):
    return "(" + ", ".join(num(v) for v in t) + (",)" if len(t) == 1 else ")")


def say(a):
    return ", ".join(fmt(v).replace(MINUS, "minus ") for v in a.ravel())


fig = Figure(
    "02-batch-through-two-layers", "The batch keeps its rows through both layers",
    f"Every array of the forward pass for the post's batch of 3 samples, each product drawn with its left factor "
    f"beside the result and its right factor above it. Layer 1: the input X, shape (3, 4), rows one per sample, holds "
    f"{say(X)}; above the result sits W1 transposed, shape (4, 3), whose column k holds the weights of neuron k: "
    f"{say(W1.T)}; under it the bias b1, shape (3,), {say(B1)}, added to every row. The result Z1, shape (3, 3), "
    f"holds {say(Z1)}. Z1 is also the left factor of layer 2: above the second result sit W2 transposed, shape "
    f"(3, 3), {say(W2.T)}, and the bias b2, {say(B2)}. The result Z2, shape (3, 3), holds {say(Z2)}. Row 1 of Z1 "
    f"and column 1 of W2 transposed are outlined, a note explains that row i on the left meets column k above "
    f"at entry (i, k) of the result, and they meet at the top-left entry of Z2, 0.5031, which the caption "
    f"line works out: 0.1 times 4.8, minus 0.14 times 1.21, plus 0.5 times 2.385, minus 1.0.",
    subtitle=rich("The post's batch, ", var("N"), " = 3, with the values of snippets/two_layer_forward.py."),
    height=720)

C = 48                       # square cells
CW = 72                      # the wider cells of the layer-2 column, for five-decimal values
F = 14
XX, Z1X, Z2X = 136, 384, 584           # left edges of the three columns
YW1, YB, YZ = 152, 360, 432            # top of W1 transposed, of the bias strips, of the batch rows
YW2 = YB - 16 - 3 * C                  # W2 transposed ends where W1 transposed ends

gx = fig.grid(XX, YZ, 3, 4, C, values=lambda i, j: fmt(X[i, j]), fill=lambda i, j: "input-soft", font=F)
gw1 = fig.grid(Z1X, YW1, 4, 3, C, values=lambda i, j: fmt(W1.T[i, j]), fill=lambda i, j: "weight-soft", font=F)
gb1 = fig.strip(Z1X, YB, 3, C, values=[fmt(v) for v in B1], fill=lambda k: "weight-soft", font=F)
gz1 = fig.grid(Z1X, YZ, 3, 3, C, values=lambda i, j: fmt(Z1[i, j]), fill=lambda i, j: "output-soft", font=F)
gw2 = fig.grid(Z2X, YW2, 3, 3, cell_w=CW, cell_h=C, values=lambda i, j: fmt(W2.T[i, j]),
               fill=lambda i, j: "weight-soft", font=F)
gb2 = fig.strip(Z2X, YB, 3, CW, height=C, values=[fmt(v) for v in B2], fill=lambda k: "weight-soft", font=F)
gz2 = fig.grid(Z2X, YZ, 3, 3, cell_w=CW, cell_h=C, values=lambda i, j: fmt(Z2[i, j]),
               fill=lambda i, j: "output-soft", font=F, strong={(0, 0): "output"})

# the hand check: row 1 of Z1 meets column 1 of W2 transposed at the top-left of Z2
gz1.window(0, 0, 1, 3, "output")
gw2.window(0, 0, 3, 1, "weight")

# how to read the layout, in the free corner over layer 2
fig.text(Z2X, 136, "How to read it", "head")
fig.note(Box(Z2X, 128, 0, 0), [rich("Row ", var("i"), " on the left meets column ", var("k"), " above"),
                                   rich("at entry (", var("i"), ", ", var("k"), ") of the result.")])

# rows are samples
gx.row_labels(["sample 1", "sample 2", "sample 3"], side="left", style="note")

# the weights and biases: layer 1 labelled on the left, layer 2 on the right
fig.text(Z1X - 16, gw1.box.cy - 4, rich(bold("W", "1", t=True), "  ", shape(W1.T.shape)), "label",
         anchor="end", color="weight")
fig.text(Z1X - 16, gw1.box.cy + 20, rich("column ", var("k"), ": the weights of neuron ", var("k")), "note",
         anchor="end")
fig.text(Z1X - 16, YB + 29, rich("+ ", bold("b", "1"), "  ", shape(B1.shape)), "label", anchor="end", color="weight")
R = gw2.box.right + 16
fig.text(R, gw2.box.cy - 4, rich(bold("W", "2", t=True), "  ", shape(W2.T.shape)), "label", color="weight")
fig.text(R, gw2.box.cy + 20, "the same layout", "note")
fig.text(R, YB + 29, rich("+ ", bold("b", "2"), "  ", shape(B2.shape)), "label", color="weight")

# the batch rows: what each array is, and the shapes that made it
YH = gz1.box.bottom + 32
for g, head, why in ((gx, rich("Input ", bold("X"), "  ", shape(X.shape)), "3 samples, 4 features"),
                     (gz1, rich(bold("Z", "1"), "  ", shape(Z1.shape)), "(3, 4) · (4, 3) + (3,)"),
                     (gz2, rich(bold("Z", "2"), "  ", shape(Z2.shape)), "(3, 3) · (3, 3) + (3,)")):
    fig.text(g.box.x, YH, head, "head")
    fig.text(g.box.x, YH + 24, why, "note")

fig.caption(rich("Top-left of ", bold("Z", "2"), ": ",
                 f" + ".join(f"{fmt(w)} {CDOT} {fmt(z)}" for w, z in HAND).replace("+ " + MINUS, MINUS + " "),
                 f" {MINUS} ", fmt(abs(B2[0])), " = ", fmt(Z2[0, 0]), "."))
fig.write()
