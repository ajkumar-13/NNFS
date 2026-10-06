"""Post 13 section 5.3: closing neuron 2's gate zeroes its five gradients and shrinks the other ten.

Run from anywhere:  python posts/13-backprop-through-a-layer/diagrams/src/02-closed-gate.py
Writes posts/13-backprop-through-a-layer/diagrams/02-closed-gate.svg.
Both layers are those of snippets/layer_backward.py (run here): the layer of section 4 and the same layer with
neuron 2's four weights negated (the snippet's weights_off). Every value comes from the snippet's forward and
backward functions and is asserted against the lines the snippet prints for sections 5 and 5.3.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sub, isub, span, num  # noqa: E402

PARTIAL = "\u2202"
SNIP = Path(__file__).resolve().parents[2] / "snippets" / "layer_backward.py"
with contextlib.redirect_stdout(io.StringIO()) as out:
    S = runpy.run_path(str(SNIP))
OUT = out.getvalue()
X = S["inputs"]


def run(weights):
    Z, A, Y, L = S["forward"](weights, S["biases"], X)
    DZ, DW, DB = S["backward"](Z, Y, X)
    num_dW, num_db = S["numerical_gradients"](weights, S["biases"], X)
    assert S["largest_gap"](DW, DB, num_dW, num_db) < 1e-8
    return dict(Z=Z, Y=Y, L=L, G=[int(z > 0) for z in Z], DZ=DZ, DW=DW, DB=DB)


OPEN, OFF = run(S["weights"]), run(S["weights_off"])
assert S["weights_off"][1] == [-w for w in S["weights"][1]]
# the snippet's printout, sections 5 and 5.3
assert "Z = [   3.1    7.2   11.3]   A = [   3.1    7.2   11.3]   Y = 21.6   L = 466.56" in OUT
assert "Z = [   3.1   -6.8   11.3]   A = [   3.1    0.0   11.3]   Y = 14.4   L = 207.36" in OUT
assert "upstream dL/dY = 2Y = 28.8   gates = [1, 0, 1]   dL/dZ = [  28.8    0.0   28.8]" in OUT
assert "neuron 1   dL/dW = [  28.8   57.6   86.4  115.2]   dL/db =  28.8" in OUT
assert "neuron 2   dL/dW = [   0.0    0.0    0.0    0.0]   dL/db =   0.0" in OUT
assert OPEN["G"] == [1, 1, 1] and OFF["G"] == [1, 0, 1]


def f1(v):
    return num(v, 1)


def d(top, bot):
    return rich(PARTIAL, top, "/", PARTIAL, bot)


YH, LV = var("\u0177"), var("L")


def say(case):
    rows = "; ".join(f"neuron {k + 1}: z = {f1(case['Z'][k])}, gate {case['G'][k]}, dL/dz = {f1(case['DZ'][k])}, "
                     f"weight gradients {', '.join(f1(v) for v in case['DW'][k])}, bias {f1(case['DB'][k])}"
                     for k in range(3))
    return f"y-hat = {f1(case['Y'])} and 2 y-hat = {f1(2 * case['Y'])}; {rows}"


fig = Figure(
    "02-closed-gate", "A closed gate zeroes its row and shrinks the others",
    f"Two tables with the same columns: each neuron's pre-activation z, its gate, dL/dz, its four weight "
    f"gradients under the inputs 1, 2, 3 and 4, and its bias gradient. Top, all three gates open: "
    f"{say(OPEN)}. Bottom, neuron 2's weights negated: {say(OFF)}. In the bottom table neuron 2's z is shaded as "
    f"negative and its row of zero gradients is outlined, and zero gradients are left unshaded. Over the weight gradients a label reads weight gradients dL/dw_kj. Notes "
    f"say that row 2 is zero because its gate is 0 and that rows 1 and 3 fell from 43.2 to 28.8 per unit of "
    f"input because y-hat fell.",
    subtitle="Section 5.3: neuron 2's four weights negated, nothing else changed.",
    height=720)

CH, CW = 48, 64
XZ, XG, XD, XW = 144, 224, 288, 376        # columns: z, gate, dL/dz, the weight gradients
XB = XW + 4 * CW + 24                       # the bias gradients
XN = XB + CW + 32                           # the notes on the right
F = 16


def block(case, top, head):
    fig.text(40, top - 40, head, "head")
    gz = fig.strip(XZ, top, 3, cell_w=CW, cell_h=CH, vertical=True, values=[f1(v) for v in case["Z"]],
                   fill=lambda k: "negative-soft" if case["Z"][k] < 0 else "output-soft", font=F)
    gg = fig.strip(XG, top, 3, cell_w=48, cell_h=CH, vertical=True, values=[str(v) for v in case["G"]], font=F)
    soft = lambda v: "gradient-soft" if v else None      # noqa: E731  a zero gradient is left unshaded
    gd = fig.strip(XD, top, 3, cell_w=CW, cell_h=CH, vertical=True, values=[f1(v) for v in case["DZ"]],
                   fill=lambda k: soft(case["DZ"][k]), font=F)
    gw = fig.grid(XW, top, 3, 4, cell_w=CW, cell_h=CH, values=lambda k, j: f1(case["DW"][k][j]),
                  fill=lambda k, j: soft(case["DW"][k][j]), font=F)
    gb = fig.strip(XB, top, 3, cell_w=CW, cell_h=CH, vertical=True, values=[f1(v) for v in case["DB"]],
                   fill=lambda k: soft(case["DB"][k]), font=F)
    gz.col_labels([span(isub("z", "k"), color="output")], style="label")
    gg.col_labels(["gate"], style="label")
    gd.col_labels([span(d(LV, isub("z", "k")), color="gradient")], style="label")
    gw.col_labels([span(rich(sub("x", str(j + 1)), " = ", num(int(X[j]))), color="input") for j in range(4)],
                  style="label")
    gb.col_labels([span(d(LV, isub("b", "k")), color="gradient")], style="label")
    fig.text(XW, top - 40, rich("weight gradients ", d(LV, isub("w", "kj"))), "label", color="gradient")
    gz.row_labels([f"neuron {k + 1}" for k in range(3)], style="note")
    fig.text(XN, top + 29, rich(YH, " = ", f1(case["Y"])), "value", color="output")
    fig.text(XN, top + 53, rich("2", YH, " = ", f1(2 * case["Y"])), "value", color="gradient")
    return gz, gd, gw


TOP_A, TOP_B = 176, 424
block(OPEN, TOP_A, "All three gates open")
gz, gd, gw = block(OFF, TOP_B, "Neuron 2 switched off")
gw.window(1, 0, 1, 4, "negative")
fig.note(gw.box, ["Row 2: gate 0, so all five gradients are 0.",
                  rich("Rows 1 and 3: 28.8 per unit of input, because ", YH, " fell.")])

fig.caption(rich("No neuron's gradient contains another's weights, but every one contains 2", YH, "."))
fig.write()
