"""Post 01 hero: the four-input neuron of section 5, with its numbers, as a weighted sum plus one bias.

Run from anywhere:  python posts/01-neurons-and-layers/diagrams/src/01-one-neuron.py
Writes posts/01-neurons-and-layers/diagrams/01-one-neuron.svg.
Every number comes from snippets/neuron_four_inputs.py, which this script runs.
"""
import contextlib
import io
import math
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Raw, rich, var, sub, num, CDOT, SIGMA  # noqa: E402

SNIP = Path(__file__).resolve().parents[2] / "snippets" / "neuron_four_inputs.py"
with contextlib.redirect_stdout(io.StringIO()) as out:
    ns = runpy.run_path(str(SNIP))
X, Wt, B, Z = ns["inputs"], ns["weights"], ns["bias"], ns["output"]
P = [x * w for x, w in zip(X, Wt)]
S = sum(P)
assert out.getvalue().strip() == "4.8" and round(Z, 10) == 4.8
assert [round(p, 10) for p in P] == [0.2, 1.6, -1.5, 2.5] and round(S, 10) == 2.8 and B == 2.0
N_IN = len(X)
N_PARAMS = N_IN + 1                                   # one weight per input, one bias


def say(v, d=1):
    return num(v, d).replace("\u2212", "minus ")


def isub(base, s):
    """An italic letter with an italic subscript letter (x_i): the index is a variable, not a digit."""
    return Raw(f'{var(base)}<tspan class="sub i" dy="4">{s}</tspan><tspan dy="-4">\u200b</tspan>')


fig = Figure(
    "01-one-neuron", "A neuron is a weighted sum plus one bias",
    f"The four-input neuron of section 5. Inputs x1 to x4, {', '.join(say(v) for v in X)}, are multiplied row by "
    f"row by the weights w1 to w4, {', '.join(say(v) for v in Wt)}, giving the products x i w i, "
    f"{', '.join(say(v) for v in P)}. Lines carry the four products into a node marked sigma plus b; the bias b, "
    f"{say(B)}, enters it from above. An arrow labelled sigma plus b equals {say(S)} plus {say(B)} leads to the "
    f"output z, {say(Z)}. Underneath, the whole sum is written out: z equals 1.0 times 0.2 plus 2.0 times 0.8 plus "
    f"3.0 times minus 0.5 plus 2.5 times 1.0 plus 2.0, which is 4.8. Four weights and one bias make five parameters.",
    subtitle="Section 5's neuron: four inputs, one weight per input, one bias, one output.")

TOP, CH = 160, 56                                     # the three columns: rows of 56, from y = 160 to 384
XC, WC, PC, CW = 96, 208, 320, 64                     # column x positions; cells 64 wide
gx = fig.strip(XC, TOP, N_IN, vertical=True, width=CW, height=N_IN * CH, values=X, decimals=1, font=16,
               fill=lambda i: "input-soft")
gw = fig.strip(WC, TOP, N_IN, vertical=True, width=CW, height=N_IN * CH, values=Wt, decimals=1, font=16,
               fill=lambda i: "weight-soft")
gp = fig.strip(PC, TOP, N_IN, vertical=True, width=CW, height=N_IN * CH, values=P, decimals=1, font=16)
gx.row_labels([sub("x", str(i + 1)) for i in range(N_IN)], style="label")
fig.text(XC, TOP - 16, rich("Inputs ", isub("x", "i")), "head")
fig.text(WC, TOP - 16, rich("Weights ", isub("w", "i")), "head")
fig.text(PC, TOP - 16, rich("Products ", isub("x", "i"), isub("w", "i")), "head")
for i in range(N_IN):
    cy = gx.cell(i, 0).cy
    fig.text((XC + CW + WC) / 2, cy + 8, "\u00d7", "op", anchor="middle")
    fig.text((WC + CW + PC) / 2, cy + 8, "=", "op", anchor="middle")

# the node: sigma plus b, on the strips' centre line
NX, NY, R = 576, TOP + N_IN * CH // 2, 48             # (576, 272)
assert NY == 272
with fig.data():
    for i in range(N_IN):
        c = gp.cell(i, 0)
        sx, sy = c.right, c.cy
        a = math.atan2(sy - NY, sx - NX)
        ex, ey = NX + R * math.cos(a), NY + R * math.sin(a)
        fig._path(f"M{sx},{sy} L{ex:.2f},{ey:.2f}", f"nofill w15 {fig._cls('s', 'rule')}")
    fig.add(f'<circle cx="{NX}" cy="{NY}" r="{R}" class="w15 {fig._cls("f", "surface")} '
            f'{fig._cls("s", "ink-muted")}" data-fit="skip"/>')
fig.text(NX, NY + 8, rich(SIGMA, " + ", var("b")), "math", anchor="middle")

# the bias, entering from above
gb = fig.strip(NX - 32, 112, 1, width=64, height=48, values=[B], decimals=1, font=16, fill=lambda i: "weight-soft")
fig.text(NX - 48, 144, rich("Bias ", var("b")), "head", anchor="end")
fig.arrow((NX, gb.box.bottom), (NX, NY - R))

# the output; the arrow's label names both parts of what it carries: the sum and the bias
ZX = 800
gz = fig.strip(ZX, NY - 24, 1, width=80, height=48, values=[Z], decimals=1, font=18, fill=lambda i: "output-soft",
               strong={0: "output"})
fig.text(ZX, NY - 40, rich("Output ", var("z")), "head")
fig.arrow((NX + R, NY), (ZX, NY), label=rich(SIGMA, " + ", var("b"), " = ", num(S, 1), " + ", num(B, 1)))

# the whole sum, written out
terms = " + ".join(f"{num(x, 1)}{CDOT}{'(' + num(w, 1) + ')' if w < 0 else num(w, 1)}" for x, w in zip(X, Wt))
fig.text(XC, 440, rich(var("z"), " = ", terms, " + ", num(B, 1), " = ", num(Z, 1)), "math")
fig.caption(rich(f"One weight per input, one bias per neuron: {N_IN} + 1 = {N_PARAMS} parameters."))
fig.write()
