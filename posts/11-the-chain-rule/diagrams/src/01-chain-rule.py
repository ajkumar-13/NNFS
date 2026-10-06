"""Post 11 hero: values go forward through a chain of three functions, and the slope comes back as the product
of one local derivative per function, each taken at the value its function received.

Run from anywhere:  python posts/11-the-chain-rule/diagrams/src/01-chain-rule.py
Writes posts/11-the-chain-rule/diagrams/01-chain-rule.svg.
The chain is the one of section 5, h(x) = 3x + 1, g(u) = u^2, f(v) = 2v^3 at x = 1. snippets/chain_rule_checks.py
is run here: its chain_derivative gives the three local derivatives, and the values, the factors, the product and
the central difference are asserted against the lines the snippet prints. The running products 1,536, 12,288 and
36,864 under the forward values are the snippet's factors multiplied from the output end, as section 6.2 reads it.
"""
import contextlib
import io
import re
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sup, num, coef, span, TIMES, CDOT  # noqa: E402

PRIME = "\u2032"
HAIR = "\u200a"          # the italic f leans into its prime; a hair space keeps the two apart
SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "chain_rule_checks.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(SNIPPET), run_name="snippet")
    ns["main"]()
OUT = buf.getvalue()

# section 5's chain, as the snippet states it
FUNCS = [lambda x: 3 * x + 1, lambda u: u * u, lambda v: 2 * v ** 3]
DERIVS = [lambda x: 3.0, lambda u: 2 * u, lambda v: 6 * v * v]
X0 = 1.0
slope, factors = ns["chain_derivative"](FUNCS, DERIVS, X0)
values = [X0]
for fn in FUNCS:
    values.append(fn(values[-1]))
assert values == [1, 4, 16, 8192], values                     # x, u, v, y
assert factors == [3, 8, 1536] and slope == 36864              # h'(1), g'(4), f'(16), innermost first
assert "local derivatives, outermost first: 1536 * 8 * 3   (3 factors)" in OUT
m = re.search(r"chain rule 36864\.000000   central difference (36864\.\d{6})", OUT)
assert m, OUT
CD = m.group(1)
assert CD == "36864.000007"
assert "dy/dx = 36(3x + 1)^5 = 36864 at x = 1" in OUT
# the running product from the output end: dy/dy = 1, dy/dv, dy/du, dy/dx
running = [1.0]
for fct in reversed(factors):
    running.append(running[-1] * fct)
assert running == [1, 1536, 12288, 36864]
CD_TXT = "36,864." + CD.split(".")[1]

fig = Figure(
    "01-chain-rule", "A chain's slope is the product of its local slopes",
    "Top row, forward, left to right: x = 1 enters h(x) = 3x + 1, which gives u = 4; u enters g(u) = u squared, "
    "which gives v = 16; v enters f(v) = 2 v cubed, which gives y = 8,192. Bottom row, backward, right to left: "
    "starting from 1 under y, each arrow multiplies by the local derivative of the function above it, taken at the "
    "value that function received: times 1,536, which is f prime (v) = 6 v squared at v = 16, gives dy/dv = 1,536; "
    "times 8, which is g prime (u) = 2u at u = 4, gives dy/du = 12,288; times 3, which is h prime (x) = 3 at x = 1, "
    "gives dy/dx = 36,864. A line below reads dy/dx = 1,536 times 8 times 3 = 36,864, and the central difference "
    f"through the whole chain is {CD_TXT}.",
    subtitle=rich("Section 5's chain ", var("y"), " = ", var("f"), "(", var("g"), "(", var("h"), "(", var("x"),
                  "))) at ", var("x"), " = 1."))

# -- columns: four value cells and three function cards, 40 apart
CW, FW, GAP = 72, 112, 40
xs = [48]
for w in (CW, FW, CW, FW, CW, FW):
    xs.append(xs[-1] + w + GAP)
VX = xs[0::2]                       # value cells: x, u, v, y
FX = xs[1::2]                       # cards: h, g, f
assert VX == [48, 312, 576, 840] and FX == [160, 424, 688] and VX[-1] + CW == 912

NAMES = ["x", "u", "v", "y"]
FNAME = [("h", "x", rich(coef(3, "x"), " + 1")), ("g", "u", sup("u", "2")), ("f", "v", rich("2", sup("v", "3")))]
DRULE = [rich(var("h"), PRIME, "(", var("x"), ") = 3"),
         rich(var("g"), PRIME, "(", var("u"), ") = 2", var("u")),
         rich(var("f"), HAIR, PRIME, "(", var("v"), ") = 6", sup("v", "2"))]

# -- forward row
YF = 184
fig.text(40, 128, "Forward: each function hands its value on", "head")
for k, x in enumerate(VX):
    g = fig.strip(x, YF, 1, cell_w=CW, cell_h=48, values=[num(int(values[k]))], font=18,
                  fill=lambda i, k=k: "input-soft" if k == 0 else "output-soft")
    fig.text(g.box.cx, YF - 16, var(NAMES[k]), "math", anchor="middle")
for k, x in enumerate(FX):
    name, arg, body = FNAME[k]
    fig.card(x, YF, FW, 48)
    fig.text(x + FW / 2, YF + 29, rich(var(name), "(", var(arg), ") = ", body), "label", anchor="middle")
for a, b in zip(xs, xs[1:]):
    w = CW if a in VX else FW
    fig.arrow((a + w, YF + 24), (b, YF + 24))

# -- backward row: the running product under each value, one factor per function between them
YB = 328
fig.text(40, 288, "Backward: multiply by each local slope, right to left", "head", color="gradient")
DNAME = ["dy/dx", "dy/du", "dy/dv", "dy/dy"]
for k, x in enumerate(VX):
    r = running[3 - k]
    g = fig.strip(x, YB, 1, cell_w=CW, cell_h=48, values=[num(int(r))], font=18,
                  fill=lambda i: "gradient-soft", strong={0: "gradient"} if k == 0 else None)
    if k == 0:
        g.window(0, 0, color="gradient")
    fig.text(g.box.cx, YB + 76, var(DNAME[k]), "label", anchor="middle", color="gradient")
for k, x in enumerate(FX):
    fct = factors[k]
    fig.arrow((x + FW + GAP - 8, YB + 24), (x - GAP + 8, YB + 24))
    fig.text(x + FW / 2, YB + 12, rich(TIMES, " ", num(int(fct))), "value", anchor="middle", color="gradient")
    fig.text(x + FW / 2, YB + 76, rich(DRULE[k], " at ", var(NAMES[k]), " = ", num(int(values[k]))), "note",
             anchor="middle")

# -- the product and its check
fig.text(40, 460, rich(var("dy/dx"), " = ", "1,536 ", CDOT, " 8 ", CDOT, " 3 = ", span("36,864", bold=True)), "math")
fig.text(VX[1], 460, rich("central difference through the whole chain: ", CD_TXT), "note")
fig.caption("Each local slope is taken at the value its own function received in the forward pass.")
fig.write()
