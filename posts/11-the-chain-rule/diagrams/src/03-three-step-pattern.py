"""Post 11 sections 4 and 7: the three steps worked on 3(2x^2)^5, and what taking the outer derivative at the
wrong point does to the slope.

Run from anywhere:  python posts/11-the-chain-rule/diagrams/src/03-three-step-pattern.py
Writes posts/11-the-chain-rule/diagrams/03-three-step-pattern.svg.
snippets/chain_rule_checks.py is run here: its chain_derivative, applied to g(x) = 2x^2 and f(z) = 3z^5 as the
snippet does, gives the factors 240 and 4 and the slope 960 at x = 1, and the wrong-point product 60 is
f'(1) * g'(1); each is asserted against the line the snippet prints. The curve is the composition itself,
asserted equal to the expanded 96x^10 of section 4 at every plotted point.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, coef, span, CDOT  # noqa: E402

PRIME = "\u2032"
HAIR = "\u200a"          # the italic f leans into its prime; a hair space keeps the two apart
SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "chain_rule_checks.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(SNIPPET), run_name="snippet")
    ns["main"]()
OUT = buf.getvalue()

g, f = (lambda x: 2 * x * x), (lambda z: 3 * z ** 5)
dg, df = (lambda x: 4 * x), (lambda z: 15 * z ** 4)
X0 = 1.0
Z0 = g(X0)
Y0 = f(Z0)
slope, factors = ns["chain_derivative"]([g, f], [dg, df], X0)
WRONG = df(X0) * dg(X0)
assert (Z0, Y0) == (2, 96)
assert factors == [4, 240] and slope == 960 and WRONG == 60
assert "local derivatives, outermost first: 240 * 4   (2 factors)" in OUT
assert "chain rule 960.000000   central difference 960.000001" in OUT
assert "outer derivative at the wrong point, f'(x) * g'(x): 60" in OUT
assert "the closed form 960x^9 at x = 1: 960" in OUT

XL, XH = 0.85, 1.15
xs = [XL + (XH - XL) * i / 60 for i in range(61)]
ys = [f(g(x)) for x in xs]
assert all(abs(y - 96 * x ** 10) < 1e-9 * max(1, y) for x, y in zip(xs, ys))


def fp(arg):
    """f prime of arg, the italic f kept clear of its prime."""
    return rich(var("f"), HAIR, PRIME, "(", arg, ")")


def gp(arg):
    return rich(var("g"), PRIME, "(", arg, ")")


fig = Figure(
    "03-three-step-pattern", "Take the outer slope at the inner value",
    "Left, three cards work the derivative of 3 times (2x squared) to the fifth. Step 1, split it: the inner "
    "function g(x) = 2x squared and the outer function f(z) = 3z to the fifth. Step 2, differentiate each: "
    "g prime (x) = 4x and f prime (z) = 15z to the fourth. Step 3, multiply, with f prime taken at z = g(x): "
    "15 (2x squared) to the fourth times 4x = 960x to the ninth, which at x = 1, where z = 2, is f prime (2) times "
    "g prime (1) = 240 times 4 = 960. Right, the curve y = 3(2x squared) to the fifth for x from 0.85 to 1.15, "
    "with the point (1, 96). The tangent there has slope 960. A second line through the point has slope 60, "
    "f prime (1) times g prime (1), the outer derivative taken at x instead of z, and runs almost flat.",
    subtitle=rich("Section 4's ", var("y"), " = 3(", coef(2, sup("x", "2")), ")", sup("", "5", italic=False),
                  " at ", var("x"), " = 1, where the inner value is ", var("z"), " = ", var("g"), "(1) = 2."))

# -- the three steps, stacked to fill the content height
CX, CWD = 40, 304
cards = [
    ("1. Split it", [rich("inner: ", var("g"), "(", var("x"), ") = ", coef(2, sup("x", "2"))),
                     rich("outer: ", var("f"), "(", var("z"), ") = ", coef(3, sup("z", "5")))]),
    ("2. Differentiate each", [rich(gp(var("x")), " = ", coef(4, "x")),
                               rich(fp(var("z")), " = ", coef(15, sup("z", "4")))]),
    (rich("3. Multiply, with ", var("f"), HAIR, PRIME, " at ", var("z"), " = ", var("g"), "(", var("x"), ")"), [
        rich("15(", coef(2, sup("x", "2")), ")", sup("", "4", italic=False), " ", CDOT, " ", coef(4, "x"),
             " = ", coef(960, sup("x", "9"))),
        rich("at ", var("x"), " = 1: ", fp("2"), " ", CDOT, " ", gp("1"), " = 240 ", CDOT, " 4 = ",
             span("960", bold=True))]),
]
GAPV = (476 - 104 - 3 * 108) // 2
assert GAPV == 24
y = 104
for head, lines in cards:
    body = fig.card(CX, y, CWD, None, heading=head, lines=lines, fit=True)
    assert body.bottom + 16 == y + 108, body
    y += 108 + GAPV

# -- the curve, the true tangent and the wrong-point line
ax = fig.line_chart(Box(376, 104, 544, 372), [dict(xs=xs, ys=ys, color="output", points=False, label=None)],
                    x=(XL, XH, [0.9, 1.0, 1.1]), y=(0, 400, [0, 200, 400]), x_label=var("x"), y_label=var("y"),
                    label_w=8, fmt_x=lambda v: f"{v:.1f}")
ax.tangent(X0, Y0, slope=slope, half_width=0.12, color="gradient")
ax.tangent(X0, Y0, slope=WRONG, half_width=0.13, color="error")
ax.point(X0, Y0, "circle", "gradient", size=10)
ax.text(X0, Y0, "(1, 96)", "label", anchor="end", dx=-12, dy=-12)
ax.text(1.08, Y0 + slope * 0.08, "slope 960", "value", color="gradient", dx=8, dy=24)
ax.text(1.02, Y0 + WRONG * 0.02, rich("slope 60: ", fp("1"), " ", CDOT, " ", gp("1")), "value", color="error",
        dy=28)
ax.text(1.12, f(g(1.12)), rich(var("y"), " = 3(", coef(2, sup("x", "2")), ")", sup("", "5", italic=False)),
        "label", color="output", anchor="end", dx=-12)

fig.caption(rich("Taking ", fp(var("x")), " in place of ", fp(var("z")), " gives 60, a sixteenth of the slope."))
fig.write()
