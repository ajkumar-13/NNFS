"""Post 27, section 2 (hero): the five lines of the Adam update as two lanes, two corrections and one update.

Run from anywhere:  python posts/27-adam-optimiser/diagrams/src/01-adam-update.py   (about a second)
Writes posts/27-adam-optimiser/diagrams/01-adam-update.svg.
The five lines are those of section 2 of index.md (asserted). The numbers under them are the first update, t = 1,
of the constant gradient g = 0.5 that snippets/bias_correction.py traces through both averages (run here; its row
t = 1 is parsed and asserted), and the caption's 3.162 is that script's uncorrected step at t = 1, which index.md
quotes. The defaults beta_1 = 0.9, beta_2 = 0.999 and epsilon = 1e-7 are read from Optimizer_Adam in adam.py.

Layout: the gradient on the left, two lanes (first moment, second moment), each an average and its correction,
joined in the update on the right; a legend for the role colours underneath.
"""
import inspect
import math
import re
import runpy
import subprocess
import sys
import contextlib
import io
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, sup, subsup, isub, hat, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")

# -- the five lines of section 2
for line in (r"m_t &= \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t",
             r"v_t &= \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2",
             r"\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}}",
             r"\hat{v}_t &= \frac{v_t}{1 - \beta_2^{\,t}}",
             r"\theta_t &= \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}"):
    assert line in INDEX, line

# -- the class defaults
sys.path.insert(0, str(POST / "snippets"))
with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path(str(POST / "snippets" / "adam.py"), run_name="snippet")
sig = inspect.signature(s["Optimizer_Adam"].__init__).parameters
B1, B2, EPS = sig["beta_1"].default, sig["beta_2"].default, sig["epsilon"].default
assert (B1, B2, EPS) == (0.9, 0.999, 1e-7)

# -- the trace of g = 0.5 and the uncorrected step, from bias_correction.py
out = subprocess.run([sys.executable, str(POST / "snippets" / "bias_correction.py")], cwd=ROOT,
                     capture_output=True, text=True, check=True).stdout
row = re.search(r"^\s+1\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$",
                out.split("== a constant gradient g = 0.5")[1], re.M).groups()
m1, m1f, mhat1, v1, v1f, vhat1 = map(float, row)
G = 0.5
assert (m1, mhat1, v1, vhat1) == (0.05, 0.5, 0.00025, 0.25), row
assert abs(m1 - (1 - B1) * G) < 1e-12 and abs(v1 - (1 - B2) * G ** 2) < 1e-12
assert abs(mhat1 - m1 / (1 - B1)) < 1e-12 and abs(vhat1 - v1 / (1 - B2)) < 1e-12
step1 = mhat1 / (vhat1 ** 0.5 + EPS)                 # in learning rates
assert f"{step1:.6f}" == "1.000000"
unc = re.search(r"^\s+1\s+1\.000\s+(\d\.\d{3})$", out, re.M).group(1)
assert unc == "3.162" and f"{m1 / v1 ** 0.5:.3f}" == unc
assert "At $t = 1$ the factor is $0.1 / \\sqrt{0.001} = 3.162$" in INDEX

# -- symbols
M = lambda t: isub("m", t)                           # noqa: E731
V = lambda t: isub("v", t)                           # noqa: E731
MH = lambda t: isub(hat("m"), t)                     # noqa: E731
VH = lambda t: isub(hat("v"), t)                     # noqa: E731
b1, b2 = sub(var("β"), "1"), sub(var("β"), "2")
b1t = subsup("β", "1", "t", italic_sup=True)
b2t = subsup("β", "2", "t", italic_sup=True)
T1 = rich(var("t"), "−1")
g_t = isub("g", "t")
TH = lambda t: isub("θ", t)                          # noqa: E731
ALPHA, EP = var("α"), var("ε")

fig = Figure(
    "01-adam-update", "Two averages, two corrections, one update",
    "A flow diagram of the Adam update for one parameter. The gradient g t feeds two lanes. Top, the first "
    "moment with beta 1 = 0.9, m t = beta 1 m t minus 1 plus (1 minus beta 1) g t, the averaging of momentum "
    "from post 24, then its correction m hat t = m t over (1 minus beta 1 to the t). Bottom, the second moment "
    "with beta 2 = 0.999, v t = beta 2 v t minus 1 plus (1 minus beta 2) g t squared, the cache of RMSProp from "
    "post 26, then its correction v hat t = v t over (1 minus beta 2 to the t). Both feed the update theta t = "
    "theta t minus 1 minus alpha times m hat t over (root of v hat t plus epsilon), epsilon = 10 to the minus 7. "
    "For the first update of a constant gradient 0.5: m 1 = 0.05 and v 1 = 0.00025; the corrections divide by "
    "0.1 and 0.001 and give m hat 1 = 0.5 and v hat 1 = 0.25, so the step is alpha times 0.5 over (0.5 plus "
    "epsilon), one learning rate. Without the corrections it would be 3.162 learning rates.",
    subtitle=rich("One parameter, ", var("t"), " counted from 1. The numbers: the first update of a constant gradient ",
                  var("g"), " = 0.5."),
    data_w=True)

LANES = (112, 280)                                   # top of each lane's cards
CH = 136                                             # card height
EMA_X, EMA_W = 168, 280
COR_X, COR_W = 480, 208
UPD_X, UPD_W = 720, 200


def lines(x, y, rows):
    """rows of (text, style, colour) from the first baseline y, 24 apart."""
    for k, (t, st, c) in enumerate(rows):
        fig.text(x, y + 24 * k, t, st, color=c)


# the gradient
gbox = Box(40, 224, 96, 112)
fig.card(gbox.x, gbox.y, gbox.w, gbox.h, color="gradient")
fig.text(gbox.cx, gbox.y + 52, g_t, "math", anchor="middle", color="gradient")
fig.text(gbox.cx, gbox.y + 80, "gradient", "note", anchor="middle")

ema, cor = [], []
for k, top in enumerate(LANES):
    first = k == 0
    head = rich("First moment, ", b1, " = 0.9") if first else rich("Second moment, ", b2, " = 0.999")
    fig.card(EMA_X, top, EMA_W, CH, heading=head)
    if first:
        f = rich(M("t"), " = ", b1, " ", isub("m", T1), " + (1 − ", b1, ") ", g_t)
        lines(EMA_X + 16, top + 72, [(f, "math", None),
                                     ("the averaging of momentum, post 24", "note", None),
                                     (rich(sub("m", "1"), " = 0.1 · 0.5 = 0.05"), "label", None)])
    else:
        f = rich(V("t"), " = ", b2, " ", isub("v", T1), " + (1 − ", b2, ") ", subsup("g", "t", "2", italic_sub=True))
        lines(EMA_X + 16, top + 72, [(f, "math", None),
                                     ("the cache of RMSProp, post 26", "note", None),
                                     (rich(sub("v", "1"), " = 0.001 · 0.25 = 0.00025"), "label", None)])
    ema.append(Box(EMA_X, top, EMA_W, CH))
    fig.card(COR_X, top, COR_W, CH, heading=rich("Correction of ", var("m" if first else "v")), color="blue")
    if first:
        lines(COR_X + 16, top + 72, [(rich(MH("t"), " = ", M("t"), " / (1 − ", b1t, ")"), "math", None),
                                     (rich("divides by 0.1 at ", var("t"), " = 1"), "note", None),
                                     (rich(sub(hat("m"), "1"), " = 0.05 / 0.1 = 0.5"), "label", None)])
    else:
        lines(COR_X + 16, top + 72, [(rich(VH("t"), " = ", V("t"), " / (1 − ", b2t, ")"), "math", None),
                                     (rich("divides by 0.001 at ", var("t"), " = 1"), "note", None),
                                     (rich(sub(hat("v"), "1"), " = 0.25"), "label", None)])
    cor.append(Box(COR_X, top, COR_W, CH))

# the update, centred between the lanes
UY, UH = 196, 168
fig.card(UPD_X, UY, UPD_W, UH, heading=rich("Update, ", EP, " = ", sup("10", "−7", italic=False)), color="weight")
fig.text(UPD_X + 16, UY + 72, rich(TH("t"), " = ", isub("θ", T1)), "math")
fig.text(UPD_X + 32, UY + 96, rich("− ", ALPHA, " · ", MH("t"), " / (√", VH("t"), " + ", EP, ")"), "math")
lines(UPD_X + 16, UY + 128, [(rich("first step: ", ALPHA, " · 0.5 / (0.5 + ", EP, ")"), "label", None),
                             (rich("= 1.000 ", ALPHA), "label", None)])

# arrows: the gradient forks into the lanes, each average into its correction, both corrections into the update
FX = 152
for b in ema:
    fig.arrow((gbox.right, gbox.cy), b.anchor("left"), via=[(FX, gbox.cy), (FX, b.cy)])
for a, b in zip(ema, cor):
    fig.arrow(a.anchor("right"), b.anchor("left"))
JX = 704
for b, y in zip(cor, (UY + 40, UY + UH - 40)):
    fig.arrow(b.anchor("right"), (UPD_X, y), via=[(JX, b.cy), (JX, y)])

# legend: small cards in the forms the cards above use
LX = 40
for color, label in (("gradient", "gradient"), (None, "optimiser state, kept between updates"),
                     ("blue", "bias correction, new in Adam"), ("weight", "parameter")):
    fig.card(LX, 444, 32, 20, radius=4, color=color)
    fig.text(LX + 44, 460, label, "label")
    LX += 44 + 8 * math.ceil(text_width(label, 14, weight=400) / 8) + 32
fig.caption(rich("Without the corrections the first step would be 0.05 / √0.00025 = ", unc, " learning rates."))
fig.write()
