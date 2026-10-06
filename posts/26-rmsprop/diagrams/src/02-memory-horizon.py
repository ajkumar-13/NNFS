"""Post 26, section 2: how much a squared gradient k steps old still counts in the cache, for three decay factors.

Run from anywhere:  python posts/26-rmsprop/diagrams/src/02-memory-horizon.py
Writes posts/26-rmsprop/diagrams/02-memory-horizon.svg.
The curves are the formula of section 2: unrolled, the cache gives the squared gradient k steps old the weight
(1 - rho) rho^k, which is rho^k times the weight of the newest. AdaGrad's sum gives every gradient the same weight.
snippets/ema_cache.py is run here (runpy, about a second, its output captured) and the values the figure marks, rho^k
at the horizon k = 1/(1 - rho) and 1/e, are asserted against its printout and against the listing of section 2 in
index.md (which the series lint keeps equal to the snippet's output).

Layout: one chart, the relative weight against the age on a log axis.
"""
import contextlib
import io
import math
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, isup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path(str(POST / "snippets" / "ema_cache.py"), run_name="snippet")
OUT = buf.getvalue()

assert "$G_t = (1 - \\rho) \\sum_{k=0}^{t-1} \\rho^k g_{t-k}^2$" in INDEX
assert "which is $\\rho^k$ times the weight of the newest one" in INDEX
RHOS = (0.9, 0.99, 0.999)
HORIZON = {}
for rho in RHOS:
    h = round(1 / (1 - rho))
    line = (f"{rho:<7} {h:<10d} {rho ** 10:<11.4f} {rho ** 100:<11.2e} {rho ** 1000:<11.2e} "
            f"{rho ** h:<16.4f} {1 - rho ** h:.4f}")
    assert line in OUT and line in INDEX, line
    HORIZON[rho] = (h, rho ** h)
assert {r: (h, f"{v:.4f}") for r, (h, v) in HORIZON.items()} == \
       {0.9: (10, "0.3487"), 0.99: (100, "0.3660"), 0.999: (1000, "0.3677")}
E = math.exp(-1)
assert f"1/e = {E:.4f}" in OUT and f"{E:.4f}" == "0.3679"

RHO = var("ρ")
COL, ADA_C = "blue", "ink-muted"                    # RMSProp, the post's method, in blue; AdaGrad in grey
SHAPE = {0.9: "circle", 0.99: "triangle", 0.999: "diamond"}
fig = Figure(
    "02-memory-horizon", "Each decay factor forgets at its own horizon",
    "A chart of the weight a squared gradient k steps old carries in the cache, as a share of the newest one's, "
    "against the age k from 1 to 10,000 steps on a logarithmic axis. For RMSProp the share is rho to the k. With "
    "rho 0.9 it falls to 0.3487 at age 10, with 0.99 to 0.3660 at age 100, and with 0.999 to 0.3677 at age 1,000, "
    "each close to the dotted level 1/e = 0.3679; past its horizon 1/(1 minus rho) each curve drops towards zero. "
    "AdaGrad's sum is a flat line at 1: every gradient counts as much as the newest, whatever its age.",
    subtitle=rich("A squared gradient ", var("k"), " steps old: ", isup("ρ", "k"),
                  " times the newest in RMSProp, as much in AdaGrad."),
    data_w=True)

K = np.unique(np.round(np.logspace(0, 4, 400), 3))
series = [dict(xs=list(K), ys=[1.0] * len(K), color=ADA_C, points=False)]
for rho in RHOS:
    series.append(dict(xs=list(K), ys=list(rho ** K), color=COL, points=False))
ax = fig.line_chart(Box(40, 112, 880, 352), series, x=(1, 1e4, [10.0 ** k for k in range(5)]),
                    y=(0, 1.1, [0, 0.25, 0.5, 0.75, 1]), x_log=True, labels=False, label_w=120,
                    fmt_x=lambda v: num(v), fmt_y=lambda v: f"{v:g}",
                    x_label=rich("age ", var("k"), " of the squared gradient, in steps, log scale"),
                    y_label="weight as a share of the newest",
                    ref_lines=[dict(y=E, label=rich("1/", var("e"), " = 0.3679"))])
for rho in RHOS:
    h, v = HORIZON[rho]
    ax.point(h, v, SHAPE[rho], COL, size=10)
    ax.text(h, v, f"{v:.4f}", "value", color=COL, dx=10, dy=-8)
    k60 = math.log(0.6) / math.log(rho)              # the label sits left of the curve at a share of 0.6
    ax.text(k60, 0.6, rich(RHO, " = ", f"{rho:g}"), "label", color=COL, dx=-10, dy=5, anchor="end")
ax.text(1e4, 1.0, "AdaGrad", "label", color=ADA_C, dx=12, dy=5)

fig.caption(rich("Marks: the horizon ", var("k"), " = 1/(1 − ", RHO, "), where each share is close to 1/",
                 var("e"), "."))
fig.write()
