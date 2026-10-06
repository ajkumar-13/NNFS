"""Post 26 hero (sections 2 and 3): AdaGrad's sum of squared gradients against RMSProp's moving average.

Run from anywhere:  python posts/26-rmsprop/diagrams/src/01-sum-against-average.py
Writes posts/26-rmsprop/diagrams/01-sum-against-average.svg.
snippets/ema_cache.py is run here (runpy, about a second; it prints as it loads, and its output is captured). Its own
adagrad_cache and rmsprop_cache give every curve: the constant gradient 0.5 for 100,000 steps with rho = 0.9 (left),
and the gradient 2 for 1,000 steps then 0.1 for 20,000, with rho = 0.9 and 0.999 (right). Every marked value is
asserted against the snippet's printout and against section 3 of index.md (its table, and the listing of the second
stream, which the series lint keeps equal to the snippet's output).

Layout: two charts side by side, the constant gradient left (log-log) and the falling gradient right (log cache).
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(POST / "snippets" / "ema_cache.py"), run_name="snippet")
OUT = buf.getvalue()
adagrad_cache, rmsprop_cache = s["adagrad_cache"], s["rmsprop_cache"]

# -- the two rules, as the post writes them
assert "$$G_t = \\rho \\, G_{t-1} + (1 - \\rho) \\, g_t^2$$" in INDEX
assert "its cache $G_t = \\sum_{s \\le t} g_s^2$ can only grow" in INDEX

# -- left: a gradient of constant size 0.5, section 3's table
G1 = np.full(100_000, 0.5)
ADA1, RMS1 = adagrad_cache(G1), rmsprop_cache(G1, rho=0.9)
TS1 = (1, 2, 10, 100, 1_000, 10_000, 100_000)
for t in TS1:
    a, r = ADA1[t - 1], RMS1[t - 1]
    assert (f"{t:6d}   {a:9.3f}   {1 / np.sqrt(a):9.4f}   {r:9.4f}   {1 / np.sqrt(r):9.4f}") in OUT, t
    row = f"| {num(t)} | {num(round(a, 3), 3)} | {1 / np.sqrt(a):.4f} | {r:.4f} | {1 / np.sqrt(r):.4f} |"
    assert row in INDEX, row
assert np.allclose(ADA1, 0.25 * np.arange(1, 100_001)) and np.allclose(RMS1, 0.25 * (1 - 0.9 ** np.arange(1, 100_001)))
FIRST = int(np.argmax(RMS1 > 0.99 * 0.25)) + 1
assert FIRST == 44 and "first t at which RMSProp's cache is within 1 percent of 0.25: 44" in OUT
assert "it is within 1 percent of 0.25 from step 44 on" in INDEX

# -- right: 2 for 1,000 steps, then 0.1 for 20,000
G2 = np.concatenate([np.full(1_000, 2.0), np.full(20_000, 0.1)])
ADA2, FAST, SLOW = adagrad_cache(G2), rmsprop_cache(G2, rho=0.9), rmsprop_cache(G2, rho=0.999)
TS2 = (1_000, 1_010, 1_050, 1_100, 2_000, 11_000, 21_000)
for t in TS2:
    line = f"{t:6d}   {ADA2[t - 1]:9.2f}   {FAST[t - 1]:11.4f}   {SLOW[t - 1]:13.4f}"
    assert line in OUT, line
    if t not in (1_010, 21_000):
        assert line in INDEX, line                    # the listing of section 3 shows five of the seven rows
SETTLE = {rho: int(np.argmax(c[1_000:] < 1.01 * 0.01)) + 1 for rho, c in ((0.9, FAST), (0.999, SLOW))}
assert SETTLE == {0.9: 101, 0.999: 10_130}
for rho, n in SETTLE.items():
    line = f"rho {rho}: within 1 percent of the new level 0.01 after {n:,} steps"
    assert line in OUT and line in INDEX, line
MARK2 = (1_000, 2_000, 11_000, 21_000)
assert [f"{FAST[t - 1]:.4f}" for t in MARK2] == ["4.0000", "0.0100", "0.0100", "0.0100"]
assert [f"{SLOW[t - 1]:.4f}" for t in MARK2] == ["2.5292", "0.9363", "0.0101", "0.0100"]
assert [f"{ADA2[t - 1]:.2f}" for t in MARK2] == ["4000.00", "4010.00", "4100.00", "4200.00"]
assert 0.0099 < FAST[1_200:].min() and SLOW.max() < 4 and FAST.max() <= 4 + 1e-12

RHO, G = var("ρ"), var("g")
ADA_C, RMS_C = "ink-muted", "blue"                     # optimiser state: AdaGrad in grey, this post's RMSProp in blue
fig = Figure(
    "01-sum-against-average", "The sum only grows; the average settles and can fall again",
    "Two charts of the cache G against the step t, both starting at 0. Left, on log-log axes, a gradient of constant "
    "size 0.5 for 100,000 steps: AdaGrad's sum is 0.25 t, 0.25 at step 1, 2.5 at step 10, 2,500 at step 10,000 and "
    "25,000 at step 100,000; RMSProp's average with rho 0.9 is 0.025 at step 1 and 0.1628 at step 10, is within 1 "
    "percent of 0.25 from step 44 and stays at 0.25. Right, the gradient is 2 for 1,000 steps and 0.1 afterwards, "
    "on a log cache axis: AdaGrad holds 4,000 at step 1,000 and 4,200 at step 21,000; RMSProp with rho 0.9 falls "
    "from 4 to within 1 percent of the new level 0.01 in 101 steps; with rho 0.999 it has reached only 2.5292 at "
    "step 1,000, is at 0.9363 at step 2,000 and needs 10,130 steps to settle at 0.01. Marks: steps the script prints.",
    subtitle=rich("AdaGrad: ", var("G"), " ← ", var("G"), " + ", sup("g", "2"), ".   RMSProp: ", var("G"), " ← ",
                  RHO, var("G"), " + (1 − ", RHO, ")", sup("g", "2"), ".   Both caches start at 0."),
    data_w=True)

left, right = fig.row([5, 6], y=104, h=344)

# -- left
body = fig.panel(left, rich("A constant gradient, |", G, "| = 0.5"))
T1 = np.unique(np.round(np.logspace(0, 5, 301)).astype(int))
ax = fig.line_chart(body, [dict(xs=list(T1), ys=list(ADA1[T1 - 1]), color=ADA_C, points=False),
                           dict(xs=list(T1), ys=list(RMS1[T1 - 1]), color=RMS_C, points=False)],
                    x=(1, 1e5, [10.0 ** k for k in range(6)]), y=(1e-2, 1e5, [10.0 ** k for k in range(-2, 6)]),
                    x_log=True, y_log=True, labels=False, label_w=72, x_label=rich("step ", var("t"), ", log scale"),
                    y_label=rich("cache ", var("G"), ", log scale"))
for t in TS1:
    ax.point(t, ADA1[t - 1], "square", ADA_C, size=8)
    ax.point(t, RMS1[t - 1], "circle", RMS_C, size=8)
ax.text(1e5, ADA1[-1], num(25000), "value", color=ADA_C, dx=10, dy=5)
ax.text(1e5, RMS1[-1], "0.25", "value", color=RMS_C, dx=10, dy=5)
ax.text(1e3, ADA1[999], rich(var("G"), " = 0.25 ", var("t")), "note", dx=-12, dy=-10, anchor="end")
ax.text(100, 0.25, "within 1 percent of 0.25 from step 44", "note", color=RMS_C, dx=-8, dy=26)

# -- right
body = fig.panel(right, rich("|", G, "| falls from 2 to 0.1 at step 1,000"))
T2 = np.arange(1, 21_001)
ax = fig.line_chart(body, [dict(xs=list(T2), ys=list(ADA2), color=ADA_C, points=False),
                           dict(xs=list(T2), ys=list(SLOW), color=RMS_C, points=False),
                           dict(xs=list(T2), ys=list(FAST), color=RMS_C, points=False)],
                    x=(0, 21_000, [0, 5_000, 10_000, 15_000, 20_000]), y=(1e-3, 1e4, [10.0 ** k for k in range(-3, 5)]),
                    y_log=True, labels=False, label_w=64, x_label=rich("step ", var("t")),
                    y_label=rich("cache ", var("G"), ", log scale"), ref_lines=[dict(x=1_000)])
for t in MARK2:
    ax.point(t, ADA2[t - 1], "square", ADA_C, size=8)
    ax.point(t, SLOW[t - 1], "diamond", RMS_C, size=10, hollow=True)
    if t > 1_000:
        ax.point(t, FAST[t - 1], "circle", RMS_C, size=8)
ax.text(21_000, ADA2[-1], num(4200), "value", color=ADA_C, dx=10, dy=5)
ax.text(21_000, FAST[-1], "0.01", "value", color=RMS_C, dx=10, dy=5)
ax.text(1_000, ADA2[999], "4,000 at step 1,000", "note", color=ADA_C, dx=12, dy=26)
ax.text(1_000, SLOW[999], rich(RHO, " = 0.999: 2.5292 at step 1,000"), "note", color=RMS_C, dx=14, dy=-10)
ax.text(2_000, SLOW[1_999], "0.9363 at step 2,000", "note", color=RMS_C, dx=12, dy=5)
ax.text(11_000, 0.01, rich(RHO, " = 0.999: 10,130 steps"), "note", color=RMS_C, dy=-14, anchor="middle")
ax.text(1_000, 0.01, rich(RHO, " = 0.9: 101 steps"), "note", color=RMS_C, dx=12, dy=24)

fig.legend(96, 476, [dict(color=ADA_C, label="AdaGrad", mark="square"),
                     dict(color=RMS_C, label=rich("RMSProp, ", RHO, " = 0.9"), mark="circle"),
                     dict(color=RMS_C, label=rich("RMSProp, ", RHO, " = 0.999"), mark="diamond", hollow=True)],
           direction="row", gap=40)
fig.caption("Marks: steps the script prints. Step counts: steps after the change until within 1 percent of 0.01.")
fig.write()
