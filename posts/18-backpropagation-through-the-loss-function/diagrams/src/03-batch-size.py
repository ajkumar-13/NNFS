"""Post 18, section 6: the 1/N keeps what reaches the parameters independent of the batch size.

Run from anywhere:  python posts/18-backpropagation-through-the-loss-function/diagrams/src/03-batch-size.py
Writes posts/18-backpropagation-through-the-loss-function/diagrams/03-batch-size.svg.
snippets/gradient_check.py is run here (runpy): the batch of section 3.1 is repeated 1, 10 and 100 times with the
snippet's own arrays and loss object, and the three quantities drawn (the entry [0, 0], the class-0 column summed
over the rows with the 1/N, and the same sum without it) are checked against the three lines the snippet prints
and section 6 quotes.

Layout: one chart on log axes, batch size N across, the size of each quantity up; with log axes the three are
straight lines of slope -1, 0 and +1.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, num, MINUS  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "gradient_check.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

loss_fn, base_pred, base_true = s["loss_fn"], s["base_pred"], s["base_true"]
NS, ENTRY, WITH, WITHOUT = [], [], [], []
for copies in (1, 10, 100):
    pred, true = np.tile(base_pred, (copies, 1)), np.tile(base_true, copies)
    loss_fn.backward(pred, true)
    NS.append(len(pred))
    ENTRY.append(loss_fn.dinputs[0, 0])
    WITH.append(loss_fn.dinputs.sum(axis=0)[0])
    WITHOUT.append((loss_fn.dinputs * len(pred)).sum(axis=0)[0])
assert NS == [3, 30, 300]
for line in ("N =   3  loss 0.5202  entry [0, 0]: -0.476190  summed over rows with 1/N: [-0.4762 -0.5556 -0.6667]"
             "  without: [-1.43 -1.67 -2.  ]",
             "N =  30  loss 0.5202  entry [0, 0]: -0.047619  summed over rows with 1/N: [-0.4762 -0.5556 -0.6667]"
             "  without: [-14.29 -16.67 -20.  ]",
             "N = 300  loss 0.5202  entry [0, 0]: -0.004762  summed over rows with 1/N: [-0.4762 -0.5556 -0.6667]"
             "  without: [-142.86 -166.67 -200.  ]"):
    assert line in OUT, line
assert [f"{v:.6f}" for v in ENTRY] == ["-0.476190", "-0.047619", "-0.004762"]
assert [f"{v:.4f}" for v in WITH] == ["-0.4762"] * 3
assert [f"{v:.2f}" for v in WITHOUT] == ["-1.43", "-14.29", "-142.86"]
assert all(v < 0 for v in ENTRY + WITH + WITHOUT)
assert round(WITHOUT[2] / WITHOUT[0]) == 100                    # "100 times larger at 300 samples than at 3"

E_S = ["0.476", "0.0476", "0.00476"]                            # the entry's size, three significant figures
assert E_S == [f"{abs(v):.3g}" for v in ENTRY]
W_S = [f"{abs(v):.2f}" for v in WITHOUT]
assert W_S == ["1.43", "14.29", "142.86"]

fig = Figure(
    "03-batch-size", "Averaging keeps the summed gradient fixed as the batch grows",
    "A chart on log axes of the class-0 column of the cross-entropy gradient when the batch of section 3.1 is "
    "repeated to N = 3, 30 and 300 samples; sizes are drawn, every value is negative. Summed over the rows with "
    "the 1/N, the column is 0.4762 at every N, a flat line. A single entry, [0, 0], falls with 1/N: 0.476, 0.0476, "
    "0.00476. Summed without the 1/N, the column grows with N: 1.43, 14.29, 142.86, 100 times larger at 300 "
    "samples than at 3.",
    subtitle="The class-0 column of the gradient, batch of section 3.1 repeated. Sizes drawn; all values are negative.",
    data_w=True)

series = [dict(xs=NS, ys=[abs(v) for v in WITHOUT], color="error", shape="square", size=10,
               label=rich("row sum, without 1/", var("N"))),
          dict(xs=NS, ys=[abs(v) for v in ENTRY], color="ink-muted", shape="triangle", size=10,
               label="one entry, [0, 0]"),
          dict(xs=NS, ys=[abs(v) for v in WITH], color="gradient", shape="circle", size=10,
               label=rich("row sum, with 1/", var("N")))]       # drawn last: at N = 3 it covers the equal entry
ax = fig.line_chart(Box(40, 104, 880, 372), series, x=(2, 450, NS), y=(1e-3, 1e3, [10.0 ** e for e in range(-3, 4)]),
                    x_log=True, y_log=True, fmt_x=lambda v: num(v), label_w=184,
                    x_label=rich("batch size ", var("N"), " (the batch of section 3.1 repeated)"),
                    y_label="size of the class-0 value, log scale")

for n, v, t in zip(NS, WITHOUT, W_S):
    ax.text(n, abs(v), t, "value", anchor="middle", color="error", dy=-14)
for n, v, t in zip(NS[1:], ENTRY[1:], E_S[1:]):
    ax.text(n, abs(v), t, "value", anchor="middle", color="ink-muted", dy=26)
assert abs(WITH[0] - ENTRY[0]) < 1e-12                          # at N = 3 the column sum is the one entry
ax.text(NS[0], abs(WITH[0]), rich(E_S[0], ", both"), "value", anchor="middle", color="gradient", dy=28)
for n in NS[1:]:
    ax.text(n, abs(WITH[0]), E_S[0], "value", anchor="middle", color="gradient", dy=-14)

fig.caption(rich("What reaches the parameters is the sum; only the 1/", var("N"),
                 " keeps it from growing with the batch."))
fig.write()
