"""Post 13 section 8.1: in the 200-iteration loop the three pre-activations fall together, and the gates close in
order of size, neuron 1 at iteration 3 and neuron 2 at iteration 12.

Run from anywhere:  python posts/13-backprop-through-a-layer/diagrams/src/04-gates-closing.py
Writes posts/13-backprop-through-a-layer/diagrams/04-gates-closing.svg.
snippets/training_loop.py is run here and its printout kept. The loop is then rerun with the snippet's own
relu, relu_deriv and learning rate, from the arrays of snippets/layer_backward.py, recording Z at every
iteration; the rerun must print exactly the lines the snippet printed, so the curves are the snippet's.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, sub, isub, num  # noqa: E402

HERE = Path(__file__).resolve().parents[2] / "snippets"
with contextlib.redirect_stdout(io.StringIO()) as out:
    T = runpy.run_path(str(HERE / "training_loop.py"))
PRINTED = out.getvalue()
with contextlib.redirect_stdout(io.StringIO()):
    S = runpy.run_path(str(HERE / "layer_backward.py"))

relu, relu_deriv, lr = T["relu"], T["relu_deriv"], T["lr"]
inputs = np.array(S["inputs"])
weights = np.array(S["weights"])
biases = np.array(S["biases"])
assert lr == 0.001

ZS, lines, previous_gates = [], [], None
for i in range(200):                      # the snippet's loop, line for line, with Z recorded
    Z = weights @ inputs + biases
    ZS.append(Z.copy())
    A = relu(Z)
    Y = np.sum(A)
    L = Y ** 2
    dL_dZ = 2 * Y * np.ones_like(A) * relu_deriv(Z)
    dL_dW = dL_dZ.reshape(-1, 1) * inputs
    gates = relu_deriv(Z).astype(int).tolist()
    if i % 40 == 0 or i == 199 or gates != previous_gates:
        lines.append(f"iter {i:3d}  loss = {L:.6f}  Y = {Y:.6f}  gates = {gates}")
    previous_gates = gates
    weights -= lr * dL_dW
    biases -= lr * dL_dZ
assert PRINTED.startswith("\n".join(lines) + "\n"), "the rerun must print what the snippet printed"
ZS = np.array(ZS)
assert "iter   3  loss = 140.818219  Y = 11.866685  gates = [0, 1, 1]" in PRINTED
assert "iter  12  loss = 14.840513  Y = 3.852339  gates = [0, 0, 1]" in PRINTED
CLOSE1, CLOSE2 = 3, 12
assert ZS[CLOSE1 - 1, 0] > 0 > ZS[CLOSE1, 0] and ZS[CLOSE2 - 1, 1] > 0 > ZS[CLOSE2, 1]
assert f"{ZS[CLOSE1, 0]:.4f}" == "-0.2167" and f"{ZS[CLOSE2, 1]:.4f}" == "-0.2477"   # section 8.1
assert np.all(ZS[CLOSE1:, 0] == ZS[CLOSE1, 0]) and np.all(ZS[CLOSE2:, 1] == ZS[CLOSE2, 1])  # frozen
# while all three gates are open the three fall by the same amount each step (section 6)
drops = ZS[:CLOSE1 - 1] - ZS[1:CLOSE1]
assert np.allclose(drops, drops[:, :1])
assert abs((ZS[0, 0] - ZS[1, 0]) - 1.3392) < 1e-9

N = 24                                     # the iterations drawn
its = list(range(N + 1))
assert ZS[N, 2] > 0
# the constant gap: while both gates are open, z3 and z2 stay 4.1 apart
GAP_AT = 8
GAP = ZS[GAP_AT, 2] - ZS[GAP_AT, 1]
assert np.allclose(ZS[:CLOSE2, 2] - ZS[:CLOSE2, 1], 4.1) and num(GAP, 1) == "4.1"


def z(k):
    return sub("z", str(k))


fig = Figure(
    "04-gates-closing", "The smallest pre-activation runs out first",
    f"A line chart of the three pre-activations z1, z2 and z3 over iterations 0 to {N} of the training loop, "
    f"learning rate 0.001, on a vertical axis from minus 4 to 12. They start at 3.1, 7.2 and 11.3 and fall by "
    f"the same amount per step while their gates are open; a brace at iteration {GAP_AT} marks z2 and z3 as "
    f"{num(GAP, 1)} apart. z1 crosses zero first: at iteration 3 it is "
    f"minus 0.2167, its gate closes and the line stays flat. z2 crosses next: at iteration 12 it is minus "
    f"0.2477 and stays there. Two red diamonds mark the closings, labelled gate 1 closes at iteration 3 and gate 2 "
    f"closes at iteration 12. z3 keeps falling towards zero, at {ZS[N, 2]:.2f} by iteration {N}, labelled as the "
    f"only open gate. A dotted line marks zero.",
    subtitle="Section 8.1: the training loop, learning rate 0.001, first 24 of its 200 iterations.")

COL = ("blue", "orange", "green")
series = [dict(xs=its, ys=ZS[:N + 1, k].tolist(), color=COL[k], points=False, label=None) for k in range(2)]
series.append(dict(xs=its, ys=ZS[:N + 1, 2].tolist(), color=COL[2], points=False,
                   label=rich(z(3), ", the only open gate")))
ax = fig.line_chart(Box(40, 104, 880, 372), series, x=(0, N, [0, 3, 6, 9, 12, 15, 18, 21, 24]),
                    y=(-4, 12, [-4, 0, 4, 8, 12]), x_label="iteration",
                    y_label=rich("pre-activation ", isub("z", "k")), label_w=176, points=False, ref_lines=[dict(y=0, label=None)])

# the two dying neurons, labelled where their lines start to fall
ax.text(1, ZS[1, 0], rich(z(1), ", neuron 1"), "label", color=COL[0], dx=12, dy=-6)
ax.text(3, ZS[3, 1], rich(z(2), ", neuron 2"), "label", color=COL[1], dx=12, dy=-6)
# the brace over the constant gap
with fig.data():
    gx, top = ax.to_px(GAP_AT, ZS[GAP_AT, 2])
    _, bot = ax.to_px(GAP_AT, ZS[GAP_AT, 1])
    fig.brace(top + 4, bot - 4, gx + 4, label=rich(num(GAP, 1), " apart"), side="right",
              vertical=True)
# the closings
for k, it in ((0, CLOSE1), (1, CLOSE2)):
    ax.point(it, ZS[it, k], "diamond", "negative", size=10)
    ax.text(it, ZS[it, k], f"gate {k + 1} closes, iteration {it}", "note", dx=-4, dy=28)

fig.caption("Open neurons fall by the same amount per step, so their gates close in order of size.")
fig.write()
