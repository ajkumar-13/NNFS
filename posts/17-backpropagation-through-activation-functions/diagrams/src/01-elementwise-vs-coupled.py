"""Post 17 hero: the same row-times-Jacobian product for sigmoid (diagonal) and softmax (full), one sample.

Run from anywhere:  python posts/17-backpropagation-through-activation-functions/diagrams/src/01-elementwise-vs-coupled.py
Writes posts/17-backpropagation-through-activation-functions/diagrams/01-elementwise-vs-coupled.svg.
Both rows use z = (1, -2, 3) and dvalues = (5, 6, 7). The top row is section 3.2: the sigmoid Jacobian is built
here with the snippet's Activation_Sigmoid (snippets/elementwise_backward.py, run here) and the product is checked
against the element-wise line; both are asserted against what the snippet prints. The bottom row is section 4:
snippets/softmax_is_coupled.py is run and its Jacobian, full product, central difference, element-wise line and
gap are read from it and asserted against its printout and the numbers the post states.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, var, sub, isub, arr, CDOT  # noqa: E402

MINUS = "\u2212"
SNIPS = Path(__file__).resolve().parents[2] / "snippets"


def run(name):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = runpy.run_path(str(SNIPS / name), run_name="snippet")
    return ns, buf.getvalue()


E, OUT_E = run("elementwise_backward.py")
S, OUT_S = run("softmax_is_coupled.py")

# -- top row: sigmoid, section 3.2
Z = np.array([[1.0, -2.0, 3.0]])
DV = np.array([[5.0, 6.0, 7.0]])
sig = E["Activation_Sigmoid"]()
sig.forward(Z)
sig.backward(DV)
SLOPES = sig.output[0] * (1 - sig.output[0])
J_SIG = np.diagflat(SLOPES)
TOP_RES = DV[0] @ J_SIG
assert np.allclose(TOP_RES, sig.dinputs[0])                       # the product is the element-wise line
assert "[[0.196612 0.000000 0.000000]\n [0.000000 0.104994 0.000000]\n [0.000000 0.000000 0.045177]]" in OUT_E
assert "non-zero entries: 3 of 9" in OUT_E
TOP_TXT = [f"{v:.5f}" for v in TOP_RES]
assert TOP_TXT == ["0.98306", "0.62996", "0.31624"]
assert f"dvalues @ jacobian      = [{' '.join(TOP_TXT)}]" in OUT_E
assert f"dvalues * f_prime(Z)    = [{' '.join(TOP_TXT)}]" in OUT_E

# -- bottom row: softmax, section 4
J_SM, FULL, NUMER, DIAG = S["jacobian"], S["full"], S["numerical"], S["diagonal_only"]
assert S["z"].tolist() == [1.0, -2.0, 3.0] and S["dvalues"].tolist() == [5.0, 6.0, 7.0]
assert np.count_nonzero(J_SM) == 9 and np.allclose(J_SM.sum(axis=0), 0.0)
assert np.allclose(J_SM, np.diagflat(S["a"]) - np.outer(S["a"], S["a"]))
assert ("[[ 0.104457 -0.000699 -0.103758]\n [-0.000699  0.005865 -0.005166]\n [-0.103758 -0.005166  0.108924]]"
        in OUT_S)
BOT_TXT = [f"{v:.6f}" for v in FULL]
assert BOT_TXT == ["-0.208216", "-0.004467", "0.212683"]
assert "full Jacobian product : [-0.208216 -0.004467  0.212683]" in OUT_S
assert "central difference    : [-0.208216 -0.004467  0.212683]" in OUT_S
DIAG_TXT = [f"{v:.6f}" for v in DIAG]
assert DIAG_TXT == ["0.522287", "0.035190", "0.762469"]
assert "element-wise line     : [0.522287 0.035190 0.762469]" in OUT_S
GAP = np.abs(DIAG - NUMER).max()
assert f"{GAP:.3f}" == "0.731" and "element-wise line against the central difference    : 0.731" in OUT_S
assert np.abs(FULL - NUMER).max() < 1e-9
assert int(np.sum(np.sign(DIAG) != np.sign(NUMER))) == 2          # "for the first two inputs, in sign"


def m(s):
    return s.replace("-", MINUS)


def d(top, bot):
    return rich("\u2202", top, "/\u2202", bot)


def cell(v):
    return "0" if v == 0 else m(f"{v:.6f}")


def say(J):
    return "; ".join(", ".join(cell(v).replace(MINUS, "minus ") for v in row) for row in J)


fig = Figure(
    "01-elementwise-vs-coupled", "A diagonal Jacobian is one multiply; a full one is not",
    "Two rows, each the row dvalues = 5, 6, 7 times a 3 by 3 Jacobian whose entry (k, j), row a_k and column z_j, is d a_k / d z_j, with its diagonal outlined, at "
    "z = 1, minus 2, 3. Top, sigmoid: only the diagonal is non-zero, 0.196612, 0.104994 and 0.045177, the other six "
    "entries are 0, and the product 0.98306, 0.62996, 0.31624 equals dvalues * f_prime(Z), one multiply per entry. "
    f"Bottom, softmax: all nine entries are non-zero ({say(J_SM)}), a_k(1 minus a_k) on the diagonal and minus a_k a_j off it, and the product is minus 0.208216, minus "
    "0.004467, 0.212683, which the central difference matches. The element-wise line would give 0.522287, "
    "0.035190, 0.762469, off by up to 0.731.",
    subtitle=rich("One sample at ", var("z"), " = (1, ", MINUS, "2, 3) with dvalues = (5, 6, 7), as in sections 3.2 "
                  "and 4."),
    height=720, data_w=True)

XV, CV = 40, 48            # the dvalues strip
XJ, CJ = 256, 104          # the Jacobian
XR, CR = 616, 96           # the result
CH = 40


def block(top, head, J, res, notes_j, notes_r):
    fig.text(40, top - 64, head, "head")
    g = fig.grid(XJ, top, 3, 3, cell_w=CJ, cell_h=CH, values=lambda k, j: cell(J[k][j]), font=14,
                 fill=lambda k, j: "output-soft" if J[k][j] != 0 else None)
    g.col_labels([sub("z", str(j + 1)) for j in range(3)])
    g.row_labels([sub("a", str(k + 1)) for k in range(3)])
    for k in range(3):                  # the diagonal, outlined in both grids
        g.outline(k, k, 1, 1, "output", 1.5, form="grid")
    fig.text(g.box.cx, top - 32, rich("Jacobian ", d(isub("a", "k"), isub("z", "j")), "  (3, 3)"), "label",
             anchor="middle", color="output")
    v = fig.strip(XV, top + CH, 3, cell_w=CV, cell_h=CH, values=["5", "6", "7"], font=14,
                  fill=lambda i: "gradient-soft")
    v.col_labels([sub("a", str(k + 1)) for k in range(3)])
    fig.text(v.box.cx, v.box.y - 32, rich("dvalues ", d(var("L"), arr_a), "  (1, 3)"), "label", anchor="middle",
             color="gradient")
    fig.op(v.box.right + 20, top + 1.5 * CH, CDOT)      # clear of the row labels a1 to a3
    fig.op((g.box.right + XR) / 2, top + 1.5 * CH, "=")
    r = fig.strip(XR, top + CH, 3, cell_w=CR, cell_h=CH, values=[m(t) for t in res], font=14,
                  fill=lambda i: "gradient-soft")
    r.col_labels([sub("z", str(j + 1)) for j in range(3)])
    fig.text(r.box.cx, r.box.y - 32, rich("dinputs ", d(var("L"), arr_z), "  (1, 3)"), "label", anchor="middle",
             color="gradient")
    fig.note(g.box, notes_j)
    if notes_r is None:                 # the element-wise identity: "=" and the code line, set apart
        fig.text(r.box.x, r.box.bottom + 28, "=", "note")
        fig.text(r.box.x + 16, r.box.bottom + 28, "dvalues * f_prime(Z)", "code")
        fig.text(r.box.x, r.box.bottom + 48, "one multiply per entry", "note")
    else:
        fig.note(r.box, notes_r)
    return g, r


arr_a, arr_z = arr("A"), arr("Z")

block(200, "Sigmoid, element-wise: 3 of 9 entries non-zero", J_SIG, TOP_TXT,
      [rich("outlined, the diagonal: ", var("f"), " ′(", isub("z", "k"), ")"),
       rich("off it: 0, as ", isub("a", "k"), " reads ", isub("z", "k"), " alone")],
      None)
g2, r2 = block(480, "Softmax, coupled: 9 of 9 entries non-zero", J_SM, BOT_TXT,
               [rich("outlined, the diagonal: ", isub("a", "k"), "(1 ", MINUS, " ", isub("a", "k"), ")"),
                rich("off it: ", MINUS, isub("a", "k"), isub("a", "j"), ", never 0")],
               ["matches the central difference",
                "the element-wise line misses it by 0.731"])

fig.caption("Off-diagonal zeros reduce the product to one multiply per entry; softmax has none to drop.")
fig.write()
