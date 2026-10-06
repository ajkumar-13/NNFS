"""Post 05 section 3.1: a minus its per-row maximum, without keepdims (silently wrong) and with it (correct).

Run from anywhere:  python posts/05-array-summation-keepdims-and-broadcasting/diagrams/src/02-max-subtraction.py
Writes posts/05-array-summation-keepdims-and-broadcasting/diagrams/02-max-subtraction.svg.
The maxima, the stretched operand (np.broadcast_to, the array NumPy subtracts in effect) and both results are
computed here with NumPy and checked against what snippets/axis_keepdims.py prints.

Layout: two equation rows, a minus max_vals as broadcast equals the result. In each stretched operand the
cells max_vals really holds are filled and outlined; the plain cells are the copies broadcasting adds.
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, MINUS, rich, span, var, num  # noqa: E402

SNIPPETS = Path(__file__).resolve().parents[2] / "snippets"


def run_snippet(name):
    """Execute snippets/<name>.py and return (module, everything it printed)."""
    spec = importlib.util.spec_from_file_location(name, SNIPPETS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    return mod, buf.getvalue()


snip, printed = run_snippet("axis_keepdims")
a = snip.a
assert a.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

flat = np.max(a, axis=1)                         # shape (3,)
col = np.max(a, axis=1, keepdims=True)           # shape (3, 1)
assert flat.shape == (3,) and col.shape == (3, 1) and flat.tolist() == [3, 6, 9]
wrong, right = a - flat, a - col
assert f"{wrong}\n" in printed and f"{right}\n" in printed       # both results exactly as the snippet prints them
assert wrong.tolist() == [[-2, -4, -6], [1, -1, -3], [4, 2, 0]] and right.tolist() == [[-2, -1, 0]] * 3
stretch_row = np.broadcast_to(flat, a.shape)     # (3,) padded to (1, 3), then copied down the rows
stretch_col = np.broadcast_to(col, a.shape)      # (3, 1) copied across the columns
assert np.array_equal(a - stretch_row, wrong) and np.array_equal(a - stretch_col, right)
assert not right[:, -1].any() and wrong[:, -1].tolist() == [-6, -3, 0]


A, MV = span("a", mono=True), span("max_vals", mono=True)   # code names, in the code face as in the post


def say(m):
    return "; ".join(", ".join(num(int(v)).replace(MINUS, "minus ") for v in r) for r in m)


fig = Figure(
    "02-max-subtraction", "Without keepdims the row maxima line up with the columns",
    f"Two rows, each the 3 by 3 array a minus max_vals as broadcast equals the result. Top, without keepdims: "
    f"max_vals is {', '.join(str(v) for v in flat)} with shape (3,), drawn as the first row of the stretched "
    f"operand and copied down, so every row has {', '.join(str(v) for v in flat)} subtracted; the result rows "
    f"are {say(wrong)}, no error is raised, and the rows do not end in 0, because column j loses the maximum of row j. Bottom, with keepdims=True: max_vals "
    f"is a column of shape (3, 1) copied across, so row i loses its own maximum, and every row of the result "
    f"is {say(right[:1])}. A key marks the filled cells as the values max_vals holds; the plain cells are the "
    f"copies broadcasting adds.",
    subtitle=rich(span("a - np.max(a, axis=1)", mono=True), ", with and without keepdims=True."))

C = 40
GA, GS, GR = 232, 392, 552                       # a, max_vals as broadcast, the result
ROWS = (152, 320)
# column headings centred over their grids: "max_vals, broadcast" is wider than its grid
fig.text(GA + 3 * C / 2, 128, rich(A), "head", anchor="middle")
fig.text(GS + 3 * C / 2, 128, rich(MV, ", broadcast"), "head", anchor="middle")
fig.text(GR + 3 * C / 2, 128, rich(span("a - max_vals", mono=True)), "head", anchor="middle")

cases = [
    dict(head="Without keepdims", lines=[rich(MV, ": (3,)"), "copied down the rows"], stretched=stretch_row,
         own=lambda i, j: i == 0, win=(0, 0, 1, 3), result=wrong, fill="error-soft",
         verdict=("No error raised", "error"), why=["Rows do not end in 0:", rich("column ", var("j"), " loses the max of row ", var("j"), ".")]),
    dict(head="With keepdims=True", lines=[rich(MV, ": (3, 1)"), "copied across the columns"], stretched=stretch_col,
         own=lambda i, j: j == 0, win=(0, 0, 3, 1), result=right, fill="output-soft",
         verdict=("Correct", "output"), why=["Every row ends in 0:", rich("row ", var("i"), " loses its own max.")]),
]
for y, cs in zip(ROWS, cases):
    fig.text(40, y + 20, cs["head"], "head")
    for k, s in enumerate(cs["lines"]):
        fig.text(40, y + 48 + 20 * k, s, "note")
    fig.grid(GA, y, 3, 3, C, values=a.tolist())
    gs = fig.grid(GS, y, 3, 3, C, values=cs["stretched"].tolist(),
                  fill=lambda i, j, own=cs["own"]: "blue-soft" if own(i, j) else None)
    gs.window(*cs["win"], color="blue")
    gr = fig.grid(GR, y, 3, 3, C, values=cs["result"].tolist(), fill=lambda i, j, f=cs["fill"]: f)
    fig.op((GA + 3 * C + GS) // 2, y + 3 * C // 2, MINUS)          # centred in the gutters, on the grids' middle row
    fig.op((GS + 3 * C + GR) // 2, y + 3 * C // 2, "=")
    vx = gr.box.right + 24
    fig.text(vx, y + 20, cs["verdict"][0], "label", color=cs["verdict"][1])
    for k, s in enumerate(cs["why"]):
        fig.text(vx, y + 48 + 20 * k, s, "note")
# the key for the filled cells, under the max_vals column it explains, centred on it
KEY = rich("what ", MV, " holds")
fig.legend(GS + 3 * C // 2 - 92, ROWS[1] + 3 * C + 32, [dict(color="blue-soft", label=KEY, mark="swatch")])
fig.caption("The same call, one keyword apart: only the (3, 1) column subtracts each row's own maximum.")
fig.write()
