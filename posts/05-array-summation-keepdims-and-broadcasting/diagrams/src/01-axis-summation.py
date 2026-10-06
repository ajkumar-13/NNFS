"""Post 05 hero (sections 2 and 3): np.sum on the 3 x 3 array a removes the named axis; keepdims keeps it at size 1.

Run from anywhere:  python posts/05-array-summation-keepdims-and-broadcasting/diagrams/src/01-axis-summation.py
Writes posts/05-array-summation-keepdims-and-broadcasting/diagrams/01-axis-summation.svg.
Every sum and shape is computed here with NumPy and checked against what snippets/axis_keepdims.py prints.

Layout: on the left, a with its two axes; the keepdims=True results sit where their axis collapsed, the
(1, 3) column sums under a's columns and the (3, 1) row sums beside a's rows. On the right, the default
results: both 1-D sums lie flat with the same shape (3,), and the full sum is the scalar 45, shape ().
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Box, Figure, rich, var  # noqa: E402

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

total = np.sum(a)
s0, s1 = np.sum(a, axis=0), np.sum(a, axis=1)
k0, k1 = np.sum(a, axis=0, keepdims=True), np.sum(a, axis=1, keepdims=True)
assert np.shape(total) == () and s0.shape == s1.shape == (3,)
assert k0.shape == (1, 3) and k1.shape == (3, 1)
for arr in (total, s0, s1, k0, k1):                       # each value exactly as the snippet prints it
    assert f"{arr}\n" in printed, arr
assert "(3,) (3,)\n" in printed and "(1, 3) (3, 1)\n" in printed
assert (int(total), s0.tolist(), s1.tolist()) == (45, [12, 15, 18], [6, 15, 24])


def shape(t):
    return "(" + ", ".join(str(n) for n in t) + ("," if len(t) == 1 else "") + ")"


def ints(v):
    return ", ".join(str(int(x)) for x in np.ravel(v))


fig = Figure(
    "01-axis-summation", "The axis named in a reduction is the axis that disappears",
    f"The post's array a, shape {shape(a.shape)}, holding 1 to 9, with axis 0 running down its rows and axis 1 "
    f"across its columns. With keepdims=True the reduced axis stays at size 1: the column sums {ints(k0)} sit "
    f"under a's columns as a row of shape {shape(k0.shape)} (axis=0), and the row sums {ints(k1)} sit beside a's "
    f"rows as a column of shape {shape(k1.shape)} (axis=1). Without keepdims, np.sum(a, axis=0) gives {ints(s0)} "
    f"and np.sum(a, axis=1) gives {ints(s1)}, both drawn flat with the same shape {shape(s0.shape)}, and np.sum(a) "
    f"gives the scalar {int(total)}, shape (). A note says that the 1-D shape does not record which axis was "
    f"summed.",
    subtitle=rich("np.sum on the post's array ", var("a"), ", with keepdims=True on the left and without it on the right."))

C = 48
AX, AY = 184, 176                       # the array a
fig.text(40, 128, "keepdims=True: the axis stays, size 1", "head")
ga = fig.grid(AX, AY, 3, 3, C, values=a.tolist())
# the two axes of a, as arrows along its top and its left edge
fig.arrow((AX, AY - 16), (ga.box.right, AY - 16))
fig.text(ga.box.cx, AY - 28, "axis 1", "label", anchor="middle")
fig.arrow((AX - 16, AY), (AX - 16, ga.box.bottom))
fig.text(AX - 28, ga.box.cy + 4, "axis 0", "label", anchor="end")
fig.text(AX - 28, AY + 29, rich(var("a"), ", (3, 3)"), "note", anchor="end")

# keepdims=True: the column sums under the columns, the row sums beside the rows
gk1 = fig.grid(ga.box.right + 32, AY, 3, 1, C, values=k1.tolist(), fill=lambda i, j: "output-soft")
gk0 = fig.grid(AX, ga.box.bottom + 32, 1, 3, C, values=k0.tolist(), fill=lambda i, j: "output-soft")
fig.text(gk1.box.right + 16, AY + 20, "axis=1", "code")
fig.text(gk1.box.right + 16, AY + 44, "row sums", "note")
fig.text(gk1.box.right + 16, gk1.box.bottom - 8, "(3, 1)", "code", color="output")
fig.text(AX, gk0.box.bottom + 28, "axis=0", "code")
fig.text(AX + 60, gk0.box.bottom + 28, "column sums", "note")
fig.text(gk0.box.right + 16, gk0.box.cy + 5, "(1, 3)", "code", color="output")

# the default: both 1-D results lie flat; the full sum is a scalar
RX = 600
fig.text(RX, 128, "Default: the axis is gone", "head")
rows = [("np.sum(a, axis=0)", s0, 184), ("np.sum(a, axis=1)", s1, 280), ("np.sum(a)", [int(total)], 376)]
for call, vals, y in rows:
    fig.text(RX, y - 16, call, "code")
    n = len(np.ravel(vals))
    g = fig.strip(RX, y, len(np.ravel(vals)), C, values=[int(v) for v in np.ravel(vals)],
                  fill=lambda k: "output-soft")
    fig.text(RX + n * C + 16, y + 29, shape(np.shape(vals if call != "np.sum(a)" else total)), "code",
             color="output")
fig.note(Box(RX, 376, 0, C), ["Both 1-D results have shape (3,):", "nothing records which axis was summed."])
fig.caption("Kept at size 1, each sum stays lined up with the rows or columns it came from.")
fig.write()
