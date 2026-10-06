"""Post 16, section 4.1: the ReLU backward with and without .copy(), on the three-element example of section 4.

Run from anywhere:  python posts/16-coding-backpropagation/diagrams/src/02-copy-not-alias.py
Writes posts/16-coding-backpropagation/diagrams/02-copy-not-alias.svg.
snippets/what_can_go_wrong.py is run here (runpy): its two classes, Activation_ReLU and ReLU_Alias, are run again on
the inputs [1, -2, 3] and the caller's array [5, 6, 7], every array the figure draws is read from them, and the two
lines its section 2 prints are asserted. The code lines on the cards are asserted to be lines of the snippet.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, rich, span, num, text_width  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "what_can_go_wrong.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
assert ("Activation_ReLU  dinputs [[5. 0. 7.]]   caller's array afterwards [[5. 6. 7.]]   same array: False"
        in OUT)
assert "ReLU_Alias       dinputs [[5. 0. 7.]]   caller's array afterwards [[5. 0. 7.]]   same array: True" in OUT

INPUTS = np.array([[1.0, -2.0, 3.0]])
HANDED = np.array([[5.0, 6.0, 7.0]])
RUN = {}
for cls in (s["Activation_ReLU"], s["ReLU_Alias"]):
    relu, upstream = cls(), HANDED.copy()
    relu.forward(INPUTS.copy())
    relu.backward(upstream)
    RUN[cls.__name__] = (relu.dinputs[0].tolist(), upstream[0].tolist(), relu.dinputs is upstream)
assert RUN["Activation_ReLU"] == ([5.0, 0.0, 7.0], [5.0, 6.0, 7.0], False)
assert RUN["ReLU_Alias"] == ([5.0, 0.0, 7.0], [5.0, 0.0, 7.0], True)

SRC = [l.split("#")[0].strip() for l in SNIPPET.read_text(encoding="utf-8").splitlines()]
ALIAS = ["self.dinputs = dvalues", "self.dinputs[self.inputs <= 0] = 0"]
COPY = ["self.dinputs = dvalues.copy()", "self.dinputs[self.inputs <= 0] = 0"]
for line in ALIAS + COPY:
    assert line in SRC, line

G = "gradient"
fig = Figure(
    "02-copy-not-alias", "Without .copy(), the mask writes into the caller's array",
    "The ReLU backward of section 4 on inputs 1, minus 2 and 3, whose gate at minus 2 is closed, and the array 5, 6, 7 "
    "the caller hands in as dvalues. Two panels run the two lines of backward. Left, without .copy(): "
    "self.dinputs = dvalues makes dvalues and self.dinputs two names for one array, and after the masking line "
    "self.dinputs[self.inputs <= 0] = 0 that one array reads 5, 0, 7, so the caller's 6 has become 0, outlined in "
    "red. Right, with .copy(): two arrays, dvalues still 5, 6, 7, unchanged, and self.dinputs 5, 0, 7, masked.",
    subtitle="The three-element example of section 4, after both lines of backward have run.",
    data_w=True)

left, right = fig.row(2)
CELL = 40
SX = 176                                    # where a strip starts inside a panel, past the names


def name(x, y, t, color=None):
    """A name right-aligned at x, the left edge of what it names minus the gap."""
    fig.text(x, y, span(t, mono=True, color=color), "code", anchor="end")


def strip(x, y, vals, zero_white=True, role="gradient"):
    return fig.strip(x, y, 3, cell=CELL, values=[num(int(v)) for v in vals],
                     fill=lambda j: None if (zero_white and vals[j] == 0) else role + "-soft")


# -- what backward is handed
YT = 112
g_in = strip(left.x + SX, YT, INPUTS[0].tolist(), zero_white=False, role="input")
name(g_in.box.x - 16, YT + 25, "self.inputs", "input")
fig.text(g_in.box.right + 16, YT + 25, rich("gate closed at ", num(-2)), "note")
g_dv = strip(right.x + SX, YT, HANDED[0].tolist())
name(g_dv.box.x - 16, YT + 25, "dvalues", G)
fig.text(g_dv.box.right + 16, YT + 25, "from the caller", "note")

# -- the two versions
YH = 216
for panel, head, lines in ((left, "Without ", ALIAS), (right, "With ", COPY)):
    fig.text(panel.x, YH, head, "head")
    fig.text(panel.x + round(0.9 * text_width(head.strip(), 16, weight=600)) + 5, YH, ".copy()", "code16")
    fig.card(panel.x, YH + 16, panel.w, None, lines=lines, style="code", fit=True)

# left: one array, two names
YA = 368
ga = strip(left.x + SX, YA, RUN["ReLU_Alias"][1])
assert RUN["ReLU_Alias"][1] == RUN["ReLU_Alias"][0]                # the same array under both names
name(ga.box.x - 36, YA + 4, "dvalues", G)
name(ga.box.x - 36, YA + 44, "self.dinputs", G)
fig.brace(YA - 12, YA + CELL + 12, ga.box.x - 28, side="right", kind="bracket", vertical=True)
ga.window(1, 1, color="error")
fig.text(ga.box.right + 16, YA + 17, "one array,", "note")
fig.text(ga.box.right + 16, YA + 37, "two names", "note")
fig.note(ga.box, rich("the caller's 6 is now 0"))

# right: two arrays
YB0, YB1 = 336, 400
gb0 = strip(right.x + SX, YB0, RUN["Activation_ReLU"][1])
gb1 = strip(right.x + SX, YB1, RUN["Activation_ReLU"][0])
name(gb0.box.x - 16, YB0 + 25, "dvalues", G)
name(gb1.box.x - 16, YB1 + 25, "self.dinputs", G)
fig.text(gb0.box.right + 16, YB0 + 25, "unchanged", "note")
fig.text(gb1.box.right + 16, YB1 + 25, "masked", "note")

fig.caption("Both versions return the same gradient; only the copy leaves the caller's array as it was.")
fig.write()
