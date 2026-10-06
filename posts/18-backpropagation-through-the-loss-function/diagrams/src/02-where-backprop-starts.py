"""Post 18, sections 1 and 5: the loss is the one class with nothing after it, so its dvalues holds the predictions.

Run from anywhere:  python posts/18-backpropagation-through-the-loss-function/diagrams/src/02-where-backprop-starts.py
Writes posts/18-backpropagation-through-the-loss-function/diagrams/02-where-backprop-starts.svg.
snippets/loss_backward.py is run here (runpy): every code line in the class card is asserted to be a line of the
snippet's Loss_CategoricalCrossentropy, and the shapes are read from the batch the snippet runs it on (N = 3,
K = 3), where dvalues is the prediction array itself and dinputs comes back in its shape.

Layout: on top the classifier of section 1 as a row of cards, forward left to right and backward right to left,
the loss outlined; under it the loss class enlarged, with what comes in from the left, what goes back to the left,
the loss L leaving to the right and, where every other class receives its gradient, nothing.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, arr, hat, span  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "loss_backward.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()

loss_fn, P = s["loss_fn"], s["y_pred"]
loss_fn.backward(P, s["y_true_indices"])
assert loss_fn.dinputs.shape == P.shape == (3, 3)               # dinputs has the shape of the predictions
assert s["y_true_indices"].shape == (3,) and s["y_true"].shape == (3, 3)
assert not any(k for k in vars(loss_fn) if k != "dinputs")      # nothing but dinputs is stored on the object
assert "shape: (3, 3)" in OUT

# the class as the snippet has it: forward shortened to its first and last lines, backward in full
SRC = SNIPPET.read_text(encoding="utf-8").splitlines()
start = SRC.index("class Loss_CategoricalCrossentropy(Loss):")
CLASS = [l[4:] for l in SRC[start + 1:] if l.startswith("    ")]
FWD = ["def forward(self, y_pred, y_true):",
       "    samples        = len(y_pred)",
       "    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)",
       "    ...",
       "    return negative_log_likelihoods"]
BWD = ["def backward(self, dvalues, y_true):",
       "    samples = len(dvalues)",
       "    labels  = len(dvalues[0])",
       "    if len(y_true.shape) == 1:",
       "        y_true = np.eye(labels)[y_true]",
       "    self.dinputs = -y_true / dvalues",
       "    self.dinputs = self.dinputs / samples"]
for line in FWD + BWD:
    assert line.strip() == "..." or line in CLASS, line
assert CLASS.index(BWD[0]) > CLASS.index(FWD[-1])               # forward above backward, as in the class


def code(t, color=None):
    return span(t, color=color, mono=True)


YH = hat(arr("y"))
NK = rich("(", var("N"), ", ", var("K"), ")")

fig = Figure(
    "02-where-backprop-starts", "The loss is the one class with nothing after it",
    "Top: the classifier of section 1 as a row of cards, Dense, ReLU, Dense, Softmax and Loss, with forward arrows "
    "left to right, y-hat entering the loss and L leaving it, and backward arrows right to left in purple, the "
    "first being the loss's dinputs to the softmax. To the right of the loss a dashed box holds nothing. Below, "
    "the class Loss_CategoricalCrossentropy enlarged: forward(self, y_pred, y_true) clips a copy and returns the "
    "per-sample losses, whose mean is L; backward(self, dvalues, y_true) converts integer labels with np.eye, "
    "divides minus y_true by dvalues and then by samples. From the left, y-hat of shape (N, K) and y_true arrive "
    "at both methods, y-hat as backward's dvalues, and dinputs of shape (N, K) goes back to the softmax as its dvalues. From the right nothing "
    "arrives: the upstream gradient is dL/dL = 1, so the dvalues slot carries y-hat. Nothing is cached.",
    subtitle=rich("The classifier of section 1 and the loss class of section 5, for a batch of ", var("N"),
                  " samples and ", var("K"), " classes."),
    height=720, data_w=True)

# ---------------------------------------------------------------- the pipeline
YP, HP = 128, 64
XW = [(40, 104, "Dense"), (184, 96, "ReLU"), (320, 104, "Dense"), (464, 112, "Softmax"), (656, 104, "Loss")]
cards = []
for x, w, name in XW:
    fig.card(x, YP, w, HP)
    fig.text(x + w / 2, YP + 38, name, "head", anchor="middle")
    cards.append(Box(x, YP, w, HP))
NOTHING = Box(816, YP, 104, HP)
fig.outline(NOTHING, "ink-muted", width=1, dash="lead", radius=8)
fig.text(NOTHING.cx, YP + 38, "nothing", "note", anchor="middle")
LOSS = cards[-1]
fig.outline(LOSS, "gradient", width=2.5, radius=8)

fig.chain(cards + [NOTHING], labels=[None, None, None, YH, var("L")], t=0.3)
ends = fig.chain(cards, color="gradient", reverse=True, t=0.7, side="below")
(p, q) = ends[-1]
fig.text((p[0] + q[0]) / 2, p[1] + 24, "dinputs", "code", anchor="middle", color="gradient")

# ---------------------------------------------------------------- the class, enlarged
XC, YC, WC = 216, 224, 560
LINES = [SRC[start]] + ["    " + l for l in FWD + BWD]       # the class line, then its methods one level in
LEAD = 28
body = fig.card(XC, YC, WC, None, lines=LINES, style="code", fit=True, leading=LEAD)
CARD = Box(XC, YC, body.w + 32, body.bottom + 16 - YC)
B0 = body.y + 16 + LEAD                                     # first code baseline
YF = B0 - 4                                          # the forward line's centre
YB = B0 + LEAD * len(FWD) - 4                          # the backward line's centre
YO = B0 + LEAD * (len(FWD) + len(BWD) - 1) - 4         # the last line, where dinputs is set

# in from the left: the predictions and the labels, to both methods
XA = 40                                              # the left column's arrows start here
fig.arrow((XA, YF), (XC - 8, YF), label=rich(YH, " ", NK, " and ", code("y_true")))
fig.arrow((XA, YB), (XC - 8, YB))                    # the backward call: the predictions fill the dvalues slot
fig.text((XA + XC - 8) / 2, YB - 32, rich(YH, " ", NK, " as ", code("dvalues"), ","), "note", anchor="middle")
CX = (XA + XC - 8) // 2
fig.text(CX - 40, YB - 12, "and", "note")
fig.text(CX - 12, YB - 12, "y_true", "code")
# back to the left: the gradient
fig.arrow((XC - 8, YO), (XA, YO), color="gradient")
fig.text(XA, YO - 12, "self.dinputs", "code", color="gradient")
fig.text(XA + 112, YO - 12, NK, "note")
fig.text(XA, YO + 28, "to the softmax as its", "note")
fig.text(XA, YO + 48, "dvalues", "code")

# out to the right: the loss; and where the upstream gradient would arrive, nothing
XR = CARD.right
fig.arrow((XR + 8, YF), (904, YF), label="per-sample losses")
fig.text(XR + 8, YF + 28, rich("their mean is ", var("L")), "note")
EMPTY = Box(XR + 32, YB - 24, 920 - XR - 32, 48)
fig.outline(EMPTY, "ink-muted", width=1, dash="lead", radius=8)
fig.text(EMPTY.cx, YB + 5, "nothing", "note", anchor="middle")
fig.note(EMPTY, [rich("∂", var("L"), "/∂", var("L"), " = 1, so")])
fig.text(EMPTY.x, EMPTY.bottom + 48, "dvalues", "code")
fig.text(EMPTY.x + 64, EMPTY.bottom + 48, rich("holds ", YH), "note")

fig.text(XC, CARD.bottom + 28, "Nothing is cached: both inputs arrive as arguments.", "note")
fig.legend(576, CARD.bottom + 28, [dict(color="arrow", label="forward: values", mark="line"),
                                   dict(color="gradient", label="backward: gradients", mark="line")],
           direction="row")

fig.caption(rich("Every other ", code("backward"), " is handed a gradient from the right; the loss computes the first one."))
fig.write()
