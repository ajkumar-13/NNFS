"""Post 19, section 5: the combined class, what goes in and out of forward and backward, with shapes.

Run from anywhere:  python posts/19-softmax-derivatives-and-the-combined-backward-pass/diagrams/src/04-combined-class.py
Writes posts/19-softmax-derivatives-and-the-combined-backward-pass/diagrams/04-combined-class.svg.
The code in the two cards is read from snippets/combined_class.py (every line is checked to be there, in order).
The snippet is run here (runpy, as __main__): the shapes it produces on its seeded batch of 5 samples and
4 classes are checked against the symbolic shapes the figure prints, (N, K) for the logits, the cached output
and dinputs, and a single number for the loss.

Layout: the last dense layer on the left, the class's forward card above its backward card. Forward arrows in the
default stroke run left to right (logits in, loss out); backward arrows in the gradient hue: the caller hands the
cached output back as dvalues, and dinputs goes left to the dense layer. A legend says what the two colours and
the cached tag mean.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, span, var, text_width  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "combined_class.py"
SRC = SNIPPET.read_text(encoding="utf-8")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="__main__")
OUT = buf.getvalue()
logits, y = s["logits"], s["y"]
N, K = logits.shape
assert (N, K) == (5, 4) and "combined dinputs, shape (5, 4)" in OUT
sl = s["Activation_Softmax_Loss_CategoricalCrossentropy"]()     # the snippet's run, repeated on its batch
loss = sl.forward(logits, y)
sl.backward(sl.output, y)
assert sl.output.shape == sl.dinputs.shape == logits.shape and np.ndim(loss) == 0
assert f"loss: {loss:.6f}" in OUT

FWD = ["def forward(self, inputs, y_true):",
       "    self.activation.forward(inputs)",
       "    self.output = self.activation.output",
       "    return self.loss.calculate(self.output, y_true)"]
BWD = ["def backward(self, dvalues, y_true):",
       "    samples = len(dvalues)",
       "    if len(y_true.shape) == 2:",
       "        y_true = np.argmax(y_true, axis=1)",
       "    self.dinputs = dvalues.copy()",
       "    self.dinputs[range(samples), y_true] -= 1",
       "    self.dinputs /= samples"]
cls = SRC[SRC.index("class Activation_Softmax_Loss_CategoricalCrossentropy"):SRC.index("def central_difference")]
pos = 0
for line in FWD + BWD:
    pos = cls.index(line, pos)                # every line is in the class, in this order
assert ".backward(" not in cls                # neither standalone backward is called from the class


def code(t, color=None):
    return span(t, mono=True, color=color)


NK = rich("(", var("N"), ", ", var("K"), ")")


fig = Figure(
    "04-combined-class", "One class replaces two backward methods",
    "The class Activation_Softmax_Loss_CategoricalCrossentropy between the last dense layer and the loss. Its "
    "forward card takes the logits, shape (N, K), runs the softmax, stores self.output, shape (N, K), marked "
    "cached, and returns the loss, one number. The caller passes softmax_loss.output back as dvalues; the backward "
    "card converts one-hot labels to indices, copies dvalues, subtracts 1 at the true class and divides by "
    "samples, and sends dinputs, shape (N, K), to the dense layer, where it becomes that layer's dvalues. Neither "
    "standalone backward method is called. The dense layer is drawn twice, a forward card and a backward card "
    "joined by a dashed edge labelled the same object. A key on the bottom row: grey arrows forward, purple arrows "
    "backward, and the cached tag, kept by forward.",
    subtitle=rich(var("N"), " samples, ", var("K"), " classes. The caller hands the cached output back to backward."),
    height=720, data_w=True)

LH = 24                                       # code leading
XD, WD = 40, 136                              # the dense layer
XC, WC = 288, 480                             # the class's two cards
YF = 160
HF = 16 + 16 + (len(FWD) - 1) * LH + 20       # card(fit=True)'s height rule, kept explicit for the wide card
HF = -(-HF // 4) * 4
YB = YF + HF + 104
HB = -(-(16 + 16 + (len(BWD) - 1) * LH + 20) // 4) * 4

fig.text(XC, YF - 16, code("Activation_Softmax_Loss_CategoricalCrossentropy"), "code")
F = fig.card(XC, YF, WC, HF, lines=FWD, style="code")
B = fig.card(XC, YB, WC, HB, lines=BWD, style="code")


def base(card_body, k):
    """The baseline of line k of a code card."""
    return card_body.y + 16 + k * LH


def tag(x, y):
    """The cached tag: a rounded soft box in the output hue with the word in it, baseline y."""
    box = Box(x, y - 17, 64, 24)
    fig.fill(box, "output-soft", radius=4)
    fig.outline(box, "output-line", radius=4)
    fig.text(box.cx, y, "cached", "note", anchor="middle", color="output")
    return box


y_in = base(F, 1) - 5                         # the logits arrow
y_back = base(B, 6) - 5                       # the dinputs arrow

# the dense layer: one object, drawn where each pass touches it, the two cards joined by a dashed edge
D1 = Box(XD, YF, WD, HF)
D2 = Box(XD, YB + HB - HF, WD, HF)
for d in (D1, D2):
    fig.card(d.x, d.y, d.w, d.h)
fig.text(D1.cx, D1.y + 36, code("Layer_Dense"), "code", anchor="middle")
fig.text(D1.cx, D1.y + 60, "the last one", "note", anchor="middle")
fig.text(D1.cx, D1.y + 96, "forward", "head", anchor="middle")
fig.text(D2.cx, D2.y + 36, "backward", "head", anchor="middle")
fig.text(D2.cx, y_back - 28, "reads it as its", "note", anchor="middle")
fig.text(D2.cx, y_back - 4, code("dvalues", color="gradient"), "code", anchor="middle")
fig.edge((D1.cx, D1.bottom + 8), (D2.cx, D2.y - 8), color="rule", dash="lead")
ym = (D1.bottom + D2.y) / 2
fig.text(D1.cx + 12, ym - 4, "the same", "note")
fig.text(D1.cx + 12, ym + 20, "object", "note")

# cached: self.output, tagged in the output hue
y_out = base(F, 2)
tag_x = F.x + 16 + text_width(FWD[2].strip(), 14, mono=True) + 24
tag_x = -(-tag_x // 4) * 4
tag(tag_x, y_out + 1)

# forward: logits in, loss out
fig.arrow((D1.right + 8, y_in), (XC - 8, y_in))
fig.text((D1.right + XC) / 2, y_in - 12, "logits", "note", anchor="middle")
fig.text((D1.right + XC) / 2, y_in + 24, NK, "note", anchor="middle")
y_loss = base(F, 3) - 5
fig.arrow((XC + WC + 8, y_loss), (XC + WC + 64, y_loss))
fig.text(XC + WC + 72, y_loss + 5, "loss", "label")
fig.text(XC + WC + 72, y_loss + 29, "one number", "note")

# backward: the cached output comes back as dvalues, dinputs goes to the dense layer
xl = tag_x + 32
fig.arrow((xl, YF + HF + 8), (xl, YB - 8), color="gradient")
yl = (YF + HF + YB) / 2                       # the arrow's middle
fig.text(xl + 12, yl - 6, code("softmax_loss.output", color="gradient"), "code")
fig.text(xl + 12, yl + 18, rich(NK, ", passed as"), "note")
fig.text(xl + 12 + 104, yl + 18, code("dvalues"), "code")
fig.arrow((XC - 8, y_back), (D2.right + 8, y_back), color="gradient")
fig.text((D2.right + XC) / 2, y_back - 12, code("dinputs", color="gradient"), "code", anchor="middle")
fig.text((D2.right + XC) / 2, y_back + 24, NK, "note", anchor="middle")

# one row under the cards: the note, then the key
YK = YB + HB + 36
fig.text(40, YK, "Neither standalone backward method is called.", "note")
KX = 448
fig.arrow((KX, YK - 5), (KX + 40, YK - 5))
fig.text(KX + 48, YK, "forward", "note")
fig.arrow((KX + 160, YK - 5), (KX + 120, YK - 5), color="gradient")
fig.text(KX + 168, YK, "backward", "note")
kt = tag(KX + 264, YK)
fig.text(kt.right + 8, YK, "kept by forward", "note")

fig.caption("The class takes the logits and hands the last dense layer its gradient in one step.")
fig.write()
