"""Post 34 hero (section 5): the combined class, what goes in and out of forward and backward, with shapes.

Run from anywhere:  python posts/34-sigmoid-and-binary-cross-entropy/diagrams/src/01-combined-class.py
Writes posts/34-sigmoid-and-binary-cross-entropy/diagrams/01-combined-class.svg. Takes about a second.

The code in the two cards is read from snippets/binary_classes.py: every line is checked to be in the class
Activation_Sigmoid_Loss_BinaryCrossentropy, in order (comments left out, as the cards have no room for them). The
snippet's class is run on the post's worked batch (logits 2, -1, 0.5, -3; labels 1, 0, 0, 1): the loss and the
dinputs the caption prints are its values, asserted against the lines index.md quotes in sections 3 and 5, and
the shapes the figure prints, (N, 1) for the logits, the cached output and dinputs, against its arrays.

Layout: the last dense layer on the left, the class's forward card above its backward card (the form of post 19,
figure 4). Forward arrows in the default stroke run left to right (logits in, loss out); backward arrows in the
gradient hue: the caller hands the cached output back as dvalues, and dinputs goes left to the dense layer. The
gradient the last line computes is written beside it. A key says what the two colours and the cached tag mean.
"""
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, MINUS, rich, span, var, hat, num, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
SNIPPET = POST / "snippets" / "binary_classes.py"
SRC = SNIPPET.read_text(encoding="utf-8")
INDEX = (POST / "index.md").read_text(encoding="utf-8")
s = runpy.run_path(str(SNIPPET), run_name="snippet")

FWD = ["def forward(self, inputs, y_true):",
       "    self.activation.forward(inputs)",
       "    self.output = self.activation.output",
       "    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)",
       "    y_pred_clipped = np.clip(self.output, 1e-7, 1 - 1e-7)",
       "    sample_losses = -(y_true * np.log(y_pred_clipped) +",
       "                      (1 - y_true) * np.log(1 - y_pred_clipped))",
       "    return float(np.mean(sample_losses))"]
BWD = ["def backward(self, dvalues, y_true):",
       "    samples = len(dvalues)",
       "    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)",
       "    self.dinputs = (dvalues - y_true) / samples"]
cls = SRC[SRC.index("class Activation_Sigmoid_Loss_BinaryCrossentropy"):SRC.index('if __name__ == "__main__"')]
pos = 0
for line in FWD + BWD:
    # the class is indented one level deeper than the cards, and the continuation keeps its alignment
    want = "    " + line
    pos = cls.index(want, pos)
assert ".backward(" not in cls                # the sigmoid's own backward is never called by the class

# -- the worked batch, through the snippet's class
logits = np.array([[2.0], [-1.0], [0.5], [-3.0]])
y = np.array([1, 0, 0, 1])
head = s["Activation_Sigmoid_Loss_BinaryCrossentropy"]()
loss = head.forward(logits, y)
head.backward(head.output, y)
assert head.output.shape == head.dinputs.shape == logits.shape == (4, 1) and isinstance(loss, float)
assert np.allclose(head.dinputs[:, 0], (head.output[:, 0] - y) / 4)
LOSS = f"{loss:.4f}"
DIN = [f"{v:.4f}" for v in head.dinputs[:, 0]]
assert LOSS == "1.1157" and DIN == ["-0.0298", "0.0672", "0.1556", "-0.2381"]
assert "mean loss      1.1157" in INDEX
assert ("On the worked batch this returns the loss 1.1157 and stores `dinputs` $(-0.0298, 0.0672, 0.1556, "
        "-0.2381)$ with shape $(4, 1)$") in INDEX
assert "**The clip is in the loss and not in the gradient.** `backward` reads the unclipped output." in INDEX
assert "loss_activation.backward(loss_activation.output, y)" in INDEX


def code(t, color=None):
    return span(t, mono=True, color=color)


N1 = rich("(", var("N"), ", 1)")
GRAD = rich("(", hat("y"), " ", MINUS, " ", var("y"), ")/", var("N"))
DINV = ", ".join(v.replace("-", MINUS) for v in DIN)

fig = Figure(
    "01-combined-class", "One class returns the loss and the gradient at the logit",
    "The class Activation_Sigmoid_Loss_BinaryCrossentropy between the last dense layer and the loss. Its forward "
    "card takes the logits, shape N by 1, runs the sigmoid, stores self.output, shape N by 1, marked cached, "
    "reshapes the labels to a float column, clips the output to 1e-7 and 1 - 1e-7, and returns the mean binary "
    "cross-entropy, a float. The caller passes loss_activation.output back as dvalues; the backward card reshapes "
    "the labels and computes (dvalues - y_true) / samples, which is y-hat minus y over N, and sends dinputs, shape "
    "N by 1, to the dense layer. The dense layer is drawn twice, a forward card and a backward card joined by a "
    "dashed edge labelled the same object. A note says the clip acts in forward only, and a key gives grey for "
    "forward, purple for backward and the cached tag. On the post's worked batch of 4 samples the class returns "
    f"the loss {LOSS} and dinputs {', '.join(DIN).replace('-', 'minus ')}.",
    subtitle=rich(var("N"), " samples, one logit each. The caller hands the cached output back to backward."),
    height=720, data_w=True)

LH = 24                                       # code leading
XD, WD = 40, 112                              # the dense layer
XC = 232                                      # the class's two cards
YF = 160
HF = -(-(16 + 16 + (len(FWD) - 1) * LH + 20) // 4) * 4
YB = YF + HF + 104
HB = -(-(16 + 16 + (len(BWD) - 1) * LH + 20) // 4) * 4
CONT = "sample_losses = -("                   # the continuation line starts under the first y_true
widths = [16 + text_width(l.strip(), 14, mono=True) for l in FWD[1:] + BWD[1:] if not l.startswith("      ")]
widths.append(16 + text_width(CONT, 14, mono=True) + text_width(FWD[6].strip(), 14, mono=True))
WC = -(-(max(widths) + 32) // 8) * 8
assert XC + WC <= 816, WC

fig.text(XC, YF - 16, code("Activation_Sigmoid_Loss_BinaryCrossentropy"), "code")
F = fig.card(XC, YF, WC, HF)
B = fig.card(XC, YB, WC, HB)


def base(card_y, k):
    """The baseline of line k of a code card whose top is card_y."""
    return card_y + 32 + k * LH


def lines(card_y, ls):
    """The code lines, one level (16) per four leading spaces; the continuation aligned by its x offset."""
    x0 = XC + 16
    with fig.data():
        for k, line in enumerate(ls):
            body = line.strip()
            if line.startswith("      "):
                x = x0 + 16 + text_width(CONT, 14, mono=True)
            else:
                x = x0 + 16 * ((len(line) - len(line.lstrip(" "))) // 4)
            fig.text(x, base(card_y, k), body, "code", snap=False)


lines(YF, FWD)
lines(YB, BWD)


def tag(x, y):
    """The cached tag: a rounded soft box in the output hue with the word in it, baseline y."""
    box = Box(x, y - 17, 64, 24)
    fig.fill(box, "output-soft", radius=4)
    fig.outline(box, "output-line", radius=4)
    fig.text(box.cx, y, "cached", "note", anchor="middle", color="output")
    return box


y_in = base(YF, 1) - 5                        # the logits arrow
y_back = base(YB, 3) - 5                      # the dinputs arrow

# the dense layer: one object, drawn where each pass touches it, the two cards joined by a dashed edge
D1 = Box(XD, YF, WD, HB)
D2 = Box(XD, YB, WD, HB)
for d in (D1, D2):
    fig.card(d.x, d.y, d.w, d.h)
fig.text(D1.cx, D1.y + 36, code("Layer_Dense"), "code", anchor="middle")
fig.text(D1.cx, D1.y + 60, "the last one", "note", anchor="middle")
fig.text(D1.cx, D1.y + 96, "forward", "head", anchor="middle")
fig.text(D2.cx, D2.y + 36, "backward", "head", anchor="middle")
fig.text(D2.cx, D2.y + 72, "reads it as its", "note", anchor="middle")
fig.text(D2.cx, D2.y + 96, code("dvalues", color="gradient"), "code", anchor="middle")
fig.edge((D1.cx, D1.bottom + 8), (D2.cx, D2.y - 8), color="rule", dash="lead")
ym = (D1.bottom + D2.y) / 2
fig.text(D1.cx + 12, ym - 4, "the same", "note")
fig.text(D1.cx + 12, ym + 20, "object", "note")

# cached: self.output, tagged in the output hue
y_out = base(YF, 2)
tag_x = XC + 16 + 16 + text_width(FWD[2].strip(), 14, mono=True) + 24
tag_x = -(-tag_x // 4) * 4
tag(tag_x, y_out + 1)

# forward: logits in, loss out
fig.arrow((D1.right + 8, y_in), (XC - 8, y_in))
fig.text((D1.right + XC) / 2, y_in - 12, "logits", "note", anchor="middle")
fig.text((D1.right + XC) / 2, y_in + 24, N1, "note", anchor="middle")
y_loss = base(YF, 7) - 5
fig.arrow((XC + WC + 8, y_loss), (XC + WC + 48, y_loss))
fig.text(XC + WC + 56, y_loss + 5, "loss", "label")
fig.text(XC + WC + 56, y_loss + 29, "a float", "note")

# backward: the cached output comes back as dvalues, dinputs goes to the dense layer
xl = tag_x + 32
fig.arrow((xl, YF + HF + 8), (xl, YB - 8), color="gradient")
yl = (YF + HF + YB) / 2
fig.text(xl + 12, yl - 6, code("loss_activation.output", color="gradient"), "code")
PASSED = rich(N1, ", passed as")
fig.text(xl + 12, yl + 18, PASSED, "note")
fig.text(xl + 12 + 104, yl + 18, code("dvalues"), "code")    # the note renders about 100 wide
fig.arrow((XC - 8, y_back), (D2.right + 8, y_back), color="gradient")
fig.text((D2.right + XC) / 2, y_back - 12, code("dinputs", color="gradient"), "code", anchor="middle")
fig.text((D2.right + XC) / 2, y_back + 24, N1, "note", anchor="middle")
fig.text(XC + WC + 16, base(YB, 3), rich("= ", GRAD), "math", color="gradient")

# one row under the cards: the note, then the key
YK = YB + HB + 40
fig.text(40, YK, "The clip acts in forward only.", "note")
KX = 448
fig.arrow((KX, YK - 5), (KX + 40, YK - 5))
fig.text(KX + 48, YK, "forward", "note")
fig.arrow((KX + 160, YK - 5), (KX + 120, YK - 5), color="gradient")
fig.text(KX + 168, YK, "backward", "note")
kt = tag(KX + 264, YK)
fig.text(kt.right + 8, YK, "kept by forward", "note")

fig.caption(rich("On the worked batch, ", var("N"), " = 4: loss ", LOSS, ", gradient ", DINV, "."))
fig.write()
