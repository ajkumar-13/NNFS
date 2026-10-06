"""Post 20, sections 2 to 5: four forward calls, four backward calls, and one update of four parameter arrays.

Run from anywhere:  python posts/20-assembling-full-backpropagation/diagrams/src/01-full-backprop-pipeline.py
Writes posts/20-assembling-full-backpropagation/diagrams/01-full-backprop-pipeline.svg.
snippets/assemble.py is run here (runpy): every shape is read from the arrays it leaves behind and asserted against
the shape table it prints, the four update lines are asserted to be lines of the snippet, and the two losses are
the snippet's own printout.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, arr, hat, span, sup, var  # noqa: E402

SNIPPET = Path(__file__).resolve().parents[2] / "snippets" / "assemble.py"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = runpy.run_path(str(SNIPPET), run_name="snippet")
OUT = buf.getvalue()
SRC = SNIPPET.read_text(encoding="utf-8").splitlines()

X, d1, a1, d2, la = s["X"], s["dense1"], s["activation1"], s["dense2"], s["loss_activation"]
N = len(s["y"])
assert N == 4 and X.shape == (4, 2)


def shape(t):
    return f"({t[0]}, {t[1]})"


# what each backward call stores, with the shapes the snippet prints in its table of section 3
STORES = {
    "dense1": [("dweights", d1.dweights.shape), ("dbiases", d1.dbiases.shape), ("dinputs", d1.dinputs.shape)],
    "activation1": [("dinputs", a1.dinputs.shape)],
    "dense2": [("dweights", d2.dweights.shape), ("dbiases", d2.dbiases.shape), ("dinputs", d2.dinputs.shape)],
    "loss_activation": [("dinputs", la.dinputs.shape)],
}
for obj, items in STORES.items():
    for name, t in items:
        line = f"{obj + '.' + name:<24} {str(t):<9} "
        assert line in OUT, line
assert STORES["dense1"][0][1] == d1.weights.shape and STORES["dense2"][0][1] == d2.weights.shape
FWD = {"Z1": d1.output.shape, "A1": a1.output.shape, "Z2": d2.output.shape, "yhat": la.output.shape}
assert set(FWD.values()) == {(4, 3)}
assert f"{'dense1.output':<24} (4, 3)" in OUT and f"{'loss_activation.output':<24} (4, 3)" in OUT
assert "closed ReLU gates: 5 of 12; zeros in activation1.dinputs: 5" in OUT
CLOSED = 5

UPDATE = ["dense1.weights -= learning_rate * dense1.dweights",
          "dense1.biases -= learning_rate * dense1.dbiases",
          "dense2.weights -= learning_rate * dense2.dweights",
          "dense2.biases -= learning_rate * dense2.dbiases"]
for line in UPDATE:
    assert line in [l.strip() for l in SRC], line
assert s["learning_rate"] == 0.01
LOSS0, LOSS1 = f"{s['loss']:.6f}", f"{s['new_loss']:.6f}"
CHANGE = s["new_loss"] - s["loss"]
assert f"loss before {LOSS0}   after {LOSS1}   change {CHANGE:.3e}" in OUT
assert (LOSS0, LOSS1, f"{CHANGE:.3e}") == ("1.098629", "1.098210", "-4.190e-04")
MANT, EXP = f"{CHANGE:.3e}".split("e")
CHANGE_TXT = rich("change −", MANT.lstrip("-"), " × ", sup("10", "−" + EXP.lstrip("-0"), italic=False))

fig = Figure(
    "01-full-backprop-pipeline", "Four calls forward, four calls back, one update",
    "The network of the post, on its batch of N = 4 samples, as a row of five cards: the data X of shape (4, 2), "
    "dense1 (a Layer_Dense), activation1 (an Activation_ReLU), dense2 (a Layer_Dense) and loss_activation "
    "(softmax and the loss). Grey forward arrows run left to right, labelled Z1, A1 and Z2, each of shape (4, 3). "
    "Purple backward arrows run right to left, labelled dL/dZ2, dL/dA1, dL/dZ1 and, dashed, dL/dX, which nothing "
    "reads. Under each object, what its backward call stores: dense1 dweights (2, 3), dbiases (1, 3) and dinputs "
    "(4, 2); activation1 dinputs (4, 3), zero at 5 closed gates; dense2 dweights (3, 3), dbiases (1, 3) and "
    "dinputs (4, 3); loss_activation dinputs (4, 3), computed from the predictions y-hat and the labels y and "
    "divided by N = 4. The parameter gradients of the two dense layers go down to a card with the four update "
    "lines, each parameter minus learning_rate times its gradient, with learning_rate = 0.01, and a second "
    "forward pass gives the loss before, 1.098629, and after, 1.098210, a change of minus 4.190 times 10 to the "
    "minus 4.",
    subtitle=rich("The network of section 2 on its batch of ", var("N"), f" = {N} samples. Shapes are (rows, columns)."),
    height=720, data_w=True)

fig.legend(40, 120, [dict(color="arrow", label="forward: each call reads the output of the call before", mark="line"),
                     dict(color="gradient", label="backward: each dinputs is the next call's dvalues", mark="line")],
           direction="row", gap=32)

# -- the row of objects
YC, CH = 152, 120
CARDS = [("X", 40, 48), ("dense1", 176, 104), ("activation1", 368, 128), ("dense2", 584, 104),
         ("loss_activation", 776, 144)]
SUB = {"dense1": "Layer_Dense", "activation1": "Activation_ReLU", "dense2": "Layer_Dense",
       "loss_activation": "softmax + loss"}
assert (d1.weights.shape, d2.weights.shape) == ((2, 3), (3, 3))
boxes = {}
for name, x, w in CARDS:
    b = Box(x, YC, w, CH)
    boxes[name] = b
    if name == "X":
        fig.card(x, YC, w, CH, color="input")
        fig.text(b.cx, YC + 52, arr("X"), "head", anchor="middle")
        fig.text(b.cx, YC + 76, shape(X.shape), "note", anchor="middle")
    else:
        fig.card(x, YC, w, CH)
        fig.text(b.cx, YC + 52, name, "code", anchor="middle")
        fig.text(b.cx, YC + 76, SUB[name], "note", anchor="middle")
row = [boxes[n] for n, _, _ in CARDS]


def dL(a):
    return rich("∂", var("L"), "/∂", a)


fig.chain(row, labels=[None] + [rich(a, " ", shape(FWD[k])) for a, k in
                              ((arr("Z", sub="1"), "Z1"), (arr("A", sub="1"), "A1"), (arr("Z", sub="2"), "Z2"))], t=0.3)
fig.chain(row[1:], labels=[dL(arr("Z", sub="1")), dL(arr("A", sub="1")), dL(arr("Z", sub="2"))],
          color="gradient", reverse=True, t=0.7, side="below")
fig.chain(row[:2], labels=[dL(arr("X"))], color="gradient", reverse=True, t=0.7, side="below", dash="proj")
fig.text(40, YC + CH + 28, "nothing", "note")
fig.text(40, YC + CH + 48, "reads it", "note")

# -- what each backward call stores
YS = 336
stores = {}
for name, w in (("dense1", 160), ("activation1", 160), ("dense2", 160), ("loss_activation", 144)):
    b = boxes[name]
    items = STORES[name]
    x = min(b.cx - w / 2, 920 - w)
    h = 56 + 24 * (len(items) - 1)
    fig.card(x, YS, w, h)
    for k, (nm, t) in enumerate(items):
        fig.text(x + 16, YS + 32 + 24 * k, span(nm, color="gradient", mono=True), "code")
        fig.text(x + w - 16, YS + 32 + 24 * k, shape(t), "label", anchor="end")
    stores[name] = Box(x, YS, w, h)
    fig.arrow((b.cx, YC + CH + 8), (b.cx, YS - 8), color="gradient",
              label="stores" if name == "dense1" else None, side="right")
fig.note(stores["activation1"], [f"zero at the {CLOSED}", "closed gates"])
fig.note(stores["loss_activation"], [rich("from ", hat(arr("y")), " and ", arr("y")), rich("divided by ", var("N"), f" = {N}")])

# -- one update of the four parameter arrays, and what it does to the loss
YU = 496
UX, UW = 176, 512
code = [rich(span(l.split(" -= ")[0], color="weight", mono=True), span(" -= learning_rate * ", mono=True),
             span(l.split(" * ")[1], color="gradient", mono=True)) for l in UPDATE]
ub = fig.card(UX, YU, UW, None, heading="The update, learning_rate = 0.01", lines=code, style="code", fit=True)
for name in ("dense1", "dense2"):
    cx = boxes[name].cx
    fig.arrow((cx, stores[name].bottom + 8), (cx, YU - 8), color="gradient")
RX = 744
rb = fig.card(RX, YU, 920 - RX, 136, heading="The loss")
for k, (lab, val) in enumerate((("before", LOSS0), ("after", LOSS1))):
    fig.text(rb.x, rb.y + 16 + 24 * k, lab, "label")
    fig.text(rb.right, rb.y + 16 + 24 * k, val, "label", anchor="end")
fig.text(rb.x, rb.y + 64, CHANGE_TXT, "label")
fig.arrow((UX + UW + 8, YU + 72), (RX - 8, YU + 72))

fig.caption("Only the two dense layers store parameter gradients, so the update has four lines.")
fig.write()
