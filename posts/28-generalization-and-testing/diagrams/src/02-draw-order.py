"""Post 28, sections 3 and 8: which draws of the random stream the training data, the weights and the test data take.

Run from anywhere:  python posts/28-generalization-and-testing/diagrams/src/02-draw-order.py   (a few seconds)
Writes posts/28-generalization-and-testing/diagrams/02-draw-order.svg.
The draw counts are measured here with the post's own code: snippets/network.py's Layer_Dense and nnfs's
spiral_data are called in each of the three orders after np.random.seed(0), and every array is rebuilt from one
block of np.random.randn(920) (the generator's formula, the weights as 0.01 times the draws), so the positions 1 to
300, 301 to 620 and 621 to 920 are asserted, not typed. That train() draws nothing is checked on two epochs. The
three gaps are those of the test_pass.py and what_can_go_wrong.py listings in index.md (the lint keeps them equal to
the scripts' output).

Layout: three rows, one per order of the calls; each block is as wide as the number of draws it takes, in call order.
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
INDEX = (POST / "index.md").read_text(encoding="utf-8")
import nnfs  # noqa: E402
from nnfs.datasets import spiral_data  # noqa: E402
from network import Layer_Dense, build, train  # noqa: E402

nnfs.init()


def spiral_from(r):
    """spiral_data(samples=100, classes=3) rebuilt from 300 given normal draws (the generator's formula)."""
    X = np.zeros((300, 2))
    for k in range(3):
        rad = np.linspace(0.0, 1, 100)
        t = np.linspace(k * 4, (k + 1) * 4, 100) + r[100 * k:100 * (k + 1)] * 0.2
        X[100 * k:100 * (k + 1)] = np.c_[rad * np.sin(t * 2.5), rad * np.cos(t * 2.5)]
    return X


np.random.seed(0)
R = np.random.randn(920)

# the documented order: training data, the two layers, (the loop), the test data
np.random.seed(0)
X, _ = spiral_data(samples=100, classes=3)
d1, d2 = Layer_Dense(2, 64), Layer_Dense(64, 3)
X_test, _ = spiral_data(samples=100, classes=3)
assert np.allclose(X, spiral_from(R[0:300]), atol=1e-12)
assert np.array_equal(d1.weights, 0.01 * R[300:428].reshape(2, 64))
assert np.array_equal(d2.weights, 0.01 * R[428:620].reshape(64, 3))
assert np.allclose(X_test, spiral_from(R[620:920]), atol=1e-12)
assert (d1.weights.size, d2.weights.size) == (128, 192)

# the training loop draws nothing: two epochs leave the stream where it was
Xb, yb, *net, opt = build(0)
state = np.random.get_state()
train(Xb, yb, *net, opt, epochs=2)
after = np.random.get_state()
assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(state, after))

# section 8: the seed set again before the test draw, and the test data drawn before the layers
np.random.seed(0)
X_again, _ = spiral_data(samples=100, classes=3)
assert np.array_equal(X_again, X)
np.random.seed(0)
_ = spiral_data(samples=100, classes=3)
X_early, _ = spiral_data(samples=100, classes=3)
e1, e2 = Layer_Dense(2, 64), Layer_Dense(64, 3)
assert np.allclose(X_early, spiral_from(R[300:600]), atol=1e-12)
assert np.array_equal(e1.weights, 0.01 * R[600:728].reshape(2, 64))
assert np.array_equal(e2.weights, 0.01 * R[728:920].reshape(64, 3))

for line in ("gap            loss +1.0362  acc 0.1400  (14.00 percentage points)",
             "identical to the training data: True",
             "'test' loss 0.0806  acc 0.9633  gap 0.00 points",
             "same training data: True; same test data: False",
             "training loss 0.0951  acc 0.9733;  test loss 0.9650  acc 0.7867;  gap 18.67 points"):
    assert line in INDEX, line
assert "the training data takes draws 1 to 300 and the two weight arrays take the next $128 + 192 = 320$" in INDEX
assert "therefore takes draws 621 to 920" in INDEX and "It gives the test set draws 301 to 600" in INDEX

PX = 0.65                       # viewBox units per draw
X0, LOOP_W, Y_STRIP = 40, 24, 48
TRAIN = ("neutral-soft", "rule", "ink")
WEIGHTS = ("weight-soft", "weight-line", "weight-ink")
TEST = ("blue-soft", "blue-line", "blue-ink")

fig = Figure(
    "02-draw-order", "When the test points are drawn decides what they are",
    "Three rows, one per order of the calls after np.random.seed(0); each block is as wide as the normal draws it "
    "takes. Documented order: training data draws 1 to 300, the weights of the two layers 301 to 620 (128 + 192), "
    "the training loop no draws, the test data 621 to 920; gap 14.00 points. Seed set again before the test draw: "
    "the test data takes draws 1 to 300 again and is the training set; gap 0.00 points. Test data drawn before the "
    "layers: test data 301 to 600, weights 601 to 920, so a different network is trained; train 97.33, test 78.67, "
    "gap 18.67 points. A short rule in each row marks the training loop.",
    subtitle="The calls in order after the seed is set; each block is as wide as the normal draws it takes.",
    data_w=True)


def block(x, y, n, style, inside, under):
    w = n * PX
    b = Box(x, y, w, Y_STRIP)
    with fig.data():
        fig.fill(b, style[0], fit=False)
        fig.outline(b, style[1], width=1)
        fig.text(b.cx, y + 29, inside, "label", anchor="middle", snap=False)
        fig.text(b.cx, y + Y_STRIP + 20, under, "tick", anchor="middle", snap=False)
    return x + w


def loop(x, y):
    """The training loop: a gap in the strip with a short rule, no draws."""
    with fig.data():
        fig.edge((x + LOOP_W / 2, y + 8), (x + LOOP_W / 2, y + Y_STRIP - 8), color="rule", width=1)
    return x + LOOP_W


ROWS_Y = (136, 264, 392)
XR = 712
# -- the documented order
y = ROWS_Y[0]
fig.text(40, y - 12, "The documented order", "head")
x = block(X0, y, 300, TRAIN, "training data", "draws 1 to 300")
x = block(x, y, 320, WEIGHTS, "weights, 128 + 192", "301 to 620")
LOOP_X = x
x = loop(x, y)
block(x, y, 300, TEST, "test data", "621 to 920")
fig.text(XR, y + 20, "gap 14.00 points", "value", color="ink", bold=True)
fig.text(XR, y + 44, "96.33 against 82.33", "note")
with fig.data():
    fig.text(LOOP_X + LOOP_W / 2, y + Y_STRIP + 20, "loop", "tick", anchor="middle", snap=False)

# -- the seed set again
y = ROWS_Y[1]
fig.text(40, y - 12, "The seed set again before the test draw", "head")
x = block(X0, y, 300, TRAIN, "training data", "draws 1 to 300")
x = block(x, y, 320, WEIGHTS, "weights, 128 + 192", "301 to 620")
x = loop(x, y)
with fig.data():
    fig.edge((x + 4, y - 4), (x + 4, y + Y_STRIP + 4), color="ink", width=1.5)
    fig.text(x + 4, y - 12, "np.random.seed(0)", "code", anchor="middle", snap=False)
block(x + 8, y, 300, TEST, "test data = training data", "draws 1 to 300 again")
fig.text(XR, y + 20, "gap 0.00 points", "value", color="ink")
fig.text(XR, y + 44, "it reads the training set", "note")

# -- the test data drawn before the layers
y = ROWS_Y[2]
fig.text(40, y - 12, "The test data drawn before the layers", "head")
x = block(X0, y, 300, TRAIN, "training data", "draws 1 to 300")
x = block(x, y, 300, TEST, "test data", "301 to 600")
x = block(x, y, 320, WEIGHTS, "other weights", "601 to 920")
loop(x, y)
fig.text(XR, y + 20, "gap 18.67 points", "value", color="ink")
fig.text(XR, y + 44, "97.33 against 78.67,", "note")
fig.text(XR, y + 64, "another network", "note")

fig.caption("The short grey rule in each row is the training loop, which draws no random numbers.")
fig.write()
