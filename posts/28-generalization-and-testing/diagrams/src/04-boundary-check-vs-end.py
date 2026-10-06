"""Post 28, section 5.2: seed 0's decision regions at the check of lowest test loss (epoch 700) and at the end.

Run from anywhere:  python posts/28-generalization-and-testing/diagrams/src/04-boundary-check-vs-end.py
(about 15 to 30 s). Writes posts/28-generalization-and-testing/diagrams/04-boundary-check-vs-end.svg.
The run is snippets/seed_spread.py's run(0), made here with the snippet's own build, train, evaluate,
boundary_length and weight_norm from snippets/network.py: the test points are drawn before the loop as run() draws
them, and a check hook at epoch 700 records what run() records there, plus the classes on network.py's 201 by 201
grid and the training predictions. Both checks are asserted against the seed 0 row of the section 5.1 listing in
index.md (epoch 700, test loss and accuracy, boundary 2161 -> 2346, weight norm 54.75 -> 187.46), and the end against
the section 3 listing (96.33 percent, 11 of 300 training points wrong).

Layout: two scatter plots of the plane, each tinted by the class the network answers on the grid, with the 300
training points on top and the wrong ones ringed; two note lines under each.
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
from network import GRID, boundary_length, build, evaluate, train, weight_norm  # noqa: E402

nnfs.init()
X, y, dense1, activation1, dense2, loss_activation, optimizer = build(0, 64)
X_test, y_test = spiral_data(samples=100, classes=3)            # as run() draws them, before the loop
network = (dense1, activation1, dense2, loss_activation)
CHECK = 700


def snapshot():
    test_loss, test_accuracy = evaluate(X_test, y_test, *network)
    train_loss, train_accuracy = evaluate(X, y, *network)
    wrong = np.argmax(loss_activation.output, axis=1) != y      # from the training-data pass just made
    dense1.forward(GRID)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    classes = np.argmax(dense2.output, axis=1).reshape(201, 201)
    return dict(test_loss=test_loss, test_accuracy=test_accuracy, train_accuracy=train_accuracy, wrong=wrong,
                classes=classes, boundary=boundary_length(dense1, activation1, dense2),
                weight_norm=weight_norm(dense1, dense2))


SNAP = {}


def on_check(epoch):
    if epoch == CHECK:
        SNAP[CHECK] = snapshot()


train(X, y, *network, optimizer, on_check=on_check)
SNAP[10000] = snapshot()                     # after the last update, as run() reads its final figures
A, B = SNAP[CHECK], SNAP[10000]
row = (f"   0    700  {A['test_loss']:.4f} -> {B['test_loss']:.4f}     {A['test_accuracy']:.4f} -> "
       f"{B['test_accuracy']:.4f}")
assert row in INDEX, row
assert (A["boundary"], B["boundary"]) == (2161, 2346)
assert (f"{A['weight_norm']:.2f}", f"{B['weight_norm']:.2f}") == ("54.75", "187.46")
assert "2161 ->  2346      54.75 -> 187.46" in INDEX
assert f"{A['train_accuracy']:.4f}" == "0.8900" and "  700      0.2872     0.8900     0.4900    0.8133" in INDEX
assert f"{B['train_accuracy']:.4f}" == "0.9633" and int(B["wrong"].sum()) == 11 and int(A["wrong"].sum()) == 33
assert "(11 of 300 are still wrong on seed 0)" in INDEX
RB, RW = B["boundary"] / A["boundary"], B["weight_norm"] / A["weight_norm"]
assert (f"{RB:.2f}", f"{RW:.2f}") == ("1.09", "3.42")


AXIS = np.linspace(-1, 1, 201)
H = (AXIS[1] - AXIS[0]) / 2


def regions(classes):
    """One rectangle per run of equal classes along a grid row, each cell centred on its grid point; rows overlap
    by a fifth of a cell so that no seam shows between two rects of one colour."""
    out = []
    for i in range(201):
        j = 0
        while j < 201:
            k = j
            while k + 1 < 201 and classes[i, k + 1] == classes[i, j]:
                k += 1
            x0, x1 = AXIS[j] - H, AXIS[k] + H
            y0, y1 = AXIS[i] - 1.2 * H, AXIS[i] + 1.2 * H
            out.append(([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], CLASS_COLORS[classes[i, j]]))
            j = k + 1
    return out


CLASS_COLORS = ["blue", "orange", "green"]
SHAPES = ["circle", "square", "triangle"]
pct = lambda v: f"{100 * v:.2f}%"            # noqa: E731

fig = Figure(
    "04-boundary-check-vs-end", "Boundary 1.09 times as long, weights 3.42 times as large",
    "Two plots of the plane from minus 1 to 1 on both axes for seed 0, each tinted by the class the network "
    "answers on a 201 by 201 grid, blue, orange and green for classes 0, 1 and 2, with the 300 training points "
    "on top as blue circles, orange squares and green triangles and the misclassified ones ringed. Left, epoch "
    "700, the check of lowest test loss: 33 training points wrong, training accuracy 89.00 percent, test "
    "accuracy 81.33 percent, boundary length 2,161, weight norm 54.75. Right, epoch 10,000, after the last "
    "update: 11 wrong, 96.33 and 82.33 percent, boundary length 2,346, weight norm 187.46. From the left to "
    "the right the boundary is 1.09 times as long and the weights 3.42 times as large.",
    subtitle="Seed 0: the class the network answers at each point of the plane, and the 300 training points.",
    height=720, data_w=True)

S = 344
TICKS = [-1, -0.5, 0, 0.5, 1]
fmt_t = lambda t: num(t) if t != int(t) else num(int(t))       # noqa: E731
PANELS = ((40, CHECK, "Epoch 700, the check of lowest test loss"), (496, 10000, "Epoch 10,000, the end of the run"))
for bx, epoch, heading in PANELS:
    snap = SNAP[epoch]
    fig.text(bx, 120, heading, "head")
    ax = fig.scatter(Box(bx, 128, 56 + S + 24, 24 + S + 48), [], x=(-1, 1, TICKS), y=(-1, 1, TICKS),
                     x_label="first input", y_label="second input", fmt_x=fmt_t, fmt_y=fmt_t,
                     regions=regions(snap["classes"]), vgrid=False)
    for k in range(3):
        for (px, py) in X[y == k]:
            ax.point(float(px), float(py), SHAPES[k], CLASS_COLORS[k], size=8)
    for (px, py) in X[snap["wrong"]]:
        ax.point(float(px), float(py), "circle", "ink", size=14, hollow=True)
    nx = ax.x
    fig.text(nx, 584, f"{int(snap['wrong'].sum())} of 300 training points wrong (ringed)", "label")
    fig.text(nx, 608, f"train {pct(snap['train_accuracy'])}, test {pct(snap['test_accuracy'])}", "note")
    fig.text(nx, 632, f"boundary length {num(snap['boundary'])}, weight norm {snap['weight_norm']:.2f}", "note")

fig.caption("Tints: the class answered at each point. End over epoch 700: boundary × 1.09, weight norm × 3.42.")
fig.write()
