"""Post 31, section 10: the mistakes a dropout layer invites, each one run.

Run from the series root:
    python posts/31-dropout/snippets/what_can_go_wrong.py

Parts 1 to 4 run in float64 before nnfs.init() is called; parts 5 and 6 call it and train the
network of network.py twice on seed 0.

Needs NumPy and the nnfs package. Takes 20 to 40 seconds.
"""
import numpy as np
import nnfs

from gradient_check import build as build_small, check
from network import SETUP, Layer_Dropout, build, evaluate, train

np.set_printoptions(suppress=True, linewidth=120)

print("== 1. Three wrong backward methods, gradient check of gradient_check.py, seed 0")


def no_mask(layer, dvalues):
    layer.dinputs = dvalues.copy()


def no_scale(layer, dvalues):
    layer.dinputs = dvalues * (layer.binary_mask > 0)


def new_mask(layer, dvalues):
    layer.dinputs = dvalues * np.random.binomial(1, layer.rate, size=dvalues.shape) / layer.rate


print("gradient         correct              no mask               mask without the scale   a new mask")
columns = [check(*build_small(0), dropout_backward=wrong) for wrong in (None, no_mask, no_scale, new_mask)]
for index in range(4):
    cells = "   ".join(f"{column[index][1]:.1e} ({column[index][2]:.1e})" for column in columns)
    print(f"{columns[0][index][0]:<16} {cells}")

print()
print("== 2. backward after an evaluation forward")
a = np.ones((1, 6))
dvalues = np.ones((1, 6))
layer = Layer_Dropout(0.5)
layer.forward(a, training=False)
try:
    layer.backward(dvalues)
except AttributeError as error:
    print("a layer that has never run in training mode:", type(error).__name__ + ":", error)
np.random.seed(0)
layer.forward(a, training=True)
print("training forward, mask:          ", layer.binary_mask[0])
layer.forward(a, training=False)
layer.backward(dvalues)
print("evaluation forward, then backward:", layer.dinputs[0], " (the output was", layer.output[0], ")")

print()
print("== 3. The seed set inside the loop")
masks = []
for epoch in range(3):
    np.random.seed(0)                       # the mistake: every epoch restarts the stream
    layer.forward(a, training=True)
    masks.append(layer.binary_mask[0])
    print(f"epoch {epoch} mask:", masks[-1])
print("all three masks equal:", all(np.array_equal(masks[0], mask) for mask in masks))

print()
print("== 4. A drop rate of 1")
with np.errstate(divide="ignore", invalid="ignore"):
    layer = Layer_Dropout(1.0)
    layer.forward(a, training=True)
print("Layer_Dropout(1.0) stores self.rate =", layer.rate, "and outputs", layer.output[0])

nnfs.init()
print()
print("== 5. The mask left on in the test pass (seed 0, Layer_Dropout(0.1), trained)")
print(SETUP)
X, y, X_test, y_test, *network, optimizer = build(seed=0, rate=0.1)
train(X, y, *network, optimizer)
loss_off, accuracy_off = evaluate(X_test, y_test, *network)
on = [evaluate(X_test, y_test, *network, training=True) for _ in range(100)]
on_accuracy = [accuracy for _, accuracy in on]
on_loss = [loss for loss, _ in on]
print(f"training=False: test accuracy {accuracy_off:.4f}, test loss {loss_off:.4f}, the same on every pass")
print(f"training=True, 100 passes: test accuracy {min(on_accuracy):.4f} to {max(on_accuracy):.4f} "
      f"(mean {np.mean(on_accuracy):.4f}), test loss {min(on_loss):.4f} to {max(on_loss):.4f}; "
      f"passes below the training=False accuracy: {sum(value < accuracy_off for value in on_accuracy)}")

print()
print("== 6. A drop rate of 0 is not the network without the layer, bit for bit (seed 0)")
dense1, activation1, dropout1 = network[0], network[1], network[2]
print(f"dtypes with Layer_Dropout: ReLU output {activation1.output.dtype}, mask {dropout1.binary_mask.dtype}, "
      f"dropout output {dropout1.output.dtype}, dense1.dbiases {dense1.dbiases.dtype}, "
      f"dense1.dweights {dense1.dweights.dtype}")
for rate, label in ((None, "no dropout layer  "), (0.0, "Layer_Dropout(0.0)")):
    X, y, X_test, y_test, *network, optimizer = build(seed=0, rate=rate)
    train(X, y, *network, optimizer)
    _, train_accuracy = evaluate(X, y, *network)
    _, test_accuracy = evaluate(X_test, y_test, *network)
    print(f"{label}: dense1.dbiases {network[0].dbiases.dtype}, training accuracy {train_accuracy:.4f}, "
          f"test accuracy {test_accuracy:.4f}")
