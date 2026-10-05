"""Post 12: four ways the single-neuron backward pass goes wrong, each reproduced.

Run from the series root:
    python posts/12-backprop-through-a-single-neuron/snippets/what_can_go_wrong.py

Contents (section 10): a sweep over the learning rate with the target at 0 and at 1; a neuron
that starts with a negative pre-activation; a bias gradient wrongly multiplied by an input,
caught by a central-difference check; and a central difference that straddles the ReLU corner.

Needs NumPy. Nothing here is random, so the output is the same on every run.
"""
import numpy as np

INPUTS = np.array([1.0, -2.0, 3.0])
WEIGHTS = np.array([-3.0, -1.0, 2.0])
BIAS = 1.0


def forward(weights, bias, target):
    z = float(np.dot(INPUTS, weights)) + bias
    yhat = max(0.0, z)
    return z, yhat, (yhat - target) ** 2


def gradients(z, yhat, target):
    upstream = 2.0 * (yhat - target) * (1.0 if z > 0 else 0.0)
    return upstream * INPUTS, upstream * 1.0


def train(lr, target, weights=WEIGHTS, bias=BIAS, steps=200):
    """Plain gradient descent. Returns the list of z after each update and the final loss."""
    weights = weights.copy()
    zs = []
    for _ in range(steps):
        z, yhat, _ = forward(weights, bias, target)
        dweights, dbias = gradients(z, yhat, target)
        weights -= lr * dweights
        bias -= lr * dbias
        zs.append(forward(weights, bias, target)[0])
    return zs, forward(weights, bias, target)[2]


def main():
    print("== 1. The learning rate: z after the first update, and the loss after 200 updates")
    print("each update multiplies (z - y) by 1 - 30 * lr while z stays positive")
    for target in (0.0, 1.0):
        print(f"target y = {target:.0f}")
        for lr in (0.01, 0.035, 0.05, 0.1, 1.0):
            zs, final_loss = train(lr, target)
            print(f"  lr = {lr:<5}  factor = {1 - 30 * lr:6.2f}   z after 1 update = {zs[0]:7.2f}"
                  f"   loss after 200 = {final_loss:.3e}")

    print()
    print("== 2. A neuron that starts dead: weights (-3, -1, -2), bias 1, target 1")
    dead_weights = np.array([-3.0, -1.0, -2.0])
    z, yhat, loss = forward(dead_weights, BIAS, 1.0)
    dweights, dbias = gradients(z, yhat, 1.0)
    print(f"z = {z:.1f}   yhat = {yhat:.1f}   loss = {loss:.1f}")
    print(f"dweights = {dweights + 0.0}   dbias = {dbias + 0.0}")   # + 0.0 turns -0.0 into 0.0
    _, final_loss = train(0.01, 1.0, weights=dead_weights)
    print(f"loss after 200 updates with lr = 0.01: {final_loss:.1f}")

    print()
    print("== 3. A bias gradient multiplied by an input, against a central difference (h = 1e-5)")
    z, yhat, _ = forward(WEIGHTS, BIAS, 0.0)
    _, dbias = gradients(z, yhat, 0.0)
    h = 1e-5
    numeric = (forward(WEIGHTS, BIAS + h, 0.0)[2] - forward(WEIGHTS, BIAS - h, 0.0)[2]) / (2 * h)
    for label, value in (("upstream * 1    (correct)", dbias),
                         ("upstream * x_2  (wrong)  ", dbias * INPUTS[1]),
                         ("upstream * x_3  (wrong)  ", dbias * INPUTS[2])):
        rel = abs(value - numeric) / max(abs(value), abs(numeric))
        print(f"dbias = {label} = {value:6.1f}   central difference {numeric:.6f}   relative error {rel:.1e}")

    print()
    print("== 4. A central difference across the ReLU corner: weights (-3, -1, 0), bias 1, target 1")
    corner_weights = np.array([-3.0, -1.0, 0.0])
    z, yhat, loss = forward(corner_weights, BIAS, 1.0)
    dweights, dbias = gradients(z, yhat, 1.0)
    numeric = (forward(corner_weights, BIAS + h, 1.0)[2] - forward(corner_weights, BIAS - h, 1.0)[2]) / (2 * h)
    print(f"z = {z:.1f}   loss = {loss:.1f}")
    print(f"dbias from the backward pass (ReLU derivative 0 at z = 0): {dbias + 0.0:.1f}")
    print(f"dbias from the central difference:                        {numeric:.6f}")
    right = (forward(corner_weights, BIAS + h, 1.0)[2] - loss) / h
    left = (loss - forward(corner_weights, BIAS - h, 1.0)[2]) / h
    print(f"one-sided slopes of the loss in b: {right:.5f} on the right, {left + 0.0:.5f} on the left")


if __name__ == "__main__":
    main()
