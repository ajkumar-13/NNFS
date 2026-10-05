"""Post 12: the forward pass, the four chain-rule factors and one update, checked in code.

Run from the series root:
    python posts/12-backprop-through-a-single-neuron/snippets/by_hand.py

Contents: the forward pass of section 3; the four local derivatives and the four gradients of
sections 4 and 5; a central-difference check of those gradients with h = 1e-5 (post 10,
section 6.1); the single gradient-descent step of section 6; and the shrink factor 0.7 of the
pre-activation that section 6 derives.

Needs NumPy. Nothing here is random, so the output is the same on every run.
"""
import numpy as np


def forward(weights, bias, inputs, target):
    """One neuron: weighted sum plus bias, ReLU, squared error. Returns (z, yhat, loss)."""
    z = float(np.dot(inputs, weights)) + bias
    yhat = max(0.0, z)
    loss = (yhat - target) ** 2
    return z, yhat, loss


def backward(z, yhat, inputs, target):
    """The chain rule, right to left. Returns (upstream, dweights, dbias)."""
    dloss_dyhat = 2.0 * (yhat - target)       # factor 1: squared error
    dyhat_dz = 1.0 if z > 0 else 0.0          # factor 2: ReLU
    dz_dterm = 1.0                            # factor 3: a sum passes the gradient to each term
    upstream = dloss_dyhat * dyhat_dz * dz_dterm
    dweights = upstream * inputs              # factor 4 for weight i is the input x_i
    dbias = upstream * 1.0                    # factor 4 for the bias is 1
    return upstream, dweights, dbias


def numerical_gradients(weights, bias, inputs, target, h=1e-5):
    """Central difference on each parameter in turn, the others held fixed."""
    dweights = np.zeros_like(weights)
    for i in range(weights.size):
        step = np.zeros_like(weights)
        step[i] = h
        plus = forward(weights + step, bias, inputs, target)[2]
        minus = forward(weights - step, bias, inputs, target)[2]
        dweights[i] = (plus - minus) / (2 * h)
    plus = forward(weights, bias + h, inputs, target)[2]
    minus = forward(weights, bias - h, inputs, target)[2]
    return dweights, (plus - minus) / (2 * h)


def main():
    inputs = np.array([1.0, -2.0, 3.0])
    weights = np.array([-3.0, -1.0, 2.0])
    bias = 1.0
    target = 0.0
    lr = 0.01

    print("== Section 3: the forward pass")
    z, yhat, loss = forward(weights, bias, inputs, target)
    print(f"products x_i * w_i = {inputs * weights}")
    print(f"z = {z:.2f}   yhat = ReLU(z) = {yhat:.2f}   L = (yhat - y)^2 = {loss:.2f}")

    print()
    print("== Sections 4 and 5: four factors, four gradients")
    upstream, dweights, dbias = backward(z, yhat, inputs, target)
    print(f"factor 1  dL/dyhat = 2 (yhat - y) = {2.0 * (yhat - target):.1f}")
    print(f"factor 2  dyhat/dz = {1.0 if z > 0 else 0.0:.1f}   (ReLU, z > 0)")
    print("factor 3  dz/d(x_i w_i) = 1.0   (a sum)")
    print(f"upstream gradient = factor 1 * factor 2 * factor 3 = {upstream:.1f}")
    for i in range(3):
        print(f"dL/dw_{i + 1} = upstream * x_{i + 1} = {upstream:.1f} * {inputs[i]:4.1f} = {dweights[i]:6.1f}")
    print(f"dL/db   = upstream * 1   = {dbias:6.1f}")

    print()
    print("== The same four numbers measured: central difference, h = 1e-5")
    num_w, num_b = numerical_gradients(weights, bias, inputs, target)
    for i in range(3):
        print(f"dL/dw_{i + 1}  chain rule {dweights[i]:6.1f}   central difference {num_w[i]:14.9f}")
    print(f"dL/db    chain rule {dbias:6.1f}   central difference {num_b:14.9f}")
    gap = max(np.max(np.abs(num_w - dweights)), abs(num_b - dbias))
    print(f"largest gap between the two columns: {gap:.1e}")

    print()
    print("== Section 6: one gradient-descent step, learning rate 0.01")
    new_weights = weights - lr * dweights
    new_bias = bias - lr * dbias
    for i in range(3):
        print(f"w_{i + 1}: {weights[i]:5.2f} - 0.01 * {dweights[i]:6.1f} = {new_weights[i]:5.2f}")
    print(f"b  : {bias:5.2f} - 0.01 * {dbias:6.1f} = {new_bias:5.2f}")
    new_z, new_yhat, new_loss = forward(new_weights, new_bias, inputs, target)
    print(f"z: {z:.2f} -> {new_z:.2f}   L: {loss:.2f} -> {new_loss:.2f}")
    norm_sq = float(np.dot(inputs, inputs))
    factor = 1 - lr * 2 * (norm_sq + 1)
    print(f"||x||^2 + 1 = {norm_sq:.0f} + 1 = {norm_sq + 1:.0f}")
    print(f"predicted shrink of z per step: 1 - 0.01 * 2 * {norm_sq + 1:.0f} = {factor:.2f}   measured: {new_z / z:.2f}")
    print(f"predicted shrink of L per step: {factor:.2f}^2 = {factor ** 2:.2f}   measured: {new_loss / loss:.2f}")

    print()
    print("== Section 6: the same step repeated, L_t = 36 * 0.49^t")
    w, b = weights.copy(), bias
    for t in range(6):
        z_t, yhat_t, loss_t = forward(w, b, inputs, target)
        print(f"t = {t}   z = {z_t:.4f}   L = {loss_t:8.4f}   36 * 0.49^t = {36 * 0.49 ** t:8.4f}")
        _, dw, db = backward(z_t, yhat_t, inputs, target)
        w -= lr * dw
        b -= lr * db


if __name__ == "__main__":
    main()
