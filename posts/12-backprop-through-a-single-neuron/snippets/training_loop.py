"""Post 12: 200 iterations of gradient descent on one three-input ReLU neuron.

Run from the series root:
    python posts/12-backprop-through-a-single-neuron/snippets/training_loop.py

Contents: the loop of section 9.2 (forward pass, backward pass, update), the loss at iterations
0, 4, ..., 20, then every twentieth up to 120 and at 199, the first iteration at which the loss is exactly 0.0,
and the parameters at the end.

Needs NumPy. Nothing here is random, so the output is the same on every run.
"""
import numpy as np

inputs = np.array([1.0, -2.0, 3.0])
weights = np.array([-3.0, -1.0, 2.0])
bias = 1.0
target = 0.0
lr = 0.01


def relu(z):
    return max(0.0, z)


def relu_deriv(z):
    return 1.0 if z > 0 else 0.0


first_zero = None
for i in range(200):
    # Forward pass.
    z = float(np.dot(inputs, weights)) + bias
    yhat = relu(z)
    loss = (yhat - target) ** 2

    # Backward pass: the local derivatives, multiplied right to left.
    dloss_dyhat = 2.0 * (yhat - target)     # factor 1
    dyhat_dz = relu_deriv(z)                # factor 2
    upstream = dloss_dyhat * dyhat_dz       # factor 3 is 1 for a sum
    dweights = upstream * inputs            # factor 4 for weight i: the input x_i
    dbias = upstream * 1.0                  # factor 4 for the bias: 1

    # Update.
    weights -= lr * dweights
    bias -= lr * dbias

    if first_zero is None and loss == 0.0:
        first_zero = i
    if i <= 20 and i % 4 == 0:
        print(f"iter {i:3d}  loss = {loss:.4f}   (exactly {loss:.3e})")
    elif (i % 20 == 0 and i <= 120) or i == 199:
        print(f"iter {i:3d}  loss = {loss:.4f}   (exactly {loss:.3e})   z = {z:.3e}")

print(f"first iteration with a loss of exactly 0.0: {first_zero}")
print(f"final weights = {weights}")
print(f"final bias    = {bias:.6f}")
print(f"final z       = {float(np.dot(inputs, weights)) + bias:.3e}")
