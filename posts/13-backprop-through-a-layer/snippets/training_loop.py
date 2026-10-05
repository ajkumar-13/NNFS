"""Post 13: 200 iterations of gradient descent on the three-neuron layer (section 8.1).

Run from the series root:
    python posts/13-backprop-through-a-layer/snippets/training_loop.py

Prints the loss every 40 iterations and at every iteration where a ReLU gate changes, then the
loss and the pre-activations the loop ends with.

Needs only NumPy. Nothing is random, so every run prints the same numbers.
"""
import numpy as np

np.set_printoptions(suppress=True)

inputs = np.array([1, 2, 3, 4], dtype=float)
weights = np.array([[0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2]])
biases = np.array([0.1, 0.2, 0.3])
lr = 0.001


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return np.where(x > 0, 1.0, 0.0)


previous_gates = None
for i in range(200):
    # Forward.
    Z = weights @ inputs + biases
    A = relu(Z)
    Y = np.sum(A)
    L = Y ** 2

    # Backward: one upstream value, shared by all three neurons.
    dL_dY = 2 * Y                               # scalar
    dY_dA = np.ones_like(A)                     # all 1s
    dA_dZ = relu_deriv(Z)                       # one gate per neuron
    dL_dZ = dL_dY * dY_dA * dA_dZ               # shape (3,)

    # One gradient per weight: every dL_dZ entry times every input.
    dL_dW = dL_dZ.reshape(-1, 1) * inputs       # shape (3, 4)
    dL_db = dL_dZ                               # shape (3,)

    gates = dA_dZ.astype(int).tolist()
    if i % 40 == 0 or i == 199 or gates != previous_gates:
        print(f"iter {i:3d}  loss = {L:.6f}  Y = {Y:.6f}  gates = {gates}")
    previous_gates = gates

    # Update, only after every gradient has been computed.
    weights -= lr * dL_dW
    biases -= lr * dL_db

Z = weights @ inputs + biases
print()
print(f"loss at iteration 199, in full: {L:.3e}")
print("final Z      :", np.round(Z, 6))
