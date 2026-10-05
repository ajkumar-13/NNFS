"""Post 13: three ways the layer backward pass goes wrong (section 9).

Run from the series root:
    python posts/13-backprop-through-a-layer/snippets/what_can_go_wrong.py

Contents: a learning rate ten times larger, which switches every neuron off in one step; the
two shape mistakes around the reshape, and the square layer that hides one of them; and one
ReLU gate used for the whole layer, caught by a central difference.

Needs only NumPy. Nothing is random, so every run prints the same numbers.
"""
import numpy as np

np.set_printoptions(suppress=True)

inputs = np.array([1, 2, 3, 4], dtype=float)
weights = np.array([[0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2]])
biases = np.array([0.1, 0.2, 0.3])


def loss(weights, biases, inputs):
    return np.sum(np.maximum(0, weights @ inputs + biases)) ** 2


def gradients(weights, biases, inputs):
    Z = weights @ inputs + biases
    Y = np.sum(np.maximum(0, Z))
    dL_dZ = 2 * Y * np.where(Z > 0, 1.0, 0.0)
    return dL_dZ.reshape(-1, 1) * inputs, dL_dZ


print("== 1. A learning rate of 0.01 instead of 0.001")
W, b = weights.copy(), biases.copy()
for i in range(3):
    Z = W @ inputs + b
    dL_dW, dL_db = gradients(W, b, inputs)
    print(f"iter {i}  Z = {np.round(Z, 3)}  loss = {loss(W, b, inputs):.2f}  "
          f"largest |gradient| = {np.abs(dL_dW).max():.1f}")
    W -= 0.01 * dL_dW
    b -= 0.01 * dL_db

print()
print("== 2. The reshape, done wrong")
dL_dW, dL_dZ = gradients(weights, biases, inputs)
try:
    dL_dZ * inputs
except ValueError as error:
    print("dL_dZ * inputs               ->", error)
flipped = inputs.reshape(-1, 1) * dL_dZ
print("inputs.reshape(-1, 1) * dL_dZ -> shape", flipped.shape,
      " equal to the transpose of the right answer:", np.array_equal(flipped, dL_dW.T))
try:
    weights.copy() - 0.001 * flipped
except ValueError as error:
    print("weights - lr * flipped        ->", error)

print("a square layer, 3 inputs and 3 neurons, raises nothing:")
square_inputs = np.array([1.0, 2.0, 3.0])
square_dL_dZ = np.array([10.0, 0.0, 30.0])
print("right, dL_dZ.reshape(-1, 1) * inputs:")
print(square_dL_dZ.reshape(-1, 1) * square_inputs)
print("wrong, inputs.reshape(-1, 1) * dL_dZ:")
print(square_inputs.reshape(-1, 1) * square_dL_dZ)

print()
print("== 3. One gate for the whole layer (neuron 2 switched off)")
weights_off = weights.copy()
weights_off[1] = -weights_off[1]
Z = weights_off @ inputs + biases
Y = np.sum(np.maximum(0, Z))
right, _ = gradients(weights_off, biases, inputs)
one_gate = (2 * Y * (1.0 if Y > 0 else 0.0)) * np.ones(3).reshape(-1, 1) * inputs
h = 1e-5
numerical = np.zeros_like(weights_off)
for k in range(3):
    for j in range(4):
        up, down = weights_off.copy(), weights_off.copy()
        up[k, j] += h
        down[k, j] -= h
        numerical[k, j] = (loss(up, biases, inputs) - loss(down, biases, inputs)) / (2 * h)
print("Z =", Z)
print("neuron 2 row with one gate per neuron :", right[1])
print("neuron 2 row with one gate for all    :", one_gate[1])
print("neuron 2 row by central difference    :", np.round(numerical[1], 6))
print(f"largest gap, one gate per neuron: {np.abs(right - numerical).max():.1e}")
print(f"largest gap, one gate for all   : {np.abs(one_gate - numerical).max():.1f}")
