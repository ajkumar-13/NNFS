"""Post 13: the backward pass through a layer of three ReLU neurons, by hand and checked.

Run from the series root:
    python posts/13-backprop-through-a-layer/snippets/layer_backward.py

Contents: the forward pass of section 4; the fifteen gradients of section 5 from a plain Python
loop over neurons and inputs (section 5.1), compared with a central difference on each of the
fifteen parameters (section 5.2); the same layer with one neuron switched off (section 5.3);
one gradient-descent step (section 6); and the loop against its one-line NumPy forms
(section 8.2).

Needs only NumPy. Nothing is random, so every run prints the same numbers.
"""
import numpy as np

inputs = [1.0, 2.0, 3.0, 4.0]

weights = [[0.1, 0.2, 0.3, 0.4],     # neuron 1
           [0.5, 0.6, 0.7, 0.8],     # neuron 2
           [0.9, 1.0, 1.1, 1.2]]     # neuron 3

biases = [0.1, 0.2, 0.3]


def forward(weights, biases, inputs):
    """Weighted sum and ReLU for each neuron, then the layer sum Y and the loss L = Y ** 2."""
    Z = []
    for k in range(len(weights)):                # one neuron at a time
        z = biases[k]
        for j in range(len(inputs)):
            z += weights[k][j] * inputs[j]
        Z.append(z)
    A = [max(0.0, z) for z in Z]
    Y = sum(A)
    L = Y ** 2
    return Z, A, Y, L


def backward(Z, Y, inputs):
    """The single-neuron recipe of post 12, run once per neuron with a shared upstream value."""
    dL_dY = 2 * Y                                # computed once, at the loss
    dL_dZ = []
    dL_dW = []
    dL_db = []
    for k in range(len(Z)):                      # one neuron at a time
        gate = 1.0 if Z[k] > 0 else 0.0          # this neuron's own ReLU gate
        dL_dZ_k = dL_dY * 1.0 * gate             # loss factor, sum factor, gate
        dL_dZ.append(dL_dZ_k)
        dL_dW.append([dL_dZ_k * x for x in inputs])   # one gradient per weight: times its input
        dL_db.append(dL_dZ_k)                    # the bias multiplies a constant 1
    return dL_dZ, dL_dW, dL_db


def numerical_gradients(weights, biases, inputs, h=1e-5):
    """Central difference on each of the fifteen parameters in turn, the others held fixed."""
    num_dW = [[0.0] * len(inputs) for _ in weights]
    num_db = [0.0] * len(biases)
    for k in range(len(weights)):
        for j in range(len(inputs)):
            up = [row[:] for row in weights]
            down = [row[:] for row in weights]
            up[k][j] += h
            down[k][j] -= h
            num_dW[k][j] = (forward(up, biases, inputs)[3] - forward(down, biases, inputs)[3]) / (2 * h)
        up = biases[:]
        down = biases[:]
        up[k] += h
        down[k] -= h
        num_db[k] = (forward(weights, up, inputs)[3] - forward(weights, down, inputs)[3]) / (2 * h)
    return num_dW, num_db


def largest_gap(dL_dW, dL_db, num_dW, num_db):
    """Largest absolute difference between the analytic and the numerical gradients."""
    gaps = [abs(a - n) for row_a, row_n in zip(dL_dW, num_dW) for a, n in zip(row_a, row_n)]
    gaps += [abs(a - n) for a, n in zip(dL_db, num_db)]
    return max(gaps)


def show(values, width=6, digits=1):
    return "[" + " ".join(f"{v:{width}.{digits}f}" for v in values) + "]"


def report(weights, biases, inputs):
    Z, A, Y, L = forward(weights, biases, inputs)
    dL_dZ, dL_dW, dL_db = backward(Z, Y, inputs)
    num_dW, num_db = numerical_gradients(weights, biases, inputs)
    print(f"Z = {show(Z)}   A = {show(A)}   Y = {Y:.1f}   L = {L:.2f}")
    print(f"upstream dL/dY = 2Y = {2 * Y:.1f}   gates = {[int(z > 0) for z in Z]}   dL/dZ = {show(dL_dZ)}")
    for k in range(len(weights)):
        print(f"neuron {k + 1}   dL/dW = {show(dL_dW[k])}   dL/db = {dL_db[k]:5.1f}")
    print(f"largest gap to a central difference (h = 1e-5) over 15 parameters: "
          f"{largest_gap(dL_dW, dL_db, num_dW, num_db):.1e}")
    return dL_dZ, dL_dW, dL_db


print("== Sections 4 and 5: forward pass, then the fifteen gradients from the loop")
dL_dZ, dL_dW, dL_db = report(weights, biases, inputs)

print()
print("== Section 5.3: neuron 2 switched off (its four weights negated)")
weights_off = [weights[0][:], [-w for w in weights[1]], weights[2][:]]
report(weights_off, biases, inputs)

print()
print("== Section 6: one gradient-descent step with learning rate 0.001")
lr = 0.001
new_weights = [[w - lr * g for w, g in zip(row_w, row_g)] for row_w, row_g in zip(weights, dL_dW)]
new_biases = [b - lr * g for b, g in zip(biases, dL_db)]
Z, A, Y, L = forward(weights, biases, inputs)
Z_new, A_new, Y_new, L_new = forward(new_weights, new_biases, inputs)
print("neuron 1 weights after the step:", show(new_weights[0], width=6, digits=4),
      f"  bias {new_biases[0]:.4f}")
print(f"Z before = {show(Z)}   Z after = {show(Z_new, width=6, digits=4)}")
print(f"Y  {Y:.4f} -> {Y_new:.4f}")
print(f"L  {L:.4f} -> {L_new:.4f}   ratio {L_new / L:.4f}")

print()
print("== Section 8.2: the loop against its one-line NumPy forms")
x = np.array(inputs)
dz = np.array(dL_dZ)
by_broadcast = dz.reshape(-1, 1) * x             # (3, 1) column times (4,) row
by_outer = np.outer(dz, x)                       # the outer product, by name
by_matmul = dz.reshape(3, 1) @ x.reshape(1, 4)   # a (3, 1) by (1, 4) matrix product
print("shape of the broadcast result:", by_broadcast.shape)
print("loop equals broadcast:", np.array_equal(np.array(dL_dW), by_broadcast))
print("broadcast equals np.outer:", np.array_equal(by_broadcast, by_outer))
print("broadcast equals the matrix product:", np.array_equal(by_broadcast, by_matmul))
