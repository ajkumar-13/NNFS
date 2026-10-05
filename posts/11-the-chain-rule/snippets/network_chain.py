"""Post 11: the four chain-rule factors of a two-layer network, each one measured.

Run from the series root:
    python posts/11-the-chain-rule/snippets/network_chain.py

Contents: one sample through the 21-parameter network of post 09 (2 inputs, 3 hidden neurons,
3 classes) with fixed weights. Every local derivative of section 6 is measured on its own with
central differences, the four tables are multiplied, and the product is compared with a central
difference taken through the whole network (section 8.2). The script then repeats the check for
W2 and b1, with their factor counts (section 6.1), and drops one factor (section 9).

No formula for a local derivative is used anywhere: posts 12 to 19 derive those. Needs only
NumPy, in float64. Nothing here is random, so the output is the same on every run.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

X = np.array([1.0, -2.0])                        # one sample, 2 features
W1 = np.array([[0.5, -1.0, 0.2],
               [-0.3, 0.4, -0.8]])               # (n_inputs, n_neurons) = (2, 3)
B1 = np.array([0.1, 0.2, -0.1])
W2 = np.array([[0.3, -0.2, 0.5],
               [0.7, 0.1, -0.4],
               [-0.6, 0.9, 0.2]])                # (3, 3)
B2 = np.array([0.0, 0.1, -0.1])
TRUE_CLASS = 0


def dense1(w1_flat, b1=B1):
    """z1 as a function of the six entries of W1, with the input held fixed."""
    return X @ w1_flat.reshape(2, 3) + b1


def relu(z1):
    return np.maximum(0.0, z1)


def dense2(a1, w2=W2):
    return a1 @ w2 + B2


def softmax_cross_entropy(z2):
    """The loss of one sample as a function of the logits: softmax, then -log of the true class."""
    shifted = np.exp(z2 - np.max(z2))
    probabilities = shifted / np.sum(shifted)
    return np.array([-np.log(probabilities[TRUE_CLASS])])


def local_derivatives(function, point, h=1e-5):
    """Table of d output_i / d input_j at point, one central difference per input (post 10)."""
    point = np.asarray(point, dtype=np.float64)
    columns = []
    for j in range(point.size):
        step = np.zeros_like(point)
        step[j] = h                              # move input j only
        columns.append((function(point + step) - function(point - step)) / (2 * h))
    return np.stack(columns, axis=1)             # shape (n_outputs, n_inputs)


def main():
    w1 = W1.reshape(-1)
    z1 = dense1(w1)
    a1 = relu(z1)
    z2 = dense2(a1)
    loss = softmax_cross_entropy(z2)[0]
    print("== Forward pass, one sample")
    print(f"z1 = x W1 + b1  = {z1}")
    print(f"a1 = ReLU(z1)   = {a1}")
    print(f"z2 = a1 W2 + b2 = {z2}")
    print(f"softmax(z2)     = {np.exp(z2) / np.sum(np.exp(z2))}")
    print(f"L               = {loss:.6f}   (-log of the probability of class {TRUE_CLASS})")

    print()
    print("== The four local derivatives, each measured on its own function")
    dL_dz2 = local_derivatives(softmax_cross_entropy, z2)
    dz2_da1 = local_derivatives(dense2, a1)
    da1_dz1 = local_derivatives(relu, z1)
    dz1_dW1 = local_derivatives(dense1, w1)
    for name, table in (("dL/dz2 ", dL_dz2), ("dz2/da1", dz2_da1), ("da1/dz1", da1_dz1), ("dz1/dW1", dz1_dW1)):
        print(f"{name}  shape {table.shape}")
        print(table + 0.0)

    print()
    print("== Section 8.2: the product of the four against one measurement through the whole network")
    chain = dL_dz2 @ dz2_da1 @ da1_dz1 @ dz1_dW1

    def whole_network(w1_flat):
        return softmax_cross_entropy(dense2(relu(dense1(w1_flat))))

    direct = local_derivatives(whole_network, w1)
    print("product of the four factors, reshaped to the shape of W1:")
    print(chain.reshape(2, 3) + 0.0)
    print("central difference through the whole network:")
    print(direct.reshape(2, 3) + 0.0)
    print(f"largest gap between the two: {np.max(np.abs(chain - direct)):.1e}")

    print()
    print("== Section 6.1: the number of factors depends on where the parameter enters")
    dz2_dW2 = local_derivatives(lambda w2_flat: dense2(a1, w2_flat.reshape(3, 3)), W2.reshape(-1))
    chain_w2 = dL_dz2 @ dz2_dW2
    direct_w2 = local_derivatives(
        lambda w2_flat: softmax_cross_entropy(dense2(a1, w2_flat.reshape(3, 3))), W2.reshape(-1))
    print(f"W2: 2 factors, shapes {dL_dz2.shape} {dz2_dW2.shape}, largest gap {np.max(np.abs(chain_w2 - direct_w2)):.1e}")
    dz1_db1 = local_derivatives(lambda b1: dense1(w1, b1), B1)
    chain_b1 = dL_dz2 @ dz2_da1 @ da1_dz1 @ dz1_db1
    direct_b1 = local_derivatives(
        lambda b1: softmax_cross_entropy(dense2(relu(dense1(w1, b1)))), B1)
    print(f"b1: 4 factors, shapes {dL_dz2.shape} {dz2_da1.shape} {da1_dz1.shape} {dz1_db1.shape}, largest gap {np.max(np.abs(chain_b1 - direct_b1)):.1e}")
    print(f"dL/db1 = {chain_b1[0] + 0.0}   (the middle hidden neuron is switched off)")

    print()
    print("== Section 9: one factor dropped (the ReLU factor)")
    dropped = dL_dz2 @ dz2_da1 @ dz1_dW1
    print("product without da1/dz1, reshaped to the shape of W1:")
    print(dropped.reshape(2, 3) + 0.0)
    print(f"largest gap to the measured gradient: {np.max(np.abs(dropped - direct)):.4f}")


if __name__ == "__main__":
    main()
