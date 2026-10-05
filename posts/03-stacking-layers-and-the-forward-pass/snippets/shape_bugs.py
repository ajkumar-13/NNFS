"""Post 03: four mistakes in a two-layer forward pass, and which of them NumPy reports.

Run from the series root:  python posts/03-stacking-layers-and-the-forward-pass/snippets/shape_bugs.py
"""
import numpy as np

inputs = np.array([[ 1.0,  2.0,  3.0,  2.5],
                   [ 2.0,  5.0, -1.0,  2.0],
                   [-1.5,  2.7,  3.3, -0.8]])

weights1 = np.array([[ 0.2,   0.8,  -0.5,   1.0 ],
                     [ 0.5,  -0.91,  0.26, -0.5 ],
                     [-0.26, -0.27,  0.17,  0.87]])
biases1  = np.array([2.0, 3.0, 0.5])

weights2 = np.array([[ 0.1,  -0.14,  0.5 ],
                     [-0.5,   0.12, -0.33],
                     [-0.44,  0.73, -0.13]])
biases2  = np.array([-1.0, 2.0, -0.5])

layer1_outputs = np.dot(inputs,         weights1.T) + biases1
layer2_outputs = np.dot(layer1_outputs, weights2.T) + biases2
print("correct layer-2 output, sample 1:", layer2_outputs[0])

# 1. Layer 2 sized for the 4 original features instead of the 3 outputs of layer 1.
wrong_size = np.ones((3, 4))
try:
    np.dot(layer1_outputs, wrong_size.T)
except ValueError as error:
    print("1. W2 of shape (3, 4) raises ValueError:", error)

# 2. The transpose forgotten on layer 2. W2 is square, so the shapes still line up.
no_transpose = np.dot(layer1_outputs, weights2) + biases2
print("2. no .T on the square W2 runs; sample 1:", no_transpose[0])

# 3. The transpose forgotten on layer 1, whose weights are not square.
try:
    np.dot(inputs, weights1)
except ValueError as error:
    print("3. no .T on the (3, 4) W1 raises ValueError:", error)

# 4. Biases stored as a column, shape (3, 1), against an output of shape (3, 3).
column_biases = biases2.reshape(3, 1)
with_column = np.dot(layer1_outputs, weights2.T) + column_biases
print("4. column bias on 3 samples runs; sample 1:", with_column[0])
try:
    np.dot(layer1_outputs[:2], weights2.T) + column_biases
except ValueError as error:
    print("   column bias on 2 samples raises ValueError:", str(error).strip())
one_sample = np.dot(layer1_outputs[:1], weights2.T) + column_biases
print("   column bias on 1 sample runs; output shape:", one_sample.shape)

# 5. The same inputs and weights always give the same numbers, bit for bit.
again = np.dot(np.dot(inputs, weights1.T) + biases1, weights2.T) + biases2
print("5. second forward pass identical to the first:", np.array_equal(again, layer2_outputs))
