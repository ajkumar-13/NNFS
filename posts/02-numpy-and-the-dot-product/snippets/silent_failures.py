import numpy as np

inputs = np.array([[ 1.0,  2.0,  3.0,  2.5],
                   [ 2.0,  5.0, -1.0,  2.0],
                   [-1.5,  2.7,  3.3, -0.8]])       # (3, 4): 3 samples, 4 features

weights = np.array([[ 0.2,   0.8,  -0.5,   1.0 ],
                    [ 0.5,  -0.91,  0.26, -0.5 ],
                    [-0.26, -0.27,  0.17,  0.87]])    # (3, 4): 3 neurons, 4 inputs

# 1. A missing transpose is loud here: the inner sizes are 4 and 3.
try:
    np.dot(inputs, weights)
except ValueError as error:
    print("np.dot(inputs, weights):", error)

# 2. A square weight matrix makes the same mistake silent.
square = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])                 # 3 neurons, 3 inputs
x = np.array([[1.0, 0.0, 0.0]])                     # one sample: only input 1 is on
print("with .T:   ", np.dot(x, square.T))           # every neuron's first weight
print("without .T:", np.dot(x, square))             # neuron 1's three weights

# 3. The star operator is silent whenever the two shapes are equal or broadcast.
print("inputs * weights:", (inputs * weights).shape)
try:
    np.ones((5, 4)) * weights
except ValueError as error:
    print("(5, 4) * (3, 4):", str(error).strip())

# 4. np.dot and @ part ways above two dimensions, and on scalars.
P = np.ones((2, 3, 4))
Q = np.ones((2, 4, 5))
print("np.dot:", np.dot(P, Q).shape, "  @:", (P @ Q).shape)
print("np.dot(2, square):", np.dot(2, square).shape)
try:
    np.matmul(2, square)
except ValueError:
    print("np.matmul(2, square): ValueError")
