import numpy as np

inputs = np.array([[ 1.0,  2.0,  3.0,  2.5],     # Sample 1
                   [ 2.0,  5.0, -1.0,  2.0],     # Sample 2
                   [-1.5,  2.7,  3.3, -0.8]])    # Sample 3

weights = np.array([[ 0.2,   0.8,  -0.5,   1.0 ],
                    [ 0.5,  -0.91,  0.26, -0.5 ],
                    [-0.26, -0.27,  0.17,  0.87]])

biases = np.array([2.0, 3.0, 0.5])

outputs = np.dot(inputs, weights.T) + biases
print(outputs)

print(inputs.shape, weights.T.shape, outputs.shape)
print(np.dot(weights, inputs[0]) + biases)       # sample 1 alone: no transpose
print(np.allclose(np.dot(weights, inputs.T).T, np.dot(inputs, weights.T)))
