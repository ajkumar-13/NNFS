"""Post 04, section 4: one layer stored in both weight layouts gives the same numbers.

Run from the series root:
    python posts/04-dense-layer-class-and-spiral-data/snippets/weight_convention.py

The batch, the weights and the biases are those of the first layer in post 03.
"""
import numpy as np

inputs = np.array([[ 1.0,  2.0,  3.0,  2.5],
                   [ 2.0,  5.0, -1.0,  2.0],
                   [-1.5,  2.7,  3.3, -0.8]])     # (3, 4): 3 samples, 4 features
biases = np.array([[2.0, 3.0, 0.5]])               # (1, 3): one bias per neuron

# Old layout (posts 01 to 03): one ROW per neuron, shape (n_neurons, n_inputs) = (3, 4).
weights_old = np.array([[ 0.2,   0.8,  -0.5,   1.0 ],
                        [ 0.5,  -0.91,  0.26, -0.5 ],
                        [-0.26, -0.27,  0.17,  0.87]])
output_old = np.dot(inputs, weights_old.T) + biases

# New layout (post 04 onward): one COLUMN per neuron, shape (n_inputs, n_neurons) = (4, 3).
weights_new = np.array([[ 0.2,   0.5,  -0.26],
                        [ 0.8,  -0.91, -0.27],
                        [-0.5,   0.26,  0.17],
                        [ 1.0,  -0.5,   0.87]])
output_new = np.dot(inputs, weights_new) + biases

print("old layout:", weights_old.shape, "forward: np.dot(inputs, weights.T) + biases")
print(output_old)
print("new layout:", weights_new.shape, "forward: np.dot(inputs, weights) + biases")
print(output_new)
print("new weights are the old ones transposed:", np.array_equal(weights_new, weights_old.T))
print("outputs identical:", np.array_equal(output_old, output_new))
