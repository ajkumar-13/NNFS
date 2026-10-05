"""Post 03: a batch of three samples through two stacked dense layers.

Run from the series root:  python posts/03-stacking-layers-and-the-forward-pass/snippets/two_layer_forward.py
"""
import numpy as np

# A batch of 3 samples, 4 features each.
inputs = np.array([[ 1.0,  2.0,  3.0,  2.5],
                   [ 2.0,  5.0, -1.0,  2.0],
                   [-1.5,  2.7,  3.3, -0.8]])

# Layer 1: 3 neurons, 4 weights each.
weights1 = np.array([[ 0.2,   0.8,  -0.5,   1.0 ],
                     [ 0.5,  -0.91,  0.26, -0.5 ],
                     [-0.26, -0.27,  0.17,  0.87]])
biases1  = np.array([2.0, 3.0, 0.5])

# Layer 2: 3 neurons, 3 weights each.
weights2 = np.array([[ 0.1,  -0.14,  0.5 ],
                     [-0.5,   0.12, -0.33],
                     [-0.44,  0.73, -0.13]])
biases2  = np.array([-1.0, 2.0, -0.5])

# Forward pass.
layer1_outputs = np.dot(inputs,         weights1.T) + biases1
layer2_outputs = np.dot(layer1_outputs, weights2.T) + biases2

print("Layer 1 outputs:", layer1_outputs.shape)
print(layer1_outputs)
print("\nLayer 2 outputs:", layer2_outputs.shape)
print(layer2_outputs)
