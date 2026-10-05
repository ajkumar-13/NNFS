"""Post 01, section 9: a layer of three neurons as one NumPy dot product plus the biases.

Run from the series root:  python posts/01-neurons-and-layers/snippets/layer_numpy.py
"""
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

print(np.dot(weights, inputs) + biases)
