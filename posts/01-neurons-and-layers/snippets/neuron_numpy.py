"""Post 01, section 9: one neuron as a NumPy dot product plus a bias.

Run from the series root:  python posts/01-neurons-and-layers/snippets/neuron_numpy.py
"""
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

print(np.dot(weights, inputs) + bias)
