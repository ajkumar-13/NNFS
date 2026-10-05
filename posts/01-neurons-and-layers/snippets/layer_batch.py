"""Post 01, section 10: a batch of three samples through the three-neuron layer in one call.

Run from the series root:  python posts/01-neurons-and-layers/snippets/layer_batch.py
"""
import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],     # Sample 1
          [2.0, 5.0, -1.0, 2.0],    # Sample 2
          [-1.5, 2.7, 3.3, -0.8]]   # Sample 3

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

print(np.dot(inputs, np.array(weights).T) + biases)
