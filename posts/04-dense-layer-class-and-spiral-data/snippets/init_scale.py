"""Post 04, section 5.1: what the number in front of randn does as layers are stacked.

Run from the series root:
    python posts/04-dense-layer-class-and-spiral-data/snippets/init_scale.py

Six Layer_Dense layers of 64 neurons each are stacked on the spiral data with nothing between
them, once for each weight scale, and the spread (standard deviation) of every layer's output
is printed. The random draws are the same for every scale, so only the scale differs.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Small random weights; zero biases.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


N_LAYERS = 6
WIDTH = 64
SCALES = [0.01, 1.0, 0.125]          # 0.125 is 1 / sqrt(64)

nnfs.init()


def output_spreads(scale):
    """Standard deviation of each layer's output when every weight is scale * randn."""
    np.random.seed(0)                # replay the same data and the same draws for every scale
    X, y = spiral_data(samples=100, classes=3)
    spreads = []
    values, n_inputs = X, 2
    for _ in range(N_LAYERS):
        layer = Layer_Dense(n_inputs, WIDTH)
        layer.weights = layer.weights * (scale / 0.01)   # the class draws 0.01 * randn; rescale it
        layer.forward(values)
        spreads.append(float(np.std(layer.output)))
        values, n_inputs = layer.output, WIDTH
    return spreads


table = {scale: output_spreads(scale) for scale in SCALES}

print("standard deviation of each layer's output, 64 neurons per layer")
print("layer   " + "   ".join(f"scale {scale:<7}" for scale in SCALES).rstrip())
for i in range(N_LAYERS):
    print(f"{i + 1:<8}" + "   ".join(f"{table[scale][i]:<13.3e}" for scale in SCALES).rstrip())

print("average change per extra layer (layer 6 over layer 1, fifth root):")
for scale in SCALES:
    factor = (table[scale][-1] / table[scale][0]) ** (1 / (N_LAYERS - 1))
    print(f"  scale {scale}: x {factor:.3f}   (scale * sqrt(64) = {scale * np.sqrt(WIDTH):.2f})")
