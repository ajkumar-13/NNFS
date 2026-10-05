"""Post 27, section 3.2: Adam with the two bias-correction lines removed, over five seeds.

Run from the series root:
    python posts/27-adam-optimiser/snippets/no_bias_correction.py

Compare with the output of seeds_adam.py: same setup, same seeds, same learning rate and decay.

Five full runs of the shared setup: about 40 seconds. Needs NumPy and the nnfs package.
"""
import numpy as np

from adam import Optimizer_Adam
from all_six import spread


class Optimizer_Adam_Uncorrected(Optimizer_Adam):
    """Optimizer_Adam with m and v used as they are; nothing else differs."""

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache     = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)
            layer.bias_cache       = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums   = self.beta_1 * layer.bias_momentums + \
                                 (1 - self.beta_1) * layer.dbiases
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache   = self.beta_2 * layer.bias_cache + \
                             (1 - self.beta_2) * layer.dbiases ** 2

        layer.weights -= self.current_learning_rate * layer.weight_momentums / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.bias_momentums / \
                         (np.sqrt(layer.bias_cache)   + self.epsilon)


if __name__ == "__main__":
    print("optimiser: Adam without bias correction, learning_rate=0.02, decay=1e-5")
    spread("Adam, uncorrected", build=lambda: Optimizer_Adam_Uncorrected(learning_rate=0.02, decay=1e-5))
