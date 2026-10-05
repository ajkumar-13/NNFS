"""Post 27, section 3.2: Adam without bias correction at a learning rate of 0.1, over five seeds.

Run from the series root:
    python posts/27-adam-optimiser/snippets/no_bias_correction_high.py

Compare with the output of rate_high.py: same setup, seeds, learning rate and decay, with the correction.

Five full runs of the shared setup: about 40 seconds. Needs NumPy and the nnfs package.
"""
from all_six import spread
from no_bias_correction import Optimizer_Adam_Uncorrected

if __name__ == "__main__":
    print("optimiser: Adam without bias correction, learning_rate=0.1, decay=1e-5")
    spread("Adam, uncorrected, 0.1", build=lambda: Optimizer_Adam_Uncorrected(learning_rate=0.1, decay=1e-5))
