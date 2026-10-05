"""Post 27, section 7: Adam at the class default learning rate of 0.001, twenty times below the documented run.

Run from the series root:
    python posts/27-adam-optimiser/snippets/rate_low.py

Same setup, seeds and decay as seeds_adam.py; only the learning rate differs.

Five full runs of the shared setup: about 40 seconds. Needs NumPy and the nnfs package.
"""
from adam import Optimizer_Adam
from all_six import spread

if __name__ == "__main__":
    print("optimiser: Optimizer_Adam(learning_rate=0.001, decay=1e-5)")
    spread("Adam, rate 0.001", build=lambda: Optimizer_Adam(learning_rate=0.001, decay=1e-5))
