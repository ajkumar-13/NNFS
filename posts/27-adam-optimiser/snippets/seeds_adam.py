"""Post 27, section 7: Adam over five seeds, with the settings of all_six.py.

Run from the series root:
    python posts/27-adam-optimiser/snippets/seeds_adam.py

Five full runs of the shared setup: about 40 seconds. Needs NumPy and the nnfs package.
"""
from all_six import spread

if __name__ == "__main__":
    spread("Adam (post 27)")
