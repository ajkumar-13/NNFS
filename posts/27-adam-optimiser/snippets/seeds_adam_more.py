"""Post 27, section 7: Adam on five further seeds, 5 to 9, with the settings of all_six.py.

Run from the series root:
    python posts/27-adam-optimiser/snippets/seeds_adam_more.py

Post 26 runs RMSProp on the same five seeds.

Five full runs of the shared setup: about 40 seconds. Needs NumPy and the nnfs package.
"""
from all_six import spread

if __name__ == "__main__":
    spread("Adam (post 27)", seeds=(5, 6, 7, 8, 9))
