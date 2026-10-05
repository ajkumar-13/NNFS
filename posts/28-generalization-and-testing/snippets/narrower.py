"""Post 28, sections 4 and 6: the same run with 8 hidden neurons in place of 64.

Run from the series root:
    python posts/28-generalization-and-testing/snippets/narrower.py

Everything except the width is the setup of seed_spread.py: the same five seeds, 10,001 epochs,
and 300 test points drawn after the weights. Compare the rows with those of seed_spread.py.

Needs NumPy and the nnfs package. Takes about 25 seconds.
"""
import nnfs

from network import SETUP
from seed_spread import SEEDS, final_table

nnfs.init()
print(SETUP)
print(f"changed: Layer_Dense(2, 8) and Layer_Dense(8, 3), 51 parameters; seeds {SEEDS}")
print()
print("== 8 neurons, after the last update")
results = final_table(8, follow=False)
dead = [final["dead"] for final, _ in results]
print("hidden neurons whose output is zero on all 300 training points, after the last update, by seed: "
      + "  ".join(str(d) for d in dead))
print(f"neurons still alive: {8 - max(dead)} to {8 - min(dead)} of 8")
