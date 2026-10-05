"""Post 24, section 8: the momentum coefficients 0.5 and 0.99 over the five seeds of seed_spread.py.

Run from the series root:
    python posts/24-momentum/snippets/beta_sweep.py

Same setup, same 10,001 epochs, same decay of 1e-3. seed_spread.py holds the rows for no
momentum and for 0.9; this script adds 0.5 and 0.99.

Needs NumPy and the nnfs package. Ten full runs: about 70 seconds.
"""
import nnfs

from momentum_sgd import SETUP, Optimizer_SGD, train

SEEDS = [0, 1, 2, 3, 4]

if __name__ == "__main__":
    nnfs.init()
    print(SETUP)
    print(f"seeds: {SEEDS}, each set with np.random.seed after nnfs.init()")
    print()
    print("seed  momentum  final loss  accuracy  peak loss  step against the last")

    rows = {}
    for momentum in (0.5, 0.99):
        for seed in SEEDS:
            optimizer = Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=momentum)
            r = train(optimizer, seed=seed)
            rows[momentum, seed] = r
            print(f"{seed:4d}  {momentum:8.2f}  {r['loss']:10.4f}  {r['accuracy']:8.4f}  {r['peak_loss']:9.3f}"
                  f"  {r['opposed']:21,d}", flush=True)

    print()
    for momentum in (0.5, 0.99):
        for key, label in (("loss", "final loss"), ("accuracy", "accuracy")):
            values = [rows[momentum, seed][key] for seed in SEEDS]
            print(f"momentum {momentum:<4}  {label:10s}  {min(values):.4f} to {max(values):.4f}, "
                  f"mean {sum(values) / len(values):.4f}")
