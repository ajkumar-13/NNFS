"""Post 24, section 7: decay only against decay plus momentum 0.9, over five seeds.

Run from the series root:
    python posts/24-momentum/snippets/seed_spread.py

The setup and the 10,001 epochs are those of momentum_sgd.py. Each run calls np.random.seed(s)
after nnfs.init(), so the data and the initial weights both change with the seed. Seed 0
reproduces the documented run.

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
    print("seed  optimiser      loss@1000  final loss  accuracy  peak loss  loss rose  step against the last")

    rows = {}
    for seed in SEEDS:
        for name, momentum in (("decay only", 0.0), ("momentum 0.9", 0.9)):
            optimizer = Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=momentum)
            r = train(optimizer, seed=seed)
            rows[seed, name] = r
            print(f"{seed:4d}  {name:13s}  {r['losses'][1000]:9.4f}  {r['loss']:10.4f}  {r['accuracy']:8.4f}"
                  f"  {r['peak_loss']:9.3f}  {r['rises']:9,d}  {r['opposed']:21,d}", flush=True)

    print()
    for name in ("decay only", "momentum 0.9"):
        runs = [rows[seed, name] for seed in SEEDS]
        for key, label in (("loss", "final loss"), ("accuracy", "accuracy"), ("peak_loss", "peak loss"),
                           ("opposed", "steps against the last")):
            values = [r[key] for r in runs]
            if key == "opposed":
                print(f"{name:13s} {label:22s} {min(values):,d} to {max(values):,d}")
            else:
                print(f"{name:13s} {label:22s} {min(values):.4f} to {max(values):.4f}, "
                      f"mean {sum(values) / len(values):.4f}")

    def wins(better):
        return sum(bool(better(rows[seed, "momentum 0.9"], rows[seed, "decay only"])) for seed in SEEDS)

    print()
    print(f"momentum has the lower final loss in {wins(lambda m, d: m['loss'] < d['loss'])} of {len(SEEDS)} seeds")
    print(f"momentum has the higher accuracy in {wins(lambda m, d: m['accuracy'] > d['accuracy'])} of {len(SEEDS)} seeds")
    print(f"momentum has the lower loss at epoch 1000 in "
          f"{wins(lambda m, d: m['losses'][1000] < d['losses'][1000])} of {len(SEEDS)} seeds")
    print(f"momentum has the higher peak loss in {wins(lambda m, d: m['peak_loss'] > d['peak_loss'])} of {len(SEEDS)} seeds")
    print(f"momentum has fewer steps against the last in "
          f"{wins(lambda m, d: m['opposed'] < d['opposed'])} of {len(SEEDS)} seeds")
