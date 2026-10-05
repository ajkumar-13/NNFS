"""Post 27, section 7: the six optimisers of Part VI on the documented run.

Run from the series root:
    python posts/27-adam-optimiser/snippets/all_six.py

Contents: Optimizer_SGD as post 24 leaves it (posts 22 and 23 are the same class with the
later arguments at their defaults), Optimizer_Adagrad of post 25, Optimizer_RMSprop of post 26,
the settings each post documents, and one run of each on the shared setup, seed 0.
The network classes, Optimizer_Adam and train() come from adam.py. seeds_adam.py, seeds_adam_more.py,
seeds_rmsprop.py, no_bias_correction.py, no_bias_correction_high.py, rate_low.py and rate_high.py
import spread() from here and repeat one optimiser each over five seeds. nnfs.init() is called once per script and each run reseeds.

Needs NumPy and the nnfs package. Six full runs: about 45 seconds.
"""
import numpy as np
import nnfs

from adam import SETUP, Optimizer_Adam, train


class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.momentum              = momentum
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:
            # The velocity buffers are created on the first call, full of zeros.
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)

            # New velocity = beta * old velocity - current learning rate * gradient.
            weight_updates = self.momentum * layer.weight_momentums \
                           - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums \
                         - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # No momentum: the update of post 23.
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbiases

        # The step is the velocity, added.
        layer.weights += weight_updates
        layer.biases  += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:

    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.epsilon               = epsilon
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        # Lazy cache creation on first call.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        # Accumulate squared gradients.
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache   += layer.dbiases ** 2

        # Per-parameter update.
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:

    def __init__(self, learning_rate=0.02, decay=0.0,
                 epsilon=1e-7, rho=0.9):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.epsilon               = epsilon
        self.rho                   = rho
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        # Lazy cache creation on first call.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        # Moving average of squared gradients: the two statements that differ from AdaGrad.
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache   = self.rho * layer.bias_cache + \
                             (1 - self.rho) * layer.dbiases ** 2

        # Per-parameter update, as in AdaGrad.
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# The settings each post of Part VI documents. A fresh optimiser is built for every run.
OPTIMISERS = {
    "SGD (post 22)":          lambda: Optimizer_SGD(learning_rate=1.0),
    "SGD, decay (post 23)":   lambda: Optimizer_SGD(learning_rate=1.0, decay=1e-3),
    "SGD, momentum (post 24)": lambda: Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9),
    "AdaGrad (post 25)":      lambda: Optimizer_Adagrad(learning_rate=1.0, decay=1e-4),
    "RMSProp (post 26)":      lambda: Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999),
    "Adam (post 27)":         lambda: Optimizer_Adam(learning_rate=0.02, decay=1e-5),
}

HEADER = "optimiser                seed  loss@100  loss@1000  final loss  accuracy  peak acc  epoch of 90%"


def row(name, seed, r):
    reached = "never" if r["epoch_90"] is None else f"{r['epoch_90']:,d}"
    return (f"{name:24s} {seed:4d}  {r['losses'][100]:8.4f}  {r['losses'][1000]:9.4f}  {r['loss']:10.4f}  {r['accuracy']:8.4f}"
            f"  {r['peak_accuracy']:8.4f}  {reached:>12s}")


def spread(name, build=None, seeds=(0, 1, 2, 3, 4)):
    """One optimiser over several seeds: a row per seed, then the range of each column."""
    build = build or OPTIMISERS[name]
    nnfs.init()
    print(SETUP)
    print(f"seeds: {list(seeds)}, each set with np.random.seed after nnfs.init(); seed 0 is the documented run")
    print()
    print(HEADER)
    runs = []
    for seed in seeds:
        runs.append(train(build(), seed=seed))
        print(row(name, seed, runs[-1]), flush=True)
    print()
    for key, label in (("loss@100", "loss@100"), ("loss@1000", "loss@1000"), ("loss", "final loss"),
                       ("accuracy", "final accuracy"), ("peak_accuracy", "peak accuracy")):
        values = [r["losses"][int(key[5:])] if "@" in key else r[key] for r in runs]
        print(f"{label:15s} {min(values):.4f} to {max(values):.4f}, mean {sum(values) / len(values):.4f}")
    reached = [r["epoch_90"] for r in runs if r["epoch_90"] is not None]
    if reached:
        print(f"epoch of 90%    {min(reached):,d} to {max(reached):,d}, reached in {len(reached)} of {len(runs)} runs")
    else:
        print(f"epoch of 90%    reached in 0 of {len(runs)} runs")
    print()
    print("epochs 9,001 to 10,000, seed by seed")
    for label, value in (("mean accuracy", lambda r: r["accuracies"][-1000:].mean()),
                         ("lowest accuracy", lambda r: r["accuracies"][-1000:].min()),
                         ("highest loss", lambda r: r["losses"][-1000:].max())):
        values = [value(r) for r in runs]
        print(f"{label:15s} " + "  ".join(f"{v:.4f}" for v in values) + f"   mean {sum(values) / len(values):.4f}")
    return runs


if __name__ == "__main__":
    nnfs.init()
    print(SETUP)
    print("seed: 0, the seed nnfs.init() sets; one run per optimiser")
    print()
    print(HEADER)
    for name, build in OPTIMISERS.items():
        print(row(name, 0, train(build())), flush=True)
