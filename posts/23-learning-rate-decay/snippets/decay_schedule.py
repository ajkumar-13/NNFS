"""Post 23, sections 2 to 4: the decay schedule and the optimiser class that implements it.

Run from the series root:
    python posts/23-learning-rate-decay/snippets/decay_schedule.py

Contents: Optimizer_SGD with a decay argument and the three-method contract; the learning rate the
class uses at given updates for four decay rates; when the rate reaches a half, a quarter, an
eighth and a tenth; the same budget under an exponential and a step schedule; and the sum of all
the rates of a run.

Needs only NumPy. Nothing is random and nothing is trained.
"""
import numpy as np


class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases  -= self.current_learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1


def rates_of_a_run(decay, updates=10001, learning_rate=1.0):
    """The learning rate the class uses at each update of a run, read off the class itself."""
    optimizer = Optimizer_SGD(learning_rate=learning_rate, decay=decay)
    rates = np.empty(updates)
    for update in range(updates):
        optimizer.pre_update_params()
        rates[update] = optimizer.current_learning_rate     # what update_params would read
        optimizer.post_update_params()
    return rates


UPDATES = 10001                                     # updates 0 to 10000, as in the training loop
decays = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
runs = {d: rates_of_a_run(d, UPDATES) for d in decays}

print("== Section 2: the rate at update t, learning_rate=1.0")
print(("     t   " + "   ".join(f"d={d:<6g}" for d in decays)).rstrip())
for t in (0, 1, 100, 1000, 5000, 10000):
    print(f"{t:6d}   " + "   ".join(f"{runs[d][t]:8.4f}" for d in decays))

closed_form = max(np.max(np.abs(runs[d] - 1.0 / (1.0 + d * np.arange(UPDATES)))) for d in decays)
print(f"largest gap between the class and 1 / (1 + d t): {closed_form:.1e}")
growth = np.diff(1.0 / runs[1e-3])                  # the reciprocal of the rate, from one update to the next
print(f"d=1e-3: 1 / rate grows per update by {growth.min():.6f} to {growth.max():.6f}")

print()
print("== Section 2: the update at which the rate reaches a fraction of its start, d=1e-3")
for label, fraction in (("1/2", 2), ("1/4", 4), ("1/8", 8), ("1/10", 10)):
    t = int(np.argmax(runs[1e-3] <= 1.0 / fraction + 1e-12))
    print(f"alpha_0 * {label:<4}  first at t = {t:5d}   (closed form ({fraction} - 1) / d = {(fraction - 1) / 1e-3:.0f})")

print()
print("== Section 3: three schedules that all start at 1.0 and stand at 0.5 after 1,000 updates")
t_all = np.arange(UPDATES)
gamma = 0.5 ** (1 / 1000)                           # exponential: halves every 1,000 updates
inverse_time = runs[1e-3]
exponential = gamma ** t_all
step = 0.5 ** (t_all // 1000)                       # step: halved at every 1,000th update
print(f"gamma = {gamma:.6f}")
print("     t   inverse time   exponential   step")
for t in (0, 500, 1000, 3000, 7000, 10000):
    print(f"{t:6d}   {inverse_time[t]:12.4f}   {exponential[t]:11.4f}   {step[t]:.4f}")
print(f"0.99 ** 10000 = {0.99 ** 10000:.1e}   half-life of 0.99: {np.log(2) / np.log(1 / 0.99):.1f} updates")

print()
print("== Section 3: the sum of the rates of all 10,001 updates")
for d in decays:
    print(f"d={d:<6g}  sum {runs[d].sum():9.1f}   share of the constant rate {runs[d].sum() / UPDATES:6.1%}")
print(f"exponential, gamma above: sum {exponential.sum():9.1f}")

print()
print("== Section 4: what the class does at the edges")
plain = Optimizer_SGD()
plain.pre_update_params()
print(f"defaults: learning_rate {plain.learning_rate}, decay {plain.decay}, iterations {plain.iterations}, "
      f"current_learning_rate after pre_update_params {plain.current_learning_rate}")
first = Optimizer_SGD(learning_rate=1.0, decay=1e-3)
first.pre_update_params()
print(f"decay=1e-3, first update: iterations {first.iterations}, current_learning_rate {first.current_learning_rate}")
first.post_update_params()
first.pre_update_params()
print(f"decay=1e-3, second update: iterations {first.iterations}, current_learning_rate {first.current_learning_rate:.6f}")
print(f"learning_rate after two updates: {first.learning_rate}")
