"""Post 09, section 5.1: gradient descent on one parameter, and what the learning rate does to it.

Run from the series root:
    python posts/09-introduction-to-optimisation/snippets/learning_rate_1d.py

The loss is f(w) = w ** 2, whose slope at w is 2 * w (post 10 derives it). Starting from w = 5,
the update rule w = w - learning_rate * slope(w) is applied 100 times for five learning rates.
Needs nothing beyond the standard library.
"""


def f(w):
    return w ** 2


def slope(w):
    return 2 * w


# One step, written out.
w, learning_rate = 5.0, 0.1
print(f"start:    w = {w}, f(w) = {f(w)}, slope = {slope(w)}")
w = w - learning_rate * slope(w)
print(f"one step: w = {w}, f(w) = {f(w)}")

# The same step with the sign flipped climbs instead.
w = 5.0
climb = []
for step in range(3):
    w = w + learning_rate * slope(w)
    climb.append(f(w))
print("plus sign, f(w) after 1, 2, 3 steps:", ", ".join(f"{value:.4g}" for value in climb))

print("learning rate   1 step      2 steps     3 steps     100 steps")
for learning_rate in (0.001, 0.01, 0.1, 1.0, 10.0):
    w = 5.0
    path = []
    for step in range(100):
        w = w - learning_rate * slope(w)
        path.append(w)
    print(f"{learning_rate:<15} {path[0]:<11.4g} {path[1]:<11.4g} {path[2]:<11.4g} {path[99]:.4g}")

# Each step multiplies w by (1 - 2 * learning_rate), so 100 steps multiply it by that factor 100 times.
print("closed form, 5 * (1 - 2 * learning_rate) ** 100:")
for learning_rate in (0.001, 0.01, 0.1, 1.0, 10.0):
    print(f"  {learning_rate:<6} {5 * (1 - 2 * learning_rate) ** 100:.4g}")
