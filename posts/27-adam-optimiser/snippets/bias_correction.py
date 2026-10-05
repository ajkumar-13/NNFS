"""Post 27, section 3: what the zero start does to the two moving averages, and what the correction does.

Run from the series root:
    python posts/27-adam-optimiser/snippets/bias_correction.py

Contents: the correction factors 1 / (1 - beta^t); a constant gradient traced through both
averages; the mean of the first moment for a gradient that varies; the size of the step with and
without the correction, and for a gradient that flips sign; the first update of Optimizer_Adam on
gradients of very different sizes; and the horizon of each default decay rate.

Arithmetic in float64. Imports Optimizer_Adam from adam.py. Takes about a second.
"""
import numpy as np

from adam import Optimizer_Adam

beta_1, beta_2, epsilon = 0.9, 0.999, 1e-7

print("== correction factors 1 / (1 - beta^t)")
print("     t   beta_1^t  factor for m   beta_2^t  factor for v")
for t in (1, 5, 10, 20, 50, 100, 1000, 5000, 10000):
    print(f"{t:6d}  {beta_1 ** t:9.5f}  {1 / (1 - beta_1 ** t):12.3f}  {beta_2 ** t:9.5f}  {1 / (1 - beta_2 ** t):12.3f}")

print()
print("== a constant gradient g = 0.5 through both averages")
print("  t         m  (1-b1^t) g     m_hat           v  (1-b2^t) g^2     v_hat")
g = 0.5
m = v = 0.0
for t in range(1, 6):
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * g ** 2
    m_hat = m / (1 - beta_1 ** t)
    v_hat = v / (1 - beta_2 ** t)
    print(f"{t:3d}  {m:8.5f}  {(1 - beta_1 ** t) * g:10.5f}  {m_hat:8.5f}  {v:10.7f}  {(1 - beta_2 ** t) * g ** 2:12.7f}  {v_hat:8.5f}")

print()
print("== a gradient that varies: 200,000 sequences drawn from a normal with mean 1 and standard deviation 2")
rng = np.random.default_rng(0)
m = np.zeros(200_000)
print("  t  mean of m  1-b1^t  mean of m_hat")
for t in range(1, 11):
    m = beta_1 * m + (1 - beta_1) * rng.normal(1.0, 2.0, size=m.shape)
    if t in (1, 3, 10):
        print(f"{t:3d}  {m.mean():9.4f}  {1 - beta_1 ** t:6.4f}  {(m / (1 - beta_1 ** t)).mean():13.4f}")

print()
print("== size of the step in units of the learning rate, constant gradient, epsilon left out")
print("     t  corrected  uncorrected = (1-b1^t) / sqrt(1-b2^t)")
ratio = lambda t: (1 - beta_1 ** t) / np.sqrt(1 - beta_2 ** t)
for t in (1, 2, 10, 100, 1000, 5000, 10000):
    print(f"{t:6d}  {1.0:9.3f}  {ratio(t):11.3f}")
steps = np.arange(1, 10001)
print(f"largest uncorrected step: {ratio(steps).max():.3f} learning rates at t = {steps[ratio(steps).argmax()]}")
print(f"uncorrected step still above 1.1 learning rates until t = {steps[ratio(steps) > 1.1].max()}")

print()
print("== corrected step when the gradient flips sign at every update: +1, -1, +1, ...")
m = v = 0.0
for t in range(1, 1001):
    g = 1.0 if t % 2 else -1.0
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * g ** 2
    if t in (1, 2, 999, 1000):
        step = (m / (1 - beta_1 ** t)) / (np.sqrt(v / (1 - beta_2 ** t)) + epsilon)
        print(f"{t:6d}  step {step:+.4f} learning rates")
print(f"limit (1 - beta_1) / (1 + beta_1) = {(1 - beta_1) / (1 + beta_1):.4f}")


class Stub:
    """Stands in for a dense layer: four weights, one bias."""


print()
print("== the first update of Optimizer_Adam(learning_rate=0.001) on gradients of very different sizes")
layer = Stub()
layer.weights, layer.biases = np.zeros((1, 4)), np.zeros((1, 1))
layer.dweights = np.array([[-1e4, -1.0, 1e-4, 3.0]])
layer.dbiases = np.array([[1e-8]])
optimizer = Optimizer_Adam()
optimizer.pre_update_params()
optimizer.update_params(layer)
optimizer.post_update_params()
print("gradients       ", layer.dweights[0], layer.dbiases[0])
print("weights after   ", layer.weights[0])
print("bias after      ", layer.biases[0], " = -0.001 * 1e-8 / (1e-8 + 1e-7)")
print("iterations after", optimizer.iterations)

print()
print("== what each default means")
for name, beta in (("beta_1", beta_1), ("beta_2", beta_2)):
    horizon = round(1 / (1 - beta))
    print(f"{name} = {beta}: horizon 1 / (1 - beta) = {horizon} steps, "
          f"which carry {1 - beta ** horizon:.3f} of the total weight")
