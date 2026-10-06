"""Post 33, sections 2 to 5: the factors of the variance argument, each checked on random numbers.

Run from the series root:
    python posts/33-weight-initialisation/snippets/variance_factors.py

1. One linear layer: the mean square of z against n_in * Var(W) * the mean square of x.
2. What ReLU, tanh and a leaky ReLU keep of the mean square of a zero-mean normal input.
3. One linear layer backwards: the mean square of the gradient against n_out * Var(W).

float64, no nnfs.init(), NumPy only. Takes about two seconds.
"""
import numpy as np

np.random.seed(0)
N_IN, N_OUT = 64, 64
SCALES = {"small": 0.01, "xavier": np.sqrt(2 / (N_IN + N_OUT)), "he": np.sqrt(2 / N_IN)}

print("== 1. Forward through one linear layer, 64 inputs, 100,000 neurons, biases zero")
x = np.random.rand(1, N_IN) + 0.5              # one fixed input row; its mean is 1, not 0
print(f"input: mean {x.mean():.4f}, mean square {np.mean(x ** 2):.4f}, variance {x.var():.4f}")
print("init     n_in * Var(W)   mean square of z / mean square of x   variance of z / variance of x")
draws = np.random.randn(N_IN, 100000)
for name, scale in SCALES.items():
    z = np.dot(x, scale * draws)
    print(f"{name:<7}  {N_IN * scale ** 2:<14.4f}  {np.mean(z ** 2) / np.mean(x ** 2):<36.4f}  {z.var() / x.var():.4f}")

print()
print("== 2. What an activation keeps: z is 1,000,000 draws of a zero-mean normal")
z = np.random.randn(1000000)
relu = np.maximum(0, z)
print(f"ReLU: fraction of zeros {np.mean(relu == 0):.4f}")
print(f"ReLU: mean square of a / mean square of z = {np.mean(relu ** 2) / np.mean(z ** 2):.4f}   (closed form 1/2)")
print(f"ReLU: variance of a / variance of z       = {relu.var() / z.var():.4f}   "
      f"(closed form 1/2 - 1/(2 pi) = {0.5 - 0.5 / np.pi:.4f})")
print(f"ReLU: mean of a = {relu.mean():.4f}   (closed form 1/sqrt(2 pi) = {1 / np.sqrt(2 * np.pi):.4f})")
leak = 0.01
leaky = np.where(z > 0, z, leak * z)
print(f"leaky ReLU, slope {leak}: mean square kept {np.mean(leaky ** 2) / np.mean(z ** 2):.5f}   "
      f"(closed form (1 + slope^2)/2 = {(1 + leak ** 2) / 2:.5f})")
for std in (0.1, 0.5, 1.0, 2.0):
    t = np.tanh(std * z)
    print(f"tanh, z with standard deviation {std}: mean square kept {np.mean(t ** 2) / np.mean((std * z) ** 2):.4f}, "
          f"mean of a {t.mean():+.4f}")

print()
print("== 3. Backward through one linear layer, 64 outputs, 100,000 inputs")
dvalues = np.random.randn(1, N_OUT) * 0.3 + 0.2      # one fixed gradient row arriving from above
print("init     n_out * Var(W)   mean square of dinputs / mean square of dvalues")
draws = np.random.randn(100000, N_OUT)
for name, scale in SCALES.items():
    dinputs = np.dot(dvalues, (scale * draws).T)
    print(f"{name:<7}  {N_OUT * scale ** 2:<15.4f}  {np.mean(dinputs ** 2) / np.mean(dvalues ** 2):.4f}")
