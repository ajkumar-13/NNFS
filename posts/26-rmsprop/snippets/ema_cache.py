"""Post 26, sections 2, 3 and 9: the exponential moving average as a cache, on hand-made gradients.

Run from the series root:
    python posts/26-rmsprop/snippets/ema_cache.py

Contents: how fast an old sample fades for three decay factors; AdaGrad's sum against RMSProp's
average for a gradient of constant size; the same two caches when the gradient shrinks; and the
size of RMSProp's first step.

Needs only NumPy. Nothing is random, and everything is float64.
"""
import numpy as np


def adagrad_cache(gradients):
    """AdaGrad's cache after every step: the running sum of the squared gradients."""
    return np.cumsum(np.asarray(gradients, dtype=np.float64) ** 2)


def rmsprop_cache(gradients, rho):
    """RMSProp's cache after every step: an exponential moving average of the squared gradients."""
    cache = 0.0
    history = []
    for g in gradients:
        cache = rho * cache + (1 - rho) * g ** 2
        history.append(cache)
    return np.array(history)


print("== Section 2: how fast an old sample fades")
print("rho     1/(1-rho)  rho^10      rho^100     rho^1000    rho^(1/(1-rho))  weight of the newest 1/(1-rho)")
for rho in (0.9, 0.99, 0.999):
    horizon = round(1 / (1 - rho))
    print(f"{rho:<7} {horizon:<10d} {rho ** 10:<11.4f} {rho ** 100:<11.2e} {rho ** 1000:<11.2e} "
          f"{rho ** horizon:<16.4f} {1 - rho ** horizon:.4f}")
print(f"1/e = {np.exp(-1):.4f}")

weights = (1 - 0.9) * 0.9 ** np.arange(200)       # weight of the sample that is k steps old
print(f"rho 0.9: weight of the newest sample {weights[0]:.4f}, of the one 10 steps old {weights[10]:.4f}, "
      f"sum of the first 200 weights {weights.sum():.6f}")

print()
print("== Section 3: a gradient of constant size 0.5")
steps = 100_000
g = np.full(steps, 0.5)
ada = adagrad_cache(g)
rms = rmsprop_cache(g, rho=0.9)
print("     t   AdaGrad G   1/sqrt(G)   RMSProp G   1/sqrt(G)")
for t in (1, 2, 10, 100, 1_000, 10_000, 100_000):
    print(f"{t:6d}   {ada[t - 1]:9.3f}   {1 / np.sqrt(ada[t - 1]):9.4f}   "
          f"{rms[t - 1]:9.4f}   {1 / np.sqrt(rms[t - 1]):9.4f}")
closed_form = 0.25 * (1 - 0.9 ** np.arange(1, steps + 1))
print(f"largest gap between the loop and g^2 (1 - rho^t): {np.max(np.abs(rms - closed_form)):.1e}")
print(f"first t at which RMSProp's cache is within 1 percent of 0.25: {int(np.argmax(rms > 0.99 * 0.25)) + 1}")

print()
print("== Section 3: the gradient shrinks from 2 to 0.1 after 1,000 steps")
g = np.concatenate([np.full(1_000, 2.0), np.full(20_000, 0.1)])
ada = adagrad_cache(g)
print("     t   AdaGrad G   RMSProp 0.9   RMSProp 0.999")
slow = rmsprop_cache(g, rho=0.999)
fast = rmsprop_cache(g, rho=0.9)
for t in (1_000, 1_010, 1_050, 1_100, 2_000, 11_000, 21_000):
    print(f"{t:6d}   {ada[t - 1]:9.2f}   {fast[t - 1]:11.4f}   {slow[t - 1]:13.4f}")
for rho, cache in ((0.9, fast), (0.999, slow)):
    settled = int(np.argmax(cache[1_000:] < 1.01 * 0.01)) + 1
    print(f"rho {rho}: within 1 percent of the new level 0.01 after {settled:,} steps")
print(f"AdaGrad: 1/sqrt(G) is {1 / np.sqrt(ada[999]):.4f} at t = 1,000 and {1 / np.sqrt(ada[-1]):.4f} at t = 21,000; "
      f"RMSProp 0.9: {1 / np.sqrt(fast[999]):.4f} and {1 / np.sqrt(fast[-1]):.4f}")

print()
print("== Section 9: the first step, cache started at zero")
print("rho     cache after step 1   |step 1| / learning rate = 1/sqrt(1 - rho)")
for rho in (0.9, 0.99, 0.999):
    for gradient in (0.5, 0.001):
        cache = (1 - rho) * gradient ** 2
        print(f"{rho:<7} g = {gradient:<6} {cache:.3e}   {gradient / np.sqrt(cache):.2f}")
