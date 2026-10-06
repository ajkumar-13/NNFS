"""Post 32, sections 1, 3 and 5: the counts and closed forms the prose quotes. No training.

Run from the series root:
    python posts/32-mini-batching/snippets/counts.py

Needs only the standard library. Takes well under a second.
"""
import math


def batches(n_samples, batch_size):
    """Updates in one epoch, and the number of rows in the last batch."""
    n_batches = math.ceil(n_samples / batch_size)
    return n_batches, n_samples - batch_size * (n_batches - 1)


def decayed(learning_rate, decay, iterations):
    """The rate pre_update_params sets when the counter reads iterations (post 23)."""
    return learning_rate / (1.0 + decay * iterations)


print("== Section 1: arrays of one full-batch forward pass")
print(f"MNIST inputs, 60,000 x 784 float32: {60000 * 784 * 4 / 1e6:,.1f} MB")
print(f"first hidden layer of 128, 60,000 x 128 float32: {60000 * 128 * 4 / 1e6:,.1f} MB")
values = 224 * 224 * 3
print(f"one 224 x 224 x 3 image: {values:,} values")
print(f"1,281,167 such images in float32: {1281167 * values * 4 / 1e9:,.0f} GB")

print()
print("== Section 3: batches per epoch, ceil(N / B), and the last batch")
print("     N     B  batches  rows in the last")
for n_samples, batch_size in ((300, 300), (300, 100), (300, 32), (300, 8), (300, 1),
                              (60000, 128), (60000, 100), (60000, 1)):
    n_batches, last = batches(n_samples, batch_size)
    print(f"{n_samples:6,d}  {batch_size:4d}  {n_batches:7,d}  {last:16d}")
print(f"60,000 / 128 = {60000 / 128}")
print(f"20 epochs at N = 60,000, B = 128: {20 * batches(60000, 128)[0]:,} updates; full batch: 20 updates")

print()
print("== Section 5: the decayed rate counts updates")
print("post 23, learning_rate 1.0, decay 1e-3, 10,001 epochs of 300 rows")
print("  B  updates  rate in the last update")
for batch_size in (300, 32):
    updates = 10001 * batches(300, batch_size)[0]
    print(f"{batch_size:3d}  {updates:7,d}  {decayed(1.0, 1e-3, updates - 1):.4f}")
print(f"decay 1e-4 at B = 32: {decayed(1.0, 1e-4, 10001 * 10 - 1):.4f}")
print("updates until the rate has halved, 1 / decay:")
for decay in (1e-2, 1e-3, 1e-4, 1e-5):
    print(f"  decay {decay:g}: {round(1 / decay):,} updates")

print("nn-p01, learning_rate 0.001, 20 epochs of 469 updates")
print("  decay  rate in the last update")
for decay in (1e-4, 1e-3):
    print(f"  {decay:g}  {decayed(0.001, decay, 20 * 469 - 1):.6f}")
