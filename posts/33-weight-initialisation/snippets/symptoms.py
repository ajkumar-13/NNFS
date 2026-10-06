"""Post 33, section 7: the symptoms of a wrong weight scale, each produced on purpose.

Run from the series root:
    python posts/33-weight-initialisation/snippets/symptoms.py

Setup, as train_stack() in network.py builds it: nnfs.init() once (seed 0, float32, float32 np.dot),
then per run np.random.seed(s), spiral_data(samples=100, classes=3), ten Layer_Dense layers of 64
neurons with ReLU, Layer_Dense(64, 3), the combined softmax and loss, full-batch epochs.

1. Scale 0.01 (init="small"): gradients near zero, an update that changes almost nothing, and a
   loss that stays at ln 3 from the first epoch.
2. Scale 1.0 (plain randn): a finite loss on the first pass and NaN after the first update of
   plain gradient descent; under Adam no NaN, and a loss that comes down over 200 updates.

Needs NumPy and the nnfs package. Takes 30 to 40 seconds.
"""
import warnings

import numpy as np
import nnfs

from network import Optimizer_SGD, Optimizer_Adam, train_stack

nnfs.init()
EPOCHS = 201


def adam():
    return Optimizer_Adam(learning_rate=0.02, decay=1e-5)


print("== 1. Ten hidden layers, init='small'")
r = train_stack(0, 10, "small", Optimizer_SGD(learning_rate=0.0), 1)        # one pass, no change
dense = r["dense"]
before = [layer.weights.copy() for layer in dense]
optimizer = Optimizer_SGD(learning_rate=1.0)
for layer in dense:
    optimizer.update_params(layer)
changed = sum(int(np.sum(layer.weights != old)) for layer, old in zip(dense, before))
total = sum(layer.weights.size for layer in dense)
print(f"seed 0, first pass: loss {r['first_loss']:.7f} (ln 3 = {np.log(3):.7f}), "
      f"largest |dweights| over the 11 layers {r['first_gradient']:.1e}")
print(f"one step of Optimizer_SGD(learning_rate=1.0): {changed} of {total:,} weights changed; "
      f"spacing of float32 at 0.01: {np.spacing(np.float32(0.01)):.1e}")
print(f"loss at epoch {EPOCHS - 1} and accuracy, per seed (ln 3 = {np.log(3):.4f}):")
print("seed  Optimizer_SGD(learning_rate=1.0)   Optimizer_Adam(learning_rate=0.02, decay=1e-5)")
for seed in range(5):
    sgd = train_stack(seed, 10, "small", Optimizer_SGD(learning_rate=1.0), EPOCHS)
    ada = train_stack(seed, 10, "small", adam(), EPOCHS)
    print(f"{seed:4d}  {sgd['loss']:.4f}  {sgd['accuracy']:.4f}                     "
          f"{ada['loss']:.4f}  {ada['accuracy']:.4f}", flush=True)

print()
print("== 2. Ten hidden layers, every weight scale 1.0 (plain randn)")
print(f"-ln(1e-7) = {-np.log(1e-7):.3f}, the clipped loss of one confidently wrong sample; "
      f"two thirds of that = {-np.log(1e-7) * 2 / 3:.3f}")
print("seed  loss, first pass  largest |output|  largest |dweights|  first NaN loss, SGD at 1.0  "
      "loss at epoch 3 and at epoch 200, Adam at 0.02")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for seed in range(10):
        sgd = train_stack(seed, 10, 1.0, Optimizer_SGD(learning_rate=1.0), 4)
        one = train_stack(seed, 10, 1.0, Optimizer_SGD(learning_rate=0.0), 1)
        ada = train_stack(seed, 10, 1.0, adam(), 4)
        late = train_stack(seed, 10, 1.0, adam(), EPOCHS)
        print(f"{seed:4d}  {sgd['first_loss']:16.4f}  {np.abs(one['dense'][-1].output).max():16.1e}  "
              f"{sgd['first_gradient']:18.1e}  epoch {sgd['nan_epoch']!s:<21}  {ada['loss']:.4f}  {late['loss']:.4f}",
              flush=True)
print("warnings raised:", sorted({str(warning.message) for warning in caught}))
