"""Post 06, section 4: softmax by hand, the overflow of the naive formula, the max-subtraction fix,
and temperature.

Run from the series root:
    python posts/06-activation-functions-relu-and-softmax/snippets/softmax.py

Needs NumPy only. Arrays are float64 unless a line says float32.
"""
import warnings

import numpy as np


class Activation_Softmax:

    def forward(self, inputs):
        # Subtract the per-row max for stability.
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        # Exponentiate and normalise per row.
        exp_values    = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities


def naive_softmax(logits):
    """The definition with no shift: it breaks when a logit is far from zero."""
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def run_and_report(function, argument):
    """Call function(argument); return the result and the NumPy warnings raised on the way."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(argument)
    return result, [f"{w.category.__name__}: {w.message}" for w in caught]


def show(row):
    """Format one row of probabilities: five decimals, or scientific notation when tiny."""
    return "[" + "  ".join(f"{v:.5f}" if v >= 1e-3 else f"{v:.2e}" for v in row) + "]"


softmax = Activation_Softmax()

# 1. One row by hand: logits [1, 2, 3].
logits = np.array([[1.0, 2.0, 3.0]])
exp_values = np.exp(logits)
print("1. logits          ", logits[0])
print("   exp of each     ", np.round(exp_values[0], 5))
print("   their sum       ", np.round(np.sum(exp_values), 5))
print("   each over sum   ", show(exp_values[0] / np.sum(exp_values)))
softmax.forward(logits)
print("   class output    ", show(softmax.output[0]))
print("   sum of the row  ", np.round(np.sum(softmax.output), 12))

# 2. Shifting every logit of a row by the same constant changes nothing.
for row in ([1.0, 2.0, 3.0], [1001.0, 1002.0, 1003.0], [-999.0, -998.0, -997.0]):
    softmax.forward(np.array([row]))
    print("2. softmax of", row, "=", show(softmax.output[0]))

# 3. The naive formula on the logits of the stability figure.
big = np.array([[1000.0, 1001.0, 999.0]])
result, messages = run_and_report(np.exp, big)
print("3. np.exp of", big[0], "=", result[0])
result, messages = run_and_report(naive_softmax, big)
print("   naive softmax   ", result[0])
for message in messages:
    print("     " + message)
shifted = big - np.max(big, axis=1, keepdims=True)
softmax.forward(big)
print("   shifted logits  ", shifted[0])
print("   exp of those    ", np.round(np.exp(shifted[0]), 3))
print("   stable softmax  ", np.round(softmax.output[0], 3))

# 4. Where np.exp overflows: the logarithm of the largest float of each precision.
limit64 = np.log(np.finfo(np.float64).max)
limit32 = np.log(np.finfo(np.float32).max)
print(f"4. largest float64 = exp({limit64:.2f})   largest float32 = exp({limit32:.2f})")
over64, _ = run_and_report(np.exp, np.float64(710))
over32, _ = run_and_report(np.exp, np.float32(89))
print(f"   float64: exp(709) = {np.exp(np.float64(709)):.3e}   exp(710) = {over64}")
print(f"   float32: exp(88)  = {np.exp(np.float32(88)):.3e}   exp(89)  = {over32}")

# 5. Very negative logits break the naive formula too: every exponential underflows to 0.
small = np.array([[-1000.0, -999.0, -1001.0]])
result, messages = run_and_report(naive_softmax, small)
print("5. naive softmax of", small[0], "=", result[0])
for message in messages:
    print("     " + message)
softmax.forward(small)
print("   stable softmax  ", np.round(softmax.output[0], 3))

# 6. Equal logits give the uniform distribution, whatever their size.
softmax.forward(np.array([[100.0, 100.0, 100.0], [0.0, 0.0, 0.0]]))
print("6. equal logits [100, 100, 100] and [0, 0, 0]:")
print(softmax.output)

# 7. A batch: each row is normalised on its own.
batch = np.array([[1.0, 2.0, 3.0],
                  [2.0, 2.0, 2.0],
                  [0.0, 5.0, 0.0]])
softmax.forward(batch)
print("7. a batch of three rows:")
print(np.round(softmax.output, 3))
print("   row sums:", np.round(np.sum(softmax.output, axis=1), 12))

# 8. Temperature: divide the logits by T before the softmax.
for T in (0.1, 1.0, 10.0):
    softmax.forward(logits / T)
    print(f"8. T = {T:4}: {show(softmax.output[0])}")
