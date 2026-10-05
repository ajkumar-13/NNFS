"""Post 19, section 4: the combined gradient of a batch of three, in three lines.

Run from the series root:
    python posts/19-softmax-derivatives-and-the-combined-backward-pass/snippets/worked_example.py

Needs only NumPy.
"""
import numpy as np

softmax_output = np.array([[0.7,  0.1, 0.2 ],
                           [0.1,  0.5, 0.4 ],
                           [0.02, 0.9, 0.08]])

y_true = np.array([0, 1, 1])   # integer class indices

# Combined backward.
dinputs = softmax_output.copy()
dinputs[range(len(y_true)), y_true] -= 1      # subtract 1 at the true class
dinputs /= len(y_true)                         # divide by batch size

print(dinputs)

print("row sums:", dinputs.sum(axis=1).round(12) + 0.0)
print("softmax_output unchanged:", softmax_output[0])
