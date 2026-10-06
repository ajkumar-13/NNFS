"""Post 34, section 7: ten seeds of the two-moons network, once with each head.

Run from the series root:
    python posts/34-sigmoid-and-binary-cross-entropy/snippets/seeds.py

One row per seed, measured forward-only after the last update. A seed fixes the data, the split
and the initial weights together. Both loss columns are data losses, without the L2 penalty.

Float64 weights, no nnfs.init(). Needs only NumPy. Takes 15 to 40 seconds.
"""
from network import SETUP, spread

print(SETUP)
print()
sigmoid_rows = spread("sigmoid")
print()
softmax_rows = spread("softmax")

print()
print("== Seed by seed: test points correct, sigmoid head minus softmax head")
differences = [a[5] - b[5] for a, b in zip(sigmoid_rows, softmax_rows)]
print("differences:", differences)
print(f"sigmoid higher on {sum(d > 0 for d in differences)} seeds, equal on {sum(d == 0 for d in differences)}, "
      f"softmax higher on {sum(d < 0 for d in differences)}")
print(f"lower test data loss: sigmoid on {sum(a[4] < b[4] for a, b in zip(sigmoid_rows, softmax_rows))} seeds, "
      f"softmax on {sum(a[4] > b[4] for a, b in zip(sigmoid_rows, softmax_rows))}")
print(f"lower train data loss: sigmoid on {sum(a[1] < b[1] for a, b in zip(sigmoid_rows, softmax_rows))} seeds, "
      f"softmax on {sum(a[1] > b[1] for a, b in zip(sigmoid_rows, softmax_rows))}")
