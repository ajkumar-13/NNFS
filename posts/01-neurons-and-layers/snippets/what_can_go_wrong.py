"""Post 01, section 13: every failure the post describes, run on the arrays of this post.

Run from the series root:  python posts/01-neurons-and-layers/snippets/what_can_go_wrong.py

Each case prints either the result of a call or the exception it raises. Nothing here is random.
"""
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
batch = [[1.0, 2.0, 3.0, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]


def attempt(call, function):
    try:
        result = function()
    except Exception as error:
        print(f"{call}\n  raises {type(error).__name__}: {str(error).strip()}")
    else:
        print(f"{call}\n  returns shape {np.shape(result)}:\n{result}")


print("1. One sample: the weights go first")
attempt("np.dot(weights, inputs) + biases", lambda: np.dot(weights, inputs) + biases)
attempt("np.dot(inputs, weights) + biases", lambda: np.dot(inputs, weights) + biases)

print("\n2. A batch: the weights need a transpose, and a list has none")
attempt("np.dot(batch, weights) + biases", lambda: np.dot(batch, weights) + biases)
attempt("np.dot(batch, weights.T) + biases", lambda: np.dot(batch, weights.T) + biases)
attempt("np.dot(batch, np.array(weights).T) + biases", lambda: np.dot(batch, np.array(weights).T) + biases)

print("\n3. Transposing the batch instead of the weights")
attempt("np.dot(weights, np.array(batch[:2]).T) + biases, 2 samples",
        lambda: np.dot(weights, np.array(batch[:2]).T) + biases)
attempt("np.dot(weights, np.array(batch).T) + biases, 3 samples",
        lambda: np.dot(weights, np.array(batch).T) + biases)
attempt("np.dot(weights, np.array(batch[:1]).T) + biases, 1 sample",
        lambda: np.dot(weights, np.array(batch[:1]).T) + biases)

print("\n4. What the bias does")
attempt("np.dot(weights, [0.0, 0.0, 0.0, 0.0]) + biases, all inputs zero",
        lambda: np.dot(weights, [0.0, 0.0, 0.0, 0.0]) + biases)
attempt("np.dot(weights, inputs), no biases", lambda: np.dot(weights, inputs))

print("\n5. Floating-point rounding")
print("0.2 + 1.6 - 1.5 =", 0.2 + 1.6 - 1.5)
products = [n_input * weight for n_input, weight in zip(inputs, weights[1])]
print("neuron 2, products added in the order 1, 2, 3, 4:",
      products[0] + products[1] + products[2] + products[3] + biases[1])
print("neuron 2, products added in the order 1, 2, 4, 3:",
      products[0] + products[1] + products[3] + products[2] + biases[1])
loops = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    loops.append(neuron_output + neuron_bias)
with_numpy = np.dot(weights, inputs) + biases
print("loops:      ", loops)
print("np.dot:     ", with_numpy.tolist())
print("largest difference:", f"{np.max(np.abs(np.array(loops) - with_numpy)):.1e}")
print("np.allclose:", np.allclose(loops, with_numpy))

print("\n6. What index 0 means")
print("np.shape(inputs) =", np.shape(inputs), " inputs[0] =", inputs[0])
print("np.shape(batch)  =", np.shape(batch), " batch[0]  =", batch[0])
