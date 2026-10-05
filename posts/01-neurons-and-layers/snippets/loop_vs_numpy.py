"""Post 01, section 9: time the nested Python loops of section 8 against np.dot.

Run from the series root:  python posts/01-neurons-and-layers/snippets/loop_vs_numpy.py

The weights, inputs, and biases are drawn from a generator seeded with 0, so the arithmetic is the
same on every run. The timings are not: they depend on the machine and on what else it is doing,
so the ratio is the number to read, not the microseconds.
"""
import timeit

import numpy as np


def layer_loops(inputs, weights, biases):
    layer_outputs = []
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)
    return layer_outputs


def layer_numpy(inputs, weights, biases):
    return np.dot(weights, inputs) + biases


def best_seconds(function, arguments, calls):
    """Shortest time per call over five rounds of `calls` calls each."""
    return min(timeit.repeat(lambda: function(*arguments), repeat=5, number=calls)) / calls


def compare(label, n_neurons, n_inputs, calls):
    rng = np.random.default_rng(0)
    weights = rng.standard_normal((n_neurons, n_inputs))
    inputs = rng.standard_normal(n_inputs)
    biases = rng.standard_normal(n_neurons)
    as_arrays = (inputs, weights, biases)
    as_lists = (inputs.tolist(), weights.tolist(), biases.tolist())

    loops = best_seconds(layer_loops, as_lists, calls)
    numpy_on_lists = best_seconds(layer_numpy, as_lists, calls)
    numpy_on_arrays = best_seconds(layer_numpy, as_arrays, 2000)
    gap = np.max(np.abs(np.array(layer_loops(*as_lists)) - layer_numpy(*as_arrays)))

    print(f"{label}: {n_neurons} neurons, {n_inputs} inputs, {n_neurons * n_inputs:,} multiplications")
    print(f"  nested Python loops on lists {loops * 1e6:10.1f} microseconds")
    print(f"  np.dot on the same lists     {numpy_on_lists * 1e6:10.1f} microseconds")
    print(f"  np.dot on NumPy arrays       {numpy_on_arrays * 1e6:10.1f} microseconds")
    print(f"  loops / np.dot on arrays     {loops / numpy_on_arrays:10.1f} times")
    print(f"  largest difference between the two results: {gap:.1e}")


compare("the layer of this post", 3, 4, calls=5000)
compare("a layer the size of an MNIST hidden layer", 128, 784, calls=3)
