"""Post 01, section 4: one neuron with three inputs, written out term by term.

Run from the series root:  python posts/01-neurons-and-layers/snippets/neuron_three_inputs.py
"""
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = (inputs[0] * weights[0]
          + inputs[1] * weights[1]
          + inputs[2] * weights[2]
          + bias)

print(output)
