"""Post 01, section 5: one neuron with four inputs, written out term by term.

Run from the series root:  python posts/01-neurons-and-layers/snippets/neuron_four_inputs.py
"""
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = (inputs[0] * weights[0]
          + inputs[1] * weights[1]
          + inputs[2] * weights[2]
          + inputs[3] * weights[3]
          + bias)

print(output)
