"""Post 03: two dense layers with no activation between them are one dense layer.

Run from the series root:  python posts/03-stacking-layers-and-the-forward-pass/snippets/linear_collapse.py

Builds the single equivalent layer of section 4 from the two layers of section 8, checks that it
reproduces the two-layer output, counts the parameters of both forms, and sets the layer-2
weights to the identity matrix to show that layer 2 then only adds its bias.
"""
import numpy as np

inputs = np.array([[ 1.0,  2.0,  3.0,  2.5],
                   [ 2.0,  5.0, -1.0,  2.0],
                   [-1.5,  2.7,  3.3, -0.8]])

weights1 = np.array([[ 0.2,   0.8,  -0.5,   1.0 ],
                     [ 0.5,  -0.91,  0.26, -0.5 ],
                     [-0.26, -0.27,  0.17,  0.87]])
biases1  = np.array([2.0, 3.0, 0.5])

weights2 = np.array([[ 0.1,  -0.14,  0.5 ],
                     [-0.5,   0.12, -0.33],
                     [-0.44,  0.73, -0.13]])
biases2  = np.array([-1.0, 2.0, -0.5])

# The two-layer chain.
layer1_outputs = np.dot(inputs,         weights1.T) + biases1
layer2_outputs = np.dot(layer1_outputs, weights2.T) + biases2

# The single equivalent layer of section 4.
weights_star = np.dot(weights1.T, weights2.T)           # (4, 3)
biases_star  = np.dot(biases1, weights2.T) + biases2    # (3,)
collapsed    = np.dot(inputs, weights_star) + biases_star

print("W_star, shape", weights_star.shape)
print(weights_star)
print("b_star, shape", biases_star.shape)
print(biases_star)
print("One layer, X . W_star + b_star:")
print(collapsed)
difference = np.max(np.abs(collapsed - layer2_outputs))
print(f"largest absolute difference from the two-layer output: {difference:.1e}")
print("W_star is the transpose of W2 . W1:",
      np.allclose(weights_star, np.dot(weights2, weights1).T))

# Parameters: every neuron owns one weight per input and one bias.
two_layer_params = weights1.size + biases1.size + weights2.size + biases2.size
one_layer_params = weights_star.size + biases_star.size
print("parameters in the two layers:", two_layer_params)
print("parameters in the equivalent layer:", one_layer_params)

# Identity experiment: with W2 = I, layer 2 only adds its bias to the layer-1 output.
identity = np.eye(3)
identity_outputs = np.dot(layer1_outputs, identity.T) + biases2
print("W2 = identity gives Z1 + b2:")
print(identity_outputs)
print("equal to layer1_outputs + biases2:",
      np.array_equal(identity_outputs, layer1_outputs + biases2))
