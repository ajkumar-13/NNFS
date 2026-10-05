# 03 - Stacking layers and the forward pass

> **TL;DR.** A deep network is the same single-layer formula applied $L$ times, with each layer's output piped into the next layer's input. This post chains two layers, traces the shapes for a batch of three samples, and shows that a stack with no activation function between its layers is still a single linear layer: the two layers built here hold 27 parameters and compute nothing that one layer of 15 cannot.
>
> **Prerequisites:** [Post 01](../01-neurons-and-layers/index.md), [Post 02](../02-numpy-and-the-dot-product/index.md).
> **Safe to skip?** Skip it if the reader can already chain two `np.dot` calls into a two-layer forward pass, state the shape of every array in it, and say why the result is still one linear layer.
>
> **After reading, you will be able to:**
>
> - Implement a two-layer forward pass in NumPy by feeding the first layer's output into the second.
> - Predict the shape of every intermediate array in a stack of dense layers before running the code.
> - Pick the weight-matrix shape of each layer from the sizes of the layers around it.
> - Explain in one sentence why dense layers stacked without activations are no more powerful than a single layer.

![A two-layer network drawn left to right. Four inputs with the shape badge (N, 4) connect to each of the three neurons of layer 1, whose formula reads Z1 = X times W1 transposed plus b1, with W1 of shape (3, 4) and b1 of shape (3,). The three values of Z1, badge (N, 3), connect to each of the three neurons of layer 2, whose formula reads Z2 = Z1 times W2 transposed plus b2, with W2 of shape (3, 3) and b2 of shape (3,). The three values of Z2, badge (N, 3), are the final output.](diagrams/01-multi-layer-anatomy.svg)

*Each layer box is the same operation: one dot product per neuron, plus that neuron's bias. The depth comes from chaining the boxes, not from adding new mathematics.*

---

## 1. The question: how does one layer feed the next?

[Post 01](../01-neurons-and-layers/index.md) and [post 02](../02-numpy-and-the-dot-product/index.md) built the smallest interesting object in deep learning: a single dense layer, a layer in which every neuron reads every input, takes a weighted sum, and adds its own bias. One such layer is enough to recognise simple linear patterns and not much else. The power of the field comes from stacking, where the output of one layer is fed as the input to the next.

Stacking raises two questions, and this post answers both. How does one layer's output become the next layer's input, and what must be true of the shapes for that to work? And once two layers are chained, has the network gained anything, or is the stack still one linear function?

The mathematics of stacking does not introduce a new operation. Each layer applies the same call:

$$\mathbf{Z}_\ell = \mathbf{X}_\ell \mathbf{W}_\ell^\top + \mathbf{b}_\ell,$$

where the subscript $\ell$ numbers the layers from 1, the product $\mathbf{X}_\ell \mathbf{W}_\ell^\top$ is the matrix product that `np.dot` computes, and $\mathbf{X}_\ell$ is either the original input (for layer 1) or the previous layer's output $\mathbf{Z}_{\ell - 1}$. Running these calls in order, from the input to the last layer, is the **forward pass**. Chaining two of them is the entire forward pass of a two-layer network. Chaining fifty of them gives a fifty-layer network. No new arithmetic is invented along the way; the only new thing to track is shape continuity, namely that the number of weights per neuron in layer $\ell$ must equal the number of neurons in layer $\ell - 1$.

The idea of stacking is older than the algorithm now used to train stacks. Ivakhnenko and Lapa published multi-layer networks fitted one layer at a time in 1965, work often cited as the origin of the group method of data handling, twenty-one years before Rumelhart, Hinton, and Williams made backpropagation the standard way to train every layer of a stack at once (Ivakhnenko and Lapa, 1965; Rumelhart, Hinton, and Williams, 1986). The forward pass was understood from the start. What took longer was an efficient, general way to learn the weights of all the layers together, and that is built step by step in posts 09 to 22.

## 2. A recap of what one layer does

A single dense layer is fully specified by three things:

| Symbol | Shape | What it is |
|---|---|---|
| $\mathbf{X}$ | $(N, n)$ | a batch of $N$ samples, each with $n$ features |
| $\mathbf{W}$ | $(m, n)$ | weights for $m$ neurons, each consuming $n$ inputs |
| $\mathbf{b}$ | $(m,)$ | one bias per neuron |

The output is:

$$\mathbf{Z} = \mathbf{X} \mathbf{W}^\top + \mathbf{b},$$

and has shape $(N, m)$: one row per sample, one column per neuron. Here $n$ and $m$ are short for the $n_\text{inputs}$ and $n_\text{neurons}$ of the [notation guide](../../notation_guide.md).

The transpose is bookkeeping, as discussed in [post 02](../02-numpy-and-the-dot-product/index.md), and exists because posts 01 to 03 store the weights with one row of $\mathbf{W}$ per neuron. PyTorch's `nn.Linear` stores its weight the same way, with shape `(out_features, in_features)`; Keras's `Dense` stores the transpose, `(input_dim, units)`, one column per neuron. [Post 04](../04-dense-layer-class-and-spiral-data/index.md) switches this series to the second layout, shape $(n, m)$, so that the forward call needs no `.T`; until then every batched call carries the transpose. The bias changes there too, from the 1-D array of shape $(m,)$ used here to a row of shape $(1, m)$, and both are added to every row of the output in the same way.

## 3. The architecture, drawn out

The network of this post has four input features, three neurons in its first layer, and three neurons in its second layer: $n = 4$, $m_1 = 3$, $m_2 = 3$, with $m_\ell$ the number of neurons in layer $\ell$. The second layer produces the network's output, so it is the output layer. The first layer sits between the input and the output layer, which makes it a **hidden layer**: its values are internal to the network and are never compared with a target. In a deeper stack every layer except the last is hidden.

The weight matrices follow from one rule:

> The number of weights per neuron in layer $\ell$ equals the number of neurons in layer $\ell - 1$.

For layer 1 the "previous layer" is the input itself, with its $n$ features. Reading the rule off the architecture:

- Layer 1 has $m_1 = 3$ neurons, each receiving $n = 4$ inputs. $\mathbf{W}_1$ has shape $(3, 4)$.
- Layer 2 has $m_2 = 3$ neurons, each receiving $m_1 = 3$ inputs. $\mathbf{W}_2$ has shape $(3, 3)$.

The biases $\mathbf{b}_1$ and $\mathbf{b}_2$ have shape $(3,)$ each, one bias per neuron in their layer.

### 3.1. What stacking does *not* do

A boundary section, because post 06 depends on it.

- **Stacking does not add expressive power on its own.** A composition of linear maps is itself a linear map: $\mathbf{X} \mapsto \mathbf{X} \mathbf{W}_1^\top \mathbf{W}_2^\top + (\text{constants})$. Strictly, a dense layer is an affine map, a linear map plus the constant shift of its bias; this series follows common usage and calls it linear. The two layers can be collapsed into a single equivalent layer of the same input and output shape; section 4 works the substitution out line by line and names that layer's weight and bias. Without a non-linearity between them, depth is decorative.
- **Stacking does not create features automatically.** The intermediate array $\mathbf{Z}_1$ is sometimes called a "representation" (the re-encoding of the input that a layer hands to the next layer), but for a stack of linear layers that representation is just another linear projection of the input. Real feature learning needs the activations introduced in [post 06](../06-activation-functions-relu-and-softmax/index.md).
- **Stacking does not change the cost of `np.dot`.** Each layer is one matrix multiplication, and the total cost is the sum over the layers: layer $\ell$ does $m_\ell \cdot m_{\ell - 1}$ multiplications per sample, so the two layers here do $3 \cdot 4 + 3 \cdot 3 = 21$. Adding a layer adds its own term to that sum. Widening is steeper: a layer that maps $m$ inputs to $m$ neurons does $m^2$ multiplications per sample, so doubling the width of every layer in a stack roughly quadruples the work.

This post leaves the activation function out on purpose. The goal here is to make the forward pass mechanical and to set up the structural argument above; activations enter in post 06, where the stack stops being linear.

The universal-approximation results of Cybenko (1989) and Hornik (1991) explain why post 06 matters: a network with a single hidden layer, a suitable non-linear activation, and enough neurons in that layer can approximate any continuous function on a closed, bounded region of its input space as closely as desired. Cybenko proved it for sigmoid-shaped activations and Hornik for a much wider class. The non-linearity is doing the heavy lifting, not the depth.

Depth still matters once activations are in place, but it is not a free win. A deeper network does not always beat a wider one: for a fixed number of parameters, depth against width is a genuine trade-off. Very deep networks gain expressive power but are harder to optimise (vanishing gradients, in particular). This series stays with shallow networks because the implementation is clearer there; the same principles scale.

## 4. The forward-pass chain

For the two-layer network above, the forward pass is two calls:

$$\mathbf{Z}_1 = \mathbf{X} \mathbf{W}_1^\top + \mathbf{b}_1$$

$$\mathbf{Z}_2 = \mathbf{Z}_1 \mathbf{W}_2^\top + \mathbf{b}_2$$

Step 1 produces an array of shape $(N, 3)$. Step 2 takes that as its input and produces another array of shape $(N, 3)$. Replacing $\mathbf{Z}_1$ in the second line with the right-hand side of the first gives:

$$\mathbf{Z}_2 = (\mathbf{X} \mathbf{W}_1^\top + \mathbf{b}_1) \mathbf{W}_2^\top + \mathbf{b}_2.$$

Multiplying out and grouping the constant pieces shows the collapse explicitly:

$$\mathbf{Z}_2 = \mathbf{X} \underbrace{(\mathbf{W}_1^\top \mathbf{W}_2^\top)}_{\mathbf{W}_\ast} + \underbrace{(\mathbf{b}_1 \mathbf{W}_2^\top + \mathbf{b}_2)}_{\mathbf{b}_\ast} = \mathbf{X} \mathbf{W}_\ast + \mathbf{b}_\ast.$$

The bias multiplies out cleanly because $\mathbf{b}_1$ is the same row for every sample: passing it through layer 2 gives every sample the same extra row $\mathbf{b}_1 \mathbf{W}_2^\top$, a constant that merges with $\mathbf{b}_2$.

![Two chains, one above the other. The upper chain is the two-layer forward pass: X of shape (N, 4) passes through layer 1, labelled W1 transposed and b1, to Z1 of shape (N, 3), then through layer 2, labelled W2 transposed and b2, to Z2 of shape (N, 3). An arrow labelled collapses to leads down to the lower chain, where the same X passes through one layer, labelled W star and b star, to the same Z2 of shape (N, 3). A band below gives the substitution: W star is W1 transposed times W2 transposed, and b star is b1 times W2 transposed plus b2.](diagrams/03-linear-collapse.svg)

*Both chains take the same input shape to the same output shape. Without an activation between the layers, the upper one has no capability the lower one lacks.*

The right-hand side is a single dense layer. Its matrix $\mathbf{W}_\ast = \mathbf{W}_1^\top \mathbf{W}_2^\top$ has shape $(4, 3)$ and already faces the input, so it is applied without a transpose; in the one-row-per-neuron layout of this post the same layer stores $\mathbf{W}_\ast^\top = \mathbf{W}_2 \mathbf{W}_1$, of shape $(3, 4)$, three neurons with four weights each. Its bias $\mathbf{b}_\ast$ has shape $(3,)$. The two layers are mathematically one, which is the point section 3.1 made, and section 8.1 confirms it on numbers. Post 06 breaks that linearity by inserting an activation function between $\mathbf{Z}_1$ and the multiplication by $\mathbf{W}_2^\top$, so $\mathbf{Z}_1$ can no longer be substituted away.

The collapse never adds anything, and it can take something away. If the hidden layer is narrower than both the input and the output, say a stack of sizes $4 \to 2 \to 3$, every sample is squeezed through two numbers on the way, so its three outputs are computed from those two numbers alone. A single layer from 4 inputs to 3 neurons has no such restriction. A linear stack is at most as powerful as one layer, never more.

For an arbitrary depth $L$, the pattern reads:

$$\mathbf{Z}_\ell = \mathbf{Z}_{\ell - 1} \mathbf{W}_\ell^\top + \mathbf{b}_\ell, \qquad \mathbf{Z}_0 = \mathbf{X}.$$

Every layer is the same line of code; only the indices change. The same substitution, repeated, collapses any number of activation-free layers into one, with $\mathbf{W}_\ast = \mathbf{W}_1^\top \mathbf{W}_2^\top \cdots \mathbf{W}_L^\top$.

## 5. Tracing the shapes

Concrete shapes for the batch case, with $N = 3$ samples flowing through the network:

| Step | Operation | Input shape | Output shape | Why |
|---|---|---|---|---|
| Input | $\mathbf{X}$ | none | $(3, 4)$ | 3 samples, 4 features each |
| Layer 1 | $\mathbf{X} \mathbf{W}_1^\top$ | $(3, 4) \cdot (4, 3)$ | $(3, 3)$ | inner sizes match, 4 and 4 |
| Layer 1 | $+\ \mathbf{b}_1$ | $(3, 3) + (3,)$ | $(3, 3)$ | bias broadcast across rows |
| Layer 2 | $\mathbf{Z}_1 \mathbf{W}_2^\top$ | $(3, 3) \cdot (3, 3)$ | $(3, 3)$ | inner sizes match, 3 and 3 |
| Layer 2 | $+\ \mathbf{b}_2$ | $(3, 3) + (3,)$ | $(3, 3)$ | bias broadcast across rows |

![A shape-flow chart read left to right for a batch of N samples. The input X of shape (N, 4) enters layer 1, which computes X times W1 transposed plus b1 with W1 of shape (3, 4), transposed to (4, 3), and b1 of shape (3,); the inner sizes 4 and 4 match. The result Z1 of shape (N, 3) enters layer 2, which computes Z1 times W2 transposed plus b2 with W2 of shape (3, 3) and b2 of shape (3,); the inner sizes 3 and 3 match. The final output Z2 has shape (N, 3). A band below states the rule: the weights per neuron in a layer equal the number of neurons in the layer before it.](diagrams/02-dimension-flow.svg)

*Every layer enforces the same shape rule. A mismatch anywhere surfaces here, on paper, before any code runs.*

A diary of intermediate shapes like this one catches a mismatch before it happens. Printing the shape of the layer-1 output $\mathbf{Z}_1$ and checking it against the expected $(N, m_1)$ is the fastest debugging move available.

Two things can be read off the table. The batch size $N$ stays on the first axis from start to finish: a layer changes how many numbers describe each sample, never how many samples there are. And the original feature count $n$ appears only once: the first matrix multiplication replaces it by $m_1$, and no later layer sees it.

## 6. The weight-matrix shape rule, restated

A table to read whenever a new layer is added.

| Layer | Receives from | Number of neurons | Weight matrix shape |
|---|---|:---:|:---:|
| 1 | the input (length $n$) | $m_1$ | $(m_1, n)$ |
| 2 | layer 1 (length $m_1$) | $m_2$ | $(m_2, m_1)$ |
| 3 | layer 2 (length $m_2$) | $m_3$ | $(m_3, m_2)$ |
| $\ell$ | layer $\ell - 1$ (length $m_{\ell - 1}$) | $m_\ell$ | $(m_\ell,\ m_{\ell - 1})$ |

The bias vectors follow at once: $\mathbf{b}_\ell$ has shape $(m_\ell,)$.

Layers can have different numbers of neurons, and they almost always do. The only constraint is shape continuity: the output size of layer $\ell$ must equal the input size of layer $\ell + 1$. Adding a layer therefore means choosing one new number, its neuron count; the size of the layer before it fixes the other dimension of the new weight matrix.

The same table counts the parameters. Layer $\ell$ owns $m_\ell \cdot m_{\ell - 1}$ weights and $m_\ell$ biases, so a stack of $L$ layers has

$$\sum_{\ell = 1}^{L} m_\ell \, (m_{\ell - 1} + 1), \qquad m_0 = n,$$

parameters in total. For the network of this post that is $3 \cdot (4 + 1) + 3 \cdot (3 + 1) = 15 + 12 = 27$. The single equivalent layer of section 4 has $3 \cdot (4 + 1) = 15$: without an activation, the 12 parameters of the second layer buy no function that the 15 could not already express.

## 7. What the forward pass means

The word "forward" refers to the direction of data flow, not of time. The data starts at the input and is repeatedly transformed by `np.dot` and a bias addition until it emerges at the output. There is no training here, no derivatives, and no loss; everything in this post happens with frozen weights.

In a trained network, the forward pass is what produces predictions at inference time. In a training loop, the forward pass is the first half of every step; the backward pass that follows (posts 12 to 21) walks through the same layers in the opposite direction, computing the gradient of the loss with respect to every weight and bias. The two halves share their architecture exactly.

Mistakes in the forward pass are of two kinds. Some stop the program: a weight matrix of the wrong size makes `np.dot` raise a `ValueError`, and a shape diary catches these before the code runs. Others run and return wrong numbers of the right shape: a transpose dropped on a square weight matrix, or a bias with the wrong orientation, raises nothing and returns an array of the expected shape, so no later line complains either. Section 9 runs both kinds. A shape check cannot expose the second kind. Checking one output entry by hand does, as section 8 does, and so does testing with a batch size of at least two that differs from every layer width, on layers that are not square, so that a wrong orientation cannot line up by accident. [Post 04](../04-dense-layer-class-and-spiral-data/index.md) reduces the room for such slips by wrapping each call in a `Layer_Dense` class that keeps a layer's weights, biases, and output together in one object.

## 8. Make it run: two layers in NumPy

The code of this post is four scripts under `snippets/`. Each needs only NumPy, finishes in under a second, and is run from the series root. The first, `python posts/03-stacking-layers-and-the-forward-pass/snippets/two_layer_forward.py`, is the forward pass itself: five arrays and two lines of arithmetic.

```python
"""Post 03: a batch of three samples through two stacked dense layers.

Run from the series root:  python posts/03-stacking-layers-and-the-forward-pass/snippets/two_layer_forward.py
"""
import numpy as np

# A batch of 3 samples, 4 features each.
inputs = np.array([[ 1.0,  2.0,  3.0,  2.5],
                   [ 2.0,  5.0, -1.0,  2.0],
                   [-1.5,  2.7,  3.3, -0.8]])

# Layer 1: 3 neurons, 4 weights each.
weights1 = np.array([[ 0.2,   0.8,  -0.5,   1.0 ],
                     [ 0.5,  -0.91,  0.26, -0.5 ],
                     [-0.26, -0.27,  0.17,  0.87]])
biases1  = np.array([2.0, 3.0, 0.5])

# Layer 2: 3 neurons, 3 weights each.
weights2 = np.array([[ 0.1,  -0.14,  0.5 ],
                     [-0.5,   0.12, -0.33],
                     [-0.44,  0.73, -0.13]])
biases2  = np.array([-1.0, 2.0, -0.5])

# Forward pass.
layer1_outputs = np.dot(inputs,         weights1.T) + biases1
layer2_outputs = np.dot(layer1_outputs, weights2.T) + biases2

print("Layer 1 outputs:", layer1_outputs.shape)
print(layer1_outputs)
print("\nLayer 2 outputs:", layer2_outputs.shape)
print(layer2_outputs)
```

It prints:

```text
Layer 1 outputs: (3, 3)
[[ 4.8    1.21   2.385]
 [ 8.9   -1.81   0.2  ]
 [ 1.41   1.051  0.026]]

Layer 2 outputs: (3, 3)
[[ 0.5031  -1.04185 -2.03875]
 [ 0.2434  -2.7332  -5.7633 ]
 [-0.99314  1.41254 -0.35655]]
```

The layer-1 block is the output that posts 01 and 02 computed for the same batch and the same layer. In both blocks each row belongs to one sample and each column to one neuron of the layer that produced it. One entry by hand: the first neuron of layer 2 has weights $[0.1, -0.14, 0.5]$ and bias $-1.0$, and the first sample arrives from layer 1 as $[4.8, 1.21, 2.385]$, so its output is $0.1 \cdot 4.8 - 0.14 \cdot 1.21 + 0.5 \cdot 2.385 - 1.0 = 0.5031$, the top-left entry of the layer-2 block.

### 8.1. One layer that does the work of two

`snippets/linear_collapse.py` starts from the same five arrays and builds the single equivalent layer of section 4:

```python
# The single equivalent layer of section 4.
weights_star = np.dot(weights1.T, weights2.T)           # (4, 3)
biases_star  = np.dot(biases1, weights2.T) + biases2    # (3,)
collapsed    = np.dot(inputs, weights_star) + biases_star
```

```text
W_star, shape (4, 3)
[[-0.18    0.0458  0.3108]
 [ 0.0724 -0.4201 -0.9812]
 [-0.0014  0.2251  0.3877]
 [ 0.605  -0.8471 -0.9181]]
b_star, shape (3,)
[-0.97   1.195  0.745]
One layer, X . W_star + b_star:
[[ 0.5031  -1.04185 -2.03875]
 [ 0.2434  -2.7332  -5.7633 ]
 [-0.99314  1.41254 -0.35655]]
largest absolute difference from the two-layer output: 8.9e-16
W_star is the transpose of W2 . W1: True
parameters in the two layers: 27
parameters in the equivalent layer: 15
```

One matrix and one bias vector reproduce the layer-2 numbers of the two-layer network. The largest difference, $8.9 \times 10^{-16}$ in this run, is floating-point rounding: the two routes group the same arithmetic differently, so they agree to about fifteen decimal places and not bit for bit, and the exact size of that residue can vary from one machine to another. The collapse is the property section 3.1 warned about. Until the activation function arrives in post 06, $\mathbf{Z}_2$ is just a linear function of $\mathbf{X}$ wearing a slightly more complicated outfit.

The script ends with an experiment that shows a layer doing nothing new without any algebra. It replaces $\mathbf{W}_2$ by the $3 \times 3$ identity matrix, which has ones on its diagonal and zeros elsewhere:

```python
# Identity experiment: with W2 = I, layer 2 only adds its bias to the layer-1 output.
identity = np.eye(3)
identity_outputs = np.dot(layer1_outputs, identity.T) + biases2
```

```text
W2 = identity gives Z1 + b2:
[[ 3.8    3.21   1.885]
 [ 7.9    0.19  -0.3  ]
 [ 0.41   3.051 -0.474]]
equal to layer1_outputs + biases2: True
```

Every entry is the layer-1 output with $\mathbf{b}_2 = [-1.0, 2.0, -0.5]$ added to its column: $4.8 - 1.0 = 3.8$, $1.21 + 2.0 = 3.21$, $2.385 - 0.5 = 1.885$. Each neuron of an identity layer copies one of its inputs, so the second layer contributes nothing but a shift. The experiment needs a square layer, here three inputs and three neurons, because only a square matrix can be an identity.

### 8.2. Extending to more layers

The pattern survives any depth without modification. `snippets/deeper_stack.py` builds five layers with random weights and chains them. `np.random.randn(m, n)` fills an array of shape $(m, n)$ with random numbers, and the `np.random.seed(0)` call at the top of the script makes every run draw the same ones.

```python
# 4 input features, then layers of 6, 5, 3, 4 and 2 neurons.
sizes = [4, 6, 5, 3, 4, 2]
X = np.random.randn(3, sizes[0])                 # a batch of 3 samples

# One row of weights per neuron: layer l has shape (sizes[l], sizes[l - 1]).
w1, w2, w3, w4, w5 = [np.random.randn(m, n) for n, m in zip(sizes[:-1], sizes[1:])]
b1, b2, b3, b4, b5 = [np.random.randn(m) for m in sizes[1:]]

z1 = np.dot(X,  w1.T) + b1
z2 = np.dot(z1, w2.T) + b2
z3 = np.dot(z2, w3.T) + b3
z4 = np.dot(z3, w4.T) + b4
z5 = np.dot(z4, w5.T) + b5    # final output
```

Each of the five lines is structurally identical. The only constraint is that every weight matrix has as many columns as the previous layer has neurons, which is the rule of section 6 and is what the list comprehension builds. The script prints every shape:

```text
X (3, 4)
layer 1: W (6, 4), b (6,) -> Z (3, 6)
layer 2: W (5, 6), b (5,) -> Z (3, 5)
layer 3: W (3, 5), b (3,) -> Z (3, 3)
layer 4: W (4, 3), b (4,) -> Z (3, 4)
layer 5: W (2, 4), b (2,) -> Z (3, 2)
the loop reproduces z5: True
W_star (4, 2) b_star (2,)
one layer reproduces z5: True
parameters in the five layers: 109
parameters in the equivalent layer: 10
```

Reading down the output, the second number of each `W` is the last number of the array on the line above it, and the batch size 3 leads every `Z`. Because the five lines differ only in their indices, the script also runs them as a loop over two lists, `weights` and `biases`, and gets the same `z5`:

```python
z = X
for w, b in zip(weights, biases):
    z = np.dot(z, w.T) + b
```

The last four lines of the output repeat the argument of section 4 at depth five. With no activation anywhere, the 109 parameters of the five layers compute what one layer from 4 inputs to 2 neurons computes, and that layer has $2 \cdot (4 + 1) = 10$ parameters.

## 9. What can go wrong?

`snippets/shape_bugs.py` runs five checks on the forward pass of section 8: four mistakes, made one at a time, and one repeat of the correct pass. It prints what NumPy does with each. The arrays `inputs`, `weights1`, `biases1`, `weights2`, and `biases2` are those of section 8.

```python
# 1. Layer 2 sized for the 4 original features instead of the 3 outputs of layer 1.
wrong_size = np.ones((3, 4))
try:
    np.dot(layer1_outputs, wrong_size.T)
except ValueError as error:
    print("1. W2 of shape (3, 4) raises ValueError:", error)

# 2. The transpose forgotten on layer 2. W2 is square, so the shapes still line up.
no_transpose = np.dot(layer1_outputs, weights2) + biases2
print("2. no .T on the square W2 runs; sample 1:", no_transpose[0])

# 3. The transpose forgotten on layer 1, whose weights are not square.
try:
    np.dot(inputs, weights1)
except ValueError as error:
    print("3. no .T on the (3, 4) W1 raises ValueError:", error)

# 4. Biases stored as a column, shape (3, 1), against an output of shape (3, 3).
column_biases = biases2.reshape(3, 1)
with_column = np.dot(layer1_outputs, weights2.T) + column_biases
print("4. column bias on 3 samples runs; sample 1:", with_column[0])
try:
    np.dot(layer1_outputs[:2], weights2.T) + column_biases
except ValueError as error:
    print("   column bias on 2 samples raises ValueError:", str(error).strip())
one_sample = np.dot(layer1_outputs[:1], weights2.T) + column_biases
print("   column bias on 1 sample runs; output shape:", one_sample.shape)

# 5. The same inputs and weights always give the same numbers, bit for bit.
again = np.dot(np.dot(inputs, weights1.T) + biases1, weights2.T) + biases2
print("5. second forward pass identical to the first:", np.array_equal(again, layer2_outputs))
```

```text
correct layer-2 output, sample 1: [ 0.5031  -1.04185 -2.03875]
1. W2 of shape (3, 4) raises ValueError: shapes (3,3) and (4,3) not aligned: 3 (dim 1) != 4 (dim 0)
2. no .T on the square W2 runs; sample 1: [-2.1744   3.21425  1.19065]
3. no .T on the (3, 4) W1 raises ValueError: shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)
4. column bias on 3 samples runs; sample 1: [ 0.5031  -4.04185 -2.53875]
   column bias on 2 samples raises ValueError: operands could not be broadcast together with shapes (2,3) (3,1)
   column bias on 1 sample runs; output shape: (3, 3)
5. second forward pass identical to the first: True
```

- **A layer sized from the wrong neighbour is caught at once.** Case 1 gives each neuron of layer 2 four weights, one per original feature, when the layer receives the three outputs of layer 1. The inner sizes are 3 and 4, `np.dot` refuses, and the message names both shapes. Errors that stop the program are the cheap ones.
- **A missing transpose is caught only when the weight matrix is not square.** Case 3 drops `.T` on $\mathbf{W}_1$, shape $(3, 4)$, and NumPy raises. Case 2 drops it on $\mathbf{W}_2$, shape $(3, 3)$, and nothing is raised: the first sample comes out as $[-2.1744, 3.21425, 1.19065]$ where the correct row is $[0.5031, -1.04185, -2.03875]$, because each neuron now reads a column of $\mathbf{W}_2$ in place of its own row. Allocating a weight matrix with its two sizes swapped fails in the same two ways. Square layers are common, since two neighbouring layers of equal width are joined by one, so this check cannot be left to NumPy.
- **A column of biases is caught only for some batch sizes.** A bias of shape $(3, 1)$ lines up with the rows of the output, not its columns, so sample $i$ receives bias $i$ on every neuron. In case 4 the first sample gets $-1.0$ three times, giving $[0.5031, -4.04185, -2.53875]$: only the first entry is right. With three samples and three neurons that runs silently. With two samples the shapes $(2, 3)$ and $(3, 1)$ cannot be combined and NumPy raises. With one sample it runs again, and the single row is stretched into an output of shape $(3, 3)$ where $(1, 3)$ was expected. A 1-D bias of shape $(m,)$ always lines up with the $m$ columns. [Post 05](../05-array-summation-keepdims-and-broadcasting/index.md) gives the full broadcasting rules.
- **Two forward passes that disagree mean something changed between them.** For the same input and the same weights the forward pass returns the same numbers bit for bit, as case 5 confirms. If two consecutive passes differ, something is being reseeded or an array is being modified in place.

## 10. Summary

| Concept | Takeaway |
|---|---|
| Forward pass | Apply the dense-layer formula once per layer, from the input to the output |
| Chaining | $\mathbf{Z}_\ell = \mathbf{Z}_{\ell-1} \mathbf{W}_\ell^\top + \mathbf{b}_\ell$, $\mathbf{Z}_0 = \mathbf{X}$ |
| Output to input | Layer $\ell$'s output is layer $\ell+1$'s input |
| Shape rule | $\mathbf{W}_\ell$ has shape $(m_\ell,\ m_{\ell-1})$; $\mathbf{b}_\ell$ has shape $(m_\ell,)$; every $\mathbf{Z}_\ell$ has shape $(N, m_\ell)$ |
| Parameters | $\sum_\ell m_\ell (m_{\ell-1} + 1)$; 27 for the $4 \to 3 \to 3$ network of this post |
| Linear stack | Without an activation between the layers, the stack equals one layer with $\mathbf{W}_\ast = \mathbf{W}_1^\top \mathbf{W}_2^\top$ and $\mathbf{b}_\ast = \mathbf{b}_1 \mathbf{W}_2^\top + \mathbf{b}_2$ |
| Any depth | Same call repeated $L$ times for $L$ layers |

## Common pitfalls

1. **Forgetting that a stack of linear layers is itself linear.** Until post 06 puts an activation between the layers, the network is one matrix and one bias vector in disguise. Activation-free layers add parameters, not capability.
2. **Sizing $\mathbf{W}_2$ from the original input instead of from layer 1's output.** $\mathbf{W}_2$ consumes the layer-1 output (length $m_1$), not the original input (length $n$).
3. **Forgetting the transpose on a later layer, or swapping the two sizes of a weight matrix.** Every `np.dot(Z, W.T)` needs its `.T`, not just the first one. A matrix of shape $(m_{\ell-1}, m_\ell)$ in place of $(m_\ell, m_{\ell-1})$ raises an error when the two sizes differ and runs silently, with the roles of neurons and inputs exchanged, when they are equal.
4. **Storing biases as a column instead of a 1-D array.** A bias of shape $(m,)$ lines up with the $m$ columns of the output. A bias of shape $(m, 1)$ lines up with the rows: an error for most batch sizes, and a silent wrong answer when $N = m$ or $N = 1$.
5. **Keeping only the final output.** The backward pass (posts 12 to 21) needs what each layer received: the weight gradient of a layer is computed from that layer's input. From post 16 on, `Layer_Dense.forward` stores it as `self.inputs`; a forward pass that throws its intermediates away has to be run again before any gradient can be computed.

## Further reading

- Cybenko, G., *"Approximation by Superpositions of a Sigmoidal Function"* (Mathematics of Control, Signals and Systems, 1989).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 6, "Deep Feedforward Networks" (MIT Press, 2016).
- Hornik, K., *"Approximation Capabilities of Multilayer Feedforward Networks"* (Neural Networks, 1991).
- Ivakhnenko, A. G. and Lapa, V. G., *"Cybernetic Predicting Devices"* (CCM Information Corporation, 1965).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 3 (2020).
- Nielsen, M., *Neural Networks and Deep Learning*, chapter 1 (online, 2015).
- Rumelhart, D., Hinton, G., and Williams, R., *"Learning Representations by Back-Propagating Errors"* (Nature, 1986).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

## What to read next

- **[Post 04 - The Dense layer class and spiral data](../04-dense-layer-class-and-spiral-data/index.md):** wraps the per-layer call of this post in a `Layer_Dense` class and introduces the spiral dataset that the series trains on from there to post 31.
- **[Post 06 - Activation functions: ReLU and Softmax](../06-activation-functions-relu-and-softmax/index.md):** adds the non-linearity between layers that stops the stack collapsing into one layer, which is what makes depth matter.
