# 01 - Neurons and layers

> **TL;DR.** A neuron is a weighted sum of its inputs plus a bias; a layer is several neurons that read the same inputs, each with its own weights and its own bias. This post codes that sentence three ways (term by term, with loops, and as the single NumPy expression `np.dot(weights, inputs) + biases`) and then sends a whole batch through with one transpose. Weights and biases are the only learnable parameters of the dense networks this series builds, and both are on the table by the end.
>
> **Prerequisites:** none. This is the first post of the series; it assumes Python lists, `for` loops, and functions, and no NumPy.
> **Safe to skip?** Skip it if the reader can already write `np.dot(inputs, weights.T) + biases` for a batch, give the shape of every array in that line, and say why a layer of three neurons over four inputs has 15 parameters.
>
> **After reading, you will be able to:**
>
> - State in one sentence what a neuron computes and what a layer computes.
> - Compute the output of a neuron or a layer by hand from its inputs, weights, and biases.
> - Count the weights and biases of a layer from its number of inputs and its number of neurons.
> - Code a neuron and a layer in plain Python and then as one NumPy dot product plus the biases.
> - Predict the shape of every intermediate array as a single sample becomes a batch.

![The four-input neuron of section 5. Inputs 1.0, 2.0, 3.0, 2.5 are multiplied row by row by the weights 0.2, 0.8, minus 0.5, 1.0, giving the products 0.2, 1.6, minus 1.5, 2.5. Lines carry the products into a node marked sigma plus b, the bias 2.0 enters from above, and an arrow labelled sigma plus b equals 2.8 plus 2.0 leads to the output z, 4.8. Under the node and the output, the whole sum is written out, ending in 4.8 under the output box, and the bottom line counts one weight per input and one bias, 4 plus 1, 5 parameters.](diagrams/01-one-neuron.svg)

*One neuron, four inputs, five parameters: a weight for each input and a single bias. Every learnable parameter of the dense networks in this series is one or the other, and the rest of the series is rules for setting them.*

---

## 1. The question: what does a neuron compute, and why build it from scratch?

A typical first encounter with a neural network is a handful of framework calls. In Keras it takes three statements.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train)
```

The fragment is shown for contrast only; it needs TensorFlow and a dataset with one-hot labels, and nothing in this series runs it. Given both, it trains a network. The questions it leaves open do not go away.

- Why Adam, and not SGD?
- What does cross-entropy actually compute?
- What does backpropagation compute, and how does an optimiser turn it into a weight update?
- When a shape mismatch is raised, which dimension is wrong, and why?

A neural network is not exotic mathematics. It is a stack of weighted sums, activation functions, and partial derivatives. Implementing each piece by hand replaces trust in a library with understanding. This series builds every component (neurons, layers, activations, losses, backpropagation, optimisers, regularisation) in Python and NumPy, without relying on a framework to do the thinking.

This first post stays at the most basic operation in the stack and asks one question: what does a single neuron compute, and what changes when several neurons share the same inputs? The answer is the **forward pass through a single neuron, then a layer**. Activation functions arrive in post 06, the loss in post 08, and backpropagation begins in post 12. Everything that comes later is layered on top of what is built here.

## 2. Where the neuron came from

The mathematical neuron predates working hardware. Two papers do the heavy lifting.

In 1943, McCulloch and Pitts modelled a neuron as a binary threshold unit: it fires when the number of its active excitatory inputs reaches a fixed threshold, provided no inhibitory input is active (McCulloch and Pitts, 1943). The model was descriptive. It had no adjustable weights, and the thresholds and the wiring were set by hand.

Fifteen years later, Frank Rosenblatt's **perceptron** made the connection strengths adjustable and added a learning rule that changed them automatically from examples (Rosenblatt, 1958). Its arithmetic is a weighted sum compared with a threshold $\theta$, which is the same thing as adding a bias $b = -\theta$ to the weighted sum and comparing the result with zero. That weighted sum plus a bias is the arithmetic still in use in 2026, sixty-eight years later. What changed is everything around it: the activation function (no longer a yes-or-no step), the loss function, the optimiser, the depth of the stack, and the scale of the data.

The name is a historical analogy. The mathematical object has very little to do with a biological neuron beyond many inputs, one output, and a threshold-like response; the word was already in use long before deep learning made it standard.

The implication for this series is small and useful. **The weighted sum inside every dense layer of a 2026 neural network is the weighted sum Rosenblatt published in 1958, repeated many times in parallel.** Everything from post 02 onward is concerned with computing those sums quickly, stacking them, and learning the weights through gradient descent.

## 3. The neuron, formally

A neuron receives a vector of inputs (an ordered list of numbers), multiplies each input by a corresponding weight, sums the products, and adds a single bias. For a neuron with $n_\text{inputs}$ inputs:

$$z = \sum_{i=1}^{n_\text{inputs}} w_i x_i + b.$$

The sum of products is the **weighted sum**, and $z$ is that sum with the bias added. There is no activation function yet; activations are the subject of post 06. For now the neuron stops at $z$, and "output" in this post always means $z$. Later posts, where an activation follows, call $z$ the pre-activation.

Each symbol carries a specific role. The shape column says how many numbers a component holds, in the notation NumPy uses from section 9 on: $(n_\text{inputs},)$ is a flat list of $n_\text{inputs}$ numbers, and a scalar is a single number. Reading the table left to right gives the entire surface area of the neuron:

| Component | Symbol | Shape (single neuron) | Role | Lifetime | What fails when it is wrong |
|---|---|---|---|---|---|
| Inputs | $x_i$ | $(n_\text{inputs},)$ | The data fed in | changes every forward pass | bad inputs give bad outputs (garbage in, garbage out) |
| Weights | $w_i$ | $(n_\text{inputs},)$ | Importance of each input, with a sign | learned during training, frozen at inference | wrong weights give a wrong prediction |
| Bias | $b$ | scalar | A constant offset, one per neuron | learned during training, frozen at inference | a missing bias forces the output to 0 whenever every input is 0 |
| Output | $z$ | scalar | The weighted sum plus the bias | recomputed every forward pass | downstream layers receive the wrong signal |

A large weight magnifies its input's contribution, a near-zero weight ignores it, and a negative weight makes the input count against the output. The bias shifts the output by the same amount whatever the inputs are. It is what allows a neuron to fit data whose target is offset from the origin: when every input is zero the products vanish and the output is exactly $b$, so a neuron without a bias could only ever answer zero there. Section 7 checks this on a layer.

The bias is one number per neuron, not one per input, because it is added after the weighted sum has collapsed the inputs to a single number: one sum, one offset. Giving every input its own offset $c_i$ would buy nothing, since $\sum_i w_i (x_i + c_i) = \sum_i w_i x_i + \sum_i w_i c_i$ and the last term is again one constant.

### 3.1. What a neuron is *not*

The boundary helps as much as the definition.

- **A neuron is not a classifier on its own.** It produces a single number, not a class label. Turning the numbers of an output layer into probabilities takes an activation function (post 06), turning them into a label takes a rule such as picking the largest, and judging how good the answer is takes a loss function (post 08).
- **A neuron is not non-linear.** The weighted sum plus bias is a straight-line function of its inputs (strictly an affine one: linear plus a constant). Stacking such neurons without activations between them gives back another function of the same kind, no more expressive than a single layer; post 03 works the collapse out.
- **A neuron is not learned in this post.** The weights and bias are *given*, not optimised. When training does start they are drawn at random: post 04 fills the weights with small random numbers and the biases with zeros, and post 33 treats initialisation in detail. Learning itself enters the series in post 09 (gradient descent) and post 12 (backpropagation).

## 4. Coding a neuron with three inputs

Plain Python lists are enough to express both the inputs and the weights.

```python
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = (inputs[0] * weights[0]
          + inputs[1] * weights[1]
          + inputs[2] * weights[2]
          + bias)

print(output)
```

**Output:**

```text
2.3
```

The arithmetic, broken out term by term:

| Step | Calculation | Value |
|---|---|---|
| $x_1 w_1$ | $1 \times 0.2$ | $0.2$ |
| $x_2 w_2$ | $2 \times 0.8$ | $1.6$ |
| $x_3 w_3$ | $3 \times (-0.5)$ | $-1.5$ |
| Sum | $0.2 + 1.6 - 1.5$ | $0.3$ |
| Plus bias | $0.3 + 2$ | $2.3$ |

Python indexing starts at zero, so `inputs[0]` is the first element, the one the mathematics calls $x_1$. That convention holds throughout the series.

## 5. Coding a neuron with four inputs

Scaling to four inputs adds one weight; the bias count stays at one. **One weight per input, one bias per neuron.** The rule never changes.

```python
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = (inputs[0] * weights[0]
          + inputs[1] * weights[1]
          + inputs[2] * weights[2]
          + inputs[3] * weights[3]
          + bias)

print(output)
```

**Output:**

```text
4.8
```

| Step | Calculation | Value |
|---|---|---|
| $x_1 w_1$ | $1.0 \times 0.2$ | $0.2$ |
| $x_2 w_2$ | $2.0 \times 0.8$ | $1.6$ |
| $x_3 w_3$ | $3.0 \times (-0.5)$ | $-1.5$ |
| $x_4 w_4$ | $2.5 \times 1.0$ | $2.5$ |
| Sum | $0.2 + 1.6 - 1.5 + 2.5$ | $2.8$ |
| Plus bias | $2.8 + 2.0$ | $4.8$ |

The shape of the operation is invariant: one weight per input, one bias per neuron, one scalar out. A neuron with $n_\text{inputs}$ inputs therefore has $n_\text{inputs} + 1$ parameters, five in this case. The figure at the top of the post draws this neuron with its numbers.

## 6. From a neuron to a layer

A **layer** is a group of neurons that all receive the same input vector while each keeps its own weights and its own bias. The figure below draws the layer of section 7, three such neurons over the four inputs of section 5, first as wiring and then as one matrix.

![Two panels. On the left, four inputs x1 to x4, holding 1.0, 2.0, 3.0, 2.5, each connect to all three neuron boxes, sigma plus b1, sigma plus b2 and sigma plus b3, twelve lines, with neuron 2's lines in the weight colour and a note that they are row 2 of W; the boxes give z1, z2 and z3. On the right, the same layer as z equals W x plus b: W is a 3 by 4 grid with one row of weights per neuron, x holds 1.0, 2.0, 3.0, 2.5, b holds 2.0, 3.0, 0.5, and z holds 4.800, 1.210, 2.385. Row 2 of W, b2 and z2 are outlined: minus 1.79 plus 3.0 is 1.21.](diagrams/02-a-layer.svg)

*Every input reaches every neuron, so the weights multiply (3 neurons times 4 inputs is 12) while the biases only add (3). Row $k$ of $\mathbf{W}$ holds the weights of neuron $k$.*

For a layer of three neurons fed by four inputs:

- Each neuron receives all four inputs.
- Each neuron owns its own four weights, giving $3 \times 4 = 12$ weights in total.
- Each neuron owns its own bias, giving three biases.
- The layer emits three outputs, one per neuron.
- **Total parameters: $3 \times 4 + 3 = 15$.**

In general a layer of $n_\text{neurons}$ neurons over $n_\text{inputs}$ inputs has $n_\text{neurons} \cdot n_\text{inputs}$ weights and $n_\text{neurons}$ biases, $n_\text{neurons}(n_\text{inputs} + 1)$ parameters in all. Only the bias count equals the neuron count. The counts grow quickly: the first layer of the series' MNIST project has 128 neurons over 784 inputs, so $128 \times 784 = 100{,}352$ weights and 128 biases.

Per-neuron arithmetic:

$$z_1 = w_{11} x_1 + w_{12} x_2 + w_{13} x_3 + w_{14} x_4 + b_1$$

$$z_2 = w_{21} x_1 + w_{22} x_2 + w_{23} x_3 + w_{24} x_4 + b_2$$

$$z_3 = w_{31} x_1 + w_{32} x_2 + w_{33} x_3 + w_{34} x_4 + b_3$$

The same three lines, written compactly:

$$\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b}.$$

The matrix $\mathbf{W}$, a rectangular grid of numbers, holds one row of weights per neuron, so its shape, rows first and then columns, is $(n_\text{neurons}, n_\text{inputs})$, here $(3, 4)$; $\mathbf{x}$ has shape $(n_\text{inputs},)$; $\mathbf{b}$ has shape $(n_\text{neurons},)$; and the output $\mathbf{z}$ has shape $(n_\text{neurons},)$. The shape diary in section 10 tracks all of this through a batch, and says how later posts store $\mathbf{W}$.

## 7. A layer, by hand

With three sets of weights, a list of lists is the natural representation.

```python
inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1],         # Neuron 1
           [0.5, -0.91, 0.26, -0.5],    # Neuron 2
           [-0.26, -0.27, 0.17, 0.87]]  # Neuron 3

biases = [2, 3, 0.5]

outputs = [
    inputs[0]*weights[0][0] + inputs[1]*weights[0][1]
    + inputs[2]*weights[0][2] + inputs[3]*weights[0][3] + biases[0],

    inputs[0]*weights[1][0] + inputs[1]*weights[1][1]
    + inputs[2]*weights[1][2] + inputs[3]*weights[1][3] + biases[1],

    inputs[0]*weights[2][0] + inputs[1]*weights[2][1]
    + inputs[2]*weights[2][2] + inputs[3]*weights[2][3] + biases[2],
]

print(outputs)
```

**Output:**

```text
[4.8, 1.21, 2.385]
```

| Neuron | Weighted sum | Plus bias | Output |
|---|---|---|---|
| 1 | $1(0.2) + 2(0.8) + 3(-0.5) + 2.5(1) = 2.8$ | $+ 2$ | $4.8$ |
| 2 | $1(0.5) + 2(-0.91) + 3(0.26) + 2.5(-0.5) = -1.79$ | $+ 3$ | $1.21$ |
| 3 | $1(-0.26) + 2(-0.27) + 3(0.17) + 2.5(0.87) = 1.885$ | $+ 0.5$ | $2.385$ |

The second column is what the layer would output with every bias set to zero: $2.8$, $-1.79$, $1.885$. Setting every input to zero instead leaves only the third column: the layer outputs its biases, $2$, $3$, $0.5$, whatever its weights are. The script `snippets/what_can_go_wrong.py` of section 13 runs both cases.

The code is correct but does not scale: fifty neurons would mean fifty hand-written summations.

## 8. The same operation, three implementations

The neuron does not care which Python construct computes it. Three implementations (the hand-written sums of section 7, two nested loops, and one NumPy call) produce the same numbers; each replaces the previous one's repetition with a stronger abstraction. The arithmetic does not change; the representation does, and with it the length of the code and its speed, which section 9 measures.

Two nested loops handle any number of neurons and any number of inputs.

```python
inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
```

**Output:**

```text
[4.8, 1.21, 2.385]
```

Tracing the outer loop once per neuron:

- Iteration 1: weights $[0.2, 0.8, -0.5, 1]$, bias $2$. Inner sum $2.8$, plus bias gives $4.8$.
- Iteration 2: weights $[0.5, -0.91, 0.26, -0.5]$, bias $3$. Inner sum $-1.79$, plus bias gives $1.21$.
- Iteration 3: weights $[-0.26, -0.27, 0.17, 0.87]$, bias $0.5$. Inner sum $1.885$, plus bias gives $2.385$.

Same numbers as section 7, with code that no longer cares about the network's width. The inner loop is still a Python loop, however, and that is the next bottleneck.

## 9. Why NumPy wins

A Python loop pays the interpreter's overhead on every pass: each multiplication and each addition is dispatched at run time, and each result becomes a new float object. NumPy is the standard Python library for numerical work, installed with `pip install numpy` and imported as `np`. Its central object is the array, a block of numbers of one type with a shape, and NumPy hands the work of the loop to compiled routines that run over the entire array at once. This strategy is called **vectorisation**.

The relevant operation is the **dot product**. The dot product of two vectors of equal length is the sum of their element-wise products, which is precisely the weighted sum a neuron computes before its bias is added:

$$\mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^{n_\text{inputs}} w_i x_i.$$

A single neuron, in NumPy:

```python
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

print(np.dot(weights, inputs) + bias)
```

**Output:**

```text
4.8
```

A full layer requires no extra code. When `weights` is two-dimensional (here a list of lists, which `np.dot` converts to a 2-D array), `np.dot` treats each row as a separate weight vector and computes its dot product with `inputs`, returning one number per row.

```python
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

print(np.dot(weights, inputs) + biases)
```

**Output:**

```text
[4.8   1.21  2.385]
```

Two lines do the work, the import and one expression, and they give the same numbers as the hand-written and looped versions. The output looks different because it is a NumPy array, which prints its elements padded to a common width and without commas.

How much faster this is depends on the size of the layer. The script `snippets/loop_vs_numpy.py`, listed in section 12, times the loops of section 8 against `np.dot` on two layers filled with seeded random numbers: the layer of this post, and a layer of 128 neurons over 784 inputs, the size of the first layer in the series' MNIST project. On the machine used to check this post (Python 3.13.5, NumPy 2.3.5), ten consecutive runs showed three things.

- **At scale NumPy wins by two orders of magnitude.** For the $100{,}352$ multiplications of the large layer, `np.dot` on NumPy arrays was between 177 and 340 times faster than the loops; the run printed in section 12 shows 265.
- **On a tiny layer there is little to win.** For the 12 multiplications of this post's layer the ratio stayed between 1.0 and 1.9, and at 1.2 or below in eight of the ten runs. The fixed cost of the two NumPy calls, `np.dot` and the addition of the biases, is about the cost of the whole loop.
- **NumPy is fast only when the numbers already sit in a NumPy array.** Handed Python lists, `np.dot` first copies every number into an array, and for the large layer that copy cost more than the loop it was meant to replace in all ten runs: 4,795.9 microseconds against 3,766.7 in the printed one. This post passes lists to `np.dot` because lists are the only container it assumes; later posts create their data as arrays.

The microseconds will differ on another machine. The two orders of magnitude at the size of a real layer are the part to remember.

## 10. Handling batches, and the shape diary

Training rarely feeds the network one sample at a time. A **batch** is a stack of input vectors, one per row.

```python
inputs = [[1.0, 2.0, 3.0, 2.5],     # Sample 1
          [2.0, 5.0, -1.0, 2.0],    # Sample 2
          [-1.5, 2.7, 3.3, -0.8]]   # Sample 3
```

Given two 2-D arrays, `np.dot` computes a matrix product: the dot product of every row of the first array with every column of the second, so a row of the first must be as long as a column of the second. With `inputs` of shape $(3, 4)$ and `weights` of shape $(3, 4)$ that fails: a row of `inputs` holds 4 numbers and a column of `weights` holds 3. These two sizes, the inner dimensions of the product, do not match. The transpose $\mathbf{W}^{\top}$, `W.T` in NumPy, is the same twelve numbers with rows and columns exchanged; its shape is $(4, 3)$ and each of its columns holds the four weights of one neuron, which realigns the product:

$$\mathbf{Z} = \mathbf{X} \mathbf{W}^{\top} + \mathbf{b}.$$

Here $\mathbf{X}$ is the batch, of shape $(N, n_\text{inputs})$ for $N$ samples, and $\mathbf{Z}$ holds one row of outputs per sample, shape $(N, n_\text{neurons})$.

The transpose is needed only because this post stores one row of weights per neuron. Posts 01 to 03 keep that layout, $(n_\text{neurons}, n_\text{inputs})$. From post 04 on, the `Layer_Dense` class stores the weights already transposed, $(n_\text{inputs}, n_\text{neurons})$, so that the same forward pass reads $\mathbf{Z} = \mathbf{X} \mathbf{W} + \mathbf{b}$ with no transpose, and it stores the biases as a $(1, n_\text{neurons})$ row instead of the flat list used here. Both layouts compute the same numbers.

The shape diary below follows the same operation at three sizes. Tracking shapes is the cheapest debugging tool in deep learning: write them down before a call and print them after it.

| Setting | `X` shape | `W` shape | `b` shape | Operation | Output shape |
|---|---|---|---|---|---|
| Single neuron, single sample | $(n_\text{inputs},)$ | $(n_\text{inputs},)$ | scalar | `np.dot(W, X) + b` | scalar |
| Layer, single sample | $(n_\text{inputs},)$ | $(n_\text{neurons}, n_\text{inputs})$ | $(n_\text{neurons},)$ | `np.dot(W, X) + b` | $(n_\text{neurons},)$ |
| Layer, batch of $N$ | $(N, n_\text{inputs})$ | $(n_\text{neurons}, n_\text{inputs})$ | $(n_\text{neurons},)$ | `np.dot(X, W.T) + b` | $(N, n_\text{neurons})$ |

The script `snippets/shape_diary.py` checks the three rows on the arrays of this post. It adds a batch of two samples, because with three samples and three neurons the output shape $(3, 3)$ cannot show which axis is which.

```text
setting                    inputs   weights  biases   output
one neuron, one sample     (4,)     (4,)     ()       ()
layer of 3, one sample     (4,)     (3, 4)   (3,)     (3,)
layer of 3, batch of 3     (3, 4)   (3, 4)   (3,)     (3, 3)
layer of 3, batch of 2     (2, 4)   (3, 4)   (3,)     (2, 3)

a batch of 2 through the layer of 3, one step at a time
  batch                        (2, 4)
  W                            (3, 4)
  W.T                          (4, 3)
  np.dot(batch, W.T)           (2, 3)
  biases                       (3,)
  np.dot(batch, W.T) + biases  (2, 3)

parameters in the layer: 12 weights + 3 biases = 15
```

NumPy writes the shape of a scalar as `()`. The batch of two comes out as $(2, 3)$: samples on the rows, neurons on the columns.

The transpose changes how the call is written, not the arithmetic: entry $(i, j)$ of the result is still the dot product of sample $i$ with the weights of neuron $j$, plus the bias of neuron $j$. NumPy's broadcasting then adds the $(n_\text{neurons},)$ bias vector to every one of the $N$ rows of the $(N, n_\text{neurons})$ product, so each neuron's bias lands in its own column; post 05 states the rule. In the code, `weights` is a Python list, and a list has no `.T`, so it is converted with `np.array(weights)` before it is transposed.

```python
import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],     # Sample 1
          [2.0, 5.0, -1.0, 2.0],    # Sample 2
          [-1.5, 2.7, 3.3, -0.8]]   # Sample 3

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

print(np.dot(inputs, np.array(weights).T) + biases)
```

**Output:**

```text
[[ 4.8    1.21   2.385]
 [ 8.9   -1.81   0.2  ]
 [ 1.41   1.051  0.026]]
```

Each row is the layer's output for one input sample; each column belongs to one neuron. The first row is the single-sample result of section 9. The figure below draws the same call as grids.

![The equation Z equals X W transposed plus b and the call np.dot(X, W.T) + b, above four grids in a row. X, 3 by 4, one sample per row, times W transposed, 4 by 3, one neuron per column, plus b broadcast to 3 by 3, every row 2.0, 3.0, 0.5, copied from the stored b of shape (3,) above it, equals Z, 3 by 3: 4.800, 1.210, 2.385, then 8.900, minus 1.810, 0.200, then 1.410, 1.051, 0.026. Sample 2's row, neuron 2's column, its bias and Z entry (2, 2) are outlined: minus 4.81 plus 3.0 is minus 1.81.](diagrams/03-batch-in-one-call.svg)

*Three samples go through the same three neurons in one call. The bias row of shape $(3,)$ is added to every row of the $(3, 3)$ product, so each neuron's bias lands in its own column.*

## 11. The core formula

Every neuron in every dense layer computes

$$z = \sum_{i=1}^{n_\text{inputs}} w_i x_i + b.$$

A layer computes $\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b}$ for one sample, and a batch goes through as $\mathbf{Z} = \mathbf{X} \mathbf{W}^{\top} + \mathbf{b}$. The forward pass built here is never replaced. The rest of the series adds three things to this core:

| What is added | When it arrives | Why |
|---|---|---|
| Activation functions (ReLU, Softmax) | Post 06 | Without them, stacking layers collapses to a single linear layer. |
| Loss functions (cross-entropy) | Post 08 | A way to score the predicted outputs against the truth. |
| Backpropagation and an optimiser | Posts 09 to 27 | A way to adjust $\mathbf{W}$ and $\mathbf{b}$ so the loss goes down. |

Posts 28 to 34 then deal with training the finished stack well: testing, validation, regularisation, mini-batches, initialisation, and a second output activation for yes-or-no problems. Post 35 says what to read after the series. Once the three additions are in place, the same forward pass that produced `[4.8, 1.21, 2.385]` becomes the inner loop of a network that can classify the spiral data, recognise handwritten digits, or predict house prices.

## 12. Make it run: ten scripts, every printed number

Every code block in this post except the Keras fragment of section 1 comes from a file under `snippets/`, and every number the post quotes is printed by one of the ten scripts there or follows from a formula stated beside it. The first four need only Python; the rest need NumPy. Each runs in about a second or less from the series root, for example `python posts/01-neurons-and-layers/snippets/layer_batch.py`.

| Script | Section | What it prints |
|---|---|---|
| `neuron_three_inputs.py` | 4 | `2.3` |
| `neuron_four_inputs.py` | 5 | `4.8` |
| `layer_by_hand.py` | 7 | `[4.8, 1.21, 2.385]` |
| `layer_loops.py` | 8 | `[4.8, 1.21, 2.385]` |
| `neuron_numpy.py` | 9 | `4.8` |
| `layer_numpy.py` | 9 | the same three numbers as a NumPy array |
| `layer_batch.py` | 10 | the $3 \times 3$ batch output |
| `shape_diary.py` | 10 | the shape diary on real arrays and the count of 15 parameters |
| `loop_vs_numpy.py` | 9 | the timing of the loops against `np.dot` |
| `what_can_go_wrong.py` | 13 | every error and every wrong answer quoted in section 13 |

The first seven are listed in the sections named. `shape_diary.py` and `what_can_go_wrong.py` only wrap calls those sections already show, so their listings are left to the folder. The timing script is listed here in full, because its numbers depend on how the timing is done. It reuses the loops of section 8 as a function, draws the weights, inputs, and biases from a generator seeded with 0, and reports the best of five rounds for each variant.

```python
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
```

The first of the ten runs of section 9 printed the following. The timings change from run to run; on that machine the multiplication counts and the two differences did not.

```text
the layer of this post: 3 neurons, 4 inputs, 12 multiplications
  nested Python loops on lists        1.8 microseconds
  np.dot on the same lists            4.3 microseconds
  np.dot on NumPy arrays              1.5 microseconds
  loops / np.dot on arrays            1.2 times
  largest difference between the two results: 2.2e-16
a layer the size of an MNIST hidden layer: 128 neurons, 784 inputs, 100,352 multiplications
  nested Python loops on lists     3766.7 microseconds
  np.dot on the same lists         4795.9 microseconds
  np.dot on NumPy arrays             14.2 microseconds
  loops / np.dot on arrays          264.7 times
  largest difference between the two results: 8.5e-14
```

The last line of each block shows that the loops and `np.dot` agree to better than $10^{-13}$, but not bit for bit. Section 13 explains why.

## 13. What can go wrong?

The script `snippets/what_can_go_wrong.py` runs every call in this section on the arrays of this post and prints what happens. The messages and arrays below are its output under NumPy 2.3.5.

**Swapping the arguments of `np.dot`.** For one neuron the order is free: two vectors of equal length give the same sum either way. With a matrix it is not. `np.dot(weights, inputs)` pairs each row of four weights with the four inputs. `np.dot(inputs, weights)` tries to pair the four inputs with the first axis of `weights`, which has length 3, and raises an error. Post 02 goes through the argument orders one by one.

```text
ValueError: shapes (4,) and (3,4) not aligned: 4 (dim 0) != 3 (dim 0)
```

**Feeding a batch without the transpose.** `np.dot(batch, weights)` with two $(3, 4)$ arrays fails for the reason given in section 10, and the message names the two sizes that had to agree: the last axis of the first array (4 inputs per sample) and the first axis of the second (3 neurons).

```text
ValueError: shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)
```

Writing `weights.T` on the list fails differently, with `AttributeError: 'list' object has no attribute 'T'`, which is why the code of section 10 wraps the list in `np.array` first.

**Transposing the batch instead of the weights.** `np.dot(weights, np.array(batch).T)` is a legal product, $(3, 4)$ times $(4, N)$, but its result has shape $(n_\text{neurons}, N)$: neurons on the rows and samples on the columns, the transpose of what the rest of the code expects. With a batch of two the mistake is caught when the biases are added:

```text
ValueError: operands could not be broadcast together with shapes (3,2) (3,)
```

With the batch of three used in this post nothing is caught. The product is $(3, 3)$, the three biases broadcast along the wrong axis (one per sample instead of one per neuron), and the call returns

```text
[[ 4.8    9.9   -0.09 ]
 [ 0.21  -1.81  -1.449]
 [ 3.885  2.7    0.026]]
```

in which only the diagonal agrees with the correct output of section 10. A batch size equal to the number of neurons hides this bug, which is why `shape_diary.py` also runs a batch of two. A batch of one raises no error either: the $(3, 1)$ product broadcasts against the $(3,)$ biases into a $(3, 3)$ array, nine numbers for a single sample, and only a look at the shape gives the mistake away.

**Reading index 0.** `inputs[0]` is the first input, `1.0`, when `inputs` holds one sample, and the whole first sample, `[1.0, 2.0, 3.0, 2.5]`, when it holds a batch. `np.shape(inputs)` says which case applies, $(4,)$ or $(3, 4)$, and it works on lists as well as on arrays.

**Expecting exact decimals.** The table of section 4 shows $0.2 + 1.6 - 1.5 = 0.3$, but Python prints `0.30000000000000004` for that sum, because 0.2 and 1.6 have no exact binary representation; adding the bias happens to round back to `2.3`. Rounding also makes the order of additions matter. Adding the four products of neuron 2 in the order 1, 2, 3, 4 and then the bias gives `1.21`; swapping the last two products gives `1.2099999999999997`. The loops use the first order. `np.dot` runs a compiled routine that is free to choose its own, and on the checking machine it returned `1.2099999999999997` for neuron 2, a difference of $2.2 \times 10^{-16}$ from the loops. Another processor or another NumPy build may return `1.21` there, because the library chooses the routine. A printed array rounds to eight decimal places, so both display as `1.21`. Results that should be equal are therefore compared with `np.allclose`, which returns `True` here, never with `==`.

## Common pitfalls

1. **Confusing the number of weights with the number of neurons.** A layer has (number of neurons) × (inputs per neuron) weights. Only the bias count equals the neuron count: three neurons over four inputs have 12 weights and 3 biases.
2. **Transposing the wrong matrix when moving to batches.** With one row of weights per neuron, the batch call is `np.dot(inputs, weights.T)`, not `np.dot(weights, inputs.T)`. The second form does not always raise an error: section 13 shows it returning wrong numbers silently when the batch size equals the number of neurons, and for a batch of one.
3. **Adding the bias inside the dot product.** The bias is added once to each neuron's weighted sum, not to each input and not to each product.
4. **Treating a layer as non-linear.** Without an activation function, stacking layers is equivalent to a single linear layer with composed weights. The non-linearity arrives in post 06.
5. **Dropping the bias to "simplify" the network.** Without a bias, every neuron's output is zero when every input is zero. The bias costs one number per neuron and almost always pays for itself.

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 6, "Deep Feedforward Networks" (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 2 (2020). The inputs, weights, and biases used throughout this post are that chapter's.
- McCulloch, W. S. and Pitts, W., *"A Logical Calculus of the Ideas Immanent in Nervous Activity"* (Bulletin of Mathematical Biophysics, 1943).
- NumPy documentation, `numpy.dot` (latest).
- Rosenblatt, F., *"The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"* (Psychological Review, 1958).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

## What to read next

- **[Post 02 - NumPy and the dot product](../02-numpy-and-the-dot-product/index.md):** the three forms of `np.dot` (vector by vector, matrix by vector, matrix by matrix) and which one is a neuron, a layer, and a batch.
- **[Post 04 - The Dense layer class and spiral data](../04-dense-layer-class-and-spiral-data/index.md):** wraps this layer in a class, stores the weights transposed so the batch call needs no `.T`, and introduces the dataset the rest of the series trains on.
