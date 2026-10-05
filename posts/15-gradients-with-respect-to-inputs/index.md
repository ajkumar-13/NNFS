# 15 - Gradients with respect to inputs

> **TL;DR.** A dense layer inside a stack owes the layer before it one more gradient than its own weights and biases need: $\partial L / \partial \mathbf{X}$, the gradient of the loss with respect to the layer's *inputs*. Each input feeds every neuron, so its gradient is a sum with one term per neuron, and the sums for all inputs and all samples are one matrix product, $\partial L / \partial \mathbf{X} = (\partial L / \partial \mathbf{Z}) \, \mathbf{W}^\top$. On the layer of posts 13 and 14 the product gives $[64.80, 77.76, 90.72, 103.68]$, and a central difference on each input agrees to $10^{-8}$. With it the backward pass of a dense layer is three lines of NumPy.
>
> **Prerequisites:** [Post 14](../14-matrices-in-backpropagation/index.md).
> **Safe to skip?** Skip it if the reader can already say why a layer returns `np.dot(dvalues, self.weights.T)`, why that array has one row per sample, and how to check it with a finite difference.
>
> **After reading, you will be able to:**
>
> - State why the input gradient is a sum over neurons while the weight gradient is a single product.
> - Derive the input gradient of a dense layer from the chain rule as the upstream gradient times the transposed weight matrix.
> - Verify an input gradient numerically with a central difference on each input.
> - Explain why the input gradient keeps one row per sample while the weight and bias gradients sum over the batch.
> - Write the three NumPy lines of a dense layer's backward pass and say which result is passed back to the layer before, and why.

![Two panels of three inputs wired to three neurons. Left: one highlighted wire from the first input to the first neuron, and a weight gradient that is a single product. Right: the three wires leaving the first input, and an input gradient that is a sum of three products. A band below gives the matrix product and its shapes.](diagrams/01-input-gradients.svg)

*A weight lies on one path through the layer; an input lies on $m$ paths, one per neuron, and its gradient adds them. The band writes the product for an array with one row of weights per neuron (section 3.1).*

---

## 1. The question: what must a layer pass back to the layer before it?

Post 14 computed the gradient of the loss with respect to a dense layer's weights and biases. For a single dense layer training on its own, the example of posts 13 and 14, those two gradients are everything.

For a network with **two or more layers** that is not enough. The weight gradient of any layer is built from that layer's **upstream gradient**. In this post, as in post 14, the term means $\partial L / \partial \mathbf{Z}$, the gradient of the loss with respect to the layer's pre-activations; post 13 used it for the single number $\partial L / \partial \hat{y}$ that the loss hands back. The last layer receives its upstream gradient from the loss. Every earlier layer has to receive it from the layer after it, and a layer that returns only its own weight and bias gradients returns nothing an earlier layer can use.

What an earlier layer can use is the gradient of the loss with respect to *its own output*. The output of one layer is the input of the next, so that quantity is the next layer's gradient with respect to its **inputs**, $\partial L / \partial \mathbf{X}$. This is the third gradient of a dense layer, and the only one of the three that leaves the layer.

![A card with the three lines of a dense layer's backward call, the upstream gradient of shape N by m arriving from the right. The weight gradient, n by m, and the bias gradient, 1 by m, stay and go to the optimiser. The input gradient, N by n, travels left to the previous layer.](diagrams/02-gradient-handoff.svg)

*Two of the three gradients are consumed by the optimiser. The third leaves the layer, and it is the only reason the layers before it can learn anything.*

In one sentence: **a layer's input gradient becomes the upstream gradient of the layer before it**, once it has passed back through whatever activation sits between the two. Without it backpropagation stops at the first boundary between layers, and every weight before that boundary keeps its initial value. In the code names of the series, a component receives `dvalues` and returns `dinputs`, and its `dinputs` is the previous component's `dvalues`. The inputs themselves are never updated: they are data, or the outputs of an earlier layer, and neither is a parameter. Section 7.3 runs the hand-off on two layers.

---

## 2. Why this gradient is a sum

The notation is that of posts 13 and 14. A layer has $n$ inputs and $m$ neurons, $x_j$ is input $j$, $z_k$ is the pre-activation of neuron $k$, and $w_{kj}$ is the weight that neuron $k$ puts on input $j$ (neuron first, input second):

$$z_k = \sum_{j=1}^{n} w_{kj} x_j + b_k.$$

The weight $w_{kj}$ appears in the pre-activation of neuron $k$ and in no other, so its gradient involves the upstream gradient of that one neuron, $\partial L / \partial w_{kj} = (\partial L / \partial z_k) \cdot x_j$.

An input $x_j$ has the opposite structure. **It feeds every neuron in the layer.** It contributes to $z_1, z_2, \dots, z_m$ through the weights $w_{1j}, w_{2j}, \dots, w_{mj}$, and the loss depends on $x_j$ through all of those pre-activations at once.

Post 11 met the smallest case of this: a variable that reaches the loss along two paths, where the chain rule is applied along each path and the two results are added. Here there are $m$ paths, one per neuron, and the results are again **added**:

$$\frac{\partial L}{\partial x_j} = \sum_{k=1}^{m} \frac{\partial L}{\partial z_k} \cdot \frac{\partial z_k}{\partial x_j} = \sum_{k=1}^{m} \frac{\partial L}{\partial z_k} \cdot w_{kj}.$$

For each neuron $k$ the local derivative $\partial z_k / \partial x_j$ is the weight $w_{kj}$, because $x_j$ appears exactly once in $z_k$, multiplied by $w_{kj}$. That is a partial derivative in the sense of post 10, section 3: every other input and every weight is held fixed.

The sum can be read off a small nudge. If $x_j$ grows by a small amount $\Delta$, each $z_k$ moves by exactly $w_{kj} \Delta$, and each of those moves changes the loss by about $\partial L / \partial z_k$ times its size. To first order the effects of several small moves made at once add, so the loss changes by about $\Delta$ times the sum above. Section 4 checks this: a nudge of $0.01$ on one input is predicted to change the loss by $0.6480$ and changes it by $0.6482$.

A sum over paths can also cancel: a positive term from one neuron and a negative term from another can leave the input gradient close to zero.

### 2.1. Why the weight gradient was not a sum

The same rule applies to a weight, but each $w_{kj}$ lies on exactly one path through the layer, from $x_j$ into $z_k$, so its sum has one term. A sum with one term is a single product, which is what post 13 derived without any sum over paths.

| Variable | Appears in | Paths to the loss | Gradient |
|---|---|:---:|---|
| weight $w_{kj}$ | $z_k$ only | 1 | $(\partial L / \partial z_k) \cdot x_j$ |
| bias $b_k$ | $z_k$ only | 1 | $\partial L / \partial z_k$ |
| input $x_j$ | $z_1, \dots, z_m$ | $m$ | $\sum_k (\partial L / \partial z_k) \cdot w_{kj}$ |

The two formulas mirror each other: the upstream gradient is multiplied by the *input* on the wire in one and by the *weight* on the wire in the other. Both follow from the same line, $z_k = \sum_j w_{kj} x_j + b_k$, differentiated once with respect to $w_{kj}$ and once with respect to $x_j$.

The count of paths in the table is for one sample. Over a batch a weight is used once per sample, and its gradient does become a sum, over the samples (section 5).

---

## 3. The matrix form

The sum of section 2 runs over the neuron index $k$, which makes it a dot product: the row vector $\partial L / \partial \mathbf{Z}$, with one entry per neuron, against the list of weights $w_{1j}, \dots, w_{mj}$ that leave input $j$. Doing it for every $j$ at once is a matrix product of the upstream row with the array that holds those lists as its columns.

That array has one row per neuron and one column per input, with $w_{kj}$ at row $k$, column $j$. In the notation of the series (`notation_guide.md`) the weight matrix $\mathbf{W}$ has shape $(n, m)$, one row per input and one column per neuron, which puts $w_{kj}$ at row $j$, column $k$ of $\mathbf{W}$. The array just described is therefore its transpose, and the input gradient is

$$\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Z}} \, \mathbf{W}^\top.$$

For one sample the shapes are $(1, m) \cdot (m, n) \rightarrow (1, n)$: the upstream row, then $\mathbf{W}^\top$ with one row per neuron, then one entry per input. The inner size $m$ that matches and disappears is the one being summed over: the $m$ neurons whose contributions the formula adds. The result has the shape of $\mathbf{X}$, as every gradient has the shape of the quantity it differentiates. None of the three gradients of a layer can be evaluated before $\partial L / \partial \mathbf{Z}$ is known, which is why the backward pass runs from the loss towards the data.

### 3.1. One formula, two array layouts

The series has stored weights in two layouts, and the code line for the input gradient depends on which one an array uses.

| Layout | Array | Shape | Forward pass | Input gradient |
|---|---|:---:|---|---|
| One row per neuron (posts 01 to 03, and the worked examples of posts 13 and 14) | `weights`, which holds $\mathbf{W}^\top$ | $(m, n)$ | `X @ weights.T` | `dL_dZ @ weights` |
| One column per neuron (`Layer_Dense`, post 04 onward) | `self.weights`, which holds $\mathbf{W}$ | $(n, m)$ | `np.dot(inputs, self.weights)` | `np.dot(dvalues, self.weights.T)` |

Both lines compute $(\partial L / \partial \mathbf{Z}) \, \mathbf{W}^\top$. In the first layout the array already *is* the transpose, so the `.T` sits in the forward line; in the second it sits in the backward line. The rule that survives both: **the input gradient multiplies by the transpose of whatever matrix the forward pass multiplied by.** Post 14, section 8, reconciles the two layouts for the weight gradient in the same way.

The worked example keeps the one-row-per-neuron array of posts 13 and 14. For two-dimensional arrays `a @ b` and `np.dot(a, b)` are the same operation.

---

## 4. Worked example: the layer of posts 13 and 14

The layer is the one post 13 trained and post 14 rewrote in matrices: four inputs, three neurons with ReLU, the three activations summed into $\hat{y}$ (`Y` in the code), and the loss $L = \hat{y}^2$.

```python
# The layer of posts 13 and 14: four inputs, three neurons, one row of weights per neuron.
X = np.array([[1.0, 2.0, 3.0, 4.0]])            # (1, 4): one sample
weights = np.array([[0.1, 0.2, 0.3, 0.4],       # neuron 1
                    [0.5, 0.6, 0.7, 0.8],       # neuron 2
                    [0.9, 1.0, 1.1, 1.2]])      # neuron 3; shape (3, 4)
biases = np.array([[0.1, 0.2, 0.3]])            # (1, 3)


def forward(X):
    """Z = X W + b with ReLU, then Y = the sum of the activations and L = Y squared."""
    Z = X @ weights.T + biases                  # (1, 3): one pre-activation per neuron
    A = np.maximum(0, Z)                        # ReLU
    Y = np.sum(A)
    return Z, Y, Y ** 2


Z, Y, L = forward(X)
dL_dZ = 2 * Y * (Z > 0)                         # (1, 3): 2Y through each ReLU gate
```

The forward pass gives $\mathbf{Z} = [3.1, 7.2, 11.3]$, $\hat{y} = 21.6$, and $L = 466.56$, the numbers of post 13. All three pre-activations are positive, so every ReLU gate is open and the upstream gradient is $2\hat{y} = 43.2$ for each neuron, $\partial L / \partial \mathbf{Z} = [43.2, 43.2, 43.2]$, as in post 14.

Section 2 as code is two loops, one over the inputs and one over the paths of each input:

```python
# Section 2: one input at a time, one path per neuron, summed.
n_neurons, n_inputs = weights.shape
by_paths = np.zeros((1, n_inputs))
for j in range(n_inputs):
    for k in range(n_neurons):
        by_paths[0, j] += dL_dZ[0, k] * weights[k, j]
```

Section 3 as code is one line, and section 3.1 is the same line for the other layout:

```python
# Section 3: the same sums as one matrix product.
dL_dX = dL_dZ @ weights                         # (1, 3) @ (3, 4) = (1, 4)

# Section 3.1: Layer_Dense stores the transposed array, one column per neuron.
W = weights.T                                   # (4, 3) = (n_inputs, n_neurons)
dinputs = np.dot(dL_dZ, W.T)                    # (1, 3) . (3, 4) = (1, 4)
```

`snippets/input_gradient.py` prints all three:

```text
== Section 2: the input gradient as a sum over paths, one path per neuron
dL/dx1 = 43.2 * 0.1 + 43.2 * 0.5 + 43.2 * 0.9 = 64.80
dL/dx2 = 43.2 * 0.2 + 43.2 * 0.6 + 43.2 * 1.0 = 77.76
dL/dx3 = 43.2 * 0.3 + 43.2 * 0.7 + 43.2 * 1.1 = 90.72
dL/dx4 = 43.2 * 0.4 + 43.2 * 0.8 + 43.2 * 1.2 = 103.68
for contrast, a weight has one path: dL/dw11 = dL/dz1 * x1 = 43.2 * 1 = 43.2

== Section 3: the same four sums as one matrix product
dL_dZ @ weights          = [64.80, 77.76, 90.72, 103.68]   shape (1, 3) @ (3, 4) = (1, 4)
np.dot(dL_dZ, W.T)       = [64.80, 77.76, 90.72, 103.68]   with W = weights.T, shape (4, 3)
largest gap, loops against matrix product: 0.0e+00
largest gap between the two layouts:       0.0e+00
```

Each entry is the sum of three contributions, one per neuron. Because the three upstream entries are equal here, each sum is $43.2$ times one column sum of `weights`: $1.5$, $1.8$, $2.1$, and $2.4$. The loss is most sensitive to $x_4$ because the weights leaving $x_4$ are the largest, not because $x_4 = 4$ is the largest input.

Each component is read like any partial derivative. The script nudges $x_1$ by $0.01$ with the other inputs fixed: the predicted change in the loss is $64.8 \times 0.01 = 0.6480$ and the actual change is $0.6482$, larger by $0.000225 = (1.5 \times 0.01)^2$, the second-order term of a squared loss, which a first derivative does not see.

---

## 5. Batches

In training, $\mathbf{X}$ is a batch of $N$ samples, one per row. Sample $i$ has inputs $x_{ij}$ and pre-activations $z_{ik}$, and the pre-activations of sample $i$ depend on the inputs of sample $i$ only. The sum over paths is therefore the same sum as before, taken inside one row:

$$\frac{\partial L}{\partial x_{ij}} = \sum_{k=1}^{m} \frac{\partial L}{\partial z_{ik}} \cdot w_{kj}.$$

The matrix form does not change; the shapes become $(N, m) \cdot (m, n) \rightarrow (N, n)$. Row $i$ of the result is row $i$ of the upstream gradient times $\mathbf{W}^\top$: the input gradient of sample $i$, computed from that sample's upstream row and from nothing else. The batch axis $N$ is on the outside of the product, so it survives; the neuron axis $m$ is on the inside, so it is summed away.

This is where the input gradient parts from the other two. In post 14 the weight gradient put the batch axis on the *inside* of its product, and the bias gradient summed over it explicitly, because a weight or a bias is one number shared by every sample: each sample pulls on it, and the pulls add. An input is not shared. Each sample has its own row of inputs, and the layer before needs the gradients row by row, because its own pre-activations are per sample too.

The code line is the one from section 4 with bigger operands:

```python
# Section 5: the input gradient of the whole batch is one product, in either layout.
dL_dX = dL_dZ @ weights                         # (3, 3) @ (3, 4) = (3, 4)
```

`snippets/batch_and_handoff.py` runs it on the three-sample batch of post 14, pushed through the layer of section 4. Each sample's activations are summed into its own $\hat{y}_i$, and the loss is the mean of $\hat{y}_i^2$ over the batch. The upstream gradient is then $2 \hat{y}_i / N$ for every open gate of sample $i$; the $1/N$ of the mean enters there, once, and no layer divides by $N$ again (post 14, section 6.3).

```text
== Section 5: a batch of three samples through the layer of posts 13 and 14
Y per sample: 18.00, 15.30, 8.22   L = mean of Y^2 = 208.5528
upstream gradient dL/dZ, shape (3, 3)
[[12.   12.   12.  ]
 [10.2  10.2  10.2 ]
 [ 5.48  5.48  5.48]]
input gradient dL_dZ @ weights, shape (3, 4)
[[18.    21.6   25.2   28.8  ]
 [15.3   18.36  21.42  24.48 ]
 [ 8.22   9.864 11.508 13.152]]
largest gap to a central difference on all 12 inputs: 2.8e-09
largest gap to the three rows computed one sample at a time: 3.6e-15
```

The first row can be confirmed by hand: $\hat{y}_1 = 18$, so the upstream entries are $2 \cdot 18 / 3 = 12$, and $12 \times [1.5, 1.8, 2.1, 2.4] = [18, 21.6, 25.2, 28.8]$, with the column sums of section 4. The last line of the output is the statement that samples do not interact: the three rows computed one sample at a time match the batched product to rounding error.

---

## 6. The complete backward toolkit for a dense layer

Posts 13 and 14 gave two gradients and this post gives the third. With $\mathbf{X}$ of shape $(N, n)$, $\mathbf{W}$ of shape $(n, m)$, and the upstream gradient $\partial L / \partial \mathbf{Z}$ of shape $(N, m)$:

| Gradient | Formula | `Layer_Dense` line | Shape | Goes to |
|---|:---:|---|:---:|---|
| Weights | $\mathbf{X}^\top \, (\partial L / \partial \mathbf{Z})$ | `np.dot(self.inputs.T, dvalues)` | $(n, m)$ | the optimiser |
| Biases | $\sum_{\text{rows}} \partial L / \partial \mathbf{Z}$ | `np.sum(dvalues, axis=0, keepdims=True)` | $(1, m)$ | the optimiser |
| Inputs | $(\partial L / \partial \mathbf{Z}) \, \mathbf{W}^\top$ | `np.dot(dvalues, self.weights.T)` | $(N, n)$ | the layer before |

The formulas are written for $\mathbf{W}$ of shape $(n, m)$; post 14 derived the weight gradient in the one-row-per-neuron layout as $(\partial L / \partial \mathbf{Z})^\top \mathbf{X}$, the transpose of the entry in the table (post 14, section 8). On the batch of section 5 the whole backward pass of the layer is:

```python
# Section 6: the three lines of the backward pass, in the layout Layer_Dense stores.
inputs, dvalues = X, dL_dZ
dweights = np.dot(inputs.T, dvalues)                    # (4, 3), the shape of W
dbiases = np.sum(dvalues, axis=0, keepdims=True)        # (1, 3), the shape of b
dinputs = np.dot(dvalues, W.T)                          # (3, 4), the shape of inputs
```

Every gradient has the shape of the array it differentiates. The first two collect the whole batch into one update and stay in the layer. The third keeps one row per sample and is the one passed back, because it is the only one the layer before can turn into its own upstream gradient (section 1).

The three lines also show what the layer must remember from its forward pass. The weight gradient needs the inputs the layer saw, and the input gradient needs the weights it used. The bias gradient needs neither.

These three lines are the **entire backward pass** of a dense layer. Post 16 puts them in a `backward` method on `Layer_Dense`.

### 6.1. What the input gradient is *not*

- **It is not the activation's backward step.** The upstream gradient used here has already come back through the layer's own activation, the factor `(Z > 0)` of the worked example. In the other direction, the input gradient a layer returns is with respect to the previous layer's *activations*, and it must pass back through that layer's activation (for ReLU, its gate) before it becomes the upstream gradient of that layer's pre-activations. Post 17 derives that step.
- **It is not meaningless at the first layer.** No layer precedes the first, so nothing in training reads its input gradient. The same array is the gradient of the loss with respect to the data, which methods outside this series (adversarial examples, saliency maps) compute.

---

## 7. Make it run: the input gradient, checked

The scripts under `snippets/` produce every number the post quotes. Each needs only NumPy, uses no random numbers, computes in float64, and runs in under a second from the series root, as `python posts/15-gradients-with-respect-to-inputs/snippets/<name>.py`: `input_gradient.py` for sections 2 to 4 and 7.1, `batch_and_handoff.py` for sections 5, 6, 7.2, and 7.3, and `what_can_go_wrong.py` for section 8.

### 7.1. A finite difference on each input

The formula of section 3 is a claim about slopes, and post 10 gave a way to measure a slope without any formula: the central difference $(L(x + h) - L(x - h)) / (2h)$ with $h = 10^{-5}$. Applied to an input gradient, it moves one input at a time, holds the others, and runs the whole forward pass twice, knowing nothing about weights or paths:

```python
def numerical_input_gradient(X, h=1e-5):
    """Central difference of the loss along each input in turn (post 10, section 2.5)."""
    grad = np.zeros_like(X)
    for index in np.ndindex(*X.shape):
        step = np.zeros_like(X)
        step[index] = h                         # move one input, hold the others
        grad[index] = (forward(X + step)[2] - forward(X - step)[2]) / (2 * h)
    return grad
```

```text
== Section 7.1: central difference on each input, h = 1e-5
dL/dx1   matrix product  64.800000   central difference  64.800000
dL/dx2   matrix product  77.760000   central difference  77.760000
dL/dx3   matrix product  90.720000   central difference  90.720000
dL/dx4   matrix product 103.680000   central difference 103.680000
largest gap: 1.0e-08
```

The two columns agree to $10^{-8}$ on numbers of size 100. With every gate open the loss is a quadratic function of each input, and a central difference has no truncation error on a quadratic; what is left is rounding error.

### 7.2. The batch

For the batch of section 5 the same check runs over all 12 inputs, and `dinputs = np.dot(dvalues, W.T)` in the `Layer_Dense` layout differs from the central difference by at most $2.8 \times 10^{-9}$. The gaps for `dweights` and `dbiases`, which post 14 checked on this batch, are $3.1 \times 10^{-9}$ and $2.3 \times 10^{-9}$.

### 7.3. Two layers: the hand-off

The last part of the script puts a second dense layer, with three inputs and two neurons, after the ReLU of the first. The backward pass now has to cross a layer boundary:

```python
dvalues2 = 2 * Y2 / N * np.ones_like(Z2)        # (3, 2): what the loss hands to layer 2
dweights2 = np.dot(A1.T, dvalues2)              # (3, 2): stays in layer 2
dinputs2 = np.dot(dvalues2, W2.T)               # (3, 3): leaves layer 2

dvalues1 = dinputs2 * (Z1 > 0)                  # through layer 1's ReLU gate
dweights1 = np.dot(X.T, dvalues1)               # (4, 3): layer 1 can now learn
dinputs1 = np.dot(dvalues1, W.T)                # (3, 4): the gradient at the data
```

Layer 2 receives a $(3, 2)$ array, one column per neuron of its own, and returns a $(3, 3)$ array, one column per neuron of layer 1; that change of width is what the product with $\mathbf{W}^\top$ does. The fifth line is the only place where layer 1 learns anything about the loss, and its right-hand side exists only because layer 2 produced `dinputs2`. Against central differences on the whole two-layer forward pass the gaps are $5.2 \times 10^{-10}$ for layer 2's weight gradient, $3.4 \times 10^{-10}$ for layer 1's, and $4.9 \times 10^{-10}$ for layer 1's input gradient. The smallest $|z|$ in layer 1 is $1.16$, so no step of $10^{-5}$ crosses a ReLU corner (post 10, section 7).

---

## 8. What can go wrong?

`snippets/what_can_go_wrong.py` reproduces five failures on the layer and the batch of section 5. Its first block is shown; the figures in the other bullets are from the same output.

```text
== A closed ReLU gate removes a path: neuron 2's weights negated, first sample only
pre-activations Z: [ 2.5 -5.6  9.5]
upstream gradient: [24.  0. 24.]
input gradient:    [24.  28.8 33.6 38.4]   gap to central difference 3.4e-09
with the gate open: [54.  64.8 75.6 86.4]
every gate closed (all weights and biases negated): input gradient [0. 0. 0. 0.]
```

- **A path is closed, or all of them are.** A neuron whose ReLU gate is closed has an upstream entry of 0 and contributes nothing to the sum of section 2. With neuron 2 switched off for the first sample (taken alone, so $N = 1$), the input gradient is the sum of two paths, $24 \times [1.0, 1.2, 1.4, 1.6]$, and not of three. If every gate of a layer is closed for a sample, the input gradient of that sample is exactly zero and no layer before it receives any signal from that sample, which is why a layer in which every ReLU neuron is dead blocks learning in everything before it.
- **The transpose is in the wrong place.** In a layer with different numbers of inputs and neurons, `np.dot(dvalues, W)` in the `Layer_Dense` layout fails at once with a shape error, which is the good case. In a square layer (3 inputs, 3 neurons in the script) both products have the same shape and the wrong one runs without complaint: the first row comes out as $[4.8, 14.4, 24.0]$ where the true gradient is $[12.0, 14.4, 16.8]$, and only the finite difference shows the gap of $7.2$. A shape check is necessary and not sufficient.
- **The input gradient is summed over the batch.** `np.sum(dinputs, axis=0)`, written by analogy with the bias gradient, has shape $(4,)$ and matches no sample's row; the layer before expects one row per sample.
- **The division by $N$ is applied twice.** The $1/N$ of a mean loss is already inside the upstream gradient (post 14, section 6.3). Dividing `dinputs` by $N$ again leaves every entry at $1/3$ of its true value on this batch, with the right shape, and every gradient in the earlier layers shrinks by the same factor.
- **The check is run in float32.** With $h = 10^{-5}$ the float32 central difference misses a correct gradient by about $1.5$ on entries no larger than $28.8$; with $h = 10^{-2}$ the gap is $1.1 \times 10^{-3}$, and in float64 with $h = 10^{-5}$ it is $2.8 \times 10^{-9}$, so a gradient check is run on float64 copies of the arrays (post 10, section 7).

---

## 9. Summary

| Concept | Takeaway |
|---|---|
| Why $\partial L / \partial \mathbf{X}$ matters | It is the only gradient that leaves the layer; without it no earlier layer receives an upstream gradient |
| Sum over paths | Each input feeds every neuron, so its gradient adds one term per neuron; a weight feeds one neuron and has one term |
| Matrix form | $\partial L / \partial \mathbf{X} = (\partial L / \partial \mathbf{Z}) \, \mathbf{W}^\top$, shape $(N, m) \cdot (m, n) \rightarrow (N, n)$ |
| Batch behaviour | One row per sample, never summed over the batch, unlike the weight and bias gradients |
| Three-line backward | Weights, biases, inputs: one NumPy expression each, each with the shape of what it differentiates |

---

## Common pitfalls

1. **Summing the input gradient over the batch.** It stays per sample, shape $(N, n)$; only the weight and bias gradients sum over the batch.
2. **Placing the transpose by habit.** The line is `np.dot(dvalues, self.weights.T)` for one column per neuron and `dL_dZ @ weights` for one row per neuron, and in a square layer the wrong one has the right shape and the wrong numbers.
3. **Using the input gradient to update something in the current layer.** It updates nothing there: `dweights` and `dbiases` go to the optimiser, and `dinputs` goes to the previous component.
4. **Dropping the input gradient at an intermediate layer.** Only the first layer's input gradient goes unread; every other one is the sole source of the upstream gradient for all the layers before it.
5. **Expecting the formula to change with the activation.** A different activation changes the upstream gradient that enters the product, not the product with $\mathbf{W}^\top$.

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.5.4 (Back-Propagation Computation in Fully-Connected MLP), where the gradient passes to the layer below through a product with the transposed weight matrix (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (Backpropagation), for the same derivation with the layout of `Layer_Dense` (2020).
- LeCun, Y., Bottou, L., Orr, G. B., and Müller, K.-R., *"Efficient BackProp"*, in *Neural Networks: Tricks of the Trade*, which states backpropagation module by module: each module turns the gradient with respect to its output into one for its parameters and one for its input (Springer, 1998).
- Nielsen, M., *Neural Networks and Deep Learning*, chapter 2 (How the backpropagation algorithm works), where a layer's error is obtained from the error of the layer after it by the input gradient of this post followed by the activation's derivative (online, 2015).
- Rumelhart, D., Hinton, G., and Williams, R., *"Learning Representations by Back-Propagating Errors"*, the paper that made backpropagation widely known, in which the error derivative is passed from each layer to the one below (Nature, 1986).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 16 - Coding backpropagation](../16-coding-backpropagation/index.md):** the three lines of section 6 become `Layer_Dense.backward`, with the inputs cached in `forward`.
- **[Post 20 - Assembling full backpropagation](../20-assembling-full-backpropagation/index.md):** the hand-off of section 7.3 repeated across every component of a whole network, from the loss back to the first layer.
