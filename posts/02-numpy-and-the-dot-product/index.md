# 02 - NumPy and the dot product

> **TL;DR.** `np.dot` runs one of three computations, chosen silently by the shapes of its arguments: two vectors give a scalar (one neuron), a matrix and a vector give a vector (a layer on one sample), and two matrices give a matrix (a layer on a batch). One rule covers all three: the last axis of the first argument is summed against the first axis of the second, so those two sizes must be equal and the sizes that are left form the result. Two vectors can be swapped freely; once a matrix is involved the order changes the answer or breaks the call, which is why a batch of row samples against row-per-neuron weights is written `np.dot(inputs, weights.T)`.
>
> **Prerequisites:** [Post 01](../01-neurons-and-layers/index.md).
> **Safe to skip?** Skip it if the reader can already predict the result shape of `np.dot` for any pair of 1-D and 2-D arrays, say which argument order a layer call needs, and explain why a batch call transposes the weights.
>
> **After reading, you will be able to:**
>
> - Name the three forms of np.dot and match each one to a neuron, a layer, or a batch.
> - Predict from the shapes alone whether a np.dot call succeeds and what shape it returns.
> - Explain why swapping the arguments is harmless for two vectors and changes or breaks the call once a matrix is involved.
> - Decide when a layer call needs the transpose of the weight matrix, and write the batch call that uses it.

![Three equations, one per form of np.dot, drawn neurons-first with the weights and inputs of section 8 and no bias: the two vector forms side by side at the top, the matrix form across the bottom, each with its call, shapes and meaning beside or under it. Vector by vector, np.dot(w, x): the weights of neuron 1 as a row meet sample 1 as a column and give 2.800; shapes (4,) and (4,) give a scalar. Matrix by vector, np.dot(W, x): the 3 by 4 weight matrix meets the same column and gives 2.800, minus 1.790 and 1.885; shapes (3, 4) and (4,) give (3,). Matrix by matrix, np.dot(W, X.T): the same weights meet a 4 by 3 matrix whose columns are samples 1 to 3 and give a 3 by 3 result with one column per sample, the first column again 2.800, minus 1.790 and 1.885; shapes (3, 4) and (4, 3) give (3, 3).](diagrams/01-three-forms.svg)

*Three forms of one call, drawn neurons-first: one neuron per row of the weights, one sample per column of the second argument. The shapes decide which arithmetic runs, and the same dot product, neuron 1 with sample 1, appears in all three results.*

---

## 1. The question: what does one `np.dot` call compute?

Post 01 implemented a layer three ways, each more compact than the last. The final NumPy version came down to one expression, `np.dot(weights, inputs) + biases`. The same function appeared there with two flat lists, with a list of lists and a flat list, and with two 2-D arguments, and it returned a number, a vector, and a matrix in turn. That leaves the question this post answers: which computation does `np.dot` run for a given pair of shapes, and which neural-network object does each one stand for?

The question matters because the call is everywhere. Every modern deep-learning framework reduces the forward pass through a dense layer to the same product in different syntax: PyTorch's `nn.Linear` computes `x @ W.T + b`, and Keras's `Dense` computes `dot(x, W) + b` with its weights stored the other way round (section 7). On a CPU each of these ends in a matrix-multiply routine of the kind standardised by the **BLAS** (Basic Linear Algebra Subprograms), an interface first specified for Fortran in 1979 (Lawson et al., 1979). That first specification covered operations on vectors, the dot product among them; routines for a matrix times a vector and for a matrix times a matrix joined the same interface in 1988 and 1990. NumPy's standard builds pass floating-point `np.dot` calls to an optimised BLAS library, and GPU libraries implement the same interface. The arithmetic has not changed since; only its packaging has.

The implication for this series is direct. Every dense layer in every post, forward and backward, goes through `np.dot`. Getting comfortable with its three forms is the highest-leverage thing this post can teach. Section 10 measures what the call buys over the loops of post 01: on a realistic batch, on the machine that ran it, the loops took about a thousand times as long.

## 2. The dot product, formally

For two vectors $\mathbf{a}$ and $\mathbf{b}$ of equal length $n$, the dot product is the sum of their element-wise products:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i.$$

The result is a single number, a scalar. The dot notation comes from the vector analysis that Josiah Willard Gibbs worked out for his physics students in the 1880s, where the product has a geometric meaning: $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$, with $\theta$ the angle between the two vectors. That geometric reading is useful in physics and less useful here, because a neuron uses the raw weighted sum of its inputs, not the angle between two vectors. The arithmetic reading (multiply, then sum) is what matters in this series.

Two consequences follow from the definition:

- **Vectors of different lengths cannot be dotted.** The sum runs over $i = 1$ to $n$; both vectors need that same $n$.
- **The order of the arguments does not matter** for two vectors: $\sum_i a_i b_i$ and $\sum_i b_i a_i$ are the same sum, because each product $a_i b_i$ equals $b_i a_i$.

The first fact returns below as the requirement that two sizes match. The second holds only for two vectors and stops holding the moment a matrix enters the call.

## 3. The three forms of `np.dot`

On the 1-D and 2-D arrays this series uses, NumPy gives one function name three distinct behaviours. Which behaviour runs is decided silently by the shapes of the arguments. Reading `np.dot(A, B)` without knowing the shapes of `A` and `B` is reading a function whose definition is unknown.

| Form | Shapes in | Shape out | What it computes | Neural-network meaning |
|---|---|---|---|---|
| Vector by vector | $(n,)$ and $(n,)$ | scalar | $\sum_i a_i b_i$ | one neuron, one sample |
| Matrix by vector | $(m, n)$ and $(n,)$ | $(m,)$ | one dot product per row of the matrix | a layer of $m$ neurons, one sample |
| Matrix by matrix | $(m, n)$ and $(n, p)$ | $(m, p)$ | one dot product per pair of a row of the first and a column of the second | a layer of $m$ neurons, $p$ samples |

The three forms share a single rule: **the last axis of the first argument is contracted with the first axis of the second argument**, where contracted means multiplied pairwise and summed. Those two axes must have the same length. They disappear from the result, and the axes that are left form the output shape. A 1-D array has only one axis, which serves as both its first and its last.

The third row is written neurons-first, as the figure at the top draws it: the weights come first and the $p$ samples are the columns of the second matrix. Section 8 turns it round into the samples-first layout the series uses, in which every sample is a row.

One function carries three behaviours for historical reasons. `np.dot` predates the `@` operator, which arrived with Python 3.5 in 2015, and the NumPy documentation now recommends `@` or `np.matmul` when both arguments are 2-D. This series uses `np.dot` because the book it follows (Kinsley and Kukieła, 2020) does; for the arrays of this series the choice is cosmetic.

### 3.1. What `np.dot` is *not*

A short boundary section, because NumPy has several operations that look similar and behave differently.

- **`np.dot` is not element-wise multiplication.** The product `A * B` requires the two arrays to have identical or broadcast-compatible shapes, multiplies them position by position, and returns an array of the shape they share. Nothing is summed. It is a different operation entirely.
- **`np.dot` is not always interchangeable with `@` (the `__matmul__` operator) or `np.matmul`.** They agree for 1-D and 2-D arrays and diverge in two places. For arrays of three or more dimensions `@` treats each argument as a stack of matrices and broadcasts over the leading axes, while `np.dot` contracts the last axis of the first argument with the second-to-last axis of the second: shapes $(2, 3, 4)$ and $(2, 4, 5)$ give $(2, 3, 5)$ under `@` and $(2, 3, 2, 5)$ under `np.dot`. And `np.dot` accepts a scalar argument, which `np.matmul` rejects. The dense layers of this series only ever pass 1-D and 2-D arrays, so the two are interchangeable here.
- **`np.dot` is not a geometric dot product when given matrices.** With 2-D arguments it is matrix multiplication, and the angle reading of section 2 does not apply.

## 4. Vector by vector: order does not matter

The simplest case. Two 1-D arrays of the same length, one number out.

```python
import numpy as np

a = [1, 2, 3]
b = [4, 5, 6]

print(np.dot(a, b))   # 1*4 + 2*5 + 3*6 = 32
print(np.dot(b, a))   # 4*1 + 5*2 + 6*3 = 32
print(np.inner(a, b), np.array(a) @ np.array(b))   # two other spellings of the same sum
```

**Output:**

```text
32
32
32 32
```

The first two calls produce the same scalar because the underlying sum is commutative. This invariance does not survive the introduction of a matrix; it is unique to the vector by vector case. The third line shows that for two 1-D arrays `np.dot`, `np.inner`, and `@` all return this same number; a codebase should pick one spelling and keep to it. The functions convert plain lists themselves, while `@` is an operator that lists do not define, so at least one side must be an array; the line wraps both.

The neural-network reading: a single neuron with $n$ inputs and $n$ weights produces its weighted sum with `np.dot(weights, inputs)` or `np.dot(inputs, weights)`. Both work. The bias is added afterwards.

## 5. Matrix by vector: order matters

Move one of the arguments up to 2-D and the symmetry collapses. With a $3 \times 3$ matrix `B` and a 3-element vector `a`, the two orderings produce two different vectors:

```python
import numpy as np

a = np.array([1, 2, 3])

B = np.array([[ 4,  5,  6],
              [ 7,  8,  9],
              [10, 11, 12]])

print(np.dot(a, B))   # vector first: one dot product per column of B
print(np.dot(B, a))   # matrix first: one dot product per row of B
```

**Output:**

```text
[48 54 60]
[32 50 68]
```

Both calls succeed and both produce a 3-element vector, but the vectors are different. The reason is that `np.dot(a, B)` and `np.dot(B, a)` compute different sums, as the figure below draws them.

![Two panels with the same vector a, holding 1, 2, 3, and the same 3 by 3 matrix B, holding 4 to 12. Left, np.dot(a, B): a drawn as a row meets each column of B and gives 48, 54, 60; column 1 of B is outlined, with the worked line 1 times 4 plus 2 times 7 plus 3 times 10 equals 48. Right, np.dot(B, a): each row of B meets a drawn as a column and gives 32, 50, 68; row 1 of B is outlined, with 4 times 1 plus 5 times 2 plus 6 times 3 equals 32. Each operand carries its shape underneath, a (3,), B (3, 3) and the result (3,), and a caption line notes that both calls run only because B is square.](diagrams/02-order-matters.svg)

*The order of the arguments decides whether the vector meets the columns of the matrix or its rows. With a matrix in the call, `np.dot` is no longer commutative.*

**`np.dot(a, B)`, vector first.** The call behaves as if `a` were a $(1, 3)$ row vector: it computes one dot product for each *column* of `B`:

| Computation | Value |
|---|---|
| `a` with column 1 of `B` | $1 \cdot 4 + 2 \cdot 7 + 3 \cdot 10 = 48$ |
| `a` with column 2 of `B` | $1 \cdot 5 + 2 \cdot 8 + 3 \cdot 11 = 54$ |
| `a` with column 3 of `B` | $1 \cdot 6 + 2 \cdot 9 + 3 \cdot 12 = 60$ |

Result: `[48 54 60]`.

**`np.dot(B, a)`, matrix first.** The call behaves as if `a` were a $(3, 1)$ column vector: it computes one dot product for each *row* of `B`:

| Computation | Value |
|---|---|
| Row 1 of `B` with `a` | $4 \cdot 1 + 5 \cdot 2 + 6 \cdot 3 = 32$ |
| Row 2 of `B` with `a` | $7 \cdot 1 + 8 \cdot 2 + 9 \cdot 3 = 50$ |
| Row 3 of `B` with `a` | $10 \cdot 1 + 11 \cdot 2 + 12 \cdot 3 = 68$ |

Result: `[32 50 68]`.

The row and the column are a way of seeing which numbers meet. NumPy leaves no size-1 axis behind: both results are 1-D, of shape $(3,)$.

Both calls succeed only because the matrix is square ($3 \times 3$) and the vector has length 3, so the vector's one axis can meet either axis of the matrix. The moment the matrix is non-square, only one of the two orderings satisfies the shape rule:

```python
C = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3): no longer square

print(np.dot(C, a))            # (2, 3) with (3,): the 3s meet, result (2,)
try:
    np.dot(a, C)               # (3,) with (2, 3): 3 meets 2
except ValueError as error:
    print("ValueError:", error)
```

**Output:**

```text
[14 32]
ValueError: shapes (3,) and (2,3) not aligned: 3 (dim 0) != 2 (dim 0)
```

The message names the two shapes and then the two sizes that failed to meet: the vector's only axis, of size 3, and the first axis of the matrix, of size 2.

This is the layer call of post 01. With `W` of shape $(m, n)$, one row per neuron, and one sample `x` of shape $(n,)$, `np.dot(W, x)` returns the $m$ weighted sums. The other order, `np.dot(x, W)`, raises unless $m = n$, and when $m = n$ it runs and returns different numbers (section 11).

## 6. Matrix by matrix: the shape rule, once and for all

For two 2-D arrays `A` of shape $(m, n)$ and `B` of shape $(n, p)$, the call `np.dot(A, B)` computes an $(m, p)$ matrix whose entry at row $i$, column $j$ is the dot product of row $i$ of `A` with column $j$ of `B`:

$$(\mathbf{A}\mathbf{B})_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}.$$

The figure below draws the product for the two matrices of the code further down, in both orders.

![Two rows. Top, np.dot(A, B): A of shape (3, 4), holding 1 to 12, times B of shape (4, 3), holding 1 to 12, gives the 3 by 3 result 70, 80, 90; 158, 184, 210; 246, 288, 330, with row 1 of A and column 1 of B outlined and the worked line 1 times 1 plus 2 times 4 plus 3 times 7 plus 4 times 10 equals 70. Bottom, np.dot(B, A): the same matrices swapped give a 4 by 4 result, drawn without values, with a note that the post prints only its shape. Every shape label sets the inner sizes in purple and the outer sizes in green, and a purple bracket labelled inner joins the two inner sizes of each product. At the right, the general rule, (m, n) times (n, p) gives (m, p), with m and p bracketed as outer, forming the result, and the two n bracketed as inner, equal and summed away.](diagrams/03-shape-rule.svg)

*The two $n$ in $(m, n) \cdot (n, p)$ must be equal; they are contracted away. The $m$ and the $p$ survive into the result, so swapping the arguments here turns a $(3, 3)$ result into a $(4, 4)$ one.*

The summary is one sentence and one rule:

> **Inner dimensions match. Outer dimensions become the result.**

```python
import numpy as np

A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])    # shape (3, 4)

B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])       # shape (4, 3)

print(np.dot(A, B))                 # (3, 4) with (4, 3): inner 4s meet, result (3, 3)
print(np.dot(B, A).shape)           # (4, 3) with (3, 4): inner 3s meet, result (4, 4)
```

**Output:**

```text
[[ 70  80  90]
 [158 184 210]
 [246 288 330]]
(4, 4)
```

The top-left entry is row 1 of `A` with column 1 of `B`: $1 \cdot 1 + 2 \cdot 4 + 3 \cdot 7 + 4 \cdot 10 = 70$. Swapping the arguments is legal here, because the inner sizes of $(4, 3)$ and $(3, 4)$ are both 3, but it is a different product with a different shape, $(4, 4)$. For two matrices the order changes the answer even when both orders run.

The same rule explains the matrix by vector case of section 5. A 1-D vector of length $n$ acts as a $(1, n)$ row when it is the first argument and as an $(n, 1)$ column when it is the second. The contracted axis is always the inner one.

## 7. The transpose, and when to reach for it

A transpose swaps the rows and columns of a matrix: entry $(i, j)$ of $\mathbf{W}^\top$ is entry $(j, i)$ of $\mathbf{W}$, so an $(m, n)$ matrix becomes $(n, m)$. In NumPy it is the attribute `.T`:

```python
import numpy as np

W = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(W.shape, W.T.shape)
print(W.T)
print(np.shares_memory(W, W.T))     # True: .T is a view of the same numbers
print(np.array([1, 2, 3]).T.shape)  # a 1-D array has no second axis to swap
```

**Output:**

```text
(3, 4) (4, 3)
[[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]
True
(3,)
```

`.T` copies nothing: it is a second view of the same numbers, so the transpose itself costs no arithmetic and no memory. Transposing a 1-D array does nothing, because there is no second axis to swap with.

A plain Python list has no `.T`; the matrix must first become an array with `np.array(...)`. Forgetting that wrapper is a common first-day error. Lists differ from arrays in a second way that raises no error at all, which is that their `+` and `*` are not arithmetic:

```python
rows = [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]]            # a plain Python list of lists

try:
    rows.T
except AttributeError as error:
    print("AttributeError:", error)
print(np.array(rows).T.shape)       # an array has .T

print([1, 2] + [3, 4])              # lists concatenate
print([1, 2] * 3)                   # lists repeat
print(np.array([1, 2]) + np.array([3, 4]))
print(np.array([1, 2]) * 3)
```

**Output:**

```text
AttributeError: 'list' object has no attribute 'T'
(4, 3)
[1, 2, 3, 4]
[1, 2, 1, 2, 1, 2]
[4 6]
[3 6]
```

Adding two lists concatenates them and multiplying a list by 3 repeats it; the same operators on arrays work element by element. `np.dot` converts list arguments itself, which is why the list-based code of post 01 ran, but any other arithmetic on them needs arrays.

The transpose enters neural-network code for one reason. Posts 01 to 03 store the **weights with one row per neuron**: $\mathbf{W}$ has shape $(m, n)$, where $m$ is the number of neurons in the layer and $n$ is the number of inputs each neuron consumes. The convention for inputs is **one row per sample**: $\mathbf{X}$ has shape $(N, n)$, where $N$ is the batch size and $n$ is the feature count. The two letters are easy to conflate in running prose, so to be explicit: capital $N$ counts samples, lowercase $n$ counts features. The [notation guide](../../notation_guide.md) writes $n_\text{neurons}$ for $m$ and $n_\text{inputs}$ for $n$.

One row per neuron is one of two conventions in use, and the frameworks do not agree on it. PyTorch's `nn.Linear` stores its weight with shape `(out_features, in_features)`, one row per neuron, and transposes in the call. Keras's `Dense` stores its kernel with shape `(input_dim, units)`, one column per neuron, and does not. The two layouts differ by a transpose and nothing else; they are matters of agreement, not necessity. Post 04 switches the layer class of this series to one column per neuron, and the `.T` leaves the forward call.

Two layouts now exist, and they must be reconciled:

- **One sample at a time.** $\mathbf{x}$ is a 1-D vector of shape $(n,)$. The call `np.dot(W, x)` is $(m, n) \cdot (n,) \to (m,)$, which lines up; no transpose is needed.
- **A batch of $N$ samples.** $\mathbf{X}$ is a 2-D matrix of shape $(N, n)$. The call `np.dot(X, W)` is $(N, n) \cdot (m, n)$, whose inner sizes are $n$ and $m$. Unless the layer happens to have as many neurons as inputs, they do not match. The fix is to transpose `W` to $(n, m)$: `np.dot(X, W.T)` is $(N, n) \cdot (n, m) \to (N, m)$.

In both cases the answer comes out the same way: every output is the dot product of one sample with one neuron's row of weights. The transpose is bookkeeping, not arithmetic. It returns in post 14, where the same shape rule decides which matrix is transposed in the weight gradient.

## 8. Batching, end to end

The matrix by matrix form of section 3 was framed neurons-first, as $(m, n) \cdot (n, p)$. A batch turns that round to samples-first, which is why the transpose appears: the inputs lead with the sample count, so the weights are transposed to bring the shared feature axis inward. A batch of three samples through a layer of three neurons, drawn in the figure and computed in the code below it:

![Section 8's batch, samples-first, laid out as one line read left to right: the inputs, shape (3, 4), one row per sample, times weights.T, shape (4, 3), one column per neuron, plus the bias row 2.0, 3.0, 0.5, added to every row, equals the outputs, shape (3, 3): 4.800, 1.210, 2.385; 8.900, minus 1.810, 0.200; 1.410, 1.051, 0.026. Above, the stored weights, shape (3, 4), one row per neuron, lead through an arrow labelled transpose down to weights.T. The row of sample 1 and the column of neuron 1 are outlined, and their output 4.800 is bold, worked out as 1.0 times 0.20 plus 2.0 times 0.80 plus 3.0 times minus 0.50 plus 2.5 times 1.00 plus the bias 2.0. A note beside the stored weights says that without the transpose the 4 meets a 3 and NumPy raises ValueError.](diagrams/04-batch-transpose.svg)

*Samples-first: one sample per row of the inputs and of the outputs, one neuron per column of `weights.T`. The transpose is what makes the inner sizes meet, and the bias row is added to every row of the product.*

```python
import numpy as np

inputs = np.array([[ 1.0,  2.0,  3.0,  2.5],     # Sample 1
                   [ 2.0,  5.0, -1.0,  2.0],     # Sample 2
                   [-1.5,  2.7,  3.3, -0.8]])    # Sample 3

weights = np.array([[ 0.2,   0.8,  -0.5,   1.0 ],
                    [ 0.5,  -0.91,  0.26, -0.5 ],
                    [-0.26, -0.27,  0.17,  0.87]])

biases = np.array([2.0, 3.0, 0.5])

outputs = np.dot(inputs, weights.T) + biases
print(outputs)
```

**Output:**

```text
[[ 4.8    1.21   2.385]
 [ 8.9   -1.81   0.2  ]
 [ 1.41   1.051  0.026]]
```

Each row of the output is one sample's result across every neuron. Each column belongs to one neuron's results across every sample. The bias vector of shape $(3,)$ is added to all three rows, so each neuron's bias lands in its own column; that is broadcasting, covered in detail in post 05. Three more lines of the same script check the bookkeeping:

```python
print(inputs.shape, weights.T.shape, outputs.shape)
print(np.dot(weights, inputs[0]) + biases)       # sample 1 alone: no transpose
print(np.allclose(np.dot(weights, inputs.T).T, np.dot(inputs, weights.T)))
```

**Output:**

```text
(3, 4) (4, 3) (3, 3)
[4.8   1.21  2.385]
True
```

The shapes follow the rule: $(3, 4) \cdot (4, 3) \to (3, 3)$. Sample 1 on its own, through the matrix by vector form with no transpose, gives the first row of the batch result, the `[4.8, 1.21, 2.385]` that post 01 computed by hand. The last line confirms that the two framings are one computation: the neurons-first product `np.dot(weights, inputs.T)` has one column per sample, and its transpose is the samples-first result, $(\mathbf{W}\mathbf{X}^{\top})^{\top} = \mathbf{X}\mathbf{W}^{\top}$.

The two equivalent ways to phrase the layer call:

$$\text{single sample:} \quad \mathbf{W}\mathbf{x} + \mathbf{b} \qquad \text{batch:} \quad \mathbf{X}\mathbf{W}^{\top} + \mathbf{b}$$

The layer code of this series takes a batch from here on, because every real training step is a batch step. Post 03 chains two calls of the batch form; from post 04 the `Layer_Dense` class stores its weights already transposed, and the same product is written `np.dot(inputs, self.weights)`.

## 9. The shape diary for `np.dot`

| Setting | First argument | Second argument | Call | Output |
|---|---|---|---|---|
| Single neuron, single sample | $(n,)$ | $(n,)$ | `np.dot(w, x)` | scalar |
| Layer of $m$, single sample | $(m, n)$ | $(n,)$ | `np.dot(W, x)` | $(m,)$ |
| Layer of $m$, batch of $N$ | $(N, n)$ | $(n, m)$ | `np.dot(X, W.T)` | $(N, m)$ |
| Two general matrices | $(m, n)$ | $(n, p)$ | `np.dot(A, B)` | $(m, p)$ |

The fastest debugging move when a shape error is raised is to print the `.shape` of every array involved, then walk down this table. The usual culprit is a missing transpose or a swapped argument order.

## 10. Make it run: every block, and the speed gap measured

Every code block in this post is a file under `snippets/`, or a contiguous part of one, and runs from the series root, for example `python posts/02-numpy-and-the-dot-product/snippets/batch_layer.py`. The files are `vector_dot.py` (section 4), `order_matters.py` (section 5), `shape_rule.py` (section 6), `transpose.py` (section 7), `batch_layer.py` (section 8), and `silent_failures.py` (section 11). They need only NumPy, and each prints the output shown under its block.

Post 01 timed its loops against `np.dot` for one sample through one layer. The batch form of this post does more work in a single call, and `snippets/loop_vs_dot.py` measures what that is worth: it computes the same batched layer with nested loops and with `np.dot`, checks that the two results agree, and reports how many times longer the loops take, for the toy layer of this post and for a batch of 1,000 samples with 100 features through 64 neurons.

```python
"""Time one batched layer call two ways: nested Python loops and np.dot.

Run from the series root:  python posts/02-numpy-and-the-dot-product/snippets/loop_vs_dot.py

The two results are compared before any time is reported. The times depend on the machine and
change from run to run; what the post quotes is the size of the gap.
"""
import time

import numpy as np


def layer_loop(samples, neurons):
    """One dot product per (sample, neuron) pair, with plain Python lists and loops."""
    outputs = []
    for sample in samples:
        row = []
        for neuron_weights in neurons:
            total = 0.0
            for x_i, w_i in zip(sample, neuron_weights):
                total += x_i * w_i
            row.append(total)
        outputs.append(row)
    return outputs


def best_of(repeats, function, *args):
    """Smallest wall-clock time of several calls, in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def compare(n_samples, n_features, n_neurons, loop_repeats, dot_repeats):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, n_features))    # one row per sample
    W = rng.standard_normal((n_neurons, n_features))    # one row per neuron
    X_list, W_list = X.tolist(), W.tolist()

    same = np.allclose(layer_loop(X_list, W_list), np.dot(X, W.T))
    loop_seconds = best_of(loop_repeats, layer_loop, X_list, W_list)
    dot_seconds = best_of(dot_repeats, np.dot, X, W.T)

    print(f"({n_samples}, {n_features}) by ({n_features}, {n_neurons}): "
          f"{n_samples * n_features * n_neurons:,d} multiply-adds, same numbers: {same}")
    print(f"  nested loops {loop_seconds * 1e3:12.4f} ms")
    print(f"  np.dot       {dot_seconds * 1e3:12.4f} ms")
    print(f"  the loops take {loop_seconds / dot_seconds:,.0f} times as long")


compare(3, 4, 3, loop_repeats=2000, dot_repeats=2000)       # the toy layer of this post
compare(1000, 100, 64, loop_repeats=3, dot_repeats=20)      # a realistic batch
```

**Output of one run** (the times differ on every run and every machine):

```text
(3, 4) by (4, 3): 36 multiply-adds, same numbers: True
  nested loops       0.0060 ms
  np.dot             0.0015 ms
  the loops take 4 times as long
(1000, 100) by (100, 64): 6,400,000 multiply-adds, same numbers: True
  nested loops     260.6873 ms
  np.dot             0.2462 ms
  the loops take 1,059 times as long
```

The large case is one call of the batch form: $1{,}000 \cdot 64 = 64{,}000$ dot products of 100 terms each. In this run the loops needed about 260 ms and `np.dot` about a quarter of a millisecond. Twenty-six runs of the script on a 6-core, 12-thread Intel laptop CPU gave ratios between about 900 and 1,600, so three orders of magnitude is the honest summary and the exact figure is not worth remembering. The loops execute several interpreter instructions and create a new Python number object for every product, while `np.dot` makes one call into compiled code that walks the arrays' memory directly, the BLAS routine of section 1.

The toy layer tells the other half. With 36 multiply-adds the loops took only about 4 times as long in nearly all of the same runs, because at that size the fixed cost of entering a NumPy call is most of the time. Vectorisation pays in proportion to the size of the arrays, and neural-network arrays are large.

## 11. What can go wrong?

A shape error is the friendly failure. The dangerous ones run without complaint, and `snippets/silent_failures.py` collects them:

```python
import numpy as np

inputs = np.array([[ 1.0,  2.0,  3.0,  2.5],
                   [ 2.0,  5.0, -1.0,  2.0],
                   [-1.5,  2.7,  3.3, -0.8]])       # (3, 4): 3 samples, 4 features

weights = np.array([[ 0.2,   0.8,  -0.5,   1.0 ],
                    [ 0.5,  -0.91,  0.26, -0.5 ],
                    [-0.26, -0.27,  0.17,  0.87]])    # (3, 4): 3 neurons, 4 inputs

# 1. A missing transpose is loud here: the inner sizes are 4 and 3.
try:
    np.dot(inputs, weights)
except ValueError as error:
    print("np.dot(inputs, weights):", error)

# 2. A square weight matrix makes the same mistake silent.
square = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])                 # 3 neurons, 3 inputs
x = np.array([[1.0, 0.0, 0.0]])                     # one sample: only input 1 is on
print("with .T:   ", np.dot(x, square.T))           # every neuron's first weight
print("without .T:", np.dot(x, square))             # neuron 1's three weights

# 3. The star operator is silent whenever the two shapes are equal or broadcast.
print("inputs * weights:", (inputs * weights).shape)
try:
    np.ones((5, 4)) * weights
except ValueError as error:
    print("(5, 4) * (3, 4):", str(error).strip())

# 4. np.dot and @ part ways above two dimensions, and on scalars.
P = np.ones((2, 3, 4))
Q = np.ones((2, 4, 5))
print("np.dot:", np.dot(P, Q).shape, "  @:", (P @ Q).shape)
print("np.dot(2, square):", np.dot(2, square).shape)
try:
    np.matmul(2, square)
except ValueError:
    print("np.matmul(2, square): ValueError")
```

**Output:**

```text
np.dot(inputs, weights): shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)
with .T:    [[1. 4. 7.]]
without .T: [[1. 2. 3.]]
inputs * weights: (3, 4)
(5, 4) * (3, 4): operands could not be broadcast together with shapes (5,4) (3,4)
np.dot: (2, 3, 2, 5)   @: (2, 3, 5)
np.dot(2, square): (3, 3)
np.matmul(2, square): ValueError
```

- **A missing transpose that raises.** `np.dot(inputs, weights)` stops with the first line above: the two shapes, then the two sizes that failed to meet, axis 1 of the first argument (the 4 features) and axis 0 of the second (the 3 neurons). This is the failure to hope for, because it stops at the faulty line.
- **A missing transpose that runs.** When a layer has as many neurons as inputs, $\mathbf{W}$ is square, and both `np.dot(X, W)` and `np.dot(X, W.T)` satisfy the shape rule. For the sample $(1, 0, 0)$ the correct call returns `[[1. 4. 7.]]`, the first weight of each neuron; without the transpose the call returns `[[1. 2. 3.]]`, the three weights of neuron 1, and nothing is raised. The square `B` of section 5 ran in both orders for the same reason: both of its axes match the vector's length, so either alignment is legal. The defence is a test with a layer whose sizes are all different.
- **`*` where `np.dot` was meant.** The star operator multiplies position by position and sums nothing. It runs here because both shapes are $(3, 4)$, and returns a $(3, 4)$ array that is not a layer output; with five samples it raises. It is silent exactly when the two shapes happen to be equal or to broadcast.
- **`np.dot` where `@` was meant.** Above two dimensions the two return different shapes, $(2, 3, 2, 5)$ against $(2, 3, 5)$ here, and only `np.dot` accepts a scalar. Code that will ever hold a stack of matrices should use `@`.
- **A 1-D result is neither a row nor a column.** `np.dot(W, x)` returns shape $(m,)$, not $(m, 1)$ or $(1, m)$, and transposing it changes nothing (section 7). An $(n,)$ array standing where a column was intended is the silent bug that post 05 takes apart.

## Common pitfalls

1. **Calling `np.dot(inputs, weights)` on a batch.** With inputs of shape $(N, n)$ and row-per-neuron weights of shape $(m, n)$ the inner sizes are $n$ and $m$, and the call raises `ValueError: shapes ... not aligned`. The fix is always to make the inner dimensions match: `np.dot(inputs, weights.T)`. For a single 1-D sample the call is `np.dot(weights, x)`, with no transpose.
2. **Trusting a call because it ran.** A square weight matrix lets a missing transpose through without an error, and two equal shapes let `*` through. Check one output against a dot product worked by hand, and test with sizes that are all different.
3. **Using `*` instead of `np.dot`.** `*` is element-wise multiplication, not matrix multiplication. It requires shapes to match or be broadcast-compatible and returns an array of that shape, not a contracted one.
4. **Forgetting `np.array()` before `.T` or before arithmetic.** Python lists have no `.T` attribute, and their `+` and `*` concatenate and repeat. `np.array(weights).T` is the cheapest fix; converting the weights once at the top of the function is cleaner.
5. **Confusing row-per-neuron with column-per-neuron.** Posts 01 to 03 store one row per neuron and transpose in the batch call; from post 04 the `Layer_Dense` class stores one column per neuron and does not. Mixing the two inside one project gives a shape error or, for a square layer, wrong numbers and no error.
6. **Trusting a 3-D `np.dot` call to do what `@` would do.** They disagree for arrays of three or more dimensions. Stick with `@` or `np.matmul` if the number of dimensions ever climbs above two.

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 2, "Linear Algebra" (MIT Press, 2016).
- Harris, C. R., et al., *"Array Programming with NumPy"* (Nature, 2020).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 2 (2020).
- Lawson, C., Hanson, R., Kincaid, D., and Krogh, F., *"Basic Linear Algebra Subprograms for FORTRAN Usage"* (ACM Transactions on Mathematical Software, 1979).
- NumPy documentation: `numpy.dot`, `numpy.matmul`, and the `@` operator (latest).
- Strang, G., *Introduction to Linear Algebra*, chapter 1 (Wellesley-Cambridge, 6th edition, 2023).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

## What to read next

- **[Post 03 - Stacking layers and the forward pass](../03-stacking-layers-and-the-forward-pass/index.md):** two calls of the batch form chained, so that the output of one layer becomes the input of the next.
- **[Post 05 - Array summation, keepdims, and broadcasting](../05-array-summation-keepdims-and-broadcasting/index.md):** the rule that let the $(3,)$ bias vector be added to every row of the $(3, 3)$ output.
