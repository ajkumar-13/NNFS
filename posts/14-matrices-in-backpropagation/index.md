# 14 - Matrices in backpropagation

> **TL;DR.** The twelve per-weight gradients of post 13 are the entries of one matrix product, $\partial L / \partial \mathbf{W} = (\partial L / \partial \mathbf{Z})^{\top} \mathbf{X}$, and the three bias gradients are one sum of the upstream gradient over the batch axis. This post derives that two-line backward pass, checks that it returns post 13's numbers exactly (43.2, 86.4, 129.6, and 172.8 in every row), and shows that the same two lines serve a batch of any size, because the product sums the per-sample contributions over the axis it contracts. A central-difference check on a three-sample batch agrees with the two lines to a relative error of about $10^{-10}$.
>
> **Prerequisites:** [Post 13](../13-backprop-through-a-layer/index.md).
> **Safe to skip?** Skip it if the reader can already write the weight and bias gradients of a dense layer as two NumPy lines in either weight layout, give the shape of each, and say why neither line changes when one sample becomes a batch.
>
> **After reading, you will be able to:**
>
> - Compute the weight and bias gradients of a dense layer for one sample or a batch with two NumPy lines.
> - Read off the shape of each gradient array from the shapes of the inputs and the layer.
> - Explain why the matrix form of the weight gradient sums the contributions of a batch automatically.
> - Convert the two gradient lines between the row-per-neuron and the column-per-neuron weight layouts.

![An outer product drawn as three grids: the upstream gradient transposed as a (3, 1) column holding 43.2 for neurons 1, 2 and 3, the input X as a (1, 4) row holding 1, 2, 3, 4 above the result, and the (3, 4) weight gradient in which every row reads 43.2, 86.4, 129.6, 172.8. The entry for neuron 2 and input 3 is outlined and worked out beside the grids as 43.2 times 3 = 129.6, with the shapes (3, 1) times (1, 4) giving (3, 4) and the code line dL_dW = dL_dZ.T @ X.](diagrams/01-matrix-weight-gradient.svg)

*One matrix product, twelve weight gradients. Each row of the result belongs to one neuron and each column to one input, the same layout as the stored weights.*

---

## 1. The question: how do twelve gradients become one matrix product?

Post 13 derived twelve weight gradients by writing one expression per weight. That worked for a layer of three neurons over four inputs. A layer of 10,000 neurons over 1,000 inputs has 10,000,000 weights, and no usable implementation computes ten million gradients one Python expression at a time.

The remedy is to express the whole computation as a single matrix product. The chain rule does not change; the bookkeeping does. The per-neuron, per-input gradient that post 13 arrived at,

$$\frac{\partial L}{\partial w_{kj}} = \frac{\partial L}{\partial z_k} \cdot x_j,$$

is the entry in row $k$ and column $j$ of a matrix whose rows are neurons and whose columns are inputs. This post derives the product that fills that matrix, checks it against the numbers of post 13, and then asks what happens when the single sample becomes a batch. The answer to that second question is the reason the matrix form is used everywhere: the same product, with no change, adds up the contributions of every sample in the batch.

---

## 2. The weight gradient as a matrix product

This post keeps the layout of [post 13](../13-backprop-through-a-layer/index.md): the weights are stored with **one row per neuron and one column per input**, shape $(m, n)$ for $m$ neurons over $n$ inputs. That is also the layout of posts 01 to 03. The `Layer_Dense` class of post 04 stores the transpose, one column per neuron, and section 8 converts every formula of this post to that layout.

Single elements keep the lowercase symbols of post 13: $x_j$ for an input, $w_{kj}$ for the weight from input $j$ to neuron $k$, $z_k$ for a pre-activation, and $a_k$ for an activation. Bold capitals stand for whole arrays. For a single sample the arrays and their shapes are:

| Quantity | Shape | Meaning |
|---|:---:|---|
| $\mathbf{X}$ | $(1, n)$ | one sample, $n$ inputs |
| $\mathbf{Z}$ | $(1, m)$ | the pre-activations, one per neuron |
| $\partial L / \partial \mathbf{Z}$ | $(1, m)$ | the upstream gradient: the sensitivity of the loss to each pre-activation (section 3) |
| $\partial L / \partial \mathbf{W}$ | $(m, n)$ | one row per neuron, one column per input |

The entry formula comes from the forward pass. Neuron $k$ computes $z_k = \sum_{j} w_{kj} x_j + b_k$, so the weight $w_{kj}$ appears in exactly one pre-activation, $z_k$, and there it multiplies $x_j$. The local derivative is $\partial z_k / \partial w_{kj} = x_j$, the chain rule (post 11) multiplies it by the gradient arriving at $z_k$, and no other path leads from $w_{kj}$ to the loss:

$$\frac{\partial L}{\partial w_{kj}} = \frac{\partial L}{\partial z_k} \cdot \frac{\partial z_k}{\partial w_{kj}} = \frac{\partial L}{\partial z_k} \cdot x_j.$$

One factor depends only on the neuron $k$ and the other only on the input $j$. A table whose $(k, j)$ entry is a number for row $k$ times a number for column $j$ is the **outer product** of a column vector and a row vector. Here the column is $(\partial L / \partial \mathbf{Z})^{\top}$, of shape $(m, 1)$, and the row is $\mathbf{X}$, of shape $(1, n)$:

$$\frac{\partial L}{\partial \mathbf{W}} = \left(\frac{\partial L}{\partial \mathbf{Z}}\right)^{\top} \mathbf{X}.$$

The shape rule of post 02 confirms it: $(m, 1) \cdot (1, n) \rightarrow (m, n)$. The inner sizes are both 1, they match and disappear, and the outer sizes $m$ and $n$ survive. The result illustrates a rule that holds for every gradient in the series: **a gradient has the shape of the array it updates**. Here that is the stored weight matrix, so the update of post 09 can be applied element by element as `weights -= lr * dL_dW`.

### 2.1. Why this is an outer product, not a dot product

Two vectors of the same length can be multiplied in two ways that are easy to confuse:

- The **dot product** $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$ returns a single number.
- The **outer product** $\mathbf{a} \mathbf{b}^{\top}$ returns a matrix whose $(i, j)$ entry is $a_i b_j$.

`snippets/single_sample.py` prints both for the same pair of vectors:

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print("dot product:", np.dot(a, b))
print("outer product:")
print(np.outer(a, b))
```

```text
dot product: 32.0
outer product:
[[ 4.  5.  6.]
 [ 8. 10. 12.]
 [12. 15. 18.]]
```

The dot product multiplies matching positions and adds the results: $4 + 10 + 18 = 32$. The outer product multiplies every position of the first vector by every position of the second and adds nothing. The gradient needs the second structure, because each of the $m \times n$ weights needs a number of its own. The outer product also does not require the two vectors to have the same length: the upstream gradient has $m$ entries and the input has $n$. In code it is written `dL_dZ.T @ X`; for two-dimensional arrays `@` is the same product as the `np.dot` of post 02.

---

## 3. Where $\partial L / \partial \mathbf{Z}$ comes from

The matrix product consumes the upstream gradient $\partial L / \partial \mathbf{Z}$. That quantity is the chain rule applied to everything between the loss and the layer's pre-activations. In the example of post 13 the three ReLU outputs $a_k$ are summed to $\hat{y}$ and the loss is $L = \hat{y}^2$, so three factors lie between $L$ and $z_k$:

$$\frac{\partial L}{\partial z_k} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_k} \cdot \frac{\partial a_k}{\partial z_k}.$$

| Factor | Value in the example |
|---|:---:|
| $\partial L / \partial \hat{y}$ | $2\hat{y} = 43.2$, one scalar shared by all neurons (post 13's layer sum was $\hat{y} = 21.6$) |
| $\partial \hat{y} / \partial a_k$ | $1$, the derivative of a sum with respect to one of its terms |
| $\partial a_k / \partial z_k$ | the ReLU gate, $1$ where $z_k > 0$ and $0$ elsewhere; here $1$ for every $k$ |

`snippets/single_sample.py` holds the inputs, weights, and biases of post 13, with the sample stored as a $(1, 4)$ row `X`, and computes the forward pass and the three factors. The code names are post 13's: `Z`, `A`, `Y` for the layer sum $\hat{y}$, and a gradient named by its fraction.

```python
# Forward pass: the three outputs are summed and the sum is squared.
Z = X @ weights.T + biases                         # shape (1, 3)
A = np.maximum(0, Z)                               # ReLU
Y = np.sum(A)
L = Y ** 2

# The upstream gradient: three chain-rule factors per neuron.
dL_dY = 2 * Y                                      # scalar
dY_dA = np.ones_like(A)                            # the sum passes 1 to each neuron
dA_dZ = np.where(Z > 0, 1.0, 0.0)                  # the ReLU gate
dL_dZ = dL_dY * dY_dA * dA_dZ                      # shape (1, 3)

print("Z:", Z, " Y:", round(Y, 6), " L:", round(L, 6))
print("dL_dZ:", dL_dZ, dL_dZ.shape)
```

```text
Z: [[ 3.1  7.2 11.3]]  Y: 21.6  L: 466.56
dL_dZ: [[43.2 43.2 43.2]] (1, 3)
```

So $\partial L / \partial \mathbf{Z} = [43.2,\ 43.2,\ 43.2]$, one value per neuron. The three values are equal only because this example shares one scalar $2\hat{y}$ and opens every gate; in a trained network they differ from neuron to neuron and from sample to sample. The figure follows the sample forward from $\mathbf{Z}$ to the loss and the gradient back again, one factor per step.

![Two lanes for the single sample. Forward, left to right: Z = 3.1, 7.2, 11.3, ReLU gives A = 3.1, 7.2, 11.3, the sum gives y-hat = 21.6, and squaring gives L = 466.56. Backward, right to left: times 2 y-hat gives 43.2, times 1 through the sum gives 43.2 for each neuron, and times the open ReLU gate gives the upstream gradient 43.2, 43.2, 43.2, outlined. A line underneath multiplies the three factors, 43.2 times 1 times 1.](diagrams/02-upstream-gradient.svg)

*The three factors of the chain rule, step by step. Whatever follows the layer reaches it as one array of shape (1, 3).*

The matrix product of section 2 never looks inside this array: whatever follows the layer is summarised in the upstream gradient, which is why one backward pass for a dense layer serves at every depth.

---

## 4. Numerical verification (single input)

With the upstream gradient in hand, the weight gradient is one line of the same script:

```python
dL_dW = dL_dZ.T @ X                                # (3, 1) @ (1, 4) = (3, 4)
print(dL_dW)
```

```text
[[ 43.2  86.4 129.6 172.8]
 [ 43.2  86.4 129.6 172.8]
 [ 43.2  86.4 129.6 172.8]]
```

These are the twelve numbers of post 13's table, arranged as a $(3, 4)$ matrix whose rows are neurons and whose columns are inputs. Column $j$ is $43.2 \cdot x_j$: $43.2$, $86.4$, $129.6$, and $172.8$ for the inputs $1$, $2$, $3$, and $4$. The figure at the top of the post draws this product with the upstream column beside the result and the input row above it, so each entry sits where its two factors meet. The script then rebuilds the same matrix the slow way, with one chain-rule product per weight, and compares:

```python
# The same twelve numbers, one chain-rule expression per weight, as post 13 wrote them.
per_weight = np.zeros((3, 4))
for k in range(3):                                 # neuron
    for j in range(4):                             # input
        per_weight[k, j] = dL_dZ[0, k] * X[0, j]
print("shape:", dL_dW.shape, "equals weights.shape:", dL_dW.shape == weights.shape)
print("identical to the twelve per-weight products:", np.array_equal(dL_dW, per_weight))
print("identical to post 13's reshape and broadcast:",
      np.array_equal(dL_dW, dL_dZ.reshape(-1, 1) * X[0]))
```

```text
shape: (3, 4) equals weights.shape: True
identical to the twelve per-weight products: True
identical to post 13's reshape and broadcast: True
```

`np.array_equal` tests every entry for exact equality, with no tolerance. The match is exact because for a single sample the matrix product performs the very same twelve multiplications, one per entry, and adds nothing. The matrix form is not an approximation of the chain rule. It is the chain rule with the twelve results stored in one array.

---

## 5. Bias gradients

The bias $b_k$ enters $z_k = \sum_j w_{kj} x_j + b_k$ with a coefficient of 1, so $\partial z_k / \partial b_k = 1$ and the chain rule gives

$$\frac{\partial L}{\partial b_k} = \frac{\partial L}{\partial z_k} \cdot 1 = \frac{\partial L}{\partial z_k}.$$

A bias behaves like a weight attached to an input that is always 1. For one sample, the bias-gradient vector of the whole layer is therefore the upstream gradient itself, and no product is needed:

```python
# Bias gradients: the input of a bias is the constant 1.
dL_db = dL_dZ.flatten()                            # shape (3,), the shape of biases
print("dL_db:", dL_db, dL_db.shape)
print("np.sum(dL_dZ, axis=0):", np.sum(dL_dZ, axis=0))
```

```text
dL_db: [43.2 43.2 43.2] (3,)
np.sum(dL_dZ, axis=0): [43.2 43.2 43.2]
```

The `flatten` call only changes the shape, from the $(1, 3)$ row of the upstream gradient to the $(3,)$ shape of the biases. The last line reaches the same array by summing over axis 0, the spelling that survives the move to a batch.

---

## 6. Extending to batches

Training feeds the layer a batch of $N$ samples, one per row. The shapes change in a predictable way:

| Quantity | Single sample | Batch of $N$ |
|---|:---:|:---:|
| $\mathbf{X}$ | $(1, n)$ | $(N, n)$ |
| $\partial L / \partial \mathbf{Z}$ | $(1, m)$ | $(N, m)$ |
| $\partial L / \partial \mathbf{W}$ | $(m, n)$ | $(m, n)$, unchanged |
| $\partial L / \partial \mathbf{b}$ | $(m,)$ | $(m,)$, unchanged |

The inputs and the upstream gradient gain rows; the two gradients do not, since a layer has one set of weights and biases however many samples pass through it.

The derivation repeats section 2 with one more index. Sample $i$ has its own inputs $x_{ij}$ and its own pre-activations, $z_{ik} = \sum_j x_{ij} w_{kj} + b_k$, and the same weight $w_{kj}$ now appears in $z_{1k}, z_{2k}, \dots, z_{Nk}$: once per sample. A variable that reaches the loss along several paths collects one chain-rule term per path, and the terms add:

$$\frac{\partial L}{\partial w_{kj}} = \sum_{i=1}^{N} \frac{\partial L}{\partial z_{ik}} \cdot x_{ij}.$$

The right-hand side is, term for term, the definition of a matrix product. Entry $(k, j)$ of $\mathbf{P}\mathbf{Q}$ is $\sum_i P_{ki} Q_{ij}$, and with $\mathbf{P} = (\partial L / \partial \mathbf{Z})^{\top}$, whose entry $(k, i)$ is $\partial L / \partial z_{ik}$, and $\mathbf{Q} = \mathbf{X}$, the sum above is entry $(k, j)$ of

$$\frac{\partial L}{\partial \mathbf{W}} = \left(\frac{\partial L}{\partial \mathbf{Z}}\right)^{\top} \mathbf{X}.$$

The formula has not changed from the single-sample case. The shapes now multiply as $(m, N) \cdot (N, n) \rightarrow (m, n)$: the transpose has moved the batch axis of the upstream gradient to the inside of the product, where it meets the batch axis of $\mathbf{X}$. A matrix product sums over the inner axis it contracts, so the product **adds up the per-sample contributions without being asked to**. For one sample the contracted axis had length 1 and there was nothing to add; for a batch it has length $N$ and the sum has $N$ terms.

For the biases the same argument applies with the constant input 1 in place of $x_{ij}$. The bias $b_k$ appears in $z_{ik}$ for every sample $i$, so its gradient is the sum of column $k$ of the upstream gradient:

$$\frac{\partial L}{\partial b_k} = \sum_{i=1}^{N} \frac{\partial L}{\partial z_{ik}}.$$

In NumPy that is `np.sum(dL_dZ, axis=0)`: axis 0 is the batch axis, and the sum removes it (post 05). Both gradients are thus summed over the batch in the same way. The weight sum is hidden inside the matrix product, where each term is scaled by an input value; the bias sum has no input to scale it and is written out.

### 6.1. Worked batch example

`snippets/batch.py` uses three samples of four inputs. The upstream gradient is made up for the purpose: every neuron receives 1 from the first sample, 2 from the second, and 3 from the third, so that each sample's share of the result is easy to follow.

```python
X = np.array([[ 1.0,  2.0,  3.0,  2.5],
              [ 2.0,  5.0, -1.0,  2.0],
              [-1.5,  2.7,  3.3, -0.8]])     # shape (3, 4): three samples, four inputs

dL_dZ = np.array([[1.0, 1.0, 1.0],
                  [2.0, 2.0, 2.0],
                  [3.0, 3.0, 3.0]])          # shape (3, 3): one row per sample, one column per neuron

dL_dW = dL_dZ.T @ X                # (3, 3) @ (3, 4) = (3, 4)
dL_db = np.sum(dL_dZ, axis=0)      # shape (3,)

print("Weight gradients:")
print(dL_dW)
print("Bias gradients:", dL_db)
```

```text
Weight gradients:
[[ 0.5 20.1 10.9  4.1]
 [ 0.5 20.1 10.9  4.1]
 [ 0.5 20.1 10.9  4.1]]
Bias gradients: [6. 6. 6.]
```

The weight-gradient matrix has shape $(3, 4)$, as in the single-sample case, and each bias gradient is $1 + 2 + 3 = 6$. The three rows are equal only because the three columns of this made-up upstream gradient are equal.

### 6.2. The sum, one sample at a time

The claim of this section is that the matrix product equals the sum of the per-sample outer products of section 4. The same script checks it with a loop:

```python
# The same result built one sample at a time: an outer product per sample, then a sum.
total = np.zeros((3, 4))
for i in range(len(X)):
    one_sample = np.outer(dL_dZ[i], X[i])    # what section 4 computed for a single sample
    print(f"sample {i + 1}, row of neuron 1:", one_sample[0])
    total += one_sample
print("sum of the three, row of neuron 1:", total[0])
print("largest difference from the matrix product:", np.max(np.abs(total - dL_dW)))
```

```text
sample 1, row of neuron 1: [1.  2.  3.  2.5]
sample 2, row of neuron 1: [ 4. 10. -2.  4.]
sample 3, row of neuron 1: [-4.5  8.1  9.9 -2.4]
sum of the three, row of neuron 1: [ 0.5 20.1 10.9  4.1]
largest difference from the matrix product: 0.0
```

Each sample contributes its input row scaled by its upstream value. By hand, the gradient of the weight from input 2 to neuron 1 is $1 \cdot 2.0 + 2 \cdot 5.0 + 3 \cdot 2.7 = 2.0 + 10.0 + 8.1 = 20.1$, the second number of the first row. The figure puts both computations on one page: the product above, with the two batch axes it contracts joined by a bracket, and row 1 of its result rebuilt below from the three scaled input rows.

![Top: the made-up upstream gradient transposed, shape (m, N) = (3, 3), one row per neuron holding 1, 2, 3, times the batch X, shape (N, n) = (3, 4), equals the weight gradient, shape (m, n) = (3, 4), whose rows all read 0.5, 20.1, 10.9, 4.1; a bracket joins the two N axes as the batch axis, contracted and summed. Bottom: 1 times row 1 of X gives 1.0, 2.0, 3.0, 2.5, 2 times row 2 gives 4.0, 10.0, minus 2.0, 4.0, 3 times row 3 gives minus 4.5, 8.1, 9.9, minus 2.4, and their sum is 0.5, 20.1, 10.9, 4.1, row 1 of the product.](diagrams/03-batch-sums-samples.svg)

*The batch axis is the inner, contracted axis of the product, so it is summed away and the gradient keeps the shape of the weights. Only the length of that axis changes with the batch size.*

### 6.3. Sum or mean?

The matrix product delivers the **sum** of the per-sample contributions. Whether the final gradient is a sum or a mean over the batch is decided earlier, by the loss. The loss of post 08 is a mean over the batch, so its derivative carries a factor $1/N$, and that factor arrives inside the upstream gradient: every entry of $\partial L / \partial \mathbf{Z}$ is already divided by $N$. The sum of $N$ contributions that each carry $1/N$ is their mean, so the two lines of this post need no division of their own, and dividing again would leave every gradient $N$ times too small (section 10).

---

## 7. The two-line backward pass

Combining the weight and the bias formulas, the backward pass for the parameters of a dense layer is two lines:

```python
dL_dW = dL_dZ.T @ X              # weight gradient: (m, n)
dL_db = np.sum(dL_dZ, axis=0)    # bias gradient:   (m,)
```

The lines are the same for any layer width and any batch size, and they do not depend on what produced `dL_dZ`. Read as shapes, the first is $(m, N) \cdot (N, n) \rightarrow (m, n)$ and the second reduces $(N, m)$ to $(m,)$. Post 16 wraps the pair, in the layout of section 8, inside `Layer_Dense.backward`, and post 21 (Coding the full backpropagation) runs it as part of a complete backward pass.

The two lines cover the layer's parameters and nothing more. They assume that the upstream gradient already includes the derivative of whatever follows the layer, as the ReLU gate was included in section 3; post 17 gives each activation a backward step of its own, and posts 18 and 19 start the chain at the loss. They also do not produce $\partial L / \partial \mathbf{X}$, the gradient a layer hands to the layer before it, which is the subject of [post 15](../15-gradients-with-respect-to-inputs/index.md).

---

## 8. The convention reconciliation

The series uses two layouts for the weights of a layer, and the two gradient lines look different in each:

| | Row per neuron | Column per neuron |
|---|:---:|:---:|
| Where | posts 01 to 03, and the layer of posts 13 and 14 | `Layer_Dense`, post 04 onward |
| `weights.shape` | $(m, n)$ | $(n, m)$ |
| Forward pass on a batch | `X @ weights.T + biases` | `X @ weights + biases` |
| Weight gradient | `dL_dZ.T @ X`, shape $(m, n)$ | `X.T @ dL_dZ`, shape $(n, m)$ |
| Bias gradient | `np.sum(dL_dZ, axis=0)`, shape $(m,)$ | `np.sum(dL_dZ, axis=0, keepdims=True)`, shape $(1, m)$ |

The two weight gradients are transposes of each other, by the rule $(\mathbf{P}\mathbf{Q})^{\top} = \mathbf{Q}^{\top}\mathbf{P}^{\top}$:

$$\left[\left(\frac{\partial L}{\partial \mathbf{Z}}\right)^{\top} \mathbf{X}\right]^{\top} = \mathbf{X}^{\top} \, \frac{\partial L}{\partial \mathbf{Z}}.$$

The right-hand side is the form the notation guide gives for the whole series. Its shapes are $(n, N) \cdot (N, m) \rightarrow (n, m)$: the batch axis is again the inner, contracted one, so everything section 6 said about the sum over the batch holds unchanged. The entry for input $j$ and neuron $k$ is still $\sum_i x_{ij} \, \partial L / \partial z_{ik}$; it has only moved from row $k$, column $j$ to row $j$, column $k$.

The row-per-neuron layout keeps each neuron's numbers in one row, which suits a hand derivation; the column-per-neuron layout removes the transpose from the forward pass (post 04, section 4). The bias line differs for a separate reason: `Layer_Dense` stores its biases as a $(1, m)$ row, and `keepdims=True` (post 05) keeps the summed axis so that the gradient comes out as a $(1, m)$ row as well.

`snippets/conventions.py` holds the same layer in both layouts and runs the batch of section 6.1 through each:

```python
# Weight gradient in each layout.
dW_rows = dL_dZ.T @ X                             # (3, 4), the shape of W_rows
dW_cols = X.T @ dL_dZ                             # (4, 3), the shape of W_cols
print("dW_rows", dW_rows.shape, "matches W_rows", W_rows.shape)
print("dW_cols", dW_cols.shape, "matches W_cols", W_cols.shape)
print(dW_cols)
print("dW_cols is the transpose of dW_rows:", np.array_equal(dW_cols, dW_rows.T))
```

```text
dW_rows (3, 4) matches W_rows (3, 4)
dW_cols (4, 3) matches W_cols (4, 3)
[[ 0.5  0.5  0.5]
 [20.1 20.1 20.1]
 [10.9 10.9 10.9]
 [ 4.1  4.1  4.1]]
dW_cols is the transpose of dW_rows: True
```

The numbers of section 6.1 reappear with rows and columns exchanged. The script also confirms that the two forward passes agree and that one update step in each layout leaves the two copies of the layer identical. Either layout is correct; mixing them inside one program is not (section 10).

---

## 9. Make it run: the two lines against a numerical gradient

Sections 4 and 6 checked the matrix form against the per-weight chain rule. Both sides of that comparison rest on the same derivation, so a mistake in the derivation would pass unnoticed. `snippets/gradient_check.py` makes an independent check: it measures each gradient with the central difference of post 10, which needs only the forward pass. It needs NumPy only and runs in under a second from the series root with `python posts/14-matrices-in-backpropagation/snippets/gradient_check.py`.

The script puts the batch of section 6.1 through the layer of post 13. Each sample's three ReLU outputs are summed to $\hat{y}_i$ (the column `Y` in the code), and the loss is the mean of $\hat{y}_i^2$ over the batch. By section 6.3 the upstream gradient therefore carries the factor $1/N$: $\partial L / \partial z_{ik} = (2 \hat{y}_i / N)$ times the ReLU gate.

```python
# Forward pass and the upstream gradient, one row per sample.
Z = X @ weights.T + biases
A = np.maximum(0, Z)
Y = np.sum(A, axis=1, keepdims=True)         # shape (N, 1)
dL_dZ = (2 * Y / N) * np.where(Z > 0, 1.0, 0.0)   # shape (N, 3)

# The two-line backward pass.
dL_dW = dL_dZ.T @ X              # weight gradient: (m, n)
dL_db = np.sum(dL_dZ, axis=0)    # bias gradient:   (m,)
```

The script then measures each of the fifteen parameters in turn: it moves that one parameter by $+h$ and by $-h$, evaluates the loss twice, and divides the difference by $2h$. The arrays are float64 and the step is $h = 10^{-5}$, the choice post 10 arrived at for a central difference.

```text
dL_dW, matrix form:
[[24.18  89.796 43.884 46.016]
 [24.18  89.796 43.884 46.016]
 [24.18  89.796 43.884 46.016]]
dL_dW, central differences:
[[24.18  89.796 43.884 46.016]
 [24.18  89.796 43.884 46.016]
 [24.18  89.796 43.884 46.016]]
dL_db, matrix form:         [27.68 27.68 27.68]
dL_db, central differences: [27.68 27.68 27.68]
largest relative error, weights: 1.1e-10
largest relative error, biases:  8.3e-11
```

The three rows of the upstream gradient are 12, 10.2, and 5.48, so the first weight gradient is $12 \cdot 1.0 + 10.2 \cdot 2.0 + 5.48 \cdot (-1.5) = 24.18$ and each bias gradient is $12 + 10.2 + 5.48 = 27.68$. The numerical gradient, which uses no chain rule at all, agrees with all fifteen analytic values to a relative error of about $10^{-10}$, far below the $10^{-7}$ that the series treats as the mark of a correct backward pass. The other four scripts in the directory run the same way: `single_sample.py` (sections 2.1 to 5), `batch.py` (section 6), `conventions.py` (section 8), and `what_can_go_wrong.py` (section 10).

---

## 10. What can go wrong?

`snippets/what_can_go_wrong.py` runs six mistakes on the single sample of section 4 and the batch of section 6.1, and prints what NumPy does with each. Several of them raise no error.

```text
== 1. The transpose is left out
one sample, (1, 3) @ (1, 4): ValueError
batch with N = m = 3, (3, 3) @ (3, 4): no error, shape (3, 4)
correct row of neuron 1: [ 0.5 20.1 10.9  4.1]
wrong   row 1:           [1.5 9.7 5.3 3.7]

== 2. The upstream gradient is one-dimensional
flat.T has shape (3,) so the transpose does nothing
one sample, (3,) @ (1, 4): ValueError
batch with N = 3, (3,) @ (3, 4): no error, shape (4,)

== 3. np.outer on a batch
one sample: np.outer gives shape (3, 4)
batch:      np.outer gives shape (9, 12) and the product gives (3, 4)

== 4. The axis is left out of the bias sum
np.sum(dL_dZ, axis=0): [6. 6. 6.]
np.sum(dL_dZ):         18.0
np.sum(dL_dZ, axis=1): [3. 6. 9.] (one number per sample, not per neuron)

== 5. The batch average is applied twice
row of neuron 1, mean loss:       [0.16666667 6.7        3.63333333 1.36666667]
row of neuron 1, divided again:   [0.05555556 2.23333333 1.21111111 0.45555556]

== 6. The two weight layouts are mixed
(3, 4) weights minus a (4, 3) gradient: ValueError
square layer: both gradients have shape (3, 3) (3, 3)
equal: False  transposes of each other: True
```

- **The transpose is left out.** `dL_dZ @ X` multiplies $(N, m)$ by $(N, n)$, whose inner sizes are $m$ and $N$. When they differ NumPy raises a `ValueError`. When the batch size happens to equal the number of neurons, as here with $N = m = 3$, the call returns an array of the right shape, $(3, 4)$, filled with the wrong numbers: its first row is 1.5, 9.7, 5.3, 3.7 where the gradient is 0.5, 20.1, 10.9, 4.1. A shape check cannot catch this; the numerical check of section 9 can, and so can a test layer in which the batch size, the number of inputs, and the number of neurons are three different numbers.
- **The upstream gradient is one-dimensional.** The transpose of a 1-D array is the same array, so `.T` does nothing to an upstream gradient of shape $(m,)$. Against a three-sample batch a length-3 vector multiplies silently and returns a vector of shape $(n,)$ where a matrix was wanted. Every quantity therefore keeps one row per sample, even when there is a single sample.
- **`np.outer` is used on a batch.** `np.outer` flattens both arguments before it multiplies. For one sample that is harmless: a $(1, 3)$ row and a $(1, 4)$ row give the correct $(3, 4)$ matrix. For the batch, the $(3, 3)$ and $(3, 4)$ arrays flatten to lengths 9 and 12 and give a $(9, 12)$ array of every pairwise product, in which nothing has been summed.
- **The axis is left out of the bias sum.** `np.sum(dL_dZ)` with no axis returns the single number 18.0, and a scalar broadcasts against the biases without complaint, so every bias moves by the same wrong amount. `axis=1` is the opposite slip: it returns one number per sample (3, 6, 9), which has the shape of the biases only when $N = m$.
- **The batch average is applied twice.** When the loss is a mean over the batch the factor $1/N$ is already inside the upstream gradient (section 6.3). Dividing the weight gradient by $N$ again leaves it $N$ times too small: 0.0556 in place of 0.1667 for the first weight here.
- **The two weight layouts are mixed.** Subtracting a $(4, 3)$ gradient from $(3, 4)$ weights raises a `ValueError`, but only because the layer is not square. For a layer with as many neurons as inputs both layouts give a $(3, 3)$ gradient, the two are transposes and not equal, and the update runs silently with each weight receiving the gradient of another.

---

## 11. Summary

| Concept | Takeaway |
|---|---|
| Outer product structure | $\partial L / \partial w_{kj} = (\partial L / \partial z_k) \cdot x_j$ is the $(k, j)$ entry of a column times a row |
| Matrix form | $\partial L / \partial \mathbf{W} = (\partial L / \partial \mathbf{Z})^{\top} \mathbf{X}$, shapes $(m, N) \cdot (N, n) \rightarrow (m, n)$ |
| Bias gradient | $\partial L / \partial b_k = \sum_i \partial L / \partial z_{ik}$, the sum of the upstream gradient over the batch axis |
| Shape invariance | Each gradient has the shape of the array it updates, whatever the batch size |
| Batches | The product contracts the batch axis, which sums the per-sample outer products; the sum is exact and needs no loop |
| Sum or mean | The layer sums; a $1/N$ comes from the loss, inside the upstream gradient |
| Two-line backward | `dL_dW = dL_dZ.T @ X` and `dL_db = np.sum(dL_dZ, axis=0)` |
| Other layout | With weights of shape $(n, m)$: `X.T @ dL_dZ` and `np.sum(dL_dZ, axis=0, keepdims=True)` |

---

## Common pitfalls

Section 10 runs each of these and shows what NumPy returns.

1. **Forgetting the transpose.** `dL_dZ @ X` is silent when the batch size equals the number of neurons.
2. **Letting an array become one-dimensional.** `.T` does nothing to a shape `(m,)` array; every quantity keeps its batch axis.
3. **Using `np.outer` on a batch.** It flattens its arguments and sums nothing, so it is right for a single sample only.
4. **Forgetting `axis=0` in the bias sum.** The result is a scalar that broadcasts across the biases.
5. **Dividing by the batch size a second time.** The $1/N$ of a mean loss is already in the upstream gradient.
6. **Mixing the two weight layouts.** `dL_dZ.T @ X` belongs to `(m, n)` weights and `X.T @ dL_dZ` to `(n, m)` weights; a square layer hides the mix-up.

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.5 (Back-Propagation and Other Differentiation Algorithms) (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (2020).
- Magnus, J. R. and Neudecker, H., *Matrix Differential Calculus with Applications in Statistics and Econometrics* (Wiley, 3rd edition, 2019).
- Petersen, K. B. and Pedersen, M. S., *The Matrix Cookbook* (2012).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 15 - Gradients with respect to inputs](../15-gradients-with-respect-to-inputs/index.md):** the third matrix product of a dense layer, the one that hands the gradient to the layer before.
- **[Post 16 - Coding backpropagation](../16-coding-backpropagation/index.md):** where the two lines of this post, in the column-per-neuron layout, become part of `Layer_Dense.backward`.
