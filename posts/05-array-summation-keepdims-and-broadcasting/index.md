# 05 - Array summation, keepdims, and broadcasting

> **TL;DR.** A NumPy reduction such as `np.sum(a, axis=k)` removes axis `k` from the shape, and `keepdims=True` keeps it as an axis of size 1, so a per-row result of shape `(N, 1)` stays lined up with the rows it came from. Broadcasting compares two shapes from the last axis backwards, pads the shorter one with 1s on the left, and stretches every size-1 axis, which is why a `(1, 3)` bias row adds to all 300 rows of a `(300, 3)` output and why a 1-D `(n,)` array always acts as a row against a 2-D array, never as a column. A per-row reduction without `keepdims` therefore lines up with the columns: NumPy raises an error when the shapes do not fit, and returns wrong numbers without one when they do.
>
> **Prerequisites:** [Post 02](../02-numpy-and-the-dot-product/index.md).
> **Safe to skip?** Skip it if the reader can already give the shape of `np.sum(a, axis=1, keepdims=True)` for a `(3, 4)` array, decide without running code whether `(5, 3)` and `(5,)` broadcast, and say why `a - np.max(a, axis=1)` is wrong for a square `a`.
>
> **After reading, you will be able to:**
>
> - Predict the shape of np.sum(a, axis=k, keepdims=...) for any 2-D array a.
> - Decide from two shapes alone whether they broadcast, and give the shape of the result.
> - Spot the silent bug where a 1-D (n,) array broadcasts as a row when a column was intended.
> - Trace how a (1, n) bias row is broadcast across the N rows of a layer's output in a forward pass.

![The 3 by 3 array a holding 1 to 9, with axis 0 running down its rows and axis 1 across its columns. With keepdims=True the column sums 12, 15, 18 sit under the columns as a row of shape (1, 3), and the row sums 6, 15, 24 sit beside the rows as a column of shape (3, 1). Without keepdims, np.sum(a, axis=0) gives 12, 15, 18 and np.sum(a, axis=1) gives 6, 15, 24, both drawn as flat, low strips with the same shape (3,), and np.sum(a) gives the scalar 45 with shape (), drawn as a bare number.](diagrams/01-axis-summation.svg)

*The axis named in the call is the one that disappears. With `keepdims=True` it stays at size 1, so the row sums remain a column beside the rows and the column sums a row under the columns.*

---

## 1. The question: what do `axis` and `keepdims` do, and which shapes combine?

Most numerical bugs make themselves known: a shape mismatch raises a `ValueError`, and a division by zero prints a `RuntimeWarning`. The two features in this post can fail more quietly. `np.sum(a, axis=1)` returns a sensible-looking 1-D result, which then lines up with the wrong axis when it is used in a later subtraction. If the shapes happen to fit, the code runs, the numbers are wrong, and nothing points at the line that caused it.

The question therefore has two halves. First, what do `axis` and `keepdims` do to the shape of a *reduction*, an operation such as `np.sum`, `np.max`, or `np.mean` that collapses many numbers into fewer? Second, which pairs of shapes will NumPy combine *element-wise*, entry by entry, and what does it do when the two shapes differ? The mechanism behind the second half is called *broadcasting*.

Softmax (post 06), categorical cross-entropy (post 08), the bias gradient (post 16), and almost every loss function the series meets require a correct reduction along a chosen axis, and most of them then combine the result with the array it came from. Settling these mechanics now is cheaper than debugging them inside a forward pass later.

Neither mechanism is new, and neither is specific to NumPy. NumPy emerged in 2005 as the unification of two earlier Python array packages, Numeric, from the mid-1990s, and Numarray, and the design of this line of tools was informed by older languages such as APL, whose notation Iverson published in 1962. The libraries that came after kept its conventions: the array types of PyTorch, TensorFlow, and JAX often mimic the NumPy interface (Harris et al., 2020). The rules in section 4 are therefore not a NumPy quirk; the same shape reasoning carries over to each of those frameworks.

---

## 2. Array summation with `axis`

Take a 3-by-3 array:

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
```

`np.sum` reduces the array along one or more axes. Three behaviours are worth distinguishing. Every code block in this section and the next is an excerpt of `snippets/axis_keepdims.py`, and section 7 gives the command that runs it.

### 2.1. No `axis`: flatten everything

```python
print(np.sum(a))             # 45
print(np.sum(a, axis=None))  # 45 (same thing)
```

Every element is summed. The result is a scalar, a single number whose shape is the empty tuple `()`.

### 2.2. `axis=0`: collapse rows, sum each column

```python
print(np.sum(a, axis=0))     # [12 15 18]
```

The result has shape `(3,)`. Each entry is one column's sum: `1+4+7=12`, `2+5+8=15`, `3+6+9=18`. The row axis is gone.

### 2.3. `axis=1`: collapse columns, sum each row

```python
print(np.sum(a, axis=1))     # [ 6 15 24]
```

Again shape `(3,)`. Each entry is one row's sum: `1+2+3=6`, `4+5+6=15`, `7+8+9=24`. The column axis is gone.

### 2.4. The mnemonic that survives every dimension

> The axis named in the call is the axis that disappears.

For a 2-D array, `axis=0` collapses the row dimension and leaves the column dimension; `axis=1` does the opposite. A common misreading takes `axis=0` to mean "the first thing in each row". It is the reverse: axis 0 is the axis whose index changes on the way down the rows, and it is collapsed by the reduction, not iterated, so what remains is one number per column.

For a 3-D batch of shape `(N, H, W)`, `axis=0` reduces across the batch, leaving a result of shape `(H, W)`. The rule does not change with the number of dimensions, as a quick check confirms:

```python
b = np.ones((4, 2, 3))               # shape (4, 2, 3)
print(np.sum(b, axis=0).shape)       # (2, 3): axis 0 is gone
print(np.sum(b, axis=1).shape)       # (4, 3): axis 1 is gone
print(np.sum(b, axis=2).shape)       # (4, 2): axis 2 is gone
print(np.sum(b, axis=-1).shape)      # (4, 2): axis -1 is the last axis
```

Axes can also be counted from the end, as list indices can: `axis=-1` is the last axis, which is axis 2 for `b` and axis 1 for any 2-D array.

### 2.5. What is *not* obvious about the result

Both `np.sum(a, axis=0)` and `np.sum(a, axis=1)` return a 1-D array of shape `(3,)`. Nothing in that shape records which axis was reduced, and that 1-D shape is the source of the silent bug that section 3 works through. The right half of the figure at the top of the post draws the two results flat, side by side, with the same shape; its left half shows the row and the column that `keepdims=True` (section 3) returns instead.

A 1-D `(n,)` array is not the same thing as a row vector of shape `(1, n)` or a column vector of shape `(n, 1)`: the first has one axis and the other two have two. Section 4 shows that broadcasting against a 2-D array treats the first exactly like the second and never like the third. At the `print` site the only trace of the difference is the bracketing: `[12 15 18]` for the 1-D array, `[[12 15 18]]` for the row vector.

---

## 3. `keepdims=True`: preserving the reduced axis

Adding `keepdims=True` keeps the reduced axis around as a length-1 dimension, instead of dropping it.

```python
print(np.sum(a, axis=0, keepdims=True))     # one row:    shape (1, 3)
print(np.sum(a, axis=1, keepdims=True))     # one column: shape (3, 1)
```

```text
[[12 15 18]]
[[ 6]
 [15]
 [24]]
```

Summary of the four combinations:

| Call | Result | Shape | Geometric interpretation |
|---|---|:---:|---|
| `np.sum(a, axis=0)` | `[12, 15, 18]` | `(3,)` | 1-D array |
| `np.sum(a, axis=0, keepdims=True)` | `[[12, 15, 18]]` | `(1, 3)` | row vector |
| `np.sum(a, axis=1)` | `[6, 15, 24]` | `(3,)` | 1-D array |
| `np.sum(a, axis=1, keepdims=True)` | `[[6], [15], [24]]` | `(3, 1)` | column vector |

The first and third have the same shape, `(3,)`, and the same printed form, a single pair of brackets. Yet they summarise opposite axes, and broadcasting will line both up with the columns of `a`. That is right for the column sums and wrong for the row sums, which is what makes the two so easy to confuse.

### 3.1. The silent-bug example: per-row max subtraction

A pattern from softmax: subtract the largest value in each row from every entry of that row. The expected result for the `a` above is three rows that each end with zero:

```text
[[-2 -1  0]
 [-2 -1  0]
 [-2 -1  0]]
```

The figure below runs the subtraction both ways and draws `max_vals` as broadcasting stretches it, which is where the two versions part.

![Two rows, each the 3 by 3 array a minus max_vals as broadcast equals the result. Without keepdims, max_vals is 3, 6, 9 with shape (3,), copied down the rows, so column j loses the maximum of row j; the result rows are -2 -4 -6, then 1 -1 -3, then 4 2 0, and no error is raised. With keepdims=True, max_vals is a column of shape (3, 1), copied across the columns, and every row of the result is -2, -1, 0.](diagrams/02-max-subtraction.svg)

*Same call style on both sides; the only difference is `keepdims=True`. One result is correct and the other is silently wrong.*

**Without `keepdims` (silently wrong):**

```python
max_vals = np.max(a, axis=1)        # [3 6 9], shape (3,)
print(a - max_vals)
```

```text
[[-2 -4 -6]
 [ 1 -1 -3]
 [ 4  2  0]]
```

NumPy treats `[3, 6, 9]` as a 1-D array, which broadcasts as a row vector against the 2-D `a`. So `3` is subtracted from column 0 of every row, `6` from column 1, `9` from column 2. The arithmetic runs without error. The numbers are wrong.

**With `keepdims=True` (correct):**

```python
max_vals = np.max(a, axis=1, keepdims=True)  # a column, shape (3, 1)
print(a - max_vals)
```

```text
[[-2 -1  0]
 [-2 -1  0]
 [-2 -1  0]]
```

Now `max_vals` is a column vector. Broadcasting extends it across the three columns, subtracting `3` from row 0, `6` from row 1, `9` from row 2. The result matches the expectation.

### 3.2. When the mistake is silent and when it is loud

The 3-by-3 example is silent for a reason: `a` is square. The three row maxima happen to fit along the three columns, so NumPy has no way to tell that a column was meant. Run the same line on an array with 4 rows and 3 columns and the four maxima cannot line up with three columns:

```python
wide = np.arange(12).reshape(4, 3)           # 4 rows, 3 columns
try:
    wide - np.max(wide, axis=1)              # shapes (4, 3) and (4,)
except ValueError as err:
    print("ValueError:", str(err).strip())
print(wide - np.max(wide, axis=1, keepdims=True))
```

```text
ValueError: operands could not be broadcast together with shapes (4,3) (4,)
[[-2 -1  0]
 [-2 -1  0]
 [-2 -1  0]
 [-2 -1  0]]
```

The error is the lucky outcome. The layer output of post 04 has 300 rows and 3 columns, so a forgotten `keepdims` in a per-row reduction on it would stop the program at once. The silent case arrives when the number of rows equals the number of columns: a quick test on 3 samples of a 3-class problem, a 10-by-10 table of counts, any square intermediate. Code that has only ever been tried on a small square example can therefore look correct with the bug in place.

`snippets/axis_keepdims.py` makes the boundary exact by running the line without `keepdims` on every shape from `(1, 1)` to `(4, 4)` and comparing it with the correct result:

```text
        1 col   2 cols  3 cols  4 cols
1 row   right   right   right   right
2 rows  wrong   wrong   error   error
3 rows  wrong   error   wrong   error
4 rows  wrong   error   error   wrong
```

NumPy raises exactly when the number of rows differs from the number of columns and neither of them is 1. The line raises nothing on a square array with more than one row and its numbers are wrong there unless every row happens to have the same maximum; the same holds for a single column, where an `(N, 1)` array minus an `(N,)` array comes back with shape `(N, N)` (section 8). It is right only for a single row, which has one maximum and nothing to misalign.

A second per-row pattern, dividing every row by its own sum, shows what "silently wrong" means for the numbers. Each row of the correct result sums to 1; the row sums of the wrong one do not:

```python
wrong = a / np.sum(a, axis=1)                    # divides column j by the sum of row j
right = a / np.sum(a, axis=1, keepdims=True)     # divides row i by the sum of row i
print(np.round(np.sum(wrong, axis=1), 3))        # [0.425 1.25  2.075]
print(np.round(np.sum(right, axis=1), 3))        # [1. 1. 1.]
```

That check, a property the result must have, is how the silent case is caught in practice.

### 3.3. The rule of thumb

> When reducing along an axis and then operating against the original 2-D array, always pass `keepdims=True`.

The rule costs nothing when it is not needed and prevents the wrong-axis broadcast when it is. Adopted everywhere, it removes this class of failure, silent or loud.

---

## 4. The broadcasting rules

Broadcasting is the mechanism that lets NumPy combine arrays of different shapes element-wise. Without it, `a + b` would only work when `a.shape == b.shape`. With it, a `(300, 3)` matrix can have a `(3,)` bias added to every row by writing `a + b`, no `np.tile` or explicit replication needed.

The rules NumPy applies are, in order:

1. **Align trailing dimensions.** If the two arrays have different numbers of dimensions, the shorter shape is left-padded with 1s until both have the same length. So `(3,)` paired with a 2-D operand is treated as `(1, 3)`.
2. **Compatibility check.** For each pair of aligned dimensions, the two sizes must be equal, or one of them must be 1. If neither holds, NumPy raises a `ValueError`.
3. **Stretch the 1s.** Any dimension of size 1 in either operand is conceptually replicated to match the other operand's size in that dimension, so the result takes the larger of the two sizes along every axis.
4. **Operate element-wise.** The (now equal-shape) operands are combined.

The figure below applies rules 1 to 3 to four pairs of shapes, writing each pair as a small table with one column per aligned axis.

![Four small tables, each with one column per aligned axis and rows for the left shape, the right shape and the result. (300, 3) plus (1, 3): the 1 stretches to 300 and the result is (300, 3). (3, 3) plus (3,): the 1-D shape is padded to (1, 3), its 1 stretches, and the result is (3, 3). (3, 1) plus (3,): the 1-D shape is padded to (1, 3), both operands stretch, and the result is (3, 3). (5, 3) plus (5,): the 1-D shape is padded to (1, 5), 3 meets 5 on axis 1 with neither equal to 1, and the result cell reads ValueError. A legend marks a stretched 1, a padded 1, the clash and the result.](diagrams/03-shape-alignment.svg)

*Against a 2-D array, a 1-D shape `(n,)` is always padded to `(1, n)`. It can never be interpreted as a column on its own.*

A few worked compatibility checks, each one confirmed by the script of section 7:

| Left shape | Right shape | Result | Reason |
|---|---|---|---|
| `(3, 3)` | `(3, 1)` | `(3, 3)` | the single column stretches across 3 columns |
| `(3, 3)` | `(1, 3)` | `(3, 3)` | the single row stretches down 3 rows |
| `(3, 3)` | `(3,)` | `(3, 3)` | `(3,)` padded to `(1, 3)`, then the row stretches |
| `(300, 3)` | `(3,)` | `(300, 3)` | same as above, with 300 rows |
| `(3, 1)` | `(3,)` | `(3, 3)` | `(3,)` padded to `(1, 3)`; the column and the row both stretch |
| `(3, 3)` | `(2, 3)` | error | row sizes are 3 and 2; neither is 1 |
| `(3, 3)` | `(3, 2)` | error | column sizes are 3 and 2; neither is 1 |
| `(5, 3)` | `(5,)` | error | `(5,)` padded to `(1, 5)`; column sizes are 3 and 5 |

For the first failing pair NumPy's message is `ValueError: operands could not be broadcast together with shapes (3,3) (2,3)`. The fifth row deserves a second look: neither operand has the result's shape, both are stretched, and no error is raised. Section 8 returns to it.

### 4.1. What broadcasting is *not*

A boundary section, because the distinction matters in code.

- **Broadcasting is not matrix multiplication.** `a + b` and `a * b` are element-wise after broadcasting; they are not `np.dot(a, b)`. For the 3-by-3 `a` of section 2, the first row of `a * a` is `[1, 4, 9]` and the first row of `np.dot(a, a)` is `[30, 36, 42]`. Mixing the two up is an easy mistake in a hand-written layer.
- **Broadcasting is not a reshape.** It does not copy data; it sets up a virtual replication that the underlying loop uses without ever materialising a stretched copy of the smaller operand. This is fast, but it means the result's shape is determined by the rules, not by what was intended.
- **Broadcasting does not depend on operand order.** `a + b` and `b + a` always give the same numbers, and the broadcast shape is determined symmetrically from both operands. The order only matters for in-place operations like `a += b`, where the left operand's shape must already be the shape of the result.

---

## 5. Broadcasting in a neural-network forward pass

The bias addition from post 04 is the textbook example. The dot product produces a `(300, 3)` matrix of layer outputs. The bias is stored as a `(1, 3)` row vector. Adding them looks like a shape mismatch and works anyway:

```python
out = np.dot(X, W) + b          # X: (300, 2), W: (2, 3), b: (1, 3)
# np.dot(X, W) has shape (300, 3)
# b has shape (1, 3)
# (300, 3) + (1, 3) broadcasts to (300, 3)
print(np.dot(X, W).shape, b.shape, out.shape)
```

In `snippets/broadcasting.py`, `X` and `W` are seeded random stand-ins for the spiral batch and the weights of post 04's `Layer_Dense(2, 3)`, and `b` is set to `[[0.1, 0.2, 0.3]]` instead of zeros so that the addition changes something. The line prints `(300, 3) (1, 3) (300, 3)`.

The bias of shape `(1, 3)` is implicitly replicated 300 times down the row axis. Every sample sees the same three bias values added to its three neuron outputs. No `for` loop. No `np.tile`. One operator. The script checks both halves of that statement: adding an explicit `np.tile(b, (300, 1))` copy gives an identical array, and `np.broadcast_to(b, (300, 3))`, which exposes the stretched operand, shares its memory with the three numbers of `b`, so the 300 rows are never written out.

The same pattern returns throughout the series, in three variants. Softmax in post 06 subtracts a per-row max and then divides by a per-row sum; both reductions use `axis=1, keepdims=True`, so each result is an `(N, 1)` column that broadcasts back against the original 2-D array. The bias gradient in post 16 is `np.sum(dvalues, axis=0, keepdims=True)`, a reduction over the batch axis kept as a `(1, n_neurons)` row so that it matches the stored bias. Categorical cross-entropy in post 08 reduces along `axis=1` and keeps the 1-D result on purpose, one loss per sample, because that result is averaged and never broadcast back. Beyond this series, batch normalisation (`cnn-014`, Convolutional Neural Networks from Scratch) subtracts a per-channel mean and divides by a per-channel standard deviation, which takes a reduction over every axis except the channel axis and a reshape before the result broadcasts back. Internalising the pattern now pays back across the rest of the series.

---

## 6. The shape diary for reductions and broadcasting

For a 2-D array `a` with $N$ rows (samples) and $n_\text{features}$ columns:

| Operation | `a` shape | Axis removed | Output shape | Notes |
|---|---|---|---|---|
| `np.sum(a)` | $(N, n_\text{features})$ | both | scalar, `()` | one number for the whole array; rarely what a per-sample or per-feature step wants |
| `np.sum(a, axis=0)` | $(N, n_\text{features})$ | axis 0, the $N$ rows | $(n_\text{features},)$ | per-feature sum across the batch |
| `np.sum(a, axis=1)` | $(N, n_\text{features})$ | axis 1, the columns | $(N,)$ | per-sample sum across features |
| `np.sum(a, axis=0, keepdims=True)` | $(N, n_\text{features})$ | axis 0, kept at size 1 | $(1, n_\text{features})$ | broadcasts cleanly back against `a` |
| `np.sum(a, axis=1, keepdims=True)` | $(N, n_\text{features})$ | axis 1, kept at size 1 | $(N, 1)$ | broadcasts cleanly back against `a` |

`snippets/broadcasting.py` prints this diary for a `(3, 4)` array, so $N = 3$ and $n_\text{features} = 4$: the shapes are `()`, `(4,)`, `(3,)`, `(1, 4)`, and `(3, 1)`, plus `(1, 1)` for `np.sum(c, keepdims=True)`, which keeps both axes at size 1. It also subtracts each result from the array. Every subtraction succeeds except the one with `np.sum(c, axis=1)`, whose 3 entries cannot line up with 4 columns.

A 1-D array of shape $(n_\text{features},)$ and a row vector of shape $(1, n_\text{features})$ print with one pair of brackets and with two. They broadcast identically against a 2-D $(N, n_\text{features})$ array, which is why the per-feature sum in the second row of the diary lines up with the columns of `a` even without `keepdims`. The trouble starts only when an operation requires the result to be a column, in which case the 1-D form does the wrong thing, silently whenever the shapes allow it.

---

## 7. Make it run: the four rules as one short function

Three scripts under `snippets/` produce every array, shape, and error message quoted in this post. Each needs only NumPy, runs from the series root in a second or two, and prints the same output on every run:

- `python posts/05-array-summation-keepdims-and-broadcasting/snippets/axis_keepdims.py` runs sections 2 and 3: the reductions of the 3-by-3 array, `keepdims`, the per-row max subtraction with and without it, and the non-square case that raises.
- `python posts/05-array-summation-keepdims-and-broadcasting/snippets/broadcasting.py` runs sections 4 to 6: the compatibility table, the bias addition, and the shape diary.
- `python posts/05-array-summation-keepdims-and-broadcasting/snippets/broadcast_traps.py` runs every failure of section 8.

The second script also states the rules of section 4 as code. `broadcast_shape` takes two shapes and applies rules 1 to 3 with nothing but tuples and a loop; rule 4, the arithmetic itself, is left to NumPy:

```python
def broadcast_shape(left, right):
    """Return the shape NumPy gives two operands of these shapes, or raise ValueError."""
    # Rule 1: left-pad the shorter shape with 1s until both have the same length.
    ndim = max(len(left), len(right))
    left = (1,) * (ndim - len(left)) + tuple(left)
    right = (1,) * (ndim - len(right)) + tuple(right)
    result = []
    for m, n in zip(left, right):
        # Rule 2: two aligned sizes must be equal, or one of them must be 1.
        if m != n and m != 1 and n != 1:
            raise ValueError(f"sizes {m} and {n} differ and neither is 1")
        # Rule 3: a size of 1 stretches to the other size.
        result.append(n if m == 1 else m)
    return tuple(result)
```

The script runs it on the eight pairs of section 4 next to NumPy itself, which is asked to add two arrays of zeros with those shapes:

```text
(3, 3)   with (3, 1) -> by hand (3, 3)   NumPy (3, 3)
(3, 3)   with (1, 3) -> by hand (3, 3)   NumPy (3, 3)
(3, 3)   with (3,)   -> by hand (3, 3)   NumPy (3, 3)
(300, 3) with (3,)   -> by hand (300, 3) NumPy (300, 3)
(3, 1)   with (3,)   -> by hand (3, 3)   NumPy (3, 3)
(3, 3)   with (2, 3) -> by hand error    NumPy error
(3, 3)   with (3, 2) -> by hand error    NumPy error
(5, 3)   with (5,)   -> by hand error    NumPy error
```

Eight pairs are an illustration, not a test, so the script then tries every shape with 0 to 3 axes whose sizes are 1, 2, or 3. There are 40 such shapes and 1,600 ordered pairs of them; the hand-written rules and NumPy agree on all 1,600, of which NumPy refuses 660. The four rules of section 4, which are also the ones NumPy's documentation states, are the whole mechanism.

---

## 8. What can go wrong?

Every item below is run by `snippets/broadcast_traps.py`, and the outputs quoted are the ones it prints.

- **A column meets a 1-D array, and both stretch.** A single-output layer returns predictions of shape `(4, 1)`; the labels arrive as `(4,)`. Their difference is not `(4, 1)`. The 1-D array is padded to `(1, 4)`, the column stretches across and the row stretches down, and the result has shape `(4, 4)`: every prediction minus every label. For predictions `0.9, 0.2, 0.8, 0.4` and labels `1, 0, 1, 0`, the mean squared difference over those 16 entries is 0.3375; with the labels reshaped to `(4, 1)` it is 0.0625, the mean over the 4 pairs that belong together. No error is raised in either case. Post 34 meets exactly this pair of shapes and reshapes its labels to `(N, 1)` so that the two match.
- **Broadcasting cannot be switched off.** NumPy has no setting that disables it. `np.add(a, row, out=out)` with a `(3,)` `row` and a `(3, 3)` `out` still broadcasts the row; the `out` argument only has to be able to hold the result. The guard that works is an explicit assertion on the shape that was meant, such as `assert max_vals.shape == (3, 1)`, which fails with a readable message the moment a 1-D `(3,)` array turns up.
- **The wrong result looks plausible.** The silent failure is easy to miss because the wrong result still has a plausible shape and plausible numbers; the per-row max subtraction of section 3.1 returns nine small integers either way. Nothing in the output points at the reduction, so the first sign tends to be indirect, such as rows of probabilities that no longer sum to 1. Checking `.shape` on the reduced array and checking one property the result must have (section 3.2) finds it directly.
- **In-place operators need room on the left.** `a + row` and `row + a` both have shape `(3, 3)` and equal values, but `row += a` raises `ValueError: non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (3,3)`, because the result has to be stored in `row`. The same rule decides how biases are stored: a bias of shape `(3,)` cannot take the `(1, 3)` gradient of post 16 in place (`doesn't match the broadcast shape (1,3)`), while the `(1, 3)` bias that `Layer_Dense` stores takes it without complaint.
- **`np.mean`, `np.max`, and `np.std` share the trap.** They take the same `axis` and `keepdims` arguments and follow the same rules: each returns shape `(3, 1)` for the 3-by-3 array with `axis=1, keepdims=True` and `(3,)` without. The bug of section 3.1 happens whenever any of them is fed back into a broadcasting operation.
- **Avoiding `keepdims=True` to save time or memory.** There is nothing to save. The output is the same numbers with one extra dimension of length 1: both forms of the row sum hold 3 elements, and no extra data is allocated.
- **Using `keepdims=True` where the 1-D form is the right one.** The 1-D form should be preferred when the result is the final output of a script or a reduction that will not be broadcast again, such as the per-sample losses of post 08. A kept axis carries its own risk there: an `(N, 1)` result that later meets an `(N,)` array is the first item of this list. For intermediates inside a layer, `keepdims=True` is the safer default.
- **Naming an axis the array does not have.** `np.sum(a, axis=2)` on a 2-D array raises `AxisError: axis 2 is out of bounds for array of dimension 2`. This one is loud, and the message says exactly what is wrong.

---

## 9. Summary

| Concept | Takeaway |
|---|---|
| `axis` | The axis named is the axis that disappears |
| `keepdims=True` | Keeps the reduced axis as a length-1 dimension, preserving the number of dimensions |
| 1-D vs row vector | An `(n,)` shape always broadcasts as a row against a 2-D array, never as a column |
| Broadcasting rules | Align trailing dims, stretch any size-1 dim, fail otherwise |
| Silent or loud | A per-row reduction without `keepdims` raises when rows and columns differ in number and neither is 1, and gives wrong numbers on a square array |
| Forward-pass example | `(300, 3) + (1, 3)` works because the row stretches across the batch |

---

## Common pitfalls

1. **Subtracting `np.max(a, axis=1)` from `a` without `keepdims=True`.** On a square array the result is wrong but raises no error; on a batch whose numbers of rows and columns differ and both exceed 1 it raises a `ValueError`. Always pass `keepdims=True` when the reduction will be broadcast back against the original.
2. **Expecting `(n,)` to broadcast as a column.** It will not. To get column behaviour, use `(n, 1)` explicitly, either via `keepdims=True` or `.reshape(-1, 1)`.
3. **Using `*` and expecting matrix multiplication.** `*` is element-wise (with broadcasting). For matrix multiplication, use `np.dot`, `@`, or `np.matmul`.
4. **Allowing broadcasting along a dimension that was not intended.** An `(N, 3)` array plus an `(N,)` array pads the second to `(1, N)` and then fails unless `N` is 3 or 1; with `N = 3` it runs and adds along the wrong axis. An `(N, 1)` array combined with an `(N,)` array never fails and returns `(N, N)`. Be deliberate about which operand is a row and which is a column.
5. **Storing biases with shape `(n_neurons,)` instead of `(1, n_neurons)`.** Both work for the forward pass, but the explicit `(1, n_neurons)` shape is clearer about intent and matches the `(1, n_neurons)` gradient of post 16, so the in-place update works; a `(n_neurons,)` bias raises on that update.
6. **Trusting a print to confirm the shape.** `[12 15 18]` and `[[12 15 18]]` differ by one pair of brackets and by one axis: the shapes are `(3,)` and `(1, 3)`. Always check `arr.shape`.

---

## Further reading

- Harris, C. R., et al., *"Array Programming with NumPy"* (Nature, 2020). The array model, reductions along axes, broadcasting, and the libraries that adopted them.
- Iverson, K. E., *A Programming Language* (Wiley, 1962). The notation behind APL, one of the array languages that informed NumPy's design.
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 4 (2020). Uses `axis=1, keepdims=True` for the per-row maximum and the per-row sum of the softmax activation.
- NumPy documentation, *"Broadcasting"* (latest). The rules of section 4 in NumPy's own words.
- NumPy documentation, `numpy.sum`, `numpy.max`, `numpy.mean` (latest). The `axis` and `keepdims` arguments of each reduction.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 06 - Activation functions: ReLU and Softmax](../06-activation-functions-relu-and-softmax/index.md):** the first real consumers of `keepdims=True`, in softmax's per-row max subtraction and per-row normalisation.
- **[Post 08 - Loss: categorical cross-entropy](../08-loss-categorical-cross-entropy/index.md):** the per-sample loss, a reduction along `axis=1` that is kept 1-D on purpose because it is averaged and never broadcast back.
