# 20 - Assembling full backpropagation

> **TL;DR.** A two-layer classifier is four objects, two dense layers, a ReLU and a combined softmax and loss, each with a `forward` and a `backward` method. The forward pass calls them from left to right, the backward pass from right to left, and the only thing that travels between two backward calls is one array: the `dinputs` a component stores is the `dvalues` of the component before it. After the four backward calls the two dense layers hold four gradient arrays, and four subtractions with a learning rate of 0.01 lower the loss on the post's batch of four samples from 1.098629 to 1.098210.
>
> **Prerequisites:** [Post 16](../16-coding-backpropagation/index.md), [Post 19](../19-softmax-derivatives-and-the-combined-backward-pass/index.md).
> **Safe to skip?** Skip it if the reader can already write, from memory, the four `forward` calls, the four `backward` calls and the four parameter subtractions of a dense, ReLU, dense, softmax classifier, and name the array each backward call is handed.
>
> **After reading, you will be able to:**
>
> - Wire Layer_Dense, Activation_ReLU and the combined softmax-loss class into a complete forward and backward pass.
> - Trace how one component's dinputs becomes the previous component's dvalues.
> - Apply a single gradient-descent update to all four parameter arrays.

![A row of five cards, the data X of shape (4, 2), dense1, activation1, dense2 and loss_activation. Grey forward arrows run left to right, labelled Z1, A1 and Z2, each of shape (4, 3); purple backward arrows run right to left, labelled dL/dZ2, dL/dA1, dL/dZ1 and, dashed, dL/dX, which nothing reads. Under each object is what its backward call stores with its shape: dweights, dbiases and dinputs for the two dense layers, dinputs alone for the ReLU, zero at its 5 closed gates, and dinputs alone for the combined class, computed from y-hat and y and divided by N = 4. The dense layers' gradients feed a card with the four update lines at learning_rate 0.01, and the loss goes from 1.098629 to 1.098210, a change of minus 4.190 times 10 to the minus 4.](diagrams/01-full-backprop-pipeline.svg)

*Every forward call has a matching backward call that runs in the opposite direction, and only the two dense layers store anything the update needs.*

---

## 1. The question: what carries the gradient from one `backward` to the next?

Posts 16 to 19 gave every component of the classifier a `backward` method, one component at a time. [Post 16](../16-coding-backpropagation/index.md) wrote it for `Layer_Dense` and `Activation_ReLU`, and [post 19](../19-softmax-derivatives-and-the-combined-backward-pass/index.md) for the class that joins softmax to the categorical cross-entropy loss. None of those posts ran all of them in one pass with the real loss at the end. The question here is: **in which order are the `backward` methods called, and what does each call hand to the next?**

| Class | `forward` | `backward` stores | Trainable |
|---|---|---|:---:|
| `Layer_Dense` | $\mathbf{Z} = \mathbf{X} \mathbf{W} + \mathbf{b}$ | `dweights`, `dbiases`, `dinputs` | yes |
| `Activation_ReLU` | $\max(0, \mathbf{Z})$ | `dinputs`, a masked copy of `dvalues` | no |
| `Activation_Softmax_Loss_CategoricalCrossentropy` | softmax, then the mean cross-entropy | `dinputs` $= (\hat{\mathbf{y}} - \mathbf{y})/N$ | no |

The network is the one of [post 07](../07-coding-the-complete-forward-pass/index.md): two inputs, a hidden layer of three ReLU neurons, an output layer of three neurons, softmax, and the cross-entropy loss. It has four parameter arrays, the weights and biases of the two dense layers, with $2 \cdot 3 + 3 + 3 \cdot 3 + 3 = 21$ numbers between them. This post runs it on a batch of four hand-written samples, small enough to print; post 21 runs the same calls on the spiral data.

---

## 2. The forward pass

```python
X = np.array([[ 1.0,  2.0],
              [-1.5,  0.5],
              [ 0.5, -2.0],
              [ 2.0,  1.0]])                    # (4, 2): four samples, two inputs
y = np.array([0, 1, 2, 1])                      # one integer label per sample

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Forward pass, left to right: each call reads the output of the call before it.
dense1.forward(X)                                   # X  -> Z1
activation1.forward(dense1.output)                  # Z1 -> A1
dense2.forward(activation1.output)                  # A1 -> Z2
loss = loss_activation.forward(dense2.output, y)    # Z2 -> predictions -> loss
```

Four objects, four calls. `loss_activation` is the combined class of post 19, which named its instance `softmax_loss` there. Each call stores its result in `self.output`, and the next call reads it. The last call differs from the other three in two ways: it also takes the labels, and it returns the loss, because the combined class runs softmax and the cross-entropy one after the other. The dense layers and the ReLU also keep the inputs they were given (the cache of post 16), and the combined class keeps its predictions; the backward pass needs both.

With $N = 4$ samples, every array between `X` and the loss has shape `(4, 3)`: `dense1.output` ($\mathbf{Z}_1$), `activation1.output` ($\mathbf{A}_1$), `dense2.output` ($\mathbf{Z}_2$) and `loss_activation.output` ($\hat{\mathbf{y}}$). The script prints the loss:

```text
loss 1.098629   ln 3 = 1.098612
```

The weights start at a scale of 0.01, so the logits are close to zero, every prediction is close to $1/3$, and the loss is close to $\ln 3$, the loss of a uniform guess over three classes (post 08).

---

## 3. The backward pass

The backward pass calls the same four objects in the opposite order.

```python
# Backward pass, right to left: each call reads the dinputs of the call before it.
loss_activation.backward(loss_activation.output, y)   # predictions, labels -> dL/dZ2
dense2.backward(loss_activation.dinputs)              # dL/dZ2 -> dweights, dbiases, dL/dA1
activation1.backward(dense2.dinputs)                  # dL/dA1 -> dL/dZ1
dense1.backward(activation1.dinputs)                  # dL/dZ1 -> dweights, dbiases, dL/dX
```

| # | Call | Is handed | Stores |
|:---:|---|---|---|
| 1 | `loss_activation.backward` | the predictions and the labels | `dinputs` $= \partial L / \partial \mathbf{Z}_2$ |
| 2 | `dense2.backward` | `loss_activation.dinputs` | `dweights`, `dbiases`, `dinputs` $= \partial L / \partial \mathbf{A}_1$ |
| 3 | `activation1.backward` | `dense2.dinputs` | `dinputs` $= \partial L / \partial \mathbf{Z}_1$ |
| 4 | `dense1.backward` | `activation1.dinputs` | `dweights`, `dbiases`, `dinputs` $= \partial L / \partial \mathbf{X}$ |

**The first call has no upstream gradient to receive.** The loss is the right-hand end of the chain, and the derivative of the loss with respect to itself is 1, so there is nothing to multiply by. The combined class goes straight to its own gradient, $(\hat{\mathbf{y}} - \mathbf{y})/N$, and for that it needs the predictions and the labels. Its first parameter is still named `dvalues`, to keep one signature across the series, but what it must be given is `loss_activation.output`. The division by $N$, which comes from the mean over the batch (post 18), happens inside this call and nowhere else in the chain.

**Calls 2 to 4 are each handed the `dinputs` of the call before.** No `backward` returns anything; each stores its results on the object, and the next line of the script reads them from there. The figure at the top of the post draws the four calls this way, with the shape of every array they store.

```text
gradient                 shape     read by
loss_activation.dinputs  (4, 3)    dense2.backward
dense2.dinputs           (4, 3)    activation1.backward
activation1.dinputs      (4, 3)    dense1.backward
dense1.dinputs           (4, 2)    nothing
dense2.dweights          (3, 3)    the update
dense2.dbiases           (1, 3)    the update
dense1.dweights          (2, 3)    the update
dense1.dbiases           (1, 3)    the update
loss_activation.dinputs
[[-0.166638  0.083295  0.083343]
 [ 0.083333 -0.166668  0.083336]
 [ 0.08334   0.083327 -0.166667]
 [ 0.083365 -0.166699  0.083335]]
dense2.dbiases [[ 0.083399 -0.166745  0.083346]]
closed ReLU gates: 5 of 12; zeros in activation1.dinputs: 5
```

The numbers can be read without a calculator. Every prediction is close to $1/3$, so a row of $(\hat{\mathbf{y}} - \mathbf{y})/4$ is close to $-1/6$ at the true class and $1/12$ elsewhere. `dense2.dbiases` is the sum of that array down each column: class 1 is the label of two samples, which gives $(1/12 + 1/12 - 1/6 - 1/6) \approx -0.1667$, and classes 0 and 2 are the label of one sample each, which gives $1/12 \approx 0.0833$. Five of the twelve ReLU gates were closed on this batch, and `activation1.dinputs` is zero in exactly those five places. The figure below shows both arrays entry by entry.

![Left, loss_activation.dinputs as a 4 by 3 grid, rows labelled y = 0, 1, 2, 1 and columns class 0 to 2. The cell of each sample's label is shaded green and holds about minus 1/6: -0.166638, -0.166668, -0.166667 and -0.166699; every other cell holds about 1/12. An arrow, sum down each column, leads to dense2.dbiases: 0.083399, -0.166745, 0.083346. Right, activation1.dinputs, samples by hidden neurons, with five cells shaded red for the closed gates, each 0, and the other seven entries small numbers copied from dense2.dinputs, such as -0.001795 and 0.001314.](diagrams/02-first-gradients.svg)

*Every row of the first gradient sums to zero, and its column sums are the bias gradient of `dense2`.*

`dense1.dinputs` is the gradient of the loss with respect to the data. It is computed because `Layer_Dense.backward` always computes it, and nothing reads it, because no component stands before `dense1`.

---

## 4. Why `dinputs` is the glue

The chain rule ([post 11](../11-the-chain-rule/index.md)) says that a component needs two things to compute its backward step: its own local derivative, which is written into its `backward` method, and the gradient of the loss with respect to its output, which arrives as `dvalues`. It needs nothing else. `dense2` does not know that a softmax follows it, or that a ReLU precedes it. That is what makes the classes interchangeable parts.

One fact turns this into a pipeline: the output of one component is the input of the next, so the gradient with respect to one component's input **is** the gradient with respect to the previous component's output. `dense2.dinputs` and the `dvalues` of `activation1` are both $\partial L / \partial \mathbf{A}_1$, the same array under two names.

The script replays the backward pass with no objects, one formula per line. The names on the left are the mathematical ones.

```python
dZ2 = loss_activation.output.copy()
dZ2[range(len(y)), y] -= 1
dZ2 /= len(y)                                         # (predictions - one-hot labels) / N
dW2 = np.dot(activation1.output.T, dZ2)
db2 = np.sum(dZ2, axis=0, keepdims=True)
dA1 = np.dot(dZ2, dense2.weights.T)                   # what dense2 hands back
dZ1 = dA1 * (dense1.output > 0)                       # what the ReLU hands back
dW1 = np.dot(X.T, dZ1)
db1 = np.sum(dZ1, axis=0, keepdims=True)
```

```text
largest difference over the four parameter gradients: 0.0e+00
```

Each line uses the result of an earlier line, and the two lines marked "hands back" are the `dinputs` of the objects. The four calls of section 3 are these nine lines, sorted into the classes that own them.

The same pattern carries any depth. Each further hidden layer adds one `Layer_Dense` and one `Activation_ReLU`: two forward calls, two backward calls, one more hand-off of each kind, and two more parameter arrays. Nothing else changes.

---

## 5. The gradient-descent update

After the backward pass, each of the four parameter arrays has a gradient of its own shape stored beside it. Gradient descent ([post 09](../09-introduction-to-optimisation/index.md)) moves every parameter a small step against its gradient. With a learning rate $\alpha = 0.01$ (`learning_rate` in the script):

```python
dense1.weights -= learning_rate * dense1.dweights
dense1.biases -= learning_rate * dense1.dbiases
dense2.weights -= learning_rate * dense2.dweights
dense2.biases -= learning_rate * dense2.dbiases
```

The sign is a minus because the gradient points in the direction in which the loss rises. `activation1` and `loss_activation` have no line, because they have no parameters. A second forward pass shows what the step did:

```text
loss before 1.098629   after 1.098210   change -4.190e-04
sum of squared gradients 0.041968   -learning_rate * sum -4.197e-04
dense2.biases after the update [[-0.000834  0.001667 -0.000833]]
```

The loss fell by $4.19 \times 10^{-4}$. The size of the fall can be predicted. For a small step, the change of the loss is close to the sum, over all 21 parameters, of gradient times change (post 10). Each change is $-\alpha$ times the gradient, so the sum is $-\alpha$ times the sum of the squared gradients: $-0.01 \cdot 0.041968 = -4.197 \times 10^{-4}$. The prediction is never positive, which is why a small enough step cannot raise the loss. The biases of `dense2` show the direction of the step: the bias of class 1, the commonest label in the batch, went up, and the other two went down.

The fall barely depends on the seed here. For seeds 0 to 9 the script prints falls between $4.169 \times 10^{-4}$ and $4.191 \times 10^{-4}$. It does depend on the batch:

```text
balanced batch of six: smallest 2.723e-07, largest 1.409e-06
```

When every class occurs equally often, the bias gradient of the last layer nearly cancels, because each class is predicted with probability about $1/3$ and is the label of a third of the samples. What is left are the weight gradients and the bias gradient of the first layer, and those are small because the weights of the 0.01 initialisation are. One step then lowers the loss by about a millionth or less. The figure below puts the falls of both batches for the ten seeds on one logarithmic axis.

![Two ranges on a logarithmic axis from 10 to the minus 7 to 10 to the minus 3, the fall in loss after one update, one tick per seed for seeds 0 to 9. On the post's batch of four, labels 0, 1, 2, 1, all ten falls sit in one place, between 4.169 times 10 to the minus 4 and 4.191 times 10 to the minus 4. On the balanced batch of six, labels 0, 1, 2, 1, 0, 2, they spread from 2.723 times 10 to the minus 7 to 1.409 times 10 to the minus 6.](diagrams/03-fall-per-seed.svg)

*The two batches differ by more than two powers of ten; the ten seeds hardly differ at all.*

A single update is a small thing; training is the cycle of forward pass, backward pass and update repeated many times, which [post 22](../22-gradient-descent-optimiser/index.md) codes as a loop, with the update packaged into an optimiser class.

The four subtractions stand after the four backward calls. Updating each dense layer straight after its own `backward` gives the same parameters, because a layer's `dinputs` already exists when its weights change (`snippets/what_can_go_wrong.py` checks it). Keeping the two phases apart is still the convention of the series: the optimisers of Part VI are handed layers whose gradients are complete.

---

## 6. Make it run: forward, backward, update

Two scripts hold every code block and every printed number of this post. Both need only NumPy, are seeded, and finish in under a second:

```text
python posts/20-assembling-full-backpropagation/snippets/assemble.py
python posts/20-assembling-full-backpropagation/snippets/what_can_go_wrong.py
```

`assemble.py` defines the classes and runs sections 2 to 5 in order. The classes are those of posts 16, 18 and 19, unchanged. `what_can_go_wrong.py` runs the mistakes of section 7 on the same batch.

---

## 7. What can go wrong?

Only one of the five mistakes below fails on the line where it is made.

**The logits handed to the first backward call.** `loss_activation.backward(dense2.output, y)` looks like the mirror of the forward call, and both arrays have shape `(4, 3)`. But `dense2.output` holds the logits, and the method subtracts the one-hot labels from whatever it is given:

```text
first row of dinputs, correct: [-0.166638  0.083295  0.083343]
first row of dinputs, wrong:   [-0.249809 -0.000009  0.000134]
row sums, correct: [0. 0. 0. 0.]   wrong: [-0.2497 -0.25   -0.2499 -0.2497]
gap to the correct gradients: dense1.dweights 5.9e-03   dense1.dbiases 5.0e-03   dense2.dweights 1.0e-02   dense2.dbiases 3.3e-01
```

Nothing is raised, and all four gradients are wrong. A correct row of this gradient sums to zero, because predictions and one-hot labels both sum to 1; the wrong rows sum to about $-0.25$, which is a cheap test.

**The ReLU left out of the backward pass.** `dense1.backward(dense2.dinputs)` skips `activation1.backward`. The shapes fit, because a ReLU does not change a shape, and all six entries of `dense1.dweights` come out different from the correct ones, some with the opposite sign. The gradients of `dense2` are unaffected, since they were computed before the mistake. Every forward call needs its backward counterpart.

**The gradient added instead of subtracted.**

```text
-=   loss before 1.098629   after 1.098210   change -4.190e-04
+=   loss before 1.098629   after 1.099050   change +4.204e-04
```

The step is as large as before and goes uphill.

**The loss handed to `dense2.backward`.** The loss is a single number, and NumPy multiplies an array by a number without complaint:

```text
dense2.dweights (3, 4)   dense2.dbiases ()   dense2.dinputs (3, 3)   activation1.inputs (4, 3)
IndexError: boolean index did not match indexed array along axis 0; size of axis is 3 but size of corresponding boolean axis is 4
```

`dense2.backward` runs and stores three arrays of the wrong shapes; they should be `(3, 3)`, `(1, 3)` and `(4, 3)`. The error appears one call later, when `activation1.backward` lays its mask of four rows on a `dinputs` of three. The update would fail as well, because `dweights` no longer has the shape of `weights`. What `dense2.backward` must be given is `loss_activation.dinputs`.

**Predictions without labels.** `loss_activation.forward` requires `y_true`, so calling it with the logits alone raises a `TypeError`. When only predictions are wanted, the softmax inside the combined class is called directly: `loss_activation.activation.forward(dense2.output)`, with the result in `loss_activation.activation.output`.

A sixth mistake, a `backward` that runs against the cache of a different batch, belongs to the classes and is measured in post 16.

---

## 8. Summary

| Concept | Takeaway |
|---|---|
| Three classes | dense layer, ReLU, and the combined softmax and loss class; only the dense layer has parameters |
| Forward order | left to right; each call reads the `output` of the call before |
| Backward order | right to left; each call is handed the `dinputs` of the call before |
| First backward call | is handed the predictions and the labels, and divides by $N$ |
| After the backward pass | `dweights` and `dbiases` on both dense layers, each with the shape of its parameter |
| Update | `parameter -= learning_rate * gradient`, four times |

---

## Common pitfalls

1. **Mirroring the forward call in the first backward call.** `loss_activation.backward` takes `loss_activation.output`, the predictions, not `dense2.output`, the logits.
2. **Skipping a component on the way back.** A ReLU changes no shape, so leaving its `backward` out raises nothing and corrupts every gradient before it.
3. **Passing the loss where a gradient belongs.** `dense2.backward` takes `loss_activation.dinputs`, an array of shape `(N, 3)`. The scalar loss is never an argument of a `backward`.
4. **Adding the gradient.** The gradient points uphill; the update subtracts it.
5. **Giving the activations an update line.** Only `Layer_Dense` stores `dweights` and `dbiases`. The other two classes store `dinputs` alone, and it is not a parameter gradient.

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 6, algorithm 6.4 (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (2020).
- Rumelhart, D., Hinton, G., and Williams, R., *"Learning Representations by Back-Propagating Errors"* (Nature, 1986).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 21 - Coding the full backpropagation](../21-coding-the-full-backpropagation/index.md):** the same calls as one short script on the spiral data, with the gradients inspected and checked.
- **[Post 22 - Gradient-descent optimiser](../22-gradient-descent-optimiser/index.md):** the four subtractions of section 5 packaged into a class that Part VI then extends.
