# 22 - Gradient-descent optimiser

> **TL;DR.** Gradients become learning once a rule turns them into new parameters, and the simplest rule subtracts the learning rate times the gradient from every parameter: `Optimizer_SGD` holds it in one method, `update_params(layer)`, which a training loop calls once per layer after every backward pass. On the spiral data, with a learning rate of 1 and 10,001 epochs, the documented run ends at a loss of 0.8737 and 64.7 percent accuracy, but its loss rose on 4,628 of the 10,000 updates, and ten seeds end anywhere between 43.0 and 87.0 percent. Rates of 0.01 and 0.1 never raise the loss and learn too slowly, a rate of 10 kills all 64 hidden neurons within 79 epochs, and five times the epochs lift every seed tried: the limit is the rule and its budget, not the data.
>
> **Prerequisites:** [Post 21](../21-coding-the-full-backpropagation/index.md).
> **Safe to skip?** Skip it if the reader can already write a full-batch training loop around an optimiser object from memory, and knows from a measurement, not from a rule of thumb, what a learning rate of 0.1, 1 and 10 does to this network.
>
> **After reading, you will be able to:**
>
> - Implement Optimizer_SGD and use it to update any layer that stores weights, biases and their gradients.
> - Write a complete training loop (forward, backward, update, log) for the spiral classifier.
> - Explain why the accuracy vanilla gradient descent reaches on the spiral is a limit of the optimiser and its budget, not of the data.

![The update rule, theta new equals theta old minus alpha times the gradient, above three small loss curves. With alpha too small the curve stays almost flat. With alpha labelled right-sized it falls slowly to about 0.87 at 10,000 epochs. With alpha too large it bounces and diverges.](diagrams/01-sgd-update-and-lr.svg)

*One rule and one setting. The middle panel draws the rate of 1 as a smooth, slow fall; the measured curve of section 4 falls on average and rises on nearly half of its steps, and section 5 measures what the right-hand panel sketches.*

---

## 1. The question: what turns four gradient arrays into a network that learns?

Post 21 ends with a gradient stored beside each of the four parameter arrays of the spiral classifier: `dense1.dweights`, `dense1.dbiases`, `dense2.dweights` and `dense2.dbiases`. Nothing has moved yet. [Post 20](../20-assembling-full-backpropagation/index.md), section 5, made one update by writing four subtractions, and left two things open: who owns those subtractions, and what happens when the cycle of forward pass, backward pass and update is repeated ten thousand times. The question of this post is: **what is the simplest rule that turns gradients into new weights, and how far does it get on the spiral?**

The rule is the one [post 09](../09-introduction-to-optimisation/index.md) introduced. For any parameter $\theta$, a weight or a bias, with gradient $g = \partial L / \partial \theta$:

$$\theta \leftarrow \theta - \alpha \, g$$

$\alpha$ is the **learning rate**, the only setting the rule has. The rule is applied to every weight and every bias once per step, and the object that applies it is called an **optimiser**. This one is **gradient descent**, also called vanilla gradient descent to set it apart from the refinements of posts 23 to 27.

The class is named `Optimizer_SGD`, for stochastic gradient descent. The stochastic variant steps on a gradient estimated from a random part of the data, an idea that goes back to the stochastic approximation method of Robbins and Monro (1951). The loop of this post uses all 300 samples for every step, so nothing in it is random after the initial weights; the update rule is the same, and the name is the one the series keeps when [post 32](../32-mini-batching/index.md) makes the batches random.

### 1.1. Why the layer and the optimiser are separate objects

| Concern | Lives in |
|---|---|
| What the gradient of this operation is | the component's `backward` |
| How a gradient becomes a new parameter | the optimiser's `update_params` |
| The learning rate, and anything remembered between steps | the optimiser |

The two sides meet in four attributes. A layer with parameters stores `weights`, `biases`, `dweights` and `dbiases`; the optimiser reads the last two and writes the first two, and knows nothing about how the gradients were computed. That is the contract every optimiser of Part VI keeps: `update_params(layer)` is called once for each layer that has parameters, after the backward pass is complete. Post 23 adds two calls around it, `pre_update_params` and `post_update_params`, once per step each; from then on a new optimiser is a change of one line in the loop.

---

## 2. The `Optimizer_SGD` class

```python
class Optimizer_SGD:

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases
```

The constructor stores one number. `update_params` is the rule, once for the weights and once for the biases. `snippets/one_update.py` calls it on numbers that can be checked by hand:

```python
layer = SimpleNamespace(weights=np.array([[0.5, -1.0], [2.0, 0.0]]),
                        biases=np.array([[0.1, -0.2]]),
                        dweights=np.array([[1.0, -2.0], [0.0, 4.0]]),
                        dbiases=np.array([[-1.0, 0.5]]))
array_before = layer.weights

optimizer = Optimizer_SGD(learning_rate=0.1)
optimizer.update_params(layer)
```

```text
weights [[0.4, -0.8], [2.0, -0.4]]
biases  [[0.2, -0.25]]
same array object as before the update: True
```

Each entry moved against its own gradient by a tenth of it: $0.5 - 0.1 \cdot 1.0 = 0.4$ and $-1.0 - 0.1 \cdot (-2.0) = -0.8$; the entry whose gradient is 0 stayed at 2.0. Three things follow from the eight lines of the class.

- **It works on anything that has the four arrays.** The `layer` above is a bare `SimpleNamespace`, not a `Layer_Dense`. `Activation_ReLU` and the combined softmax and loss class have no parameters, so they are never handed to it.
- **It writes in place.** `-=` changes the array the layer already holds, so the next forward pass sees the new parameters without anything being handed back.
- **It remembers nothing.** The only attribute is the learning rate; each call uses the gradients of the moment and forgets them. Its default of 1.0 is the value the documented run of section 3 uses, not a recommendation; section 5 measures what other values do.

The same script repeats the single update of post 20, section 5, on that post's batch and seed, with `Optimizer_SGD(learning_rate=0.01)` in place of the four written-out subtractions:

```text
loss before 1.098629   after 1.098210
largest difference from the four subtractions written out: 0.0e+00
```

These are post 20's two losses, and the parameters agree to the last bit. The class adds no arithmetic; it gives the subtractions an owner.

---

## 3. The training loop

`snippets/train_sgd.py` puts the four objects of post 20, the optimiser, and a loop over epochs together:

```python
nnfs.init()                                   # seed 0, float32 arrays, and a patched np.dot
X, y = spiral_data(samples=100, classes=3)    # X (300, 2), y (300,): three classes of 100 points

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(learning_rate=1.0)

losses, accuracies = [], []

for epoch in range(10001):
    # Forward pass.
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Accuracy: the share of samples whose largest output is the true class.
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Backward pass.
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update: one call for each layer that has parameters.
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    # Log.
    losses.append(loss)
    accuracies.append(accuracy)
    if epoch % 1000 == 0:
        print(f"epoch {epoch:5d}  loss {loss:.4f}  acc {accuracy:.4f}")
```

The forward and backward calls are those of posts 20 and 21, and the classes are those of posts 16 and 19, unchanged. Four things about the loop are fixed here, because every later post of Part VI repeats this run and compares its own optimiser with it.

**The setup.** `nnfs.init()` seeds NumPy with 0 and makes every array `float32`; the data are the 300 spiral points of post 04; the weights start as `0.01 * np.random.randn` and the biases at zero; the learning rate is 1.0; the loop runs 10,001 times. The script prints this setup.

**The hidden layer has 64 neurons, not 3.** The network has $2 \cdot 64 + 64 + 64 \cdot 3 + 3 = 387$ parameters where the network of post 20 had 21. Post 09, section 4, ran gradient descent on both for 10,000 steps: it ended at a loss of 1.0776 with 3 hidden neurons and at 0.8737 with 64.

**An epoch is one pass over the whole training set.** All 300 samples go through at once, so here one epoch is one update. With mini-batches (post 32) an epoch is several updates.

**The loss and the accuracy are measured before the update.** The numbers logged at epoch $t$ belong to the parameters after $t$ updates, so epoch 0 shows the untrained network and epoch 10,000 the result of 10,000 updates. Accuracy is only reported; the gradient comes from the loss.

---

## 4. What the run produces

```text
epoch     0  loss 1.0986  acc 0.3600
epoch  1000  loss 1.0623  acc 0.4000
epoch  2000  loss 1.0372  acc 0.4033
epoch  3000  loss 0.9854  acc 0.5433
epoch  4000  loss 0.9707  acc 0.5200
epoch  5000  loss 0.9731  acc 0.5100
epoch  6000  loss 0.9398  acc 0.4900
epoch  7000  loss 0.9050  acc 0.5933
epoch  8000  loss 0.8736  acc 0.6167
epoch  9000  loss 0.8717  acc 0.5700
epoch 10000  loss 0.8737  acc 0.6467
```

The first row is the untrained network: a loss of $\ln 3 = 1.0986$ and 108 of 300 correct. The last row, a loss of 0.8737 and 194 of 300 correct, is the result post 09 obtained with the same steps written without classes. It is the **documented run** of Part VI.

**The start is slow.** After 100 updates the loss is 1.0869, and after 1,000 it is 1.0623 with 40 percent correct. The cause is the one post 20, section 5, measured on a small batch: with weights at a scale of 0.01 and every class equally frequent, the first gradients are tiny, and a step of $\alpha$ times a tiny gradient is tiny whatever $\alpha$ is.

**The log hides most of what the loss did.** Eleven rows a thousand epochs apart read as a steady descent. The script also looks between them:

```text
the loss rose on 4,628 of 10,000 updates; largest single rise 0.0553, at epoch 1,604
lowest loss 0.8343, at epoch 7,874; highest accuracy 0.6700, at epoch 8,117
epochs 9,001 to 10,000: loss 0.8350 to 0.9330, accuracy 0.5633 to 0.6533
updates 1 to 1,000: the loss rose on 105
updates 1,001 to 2,000: the loss rose on 503
```

In its first thousand updates the loss rose on about one in ten; from then on it rose on about every second one. The curve falls on average and bounces from step to step. Over the last thousand epochs the accuracy moved between 56.3 and 65.3 percent, so the 64.7 percent of the final row sits near the top of a band nine points wide, and stopping a few epochs earlier or later would have printed a different result. The loss and the accuracy do not even move together: from epoch 3,000 to epoch 6,000 the loss fell from 0.9854 to 0.9398 while the accuracy fell from 54.3 to 49.0 percent.

**One seed is one run.** `snippets/seed_spread.py 10` repeats the run from ten seeds, calling `np.random.seed` after `nnfs.init()` so that the data and the initial weights are both redrawn; seed 0 is the documented run.

```text
seed  final loss  final acc  loss rose on  dead   lowest loss  highest loss  lowest acc  highest acc
0     0.8737      0.6467     4,628         3      0.8350       0.9330        0.5633      0.6533
1     0.5091      0.7467     4,665         6      0.4831       1.2313        0.5567      0.7967
2     0.9906      0.5367     4,154         11     0.7931       1.3265        0.4200      0.6333
3     0.3943      0.8700     4,501         2      0.3521       2.6356        0.5433      0.8867
4     0.4843      0.7867     4,765         3      0.4833       1.4611        0.5167      0.8100
5     2.2696      0.4300     4,457         4      0.6053       3.3912        0.4000      0.6900
6     0.5277      0.7433     4,858         7      0.5001       1.4217        0.5600      0.7967
7     0.6652      0.7167     4,503         2      0.6162       1.4059        0.4567      0.7467
8     0.9146      0.5867     4,539         12     0.8530       1.0577        0.4733      0.6400
9     0.4864      0.7800     4,774         5      0.4318       2.3218        0.4267      0.8433
over the 10 seeds: final loss 0.3943 to 2.2696, final accuracy 0.4300 to 0.8700
```

The last four columns are the lowest and highest loss and accuracy over epochs 9,001 to 10,000, and `dead` counts the hidden neurons that output zero for all 300 samples at the end. Three readings follow.

- **The final accuracy is anywhere from 43.0 to 87.0 percent.** The 64.7 percent of seed 0 is one draw from that range, not what this optimiser reaches on the spiral.
- **The loss rose on 4,154 to 4,858 of the 10,000 updates in every run.** The bouncing belongs to the learning rate of 1 on this network, not to one unlucky start.
- **The last epoch is an arbitrary place to stop.** Seed 5 ends at a loss of 2.2696, about twice the 1.0986 of the untrained network, only because epoch 10,000 fell on a peak: within its last thousand epochs the loss was as low as 0.6053 and the accuracy as high as 69.0 percent. The steadier figure is the mean over epochs 9,001 to 10,000, which the script prints seed by seed after the table: across the ten runs a loss of 0.4159 to 0.9071 and an accuracy of 56.1 to 84.5 percent. These means, not the final rows, are what a later optimiser is to be compared with.

![A loss curve drawn through four marked points, 1.10 at the start, 1.06 at 1,000 epochs, 0.97 at 5,000 and 0.87 at 10,000, with accuracies of 36, 40, 51 and 65 percent above them, beside three bars: plain gradient descent at 10,000 epochs 65 percent, at 50,000 epochs 91 percent, Adam at 10,000 epochs 96 percent.](diagrams/02-sgd-is-slow.svg)

*The marked points are those of the seed 0 run, and its 50,000-epoch bar is the seed 0 row of section 6. The curve between the points is drawn smooth, which the run is not, and the Adam bar is a result of post 27 that this post does not measure.*

---

## 5. The learning-rate trade-off

If a rate of 1 bounces, the obvious move is a smaller one. `snippets/learning_rate.py full` trains the same network with six rates, each from seeds 0 to 4, and prints the smallest and largest value over the five seeds:

```text
rate    final loss        final accuracy    loss rose on      largest rise    highest loss      dead of 64
0.001   1.0984 to 1.0985  0.4133 to 0.4367  292 to 462        2.4e-07         1.0986 to 1.0986  0 to 1
0.01    1.0702 to 1.0867  0.3967 to 0.4433  0 to 0            none            1.0986 to 1.0986  2 to 9
0.1     1.0014 to 1.0583  0.4367 to 0.5267  0 to 0            none            1.0986 to 1.0986  2 to 10
1.0     0.3943 to 0.9906  0.5367 to 0.8700  4,154 to 4,765    1.1e+00         1.1523 to 2.6356  2 to 11
3.0     0.3746 to 0.5482  0.7033 to 0.8100  4,690 to 4,907    1.8e+00         2.4746 to 3.8377  2 to 13
10.0    1.1854 to 1.9295  0.3333 to 0.3333  6,425 to 6,468    7.7e-01         2.2560 to 2.3889  64 to 64
learning rate 10.0: the first epoch at which all 64 hidden neurons are dead
seed 0 to 4: [64, 38, 36, 79, 38]
```

**Too small: 0.001 and 0.01.** After 10,000 updates the loss has not left the band from 1.07 to 1.10 in any seed. At 0.001 the "rises" are float32 rounding, at most $2.4 \times 10^{-7}$ each, on a loss that has moved by one or two ten-thousandths in total.

**Steady and slow: 0.1.** The loss never rose, in 10,000 updates of any of the five runs, and it ended between 1.0014 and 1.0583, with at most 52.7 percent correct. Every step helped and the budget ran out.

**Fast and unsteady: 1 and 3.** Both rates raise the loss on more than four updates in ten, with single rises as large as 1.1 and 1.8 and peaks far above the 1.0986 the run started from, and both end lower than 0.1 does in every seed. At 3 the peaks are higher and the final losses lie in a lower, narrower band than at 1. On this network the rate that gets furthest in the budget is a rate that bounces.

**Too large: 10.** All 64 hidden neurons are dead by epoch 79 at the latest, in every one of the five runs. This is the failure posts 12 and 13 showed on one neuron and one layer: a step long enough to push every pre-activation of a ReLU below zero for every sample. From then on `activation1.output` is all zeros, so `dense2.dweights` is zero and nothing reaches `dense1`; only the three biases of `dense2` still move, every sample gets the same three logits, and the accuracy is exactly 100 of 300. The loss does not become `inf` or `nan`: it ends between 1.1854 and 1.9295, still rising on about two updates in three, in a network that can no longer learn.

A handful of dead neurons, 2 to 13 of 64, appears at every rate from 0.01 to 3 and does not stop those runs; all 64 is what ends training. Post 17 names the usual remedies, a smaller learning rate and a leaky ReLU.

This is the one-parameter experiment of post 09, section 5.1, on a real network: crawl, steady descent, bouncing, and, where the parabola diverged, a network of dead ReLUs. The difference is that no rate here is both steady and quick. The rates that never raise the loss are too slow for the budget, and the rates that use the budget spend nearly half their steps going up.

---

## 6. Why the limit is the rule and not the data

A final accuracy between 43 and 87 percent invites the reading that three interleaved spirals are too hard for this network. Two measurements say otherwise.

The first is in section 4: on the same kind of data and the same architecture, seed 3 ends at 87.0 percent. The second is the budget. `snippets/seed_spread.py long` runs seeds 0 to 4 for 50,001 epochs and prints the loss and the accuracy at five epochs:

```text
seed         10,000         20,000         30,000         40,000         50,000   mean accuracy, last 1,000
0     0.8737 0.6467  0.4595 0.8100  0.3749 0.8433  0.2418 0.9000  0.2896 0.9067   0.9006
1     0.5091 0.7467  0.3855 0.8233  0.1962 0.9200  0.1707 0.9200  0.1846 0.9267   0.9179
2     0.9906 0.5367  0.7584 0.6433  0.6474 0.6933  0.6395 0.6933  0.5604 0.7433   0.7237
3     0.3943 0.8700  0.2398 0.9233  0.1551 0.9467  0.1488 0.9567  0.1196 0.9500   0.9481
4     0.4843 0.7867  0.3776 0.8567  0.1980 0.9267  0.1168 0.9633  0.1054 0.9633   0.9414
accuracy at epoch 50,000: 0.7433 to 0.9633; mean of the last 1,000 epochs: 0.7237 to 0.9481
```

In every one of the five seeds the accuracy at epoch 50,000 is higher than at epoch 10,000, and four of the five end above 90 percent; averaged over the last thousand epochs, which is steadier than one epoch, the runs stand between 72.4 and 94.8 percent. The same 387 parameters on the same 300 points get there with the same rule, given five times the updates. The network can fit the spiral; gradient descent with one constant rate is slow at finding the fit.

Three properties of the rule are the candidates for the cause, and each of the next posts changes one of them.

- **One rate for the whole run.** Section 5 showed that the rates that make progress also bounce. The rule cannot take long steps early and short ones late. Post 23 lets the rate shrink as training goes on.
- **No memory.** Each step uses the gradient of the moment and nothing else, so the rule cannot take into account that its last step raised the loss, as nearly half of them do at a rate of 1. Post 24 gives the step a memory of the steps before it.
- **One rate for all 387 parameters.** A weight whose gradient is large and a weight whose gradient is small are scaled by the same $\alpha$. Posts 25 to 27 give every parameter its own rate.

What those changes buy is measured in those posts, against the run of section 3. One thing none of them changes: gradient descent in any form walks downhill from where it starts, and post 09, section 6, already said that this guarantees no global minimum.

---

## 7. Make it run: the optimiser, the loop, and the spreads

Six scripts hold every code block and every number of this post. All run from the series root, for example `python posts/22-gradient-descent-optimiser/snippets/train_sgd.py`, and print the same output on every run. The first two need only NumPy; the others also need the `nnfs` package (`pip install nnfs`). The times are those of an idle laptop CPU; a busy machine took up to one and a half times as long.

| Script | Section | Prints | Time |
|---|:---:|---|:---:|
| `optimizer_sgd.py` | 2 | the classes of posts 16 and 19 and `Optimizer_SGD`; the other scripts import them from here | 1 s |
| `one_update.py` | 2 | the update checked by hand, and the update of post 20 made by the class | 1 s |
| `train_sgd.py` | 3, 4 | the documented run, its setup, and what happened between the logged epochs | 7 s |
| `seed_spread.py` | 4 | the run from seeds 0 to 4 | 35 s |
| `seed_spread.py 10` | 4 | the run from seeds 0 to 9: the table of section 4 | 70 s |
| `seed_spread.py long` | 6 | 50,001 epochs from seeds 0 to 4 | 160 s |
| `learning_rate.py` | 5 | the six rates after 2,001 epochs, and the epochs at which a rate of 10 kills the hidden layer | 40 s |
| `learning_rate.py full` | 5 | the six rates after 10,001 epochs: the table of section 5 | 185 s |
| `what_can_go_wrong.py` | 8 | the three failures of section 8 | 35 s |

---

## 8. What can go wrong?

`snippets/what_can_go_wrong.py` runs three mistakes in the loop. Only the first raises an error.

**The optimiser is called where it has nothing to read.**

```text
update_params(dense1) before any backward pass: AttributeError: 'Layer_Dense' object has no attribute 'dweights'
update_params(activation1): AttributeError: 'Activation_ReLU' object has no attribute 'weights'
```

A layer has no `dweights` until its `backward` has run, and an activation never has any. The first message is the lucky case. Once gradients exist, the optimiser cannot tell fresh ones from old ones: an update that is called twice, or after a forward pass with no new backward pass, steps again on whatever is stored.

**A layer is left out of the update.** With the line `optimizer.update_params(dense1)` missing, the loop runs to the end:

```text
final loss 1.0682 to 1.0825, final accuracy 0.4233 to 0.4500
```

Over seeds 0 to 4 the loss falls a little on every update, never rises, and ends above 1.06, where the complete loop ended between 0.3943 and 0.9906. The output layer is learning on top of 64 hidden features that stay at their random start. A smooth, slowly falling curve looks healthier than the correct one, and nothing in the log points at the missing line.

**The gradient is added instead of subtracted.**

```text
seed  loss at epochs 0, 10, 50, 100       first nan at epoch  accuracy at the end
0     1.0986  1.0987  10.7454  10.7454    109                 0.3333
```

The other four seeds print the same losses to within 0.0003, with the first `nan` between epochs 107 and 110. The loss climbs from the first update. By epoch 50 the network gives one and the same class a probability of 1 for all 300 samples: the 200 samples of the other two classes each cost $-\ln 10^{-7} = 16.1181$, the clipping limit of post 08, and their share of the mean is $2/3 \cdot 16.1181 = 10.7454$. The weights keep growing until `float32` overflows, the loss becomes `nan`, and the accuracy is 100 of 300, because `argmax` of a row of `nan` is 0.

---

## 9. Summary

| Concept | Takeaway |
|---|---|
| Update rule | $\theta \leftarrow \theta - \alpha g$, for every weight and bias, once per step |
| `Optimizer_SGD` | stores `learning_rate`; `update_params(layer)` reads `dweights` and `dbiases` and changes `weights` and `biases` in place |
| Contract | one `update_params` call per layer with parameters, after the whole backward pass |
| Training loop | forward, backward, update, log; one epoch is one update on the full batch |
| Documented run | seed 0, rate 1, 10,001 epochs: loss 0.8737, 64.7 percent; the loss rose on 4,628 of 10,000 updates |
| Seed spread | ten seeds end between 43.0 and 87.0 percent |
| Learning rate | 0.1 never raises the loss and ends above 1.00; 1 and 3 get further and bounce; 10 kills all 64 hidden neurons |
| Not the data | at 50,001 epochs all five seeds are higher, four of them above 90 percent |

---

## Common pitfalls

1. **Reporting the last epoch as the result.** At a rate of 1 the accuracy moved by nine points or more within the last thousand epochs of every one of the ten runs. A lowest loss or a mean over the last epochs says more than one row.
2. **Comparing two settings on one seed.** The same optimiser ends between 43.0 and 87.0 percent from ten seeds; a difference smaller than that spread, measured once, is not a finding.
3. **Leaving a layer out of the update.** No error is raised and the loss still falls. Every layer with parameters needs its `update_params` call.
4. **Raising the learning rate to go faster.** Up to a point it works and bounces; past it the hidden layer dies and no later step can revive it.
5. **Updating without a fresh backward pass.** The optimiser reads whatever gradients are stored, including stale ones.
6. **Blaming the data for a low score.** The same network on the same points passes 90 percent in four of five long runs.

---

## Further reading

- Cauchy, A.-L., *"Méthode générale pour la résolution des systèmes d'équations simultanées"* (Comptes rendus de l'Académie des sciences, 1847).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 8 (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 10 (2020).
- Robbins, H. and Monro, S., *"A Stochastic Approximation Method"* (Annals of Mathematical Statistics, 1951).
- Ruder, S., *"An Overview of Gradient Descent Optimization Algorithms"* (arXiv:1609.04747, 2016).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 23 - Learning-rate decay](../23-learning-rate-decay/index.md):** the first change to the rule, a learning rate that shrinks during the run.
- **[Post 32 - Mini-batching](../32-mini-batching/index.md):** the loop of this post with random batches, where the S of `Optimizer_SGD` becomes literal.
