# 09 - Introduction to optimisation

> **TL;DR.** Once a loss exists, the parameters have to be moved so that it falls. Drawing fresh parameters at random lowers the spiral classifier's loss from 1.0986 to 1.0981 in 10,000 tries. Nudging them at random and keeping the improvements solves an easy dataset and, over 10,000 iterations on the 21-parameter network, does no worse than **gradient descent** (1.0435 against 1.0776), but it cannot use added parameters: on a 387-parameter network it ends at 1.0730, where gradient descent reaches 0.8737 on the same budget. Gradient descent knows the downhill direction before it steps, $\theta \leftarrow \theta - \alpha \nabla L$, and that knowledge is what scales.
>
> **Prerequisites:** [Post 08](../08-loss-categorical-cross-entropy/index.md).
> **Safe to skip?** Skip it if the reader can already say why random search fails as parameters are added, write the gradient-descent update rule, and say what a learning rate that is too large does.
>
> **After reading, you will be able to:**
>
> - Explain why random selection fails to lower the loss of a 21-parameter classifier.
> - Explain why random perturbation solves easy data and keeps pace with gradient descent on a small network and short budget, yet cannot use added parameters.
> - State the gradient-descent update rule, say what each symbol means, and predict what a given learning rate does on a one-parameter loss.
> - Name which later posts supply the calculus, the backward pass, and the optimisers that the update rule relies on.

![A dot chart of the final loss on the spiral after 10,000 iterations from one start, on an axis from 0.85 to 1.10 with a dotted line at ln 3, 1.0986. With 21 parameters, random selection's best draw is 1.0981; random perturbation ends at 1.0435 in the printed run and between 1.0321 and 1.0793 in 20 other streams of nudges, median 1.0684; gradient descent ends at 1.0776 with a learning rate of 1 and 1.0797 with 0.1. With 387 parameters, the walk ends at 1.0730 and between 1.0442 and 1.0666, median 1.0595, while gradient descent ends at 0.8737 and 1.0226.](diagrams/01-strategies-compared.svg)

*Final losses after 10,000 iterations on the spiral. On 21 parameters the blind walk keeps pace with gradient descent; on 387 every stream of the walk stays above 1.04, and gradient descent at a learning rate of 1 reaches 0.8737.*

---

## 1. The question: how should 21 numbers be moved?

Eight posts in, the script has every piece of a working classifier:

- a `Layer_Dense` class that allocates random weights and computes a linear forward pass,
- two activations (`Activation_ReLU` and `Activation_Softmax`) that handle hidden and output layers,
- a `Loss_CategoricalCrossentropy` class that turns predictions into a single non-negative number,
- an end-to-end pipeline that runs all of the above on the spiral dataset.

What is missing is exactly one thing: a way to update the 21 random numbers in `dense1.weights`, `dense1.biases`, `dense2.weights`, and `dense2.biases` so that the loss goes down. With those numbers left as they are, [post 08](../08-loss-categorical-cross-entropy/index.md) measured a loss of 1.0986, which is $\ln 3$, the loss of a uniform guess over three classes, and 102 of 300 points classified correctly.

The 21 parameters of the current architecture are not many by modern standards (ResNet-18, a small image classifier, has about 11.7 million, and GPT-3 has 175 billion), but every one of them needs an update rule. The same rule has to work at all three scales.

| Parameter array | Shape | Count |
|---|:---:|:---:|
| `dense1.weights` | `(2, 3)` | 6 |
| `dense1.biases` | `(1, 3)` | 3 |
| `dense2.weights` | `(3, 3)` | 9 |
| `dense2.biases` | `(1, 3)` | 3 |
| **total** | | **21** |

The question this post sets up: **what is the right algorithm for moving these 21 numbers in the direction that lowers the loss?** Two strategies that need no mathematics are tried first, and both are measured.

---

## 2. Strategy 1: random selection

The simplest possible idea is to throw away the parameters every iteration, draw new ones from scratch, compute the loss, and keep whichever set scored best. The network, the data, and the loss object are those of post 08; `snippets/random_selection.py` adds this loop:

```python
best_loss = float('inf')

for iteration in range(draws):
    # Throw the 21 parameters away and draw 21 new ones.
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases  = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases  = 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_fn.calculate(activation2.output, y)

    if loss < best_loss:
        best_loss = loss
        # Snapshot the winning parameters.
        best_parameters = [dense1.weights.copy(), dense1.biases.copy(),
                           dense2.weights.copy(), dense2.biases.copy()]
```

With `draws = 10_000` it prints:

```text
parameters: 21
untrained network: loss 1.0986, 102 of 300 correct
uniform guess: loss log(3) = 1.0986
after      10 draws: best loss 1.0986, 100 of 300 correct
after     100 draws: best loss 1.0984, 118 of 300 correct
after   1,000 draws: best loss 1.0983, 100 of 300 correct
after  10,000 draws: best loss 1.0981, 104 of 300 correct
the best set changed 13 times, the last time at draw 5,583
losses of all 10,000 draws: from 1.0981 to 1.1075, 436 of them below log(3)
```

Ten thousand forward passes move the loss from 1.0986 to 1.0981. The number of correct answers wanders between 100 and 118 of 300 and ends at 104, which is 34.7 percent against a chance level of 33.3. The network has done nothing useful. Ten times the budget does not change the picture: run with `100000` as its argument, the script ends on `best loss 1.0980, 110 of 300 correct`.

Two things are going on, and the output separates them.

**Every draw is nearly the same network.** Parameters drawn at a scale of 0.05 are tiny, so the logits are tiny, and softmax of near-equal logits is close to $1/3$ per class whatever the input. All 10,000 losses lie between 1.0981 and 1.1075. The search is choosing among 10,000 near-uniform guessers, and only 436 of them, 4.4 percent, are even slightly better than guessing uniformly.

**Wider draws do not rescue the method.** The script repeats the search with every parameter drawn at scale 1.0 and at scale 10.0:

```text
wider draws, 10,000 each:
  scale  1.0: best loss 1.0918, 114 of 300 correct, 5 of 10,000 draws below log(3)
  scale 10.0: best loss 1.6709, 100 of 300 correct, 0 of 10,000 draws below log(3)
```

Larger parameters make the network confident, and a confident network with random parameters is confidently wrong on most points, which cross-entropy punishes heavily (post 08, section 3). At scale 1.0 only 5 draws in 10,000 beat the uniform guess, and the best of them gets 114 of 300. At scale 10.0 none does.

The reason is geometric. Twenty-one parameters define a 21-dimensional space. Drawing a point of that space at random is like throwing a dart at a 21-dimensional board and hoping it lands in a winning region whose volume is a tiny fraction of the whole. The measurement above puts a number on the outermost ring of that target: the region where the network is merely better than not guessing holds 5 darts in 10,000 at scale 1.0. The region where it classifies well is smaller again, and 10,000 or 100,000 throws do not find it.

For modern networks the situation is far worse, and section 4 says how much worse. Random selection on 21 parameters is already useless.

---

## 3. Strategy 2: random perturbation

A better idea: instead of throwing away the current parameters every iteration, *adjust* them by a small random amount. If the loss goes below the best so far, keep the change. If not, revert. `snippets/random_perturbation.py` wraps the loop in a function so that it can be run on more than one dataset and network:

```python
    # The loss to beat is the loss of the starting parameters.
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    best_loss = loss_fn.calculate(activation2.output, y)
    best_correct = int(np.sum(np.argmax(activation2.output, axis=1) == y))
    print(f"  start: loss {best_loss:.4f}, {best_correct} of {len(y)} correct")

    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases  = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases  = dense2.biases.copy()

    for iteration in range(iterations):
        # Nudge every parameter by a small random amount.
        dense1.weights += 0.05 * np.random.randn(2, n_hidden)
        dense1.biases  += 0.05 * np.random.randn(1, n_hidden)
        dense2.weights += 0.05 * np.random.randn(n_hidden, 3)
        dense2.biases  += 0.05 * np.random.randn(1, 3)

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        loss = loss_fn.calculate(activation2.output, y)

        if loss < best_loss:
            # Keep the change: the nudged parameters become the new best.
            best_loss = loss
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases  = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases  = dense2.biases.copy()
            best_correct = int(np.sum(np.argmax(activation2.output, axis=1) == y))
            kept, last_kept = kept + 1, iteration + 1
        else:
            # Revert: go back to the best parameters seen so far.
            dense1.weights = best_dense1_weights.copy()
            dense1.biases  = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases  = best_dense2_biases.copy()
```

The difference from strategy 1 is one character per line: **`+=`** instead of `=`. The parameters drift around their current values instead of being replaced wholesale, so progress made in one iteration is still there in the next. The loss to beat starts at the loss of the untrained network, so a first nudge that makes things worse is undone like any other. With `n_hidden = 3` this is the same 21-parameter network.

**On an easy dataset it works.** `vertical_data` from the `nnfs` package draws three round clusters side by side, 100 points each, centred at 0, 1/3 and 2/3 along the horizontal axis. The clusters overlap a little; two vertical cuts halfway between the centres get 277 of 300.

```text
three side-by-side clusters (vertical_data), 3 hidden neurons
  two vertical cuts halfway between the cluster centres: 277 of 300 correct
  21 parameters
  start: loss 1.0986, 26 of 300 correct
  after     100 iterations: loss 0.9883, 200 of 300 correct
  after   1,000 iterations: loss 0.2034, 277 of 300 correct
  after  10,000 iterations: loss 0.1714, 279 of 300 correct
  kept 374 of 10,000 nudges, the last at iteration 7,607
  every rejected nudge is undone, so the final loss is also the lowest loss seen
```

The loss falls from 1.0986 to 0.1714 and the network gets 279 of 300 right, 93.0 percent, as good as the two hand-placed cuts. On this data the strategy solves the problem.

**On the spiral it gets much less far.**

```text
spiral, 3 hidden neurons
  21 parameters
  start: loss 1.0986, 102 of 300 correct
  after     100 iterations: loss 1.0872, 121 of 300 correct
  after   1,000 iterations: loss 1.0746, 132 of 300 correct
  after  10,000 iterations: loss 1.0435, 122 of 300 correct
  kept 147 of 10,000 nudges, the last at iteration 9,544
  every rejected nudge is undone, so the final loss is also the lowest loss seen
```

The loss reaches 1.0435 and 122 of 300 correct, 40.7 percent, a little above the 118 of 300 of the straight-line model of post 04. The walk is still finding the occasional improvement near the end, the last at iteration 9,544, but they have become rare: 147 nudges kept in 10,000.

Two readings of that result are possible, and they call for different fixes. Either the blind search is what holds the loss up, and a method that knew where downhill was would be far below 1.04 by now, or this is simply how far 10,000 iterations go on this network, whatever the method. Section 4 puts the question to a measurement.

One property of the method is visible already. It is **direction-blind**: it cannot tell which direction is downhill without trying one, and each try costs a full forward pass. On the easy data 374 of 10,000 tries were kept; on the spiral, 147. Even where the strategy succeeds, more than 96 percent of the forward passes are thrown away.

---

## 4. Why random search cannot scale

The method that knows the downhill direction is gradient descent, the subject of section 5. `snippets/gradient_descent_preview.py` runs it from the same starting parameters and on the same budget of 10,000 iterations as the random walk, borrowing the computation of the gradient from later posts. Both methods are run on the 21-parameter network and on a wider one with 64 hidden neurons, which has $2 \cdot 64 + 64 + 64 \cdot 3 + 3 = 387$ parameters and is the network post 22 trains. Gradient descent has one setting, the learning rate $\alpha$, a positive number that sets the size of each step (section 5 defines it), and it is run with two values: 1, the value post 22 starts from, and 0.1.

The two methods do not behave alike from step to step, so the same quantities are reported for both. The random walk undoes every nudge that does not help, so its loss never goes up and its final loss is also the lowest it has seen. Gradient descent keeps every step, and a step can raise the loss; for it the table gives the final loss, the lowest loss seen on the way, and the number of steps on which the loss went up.

| Spiral data, 10,000 iterations | 21 parameters | 387 parameters |
|---|:---:|:---:|
| Random perturbation, final loss (also its lowest) | 1.0435 | 1.0730 |
| Random perturbation, correct of 300 | 122 | 129 |
| Gradient descent, $\alpha = 1$, final loss | 1.0776 | 0.8737 |
| Gradient descent, $\alpha = 1$, lowest loss on the way | 1.0712 | 0.8343 |
| Gradient descent, $\alpha = 1$, steps on which the loss rose | 4,363 | 4,628 |
| Gradient descent, $\alpha = 1$, correct of 300 | 115 | 194 |
| Gradient descent, $\alpha = 0.1$, final loss | 1.0797 | 1.0226 |
| Gradient descent, $\alpha = 0.1$, lowest loss on the way | 1.0797 | 1.0226 |
| Gradient descent, $\alpha = 0.1$, steps on which the loss rose | 923 | 0 |
| Gradient descent, $\alpha = 0.1$, correct of 300 | 113 | 133 |

Four things can be read off the table.

**On 21 parameters and this budget, the blind walk is not behind.** It ends at 1.0435; gradient descent ends at 1.0776 with a learning rate of 1 and at 1.0797 with 0.1. How large that lead is depends on the walk's draws. Run with the argument `seeds`, `random_perturbation.py` repeats both spiral runs from the same starting parameters with 20 other streams of nudges:

```text
spiral, 3 hidden neurons, 20 streams of nudges, 10,000 iterations each
  final loss: lowest 1.0321, median 1.0684, highest 1.0793
  below gradient descent at learning rate 1 (1.0776): 16 of 20 streams
  below gradient descent at learning rate 0.1 (1.0797): 20 of 20 streams
spiral, 64 hidden neurons, 20 streams of nudges, 10,000 iterations each
  final loss: lowest 1.0442, median 1.0595, highest 1.0666
  below gradient descent at learning rate 1 (0.8737): 0 of 20 streams
  below gradient descent at learning rate 0.1 (1.0226): 0 of 20 streams
```

On the small network the walk ends anywhere between 1.0321 and 1.0793, so the printed 1.0435 is one of its luckier runs and the lead of 0.03 is not a result. What holds is that the walk ends below gradient descent's 1.0776 in 16 of the 20 streams. That answers the question of section 3: blindness was not what held the walk up, because over 10,000 iterations on the small network knowing the downhill direction gave no reliable advantage.

**That is a statement about the budget, not about the network.** Neither 1.04 nor 1.08 is the lowest loss three hidden neurons can reach. Run with the argument `long`, the two scripts give both methods ten times the budget on the 21-parameter network: gradient descent with $\alpha = 0.1$ sits near 1.0797 for a long stretch and then falls to 1.0049 and 137 of 300 correct after 100,000 steps, while the random walk moves from 1.0435 to 1.0403 and 125 of 300, keeping 27 more nudges in 90,000. In this pair of runs, with enough steps, the method that knows the direction pulls ahead on the small network too.

**On 387 parameters only gradient descent uses the extra room.** With $\alpha = 1$ it ends at 0.8737 and 194 of 300 correct, 64.7 percent; with $\alpha = 0.1$ at 1.0226 and 133. The random walk ends at 1.0730 and 129 of 300, keeping 37 of its 10,000 nudges, and that printed run is its unluckiest: the 20 other streams end between 1.0442 and 1.0666. Not one of them is below gradient descent at either learning rate. Eighteen times as many parameters lowered the final loss of gradient descent by 0.2039 at $\alpha = 1$ and by 0.0571 at $\alpha = 0.1$. They moved the walk's median from 1.0684 to 1.0595, less than the distance between two of its own streams, so the walk gained next to nothing from them. The figure at the top of the post puts all of these final losses on one axis, one tick per stream of the walk.

**The learning rate of 1 bounces.** With $\alpha = 1$ the loss went up on 4,363 of the 10,000 steps on the small network and on 4,628 on the wide one, where the run ends at 0.8737 although it had touched 0.8343 at step 7,874. This is the oscillation that section 5.1 produces on one parameter. With $\alpha = 0.1$ the wide network's loss never rises, and it gets less far in the same budget: 1.0226. A large step that bounces covered more ground here than a small one that descends steadily; both settings are shown so that neither is mistaken for the method. On the small network the loss at $\alpha = 0.1$ also ticks up on 923 steps, yet its final value is its lowest to four decimals.

Early in a run the picture is different again. After 100 iterations the random walk leads on both networks (1.0872 against 1.0982 on 21 parameters and 1.0857 against 1.0869 on 387, gradient descent at $\alpha = 1$): `Layer_Dense` starts its weights at a scale of 0.01, a nudge of 0.05 is five times that, and the first steps of gradient descent from such a start are small.

That is the measured form of the scaling argument: with few parameters and few steps the blind method loses nothing, and with eighteen times the parameters it is left far behind in every stream tried. Two general statements stand behind this.

**The budget buys little, and each parameter multiplies the price.** For random selection, the best loss in section 2 after 10, 100, 1,000, 10,000 and 100,000 draws was 1.0986, 1.0984, 1.0983, 1.0981 and 1.0980: each tenfold increase in the budget bought one or two ten-thousandths, so the return on the budget is roughly logarithmic. The cost in parameters runs the other way. As an illustration, suppose a good region required nothing more than each parameter landing in the correct half of its range. One draw would then hit it with probability $(1/2)^{21} = 1/2{,}097{,}152$ for 21 parameters, about one in two million, and with probability $(1/2)^{387} \approx 3 \times 10^{-117}$ for 387. Doubling the number of draws does double such a chance, but twice a chance of that size is still nothing. Under that assumption the number of draws needed doubles with every parameter added, and no budget keeps up. The figure below draws every parameter of the small network and then both networks to one scale.

![Top: the 21 parameters of the network with 3 hidden neurons, one cell each in their array shapes, W1 of shape (2, 3), b1 (1, 3), W2 (3, 3) and b2 (1, 3), with 6 plus 3 plus 9 plus 3 equals 21 underneath. Bottom: both networks as bars on one axis from 0 to 400 parameters, 21 long for 3 hidden neurons, labelled 6 plus 3 plus 9 plus 3 equals 21, and 387 long for 64, split into 128 weights in W1, 64 biases in b1, 192 weights in W2 and 3 biases in b2.](diagrams/02-parameter-count.svg)

*Every weight and every bias is one number the update rule has to move. If each had to land in the right half of its range by chance, one draw would succeed with probability 1 in 2,097,152 for 21 parameters and about $3 \times 10^{-117}$ for 387.*

**Random perturbation asks one question per forward pass.** It proposes a direction, pays for a forward pass, and learns a single fact: better or worse. In 21 dimensions that was competitive for 10,000 iterations. In 387 it kept 37 proposals in 10,000, and a network of millions of parameters has millions of directions to ask about. The fix is not to be cleverer about sampling; the fix is to know the downhill direction *before* taking the step.

That direction is given by **calculus**.

---

## 5. The gradient: the missing piece

For a function of several variables, the **gradient** is the vector of its partial derivatives, one slope per variable ([post 10](../10-derivatives-partial-derivatives-and-gradients/index.md) builds both ideas from zero). Write $\theta_1, \theta_2, \dots, \theta_{21}$ for the 21 parameters, weights and biases alike, and $L$ for the loss. Then

$$\nabla L = \left(\frac{\partial L}{\partial \theta_1},\ \frac{\partial L}{\partial \theta_2},\ \dots,\ \frac{\partial L}{\partial \theta_{21}}\right).$$

Each component says how fast the loss changes when that one parameter is increased and the others are held still. Two facts about the vector matter.

- **The direction of $\nabla L$ is the direction of steepest *ascent*** in parameter space; the loss increases fastest if every parameter moves in proportion to its component of $\nabla L$.
- **The direction of $-\nabla L$ is therefore the direction of steepest *descent*.** Moving every parameter against its component of the gradient reduces the loss as quickly as a small step can.

This is the foundation of **gradient descent**. At every iteration the update rule is

$$\theta \leftarrow \theta - \alpha \, \nabla L,$$

where $\theta$ stands for all the parameters at once, the arrow means "is replaced by", and $\alpha$ is the **learning rate**, a small positive number that sets how big the step is. Written for one parameter, the rule is $\theta_i \leftarrow \theta_i - \alpha \, \partial L / \partial \theta_i$: every weight and every bias moves against its own slope. The minus sign is what makes the algorithm a *descent* algorithm.

Compared with random perturbation, one thing has changed. The random walk learned one fact per forward pass. The gradient delivers the slope along all 21 axes, or all 387, at once, and backpropagation computes it with one forward pass and one backward pass.

The idea is older than neural networks. Cauchy described it in 1847 as a general method for solving systems of equations by repeatedly stepping downhill on a function of several variables (Cauchy, 1847). It became the engine of neural-network training when Rumelhart, Hinton, and Williams showed that **backpropagation**, an efficient algorithm for computing $\nabla L$ through the layers of a network, lets the rule train every layer at once (Rumelhart, Hinton, and Williams, 1986). Posts 12 to 21 derive backpropagation from scratch; the present post only motivates it.

### 5.1. One parameter, by hand

The rule can be tried without any of that machinery on a loss with a single parameter, $f(w) = w^2$. Its lowest point is at $w = 0$, and its slope at any $w$ is $2w$, a fact post 10 derives.

Start at $w = 5$, where the loss is 25 and the slope is 10. With $\alpha = 0.1$ the rule gives $w \leftarrow 5 - 0.1 \cdot 10 = 4$, and the loss falls to 16. No direction was tried and rejected: the slope was positive, so the step went the other way. With a plus sign in place of the minus the same step climbs, and the loss goes 36, 51.84, 74.65 on the first three steps, rising on every one.

`snippets/learning_rate_1d.py` repeats the step 100 times for five learning rates:

```python
for learning_rate in (0.001, 0.01, 0.1, 1.0, 10.0):
    w = 5.0
    path = []
    for step in range(100):
        w = w - learning_rate * slope(w)
        path.append(w)
    print(f"{learning_rate:<15} {path[0]:<11.4g} {path[1]:<11.4g} {path[2]:<11.4g} {path[99]:.4g}")
```

```text
learning rate   1 step      2 steps     3 steps     100 steps
0.001           4.99        4.98        4.97        4.093
0.01            4.9         4.802       4.706       0.6631
0.1             4           3.2         2.56        1.019e-09
1.0             -5          5           -5          5
10.0            -95         1805        -3.43e+04   3.753e+128
```

Because the slope is $2w$, each step replaces $w$ by $w - 2\alpha w = (1 - 2\alpha)\,w$, so 100 steps multiply the starting value by $(1 - 2\alpha)^{100}$. That one factor explains every row:

- $\alpha = 0.001$ gives a factor of 0.998 per step. The direction is right and the progress is negligible: after 100 steps $w$ is still 4.093. Training **crawls**.
- $\alpha = 0.1$ gives 0.8. After 100 steps $w$ is $10^{-9}$, which is as good as the minimum.
- $\alpha = 1$ gives $-1$. Every step jumps to the mirror-image point on the other side of the valley, and $w$ **oscillates** between $-5$ and $5$ for ever with the loss stuck at 25.
- $\alpha = 10$ gives $-19$. Every step overshoots further than the last and $w$ **diverges**, to $3.75 \times 10^{128}$ after 100 steps.

On this loss the rule converges exactly when $|1 - 2\alpha| < 1$, that is for $0 < \alpha < 1$. A real loss surface has no single threshold of this kind, since its steepness differs from place to place and from direction to direction, but the behaviours carry over to a real network: in section 4 the learning rate of 1 bounced, and 0.1 descended steadily and slowly.

The figure below draws four of the five runs on the curve $f(w) = w^2$: the start and the first three steps as dots, and the value after 100 steps as a ring.

![Four charts of f(w) = w squared for w from minus 6 to 6, each with the start at w = 5 and the first steps of gradient descent as dots joined by straight jumps, and w after 100 steps as a ring. With a learning rate of 0.001 the dots 5, 4.99, 4.98 and 4.97 sit on top of one another, as a note beside them says, and the ring sits at 4.093. With 0.1 the dots step down the right side, 5, 4, 3.2, 2.56, and the ring is at the minimum. With 1 the jumps run between 5 and minus 5 at a loss of 25. With 10 the first jump leaves the chart, labelled to w = minus 95.](diagrams/03-learning-rate.svg)

*The same rule and the same start; only the learning rate differs. Each step multiplies $w$ by $1 - 2\alpha$: 0.998, 0.8, $-1$ and $-19$.*

---

## 6. What gradient descent is *not*

A boundary section, because the algorithm has well-known limitations that later posts refine.

- **It is not guaranteed to find the global minimum.** Gradient descent walks downhill from wherever it starts and stops where the ground is flat. A loss surface in deep learning has many local minima (points lower than everything immediately around them, though not the lowest overall) and saddle points; reaching the best one is a research topic, not a guarantee.
- **It is not free.** Computing $\nabla L$ requires a backward pass through the entire network at every iteration. The backward pass costs about as much as the forward pass, not 21 or 387 times as much, which is the whole reason the method is affordable; posts 12 to 21 show where that economy comes from.
- **It is not random search wearing a suit.** Random perturbation and gradient descent differ in what they know before each step. Random perturbation knows nothing and tries blindly; gradient descent knows the slope of the loss with respect to every parameter before committing to a move.
- **It is not automatically ahead of a blind search.** Section 4 found a network small enough and a budget short enough for a random walk to keep pace with it.
- **It is not a descent on every step.** The name describes the direction of each step, not its outcome: with a learning rate of 1 the loss rose on more than four steps in ten in section 4. Posts 22 to 27 introduce optimisers that manage the step size.
- **It is not the final story.** Mini-batches, momentum, Adam, and the other techniques covered later refine the simple "step against the gradient" rule. They all start from the same gradient.

---

## 7. Why this post comes before the calculus

The choice of post order is deliberate. Many introductions begin with the calculus and arrive at "and therefore gradient descent" several chapters later. This series goes the other way: **first see why blind methods fail; then learn the tools that make targeted methods possible**.

Motivation arrives first, mechanism second. Posts 10 and 11 build the calculus toolkit, posts 12 to 21 derive and code backpropagation, and posts 22 to 27 introduce successively smarter optimisers. By the end of post 27, the spiral classifier is a real, trained network.

| Posts | Topic | What it adds |
|:---:|---|---|
| 10 | Derivatives and partial derivatives | the language for talking about slopes |
| 11 | The chain rule | the rule for composing slopes through a network |
| 12 to 15 | Backpropagation, derived | the gradient of the loss with respect to every layer's inputs and weights |
| 16 to 21 | Backpropagation, coded | a `backward` method on every class |
| 22 to 27 | Optimisers | better step rules: gradient descent, decay, momentum, AdaGrad, RMSProp, Adam |

Everything in this list is in service of the one update rule of section 5. The rule itself does not change; the methods for computing $\nabla L$ and for using it both get more sophisticated.

---

## 8. The hill-in-fog metaphor

A useful picture: a hiker stands on a hill in dense fog, holding a loss meter that reads the current height above sea level. The goal is to reach the lowest point.

| Algorithm | What the hiker does | Result |
|---|---|---|
| Random selection | Teleport to a random coordinate, check the loss, repeat. | Almost never useful: the bottom is a small region, and random teleports rarely land in it. |
| Random perturbation | Take a small random step. If lower, keep it. If higher, step back. | Works on a small, gentle hill; on a large one almost every step is taken back. |
| Gradient descent | Feel the ground underfoot (the gradient). Step in the steepest downhill direction. | Heads downhill from the first step, and keeps doing so however many directions there are. |

The metaphor breaks down in high dimensions (a 21-dimensional hill is hard to visualise), but the core intuition transfers. Gradient descent works for the same reason a hiker who can feel the slope under their boots gets down a mountain faster than one who teleports at random: information about direction is what makes the difference.

---

## 9. Make it run: four scripts, every number

Every number in this post is printed by a script in `snippets/`. Each runs from the series root, for example `python posts/09-introduction-to-optimisation/snippets/random_selection.py`, and is seeded, so it prints the same output on every run. The three that use the network call `nnfs.init()` (seed 0, `float32`) and need the `nnfs` package (`pip install nnfs`) besides NumPy. The fourth needs only the standard library. The times are those of an idle laptop CPU; a busy machine took up to four times as long.

| Command | Sections | What it prints | Time |
|---|---|---|---|
| `random_selection.py` | 2, 4 | the 21 parameters, the untrained loss, the best loss after 10 to 10,000 draws, and the two wider searches | 5 s |
| `random_selection.py 100000` | 2, 4 | the same search with 100,000 draws, ending at loss 1.0980 and 110 of 300 correct; the two wider searches keep their 10,000 draws | 20 s |
| `random_perturbation.py` | 3, 4 | the three runs of random perturbation: easy data, spiral with 21 parameters, spiral with 387 | 6 s |
| `random_perturbation.py long` | 4 | 100,000 iterations of the walk on the spiral with 21 parameters | 15 s |
| `random_perturbation.py seeds` | 4 | both spiral runs repeated with 20 other streams of nudges: the block quoted in section 4 | 80 s |
| `gradient_descent_preview.py` | 4 | gradient descent on the spiral with 21 and with 387 parameters, each at learning rates 1 and 0.1 | 15 s |
| `gradient_descent_preview.py long` | 4 | 100,000 steps of gradient descent at learning rate 0.1 on the spiral with 21 parameters | 25 s |
| `learning_rate_1d.py` | 5.1 | one step on $f(w) = w^2$, the plus-sign climb, the five learning rates, and the closed form $5\,(1 - 2\alpha)^{100}$ for each | under 1 s |

The 387-parameter run of `random_perturbation.py` prints:

```text
spiral, 64 hidden neurons
  387 parameters
  start: loss 1.0986, 108 of 300 correct
  after     100 iterations: loss 1.0857, 114 of 300 correct
  after   1,000 iterations: loss 1.0789, 118 of 300 correct
  after  10,000 iterations: loss 1.0730, 129 of 300 correct
  kept 37 of 10,000 nudges, the last at iteration 6,489
  every rejected nudge is undone, so the final loss is also the lowest loss seen
```

`gradient_descent_preview.py` prints:

```text
spiral, 3 hidden neurons (21 parameters), learning rate 1.0
  after       0 steps: loss 1.0986, 102 of 300 correct
  after     100 steps: loss 1.0982, 114 of 300 correct
  after   1,000 steps: loss 1.0797, 113 of 300 correct
  after  10,000 steps: loss 1.0776, 115 of 300 correct
  lowest loss on the way: 1.0712, at step 3,851
  the loss went up on 4,363 of 10,000 steps
spiral, 3 hidden neurons (21 parameters), learning rate 0.1
  after       0 steps: loss 1.0986, 102 of 300 correct
  after     100 steps: loss 1.0986, 112 of 300 correct
  after   1,000 steps: loss 1.0982, 115 of 300 correct
  after  10,000 steps: loss 1.0797, 113 of 300 correct
  lowest loss on the way: 1.0797, at step 9,999
  the loss went up on 923 of 10,000 steps
spiral, 64 hidden neurons (387 parameters), learning rate 1.0
  after       0 steps: loss 1.0986, 108 of 300 correct
  after     100 steps: loss 1.0869, 120 of 300 correct
  after   1,000 steps: loss 1.0623, 120 of 300 correct
  after  10,000 steps: loss 0.8737, 194 of 300 correct
  lowest loss on the way: 0.8343, at step 7,874
  the loss went up on 4,628 of 10,000 steps
spiral, 64 hidden neurons (387 parameters), learning rate 0.1
  after       0 steps: loss 1.0986, 108 of 300 correct
  after     100 steps: loss 1.0985, 127 of 300 correct
  after   1,000 steps: loss 1.0867, 119 of 300 correct
  after  10,000 steps: loss 1.0226, 133 of 300 correct
  lowest loss on the way: 1.0226, at step 10,000
  the loss went up on 0 of 10,000 steps
```

The two long runs on the 21-parameter network repeat the first 10,000 iterations of the runs above and then end with:

```text
  after 100,000 iterations: loss 1.0403, 125 of 300 correct
  kept 174 of 100,000 nudges, the last at iteration 97,719
```

```text
  after 100,000 steps: loss 1.0049, 137 of 300 correct
  lowest loss on the way: 1.0049, at step 96,997
  the loss went up on 25,028 of 100,000 steps
```

The preview script computes the gradient with nine lines that this post has not derived: the gradient of the loss with respect to the logits from post 19, the dense-layer gradients from post 16, and the ReLU gradient from post 17. Its update is the rule of section 5, applied to the four parameter arrays. Nothing else in this post depends on those lines; they are there to put a number on what knowing the direction is worth, and post 22 rebuilds the same loop from classes.

All the runs on the spiral start from the same data and the same initial parameters, because each script replays the random stream from seed 0 before it builds a network. The `seeds` mode keeps that start and reseeds only the stream the nudges are drawn from.

---

## 10. What can go wrong?

- **The comparison is run in one setting only.** One network and one budget can make either method look like the winner: after 100 iterations the random walk leads on both networks, and section 4 needed the long run and the wide network before the two methods separated. A result quoted without its network, step size and budget cannot be compared with another.
- **The two methods are scored differently.** The random walk can only report its best loss so far; gradient descent naturally reports its last. On a run that bounces these differ (0.8343 against 0.8737 in section 4), so a fair table states which one it shows.
- **The loss to beat starts at infinity.** If `best_loss` is initialised to `float('inf')` in the random walk, the first nudge is always kept, even when it raises the loss above that of the starting parameters. The loop of section 3 starts from the loss of the starting parameters instead.
- **One seed is one run.** The tables print a single seeded run of each method. The `seeds` output of section 4 shows what that hides for the walk: from one stream of nudges to the next its final loss moves in the second decimal, from 1.0321 to 1.0793 on 21 parameters. A difference of 0.03 between two walk results is therefore not a finding; the 0.2 by which gradient descent leads on 387 parameters is. Gradient descent draws nothing at random once it has started, but its result depends on the starting parameters, and those were not varied here.
- **The revert restores the wrong values.** Random perturbation must go back to the parameters as they were *before* the nudge. In the loop of section 3 those are the `best_...` copies, taken with `.copy()`. Without the copy, `best_dense1_weights` is a second name for the very array that `+=` goes on changing, the revert does nothing, and the walk keeps every bad step.
- **The plus sign.** Stepping along the gradient instead of against it climbs. The symptom is unmistakable, a loss that rises on every iteration, as in the 36, 51.84, 74.65 of section 5.1.
- **The gradient is zero away from a minimum.** At a saddle point the loss rises in some directions and falls in others, and the gradient is exactly zero there, so the rule stops moving. In high dimensions saddle points are thought to be far more common than poor local minima, and the momentum of post 24 is one way to keep moving through a flat region.
- **The gradient cannot be written down.** Then it can be estimated numerically, by nudging one parameter at a time and measuring the change in the loss. Post 10 defines that finite-difference estimate. It costs at least one forward pass per parameter, 21 or 387 per step here, which is why this series uses it only to check the gradients that backpropagation computes.

Every step in this post computes the loss on all 300 points; computing it on a random subset at each step, the mini-batch of post 32, is what the word *stochastic* in stochastic gradient descent refers to.

---

## 11. Summary

| Concept | Takeaway |
|---|---|
| 21 parameters | Even the toy network poses a 21-dimensional optimisation problem |
| Random selection | 10,000 draws move the loss from 1.0986 to 1.0981; wider draws are confidently wrong |
| Random perturbation | Solves the easy data (279 of 300); on the spiral its final loss stays between 1.03 and 1.08 with 21 parameters or with 387 |
| Small network, small budget | After 10,000 iterations on 21 parameters the walk is not behind gradient descent: below its 1.0776 in 16 of 20 streams |
| More steps | After 100,000 on 21 parameters, in the one pair of runs made, gradient descent leads: 1.0049 against 1.0403 |
| More parameters | On 387, gradient descent reaches 0.8737 (1.0226 at the steadier $\alpha = 0.1$); the walk stays above 1.04 in every stream |
| Gradient | The vector of partial derivatives; its negative is the steepest-descent direction |
| Update rule | $\theta \leftarrow \theta - \alpha \, \nabla L$, with $\alpha$ the learning rate that sets the step size |
| What comes next | Calculus (posts 10 and 11), backpropagation (12 to 21), optimisers (22 to 27) |

---

## Common pitfalls

1. **Reading one comparison as a verdict on a method.** A result belongs to a method, a network, a step size, a budget and a seed together; section 4 reverses its first impression twice by changing one of them.
2. **Setting the learning rate by intuition.** On the 387-parameter network a learning rate of 1 made the loss rise on 4,628 of 10,000 steps and still ended lower than a rate of 0.1 that never rose. Neither could have been guessed; confirm with a scan over several values or start from a known default (Adam typically uses `1e-3`).
3. **Reading "subtract the gradient" as subtracting one number.** $\nabla L$ has one component per parameter, and each parameter is moved by its own component. The 21 parameters receive 21 different steps.
4. **Minimising something that has no slope.** The rule needs a loss that changes smoothly with the parameters. This is why post 08 built cross-entropy and did not train on accuracy: accuracy stays flat and then jumps, and a flat stretch gives the gradient nothing to report. A low training loss is also not the end of the story; whether it carries over to new data is the subject of [post 28](../28-generalization-and-testing/index.md).

---

## Further reading

- Cauchy, A.-L., *"Méthode générale pour la résolution des systèmes d'équations simultanées"* (Comptes rendus de l'Académie des sciences, 1847).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 4 (Numerical Computation) and chapter 8 (Optimization for Training Deep Models) (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 6 (2020). The two random strategies and the `vertical_data` example come from this chapter.
- Rumelhart, D., Hinton, G., and Williams, R., *"Learning Representations by Back-Propagating Errors"* (Nature, 1986).
- Ruder, S., *"An Overview of Gradient Descent Optimization Algorithms"* (arXiv:1609.04747, 2016).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 10 - Derivatives, partial derivatives, and gradients](../10-derivatives-partial-derivatives-and-gradients/index.md):** the calculus toolkit behind $\nabla L$, including the finite-difference estimate of a slope.
- **[Post 22 - Gradient-descent optimiser](../22-gradient-descent-optimiser/index.md):** the update rule of this post as an `Optimizer_SGD` class, trained on the spiral with the gradients of posts 12 to 21.
