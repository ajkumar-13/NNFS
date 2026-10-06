# 31 - Dropout

> **TL;DR.** Dropout sets each activation of a layer to zero with probability $p$ on every training forward pass and divides the survivors by $1 - p$, so the expected activation is unchanged and evaluation uses the layer as the identity. `Layer_Dropout` is three short methods, and the `training` argument of its `forward` is the switch between the two modes. On the 64-neuron spiral network of post 28 it did not help: at $p = 0.1$ the test accuracy fell on each of ten seeds, by 9.33 to 31.33 points, because the network stopped fitting its training data. Test accuracy exceeded training accuracy only where the training figure was measured through the mask; with the mask off, training accuracy was the higher one in every run.
>
> **Prerequisites:** [Post 28](../28-generalization-and-testing/index.md).
> **Safe to skip?** Skip it if the reader can already write an inverted-dropout layer with its backward pass from memory, say why evaluation runs it as the identity, and say why a training accuracy measured through the mask cannot be compared with a test accuracy.
>
> **After reading, you will be able to:**
>
> - Explain why random masking attacks co-adaptation and why the ensemble view makes the gain unsurprising.
> - Implement Layer_Dropout with the inverted convention: scale by 1/(1 - p) in training, identity at test time.
> - Wire dropout into the forward and backward passes of the pipeline with the train-versus-test switch.

![Two panels show the same row of eight neurons with activations of 1. In the training panel two neurons are crossed out and output 0 while the other six output 1.33, and the outputs sum to 8. In the testing panel all eight output 1 and sum to 8. A strip below contrasts training=True, mask and scale, with training=False, identity.](diagrams/01-dropout-train-vs-test.svg)

*One layer, two modes, at a drop rate of 0.25. In training a fresh mask zeroes some neurons and the rest are divided by 0.75; in evaluation the layer passes its input through. The six survivors sum to 8 on this draw; in general only the expected sum is preserved (section 4).*

---

## 1. The question: what does zeroing random activations do?

[Post 28](../28-generalization-and-testing/index.md) listed four levers against overfitting and left the last one unmeasured: dropout. A weight penalty acts on the loss. Dropout acts on the forward pass: on every training pass each activation of a layer is set to zero with probability $p$, and at evaluation nothing is removed. The question of this post is: **what does zeroing random activations during training do to a network, and why must evaluation leave every neuron on?**

Hinton and co-authors, who introduced the method in 2012, described the failure it is aimed at as **co-adaptation**: a hidden neuron that is useful only in combination with particular other neurons. Two pictures make the idea concrete; both illustrate that paper's argument, and neither is measured here.

**A neuron that leans on another.** Suppose neurons $A$ and $B$ learn similar features. Gradient descent can settle on a pair in which $B$ contributes only a small correction to what $A$ already does. On the training data the pair fits. On new data, where $A$ responds a little differently, the correction is tuned to the wrong value.

**A neuron that stores an example.** A wide layer can spend a neuron on a handful of training points, firing for them and for almost nothing else. That neuron lowers the training loss and says nothing about a point it has not seen.

Both are dependencies that the training set rewards and new data does not. Dropout makes them unreliable instead of forbidding them: $B$ cannot count on $A$, because on any pass $A$ may be missing, and the neuron that stores a point may itself be missing when that point comes round. What survives such training is a set of features that still help when a random part of the layer is gone.

---

## 2. Two readings: co-adaptation and the implicit ensemble

The papers give two explanations of why this helps. They describe the same algorithm.

**Co-adaptation.** The first is the mechanical one of section 1: no weight can become critical to the output, because the pass that needs it may not include its neuron.

**An ensemble that shares its weights.** The second is statistical. A layer of $n$ neurons has $2^n$ subsets, so a network with one dropout layer contains $2^n$ **thinned networks**, one for each choice of which neurons stay on. The count does not depend on $p$; the rate only sets how often each pattern is drawn. A training pass draws a thinned network and takes one optimiser step on it, and since all of them use the same weight arrays, the step changes every other thinned network too. The 64-neuron layer of this series has $2^{64} \approx 1.8 \times 10^{19}$ of them; the point is the sharing, not the coverage.

![A layer of five units, labelled as having 32 subsets, above four training steps that each cross out a different subset, with 3, 3, 4 and 2 units left active. A card beside them says that every step updates one subnetwork and that all subnetworks share one set of weights.](diagrams/02-implicit-ensemble.svg)

*A mask chooses which of the $2^n$ thinned networks takes the step. The figure draws one mask per step; the class of section 5 draws one per sample, as section 5 explains.*

Averaging the predictions of many models is a standard way to reduce test error, at the cost of one training run and one forward pass per model. The ensemble reading says that the full network stands in for an average over its thinned networks at the cost of a single pass. For a network with one hidden layer feeding a softmax, Hinton et al. (2012) state the relation exactly: the full network outputs the renormalised geometric mean of the class probabilities of all $2^n$ thinned networks. The reason is one line. The logits are linear in the mask, so with the scale of section 4 their expectation over masks is the logits of the full network, and the softmax of a mean of logits is the renormalised geometric mean of the softmaxes.

`snippets/ensemble.py` checks it by enumeration on a network small enough to list: 2 inputs, 8 ReLU neurons, 3 classes, random parameters at scale 1, $p = 0.5$, all 256 masks on the 300 spiral points.

```text
masks: 256, their probabilities sum to 1.000000; points: 300; p = 0.5
mean of the logits over the masks, against the full network:        largest gap 8.4e-15
renormalised geometric mean of the probabilities, against the full: largest gap 2.4e-15
arithmetic mean of the probabilities, against the full:             largest gap 0.2293
points where the arithmetic mean and the full network pick the same class: 300 of 300
one thinned network against the full network: it picks another class on 6.3 to 76.7 percent of the points (mean over the masks 37.5)
```

The first two lines are equalities up to rounding. The third shows what the identity is not: the full network is not the arithmetic mean of the thinned networks' probabilities, which differs from it by up to 0.23 in a probability, although the two choose the same class on all 300 points here. The last line is the reason the switch of section 7 matters: a single thinned network disagrees with the full one on 37.5 percent of the points on average.

The identity needs the dropped layer to feed the softmax through a linear layer. With a non-linearity after the dropout layer, as in a deeper network, the full network is only an approximation to the average; Srivastava et al. (2014) report that the approximation works well in practice. Neither reading says how much a given network gains; section 8 measures it.

---

## 3. The dropout rate

The method has one hyperparameter, the **dropout rate** $p$: the probability that an activation is set to zero on a training pass. Each entry is drawn on its own, so the number of neurons one sample keeps is itself random. For the 64-neuron layer, from `snippets/layer_dropout.py`:

```text
   p   mean kept   standard deviation   P(all 64 kept)   scale 1 / (1 - p)
 0.1        57.6                 2.40         1.18e-03               1.111
 0.2        51.2                 3.20         6.28e-07                1.25
 0.5        32.0                 4.00         5.42e-20                   2
 0.8        12.8                 3.20         1.84e-45                   5
```

The columns are closed forms: the mean is $64(1 - p)$, the standard deviation $\sqrt{64 p (1 - p)}$, and the chance that a sample passes untouched is $(1 - p)^{64}$. Even at $p = 0.1$ only about one sample in 850 sees the whole layer.

The original papers used $p = 0.5$ for hidden layers and a smaller rate, 0.2, on the inputs, in networks with hundreds or thousands of neurons per layer. This post uses 0.1, the rate of the released projects of this series (`nn-p01`, `nn-p03`), and section 8 also runs 0.2 and 0.5. The rate is a hyperparameter like $\lambda$ or the learning rate, and it is chosen on validation data ([post 29](../29-validation-and-hyperparameter-tuning/index.md)), never on the test set.

---

## 4. The inverted-dropout scale

Setting a fraction $p$ of a layer's activations to zero lowers their expected sum by the same fraction. The next layer then learns weights that suit inputs of that reduced size, and at evaluation, with every neuron on, it receives inputs that are larger by a factor $1/(1 - p)$: 25 percent larger at $p = 0.2$. Something has to rescale, and there are two places to do it.

**At evaluation.** The papers train with the bare mask and, at test time, multiply by the keep probability $1 - p$. Every evaluation pass then carries an extra step.

**In training.** **Inverted dropout** divides the surviving activations by $1 - p$ during training. With a mask entry $m \in \{0, 1\}$ that is 1 with probability $1 - p$, the output for an activation $a$ is $a m / (1 - p)$ and

$$\mathbb{E}\left[\frac{a \, m}{1 - p}\right] = \frac{a (1 - p)}{1 - p} = a, \qquad \text{Var}\left[\frac{a \, m}{1 - p}\right] = a^2 \, \frac{p}{1 - p}$$

The expected input of the next layer is the same in both modes, so evaluation is the identity. This is the convention of this series. The variance is the price: the layer's output is right on average and noisy on every pass, and at $p = 0.5$ the standard deviation of an entry equals the entry.

`snippets/layer_dropout.py` runs both statements. Five activations of 1 at $p = 0.2$, with a seed whose mask drops one of them:

```text
mask of 0 and 1:        [1 1 1 1 0]  sum of a * mask: 4.0
mask / (1 - p):         [1.25 1.25 1.25 1.25 0.  ]  sum: 5.0
```

The four survivors carry 1.25 each and the sum is back at 5. That is one draw; a mask that kept all five would give 6.25. The claim is about the mean, here over 200,000 masks:

```text
activations a:               [0.5 1.  2.  0.  3. ]
mean, mask only:             [0.3998 0.8    1.5999 0.     2.3973]  closed form (1 - p) * a: [0.4 0.8 1.6 0.  2.4]
mean, mask / (1 - p):        [0.4997 1.     1.9999 0.     2.9966]  closed form a
variance, mask / (1 - p):    [0.0626 0.25   1.0002 0.     2.2577]  closed form a^2 * p / (1 - p): [0.0625 0.25   1.     0.     2.25  ]
```

The bare mask shrinks every mean by the factor 0.8, the scaled mask returns the input to within sampling error, and the variance matches its closed form.

---

## 5. The `Layer_Dropout` class

The layer has no parameters. Its forward pass draws a mask, scales it and multiplies; its backward pass multiplies by the same array.

```python
class Layer_Dropout:

    def __init__(self, rate):
        # 'rate' is the drop probability p. The arithmetic needs the keep
        # probability 1 - p, so that is what the layer stores.
        self.rate = 1 - rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()                         # evaluation: identity, no draw
            return
        # Inverted dropout: a fresh mask of 0 and 1 / (1 - p) on every call.
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask               # the mask of the last training forward
```

**The argument is the drop probability, the attribute is the keep probability.** `Layer_Dropout(0.2)` drops 20 percent and stores `self.rate = 0.8`, because both the draw and the division need $1 - p$. The glossary, the cheatsheet and the released projects use the same convention.

**The mask has the shape of the input.** `np.random.binomial(1, self.rate, size=inputs.shape)` draws one 0 or 1 per entry, so for a batch of 300 samples and 64 neurons it draws 19,200 values and every sample gets its own pattern: one pass trains up to 300 thinned networks at once. Run on the network of section 6, `snippets/network.py` prints:

```text
training mode:   output (300, 64), mask values [0.         1.11111111], entries dropped 1927 of 19200 (0.1004), loss 1.0986
rows of the mask that differ from row 0: 299 of 299
```

Despite its name, `binary_mask` holds 0 and $1/(1 - p)$: the scale is folded into the mask.

**A new mask on every call.** The draw is inside `forward`. A stored mask reused across epochs would hand every sample the same thinned network each time, a fixed pattern the weights can adapt to.

**The backward pass is one line.** In training mode the layer computes $o = a \, m'$ entry by entry, with $m'$ the stored mask of 0 and $1/(1 - p)$. Then $\partial o / \partial a = m'$, and the chain rule of [post 16](../16-coding-backpropagation/index.md) gives `dinputs = dvalues * binary_mask`: a dropped entry passes back zero, a kept one passes back its gradient times the same $1/(1 - p)$ it was scaled by going forward. As in every layer of the series, `backward` stores `dinputs` and returns nothing.

```text
mask:    [1.25 1.25 1.25 1.25 0.  ]
dinputs: [ 0.125 -0.25   0.375  0.5   -0.   ]
```

**`training=False` makes the layer the identity.** No mask is drawn, the output is a copy of the input, and the random stream is left where it was:

```text
output: [7. 7. 7. 7. 7.]  equals the input: True  random stream untouched: True
```

The default is `training=True`, so a call that omits the argument drops. The class matches the reference implementation in `nn-p01` except for one check: the project's constructor rejects a rate outside $[0, 1)$, and this class does not (section 11).

---

## 6. Dropout in the network

The layer goes after the activation and before the next dense layer, so that it acts on the hidden layer's outputs:

```text
X -> Layer_Dense(2, 64) -> ReLU -> Layer_Dropout(0.1) -> Layer_Dense(64, 3) -> softmax and loss
```

The loop is the one of posts 27 and 28 with one object and two calls added. From `snippets/network.py`:

```python
    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dropout1.forward(activation1.output, training=True)
        dense2.forward(dropout1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
```

`dense2` now reads `dropout1.output` and `activation1.backward` reads `dropout1.dinputs`: the layer sits in both directions of the chain, in mirrored order. The optimiser is not told about it, since the layer has no parameters. The `loss` and `accuracy` of this loop are measured through the mask of that epoch; section 8 shows how far that puts them from the figures of the same weights with the mask off. The dense layers are those of post 16.

**Checking the backward pass.** A network with a dropout layer is a random function: two forward passes draw two masks, so a central difference across them would compare two different networks. The check therefore draws one mask, keeps it, and differences the loss of that one thinned network, with $h = 10^{-5}$ in float64 and without `nnfs.init()`, as in post 21. `snippets/gradient_check.py`, on 2 inputs, 4 ReLU neurons, `Layer_Dropout(0.5)` and 3 classes with parameters at scale 1:

```text
gradient         largest relative error   largest absolute gap
dense2.dweights  2.9e-09                  2.8e-11   pass
dense2.dbiases   2.4e-10                  2.0e-11   pass
dense1.dweights  4.3e-09                  3.1e-11   pass
dense1.dbiases   2.9e-10                  4.5e-11   pass
mask values [0. 2.], entries dropped 595 of 1200

== Seeds 0 to 9: the largest relative error over the four arrays
largest relative error 1.1e-06, largest absolute gap 4.6e-11, seeds above the pass mark: [2]
```

Nine of the ten seeds are below the pass mark of $10^{-7}$. Seed 2 is above it with an absolute gap of at most $4.6 \times 10^{-11}$, the rounding level of the passing checks: the tiny-gradient case of post 21, not a wrong formula. Section 11 runs the same check on three wrong `backward` methods.

---

## 7. The train-versus-test switch

The rule is short: the mask is on in the passes that are followed by an update, and off in every pass whose figures are reported.

| Pass | `training=` | The dropout layer | The network that runs |
|---|:---:|---|---|
| Training, forward | `True` | draws a mask, scales by $1/(1 - p)$ | a different thinned network for every sample |
| Training, backward | none: `backward` takes no flag | multiplies by the stored mask | the same thinned networks |
| Validation or test, forward | `False` | identity | the full network, the same on every pass |

In this series the switch is an argument of one method of one layer. `Layer_Dropout.forward` takes `training`, the other layers do not know about it, and the caller passes it. In the released projects a model object's `forward(X, y, training)` hands the flag on to each dropout layer. The test pass of post 28 gains one argument:

```python
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output, training=training)
    dense2.forward(dropout1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
```

These lines are the body of `evaluate` in `snippets/network.py`, whose `training` argument defaults to `False`. It is still forward-only.

**Why evaluation leaves every neuron on.** Three reasons, each from an earlier section. The full network is the ensemble of section 2, and a pass through a mask reports one random member of it. The scale of section 4 was chosen so that the full network needs no correction. And a figure that changes from pass to pass cannot be compared with anything.

**`backward` has no switch and must follow a training forward.** It multiplies by `self.binary_mask`, which only a training-mode `forward` writes. Called after an evaluation pass it uses the mask of an earlier pass, or fails if there has been none (section 11). In a correct loop the situation does not arise, because evaluation never calls `backward`.

---

## 8. Results on the spiral

**The setup**, printed by every training script: `nnfs.init()` once (seed 0, float32, and its own `np.dot`); then for each seed $s$, `np.random.seed(s)`, `spiral_data(samples=100, classes=3)`, `Layer_Dense(2, 64)`, ReLU, the dropout layer, `Layer_Dense(64, 3)`, the combined softmax and loss, weights at `0.01 * randn`, a test set from a second `spiral_data` call, `Optimizer_Adam(learning_rate=0.02, decay=1e-5)` and 10,001 full-batch epochs. This is the documented run of post 28 with one layer added, and that run is the baseline. Post 28 drew its test points after training; the loop now draws masks, so they are drawn just before it, which gives the same 300 points. In the baseline a pass-through object, `No_Dropout`, stands where the dropout layer will stand, and `seeds_none.py` reproduces post 28's figures digit for digit: 96.33 and 82.33 percent on seed 0, and 78.00 to 96.33 against 67.33 to 82.67 percent over seeds 0 to 4.

**Three figures per run**, all measured forward-only after the last update, on the same weights:

- the training accuracy **through the mask**, as the mean over 100 fresh masks: what a training loop prints;
- the training accuracy with the **mask off**;
- the test accuracy with the mask off.

Each cell is the range over seeds 0 to 4, in percent; the last two columns are differences in points.

| Dropout layer | Script | Training, mask on | Training, mask off | Test | Mask off minus test | Mask on minus test |
|---|---|:---:|:---:|:---:|:---:|:---:|
| none | `seeds_none.py` | 78.00 to 96.33 | 78.00 to 96.33 | 67.33 to 82.67 | 10.67 to 14.00 | 10.67 to 14.00 |
| $p = 0.1$ | `seeds_rate_10.py` | 53.07 to 69.50 | 58.00 to 74.33 | 54.00 to 66.33 | 2.00 to 11.67 | $-2.93$ to 6.92 |
| $p = 0.2$ | `seeds_rate_20.py` | 54.81 to 64.88 | 61.33 to 72.67 | 50.67 to 63.00 | 6.67 to 11.33 | $-0.89$ to 5.88 |
| $p = 0.5$ | `seeds_rate_50.py` | 56.11 to 58.44 | 64.33 to 67.00 | 53.33 to 62.33 | 3.00 to 11.00 | $-4.97$ to 2.78 |

**Dropout lowered the test accuracy on every seed, at every rate.** `snippets/jobs/paired.py` runs seeds 0 to 9 and prints every seed-by-seed count of this section. Against the baseline of the same seed, the test accuracy is lower on ten of ten seeds at each rate: by 9.33 to 31.33 points at $p = 0.1$, by 4.33 to 32.00 at $p = 0.2$ and by 5.00 to 23.67 at $p = 0.5$. The test loss, in contrast, is lower with dropout on 2 to 4 of the ten seeds at each rate: the baseline is confidently wrong on some test points (post 28, section 4), and the dropout networks are less confident.

**The cause is under-fitting, not better generalisation.** With the mask off, the training accuracy at $p = 0.1$ is 13.00 to 38.33 points below the baseline's over the ten seeds: 55.00 to 74.33 percent where the baseline has 78.00 to 96.33. The network no longer fits the data it is trained on. The gap between training and test accuracy does narrow, on all ten seeds at $p = 0.1$, and it narrows because the training figure falls further than the test figure. This is the pattern post 28 measured when it cut the layer to 8 neurons: a smaller gap and a worse model. The number of dead neurons is no explanation: 10 to 26 of the 64 without dropout and 11 to 27 at $p = 0.1$, over the ten seeds.

**Why so small a rate costs so much.** The last line of `seeds_none.py` scores each baseline network through a `Layer_Dropout(0.1)` it was not trained with: the training accuracy is then 41.10 to 47.73 percent where the mask-off figure is 78.00 to 96.33, and the loss 7.37 to 8.09 where it was 0.08 to 0.47. The fit found without dropout does not survive the mask. By section 4 a mask adds noise of variance $\frac{p}{1 - p} \sum_j w_{kj}^2 a_j^2$ to logit $k$, so the larger the terms a logit is summed from, the more a mask disturbs it, and training with the mask leaves such a fit: `jobs/paired.py` trains the fitted networks of seeds 0 to 4 for 10,001 more epochs with the mask on, and the mask-off training accuracy falls on all five, by 9.33 to 32.33 points.

**The mask alone costs 3 to 11 points of training accuracy.** On the same weights, the training accuracy through the mask is below the mask-off figure in every run: by 3.31 to 8.49 points at $p = 0.1$ and by 6.13 to 10.88 points at $p = 0.5$, over ten seeds. This difference says nothing about new data. It is the handicap of scoring thinned networks, the disagreement that section 2 measured on a small network.

**Test accuracy above training accuracy is an effect of that handicap.** The claim that goes with dropout, a test accuracy higher than the training accuracy, appears in these runs only when the training figure is taken through the mask: on 4 of ten seeds at $p = 0.1$, 4 of ten at $p = 0.2$ and 9 of ten at $p = 0.5$. With the mask off on both sides, the training accuracy is above the test accuracy in all thirty runs. A negative gap between a masked training figure and an unmasked test figure compares two different networks, and is not a sign of generalisation.

**With ten times the data the picture is sharper.** `snippets/jobs/more_data.py` repeats the first two rows with 1,000 samples per class in both sets, seeds 0 to 4:

```text
no dropout layer; 64 neurons; 1000 samples per class; learning rate 0.02; seeds (0, 1, 2, 3, 4)
train accuracy, mask on 84.50 to 88.93, mask off 84.50 to 88.93 percent; test accuracy 81.47 to 87.77 (mean 85.20) percent
mask off minus test 1.17 to 3.03 (mean 1.89) points; mask on minus test 1.17 to 3.03 (mean 1.89) points; test loss 0.329 to 0.441

Layer_Dropout(0.1); 64 neurons; 1000 samples per class; learning rate 0.02; seeds (0, 1, 2, 3, 4)
train accuracy, mask on 44.29 to 59.90, mask off 50.97 to 66.50 percent; test accuracy 50.67 to 65.13 (mean 56.86) percent
mask off minus test 0.30 to 1.37 (mean 0.79) points; mask on minus test -6.38 to -5.23 (mean -5.97) points; test loss 0.791 to 1.005
runs whose test accuracy is above the mask-on training accuracy: 5 of 5; above the mask-off training accuracy: 0 of 5
```

With 3,000 training points the gap without dropout is already 1.17 to 3.03 points, so there is little left to regularise. Dropout then lowers the test accuracy by 22.63 to 35.23 points, seed by seed. The test accuracy is above the masked training accuracy in all five runs, by 5.23 to 6.38 points, and below the mask-off training accuracy in all five, by 0.30 to 1.37 points. The whole of the "negative gap" is the mask.

**A step ten times smaller does not rescue it.** The mask makes every gradient noisy, and the learning rate of 0.02 was chosen without that noise, so the optimiser is the first suspect. `snippets/jobs/smaller_step.py` repeats the first two rows at a learning rate of 0.002. Without dropout the test accuracy is 75.67 to 84.67 percent over seeds 0 to 4; with $p = 0.1$ it is 45.67 to 52.33 percent, lower by 25.33 to 39.00 points seed by seed. The damage is not a matter of this step size, nor of dead neurons: these runs have 5 to 14 without dropout and 3 to 13 with it. Whether a much longer schedule would undo it is not measured here. Seed 1 of these runs is the only run of this post whose masked training figure, 49.50 percent, is not below the mask-off one, 49.00.

**With eight times the neurons, dropout stops hurting and does not measurably help.** `snippets/jobs/wider.py` uses 512 hidden neurons in place of 64. A width changes how many draws the weights take, so its test points are not those of the rows above.

```text
no dropout layer; 512 neurons; 100 samples per class; learning rate 0.02; seeds (0, 1, 2, 3, 4)
train accuracy, mask on 97.00 to 99.33, mask off 97.00 to 99.33 percent; test accuracy 82.00 to 88.67 (mean 84.53) percent
mask off minus test 8.33 to 17.33 (mean 13.93) points; mask on minus test 8.33 to 17.33 (mean 13.93) points; test loss 0.680 to 1.450

Layer_Dropout(0.1); 512 neurons; 100 samples per class; learning rate 0.02; seeds (0, 1, 2, 3, 4)
train accuracy, mask on 89.57 to 92.88, mask off 92.00 to 94.33 percent; test accuracy 83.33 to 88.33 (mean 85.20) percent
mask off minus test 4.33 to 9.67 (mean 7.67) points; mask on minus test 2.64 to 8.21 (mean 5.95) points; test loss 0.506 to 0.850
```

Seed by seed, the test accuracy with dropout is higher on three seeds, by 0.33, 0.67 and 2.67 points, equal on one, and lower by 0.33 points on one. One test point is 0.33 points, and a test accuracy on 300 points scatters by about 2.2 points (post 28, section 4), so this is not a measured gain. Two things did move on all five seeds. The gap narrowed, to 4.33 to 9.67 points from 8.33 to 17.33, with the mask-off training accuracy falling to 92.00 to 94.33 percent from 97.00 to 99.33. And the test loss fell. On a layer with neurons to spare, dropout at 0.1 did what a regulariser is described as doing, less fit and a smaller gap, and left the test accuracy where it was. These are five seeds at one width and one rate.

**What these runs support.** On the spiral, with one hidden layer of 64 neurons trained full-batch, dropout at 0.1 to 0.5 is the wrong tool: the network is not too large for the problem (post 28 found that 128 neurons test better than 64), and removing part of it at random on every pass costs more fit than it buys. The runs do not show that dropout is ineffective in general. The released project `nn-p01` trains two hidden layers of 128 neurons on 60,000 images with $p = 0.1$ and an L2 penalty and publishes a test accuracy of 98.00 percent, with a training accuracy of 98.47 percent measured through the mask in the last epoch; it has no run without dropout, so that figure does not measure what dropout contributed either.

---

## 9. Where dropout sits in the toolkit

Post 28 named four levers. With this post each has been measured on the spiral network, in its own post's setup; the first row is added here:

| Lever | Acts on | At evaluation | Measured on the spiral |
|---|---|---|---|
| More data | the training set | nothing changes | 1,000 samples per class: gap 1.17 to 3.03 points, from 10.67 to 14.00 (section 8) |
| Capacity | the architecture | nothing changes | 8 neurons: smaller gap, far lower test accuracy; 128 neurons: higher test accuracy (post 28) |
| Epoch budget | the schedule | nothing changes | stopping at the lowest test loss lowered the test accuracy in all five runs (post 28) |
| L1, L2 penalties | the loss and the weight gradients | the penalty is left out of the reported loss | L1 raised test accuracy on ten of ten seeds, L2 on eight (post 30) |
| Dropout | the forward and backward passes of training | the layer is the identity | lowered test accuracy on ten of ten seeds at each of three rates (section 8) |

Dropout is the only one of these that makes the training-time network differ from the evaluation-time network, which is why it is the only one that needs a switch, and why its training figures need the care of section 8.

One deliberate exception to the switch exists. Gal and Ghahramani (2016) keep the mask on at test time, run many passes and read the spread of the predictions as a measure of the model's uncertainty. The default of this series, and of its projects, is the mask off.

---

## 10. Make it run: the scripts

Every code block and every number of this post comes from a script in `snippets/`, run from the series root, for example:

```text
python posts/31-dropout/snippets/seeds_rate_10.py
```

| Script | Contents | Time |
|---|---|---|
| `network.py` | the classes, the loop, the setup; prints the two modes of section 5 | 1 s |
| `layer_dropout.py` | the mask, the scale, the expectation, the table of section 3 | 1 s |
| `ensemble.py` | all 256 thinned networks against the full one (section 2) | 1 s |
| `gradient_check.py` | the central-difference check with a fixed mask (section 6) | 1 s |
| `what_can_go_wrong.py` | the mistakes of section 11 | 30 s |
| `seeds_none.py`, `seeds_rate_10.py`, `seeds_rate_20.py`, `seeds_rate_50.py` | five seeds each (section 8) | 30 to 60 s each |
| `jobs/paired.py` | seeds 0 to 9 at every rate, the seed-by-seed counts, fitted networks trained on with the mask (section 8) | 6 to 12 min |
| `jobs/smaller_step.py` | ten runs at a learning rate of 0.002 (section 8) | 2 to 4 min |
| `jobs/more_data.py` | ten runs on 3,000 points (section 8) | 20 to 40 min |
| `jobs/wider.py` | ten runs with 512 neurons (section 8) | 25 to 45 min |

The four scripts under `jobs/` are too slow for the two-minute limit of the others and are run on their own. The seed scripts take other seeds as arguments, as in `seeds_rate_10.py 5 6 7 8 9`. All need NumPy and the `nnfs` package, which supplies `spiral_data` and `nnfs.init()`.

---

## 11. What can go wrong?

`snippets/what_can_go_wrong.py` runs each mistake.

**The mask left on in the test pass.** `forward` defaults to `training=True`, so a test pass that omits the argument drops. On the trained network of seed 0 at $p = 0.1$:

```text
training=False: test accuracy 0.6633, test loss 0.7623, the same on every pass
training=True, 100 passes: test accuracy 0.5833 to 0.6633 (mean 0.6222), test loss 0.8040 to 1.0265; passes below the training=False accuracy: 99
```

Nothing is raised. The reported figure is that of a random thinned network: it changes from pass to pass, and in 99 of 100 passes it is below the full network's, by up to 8.00 points. Two test passes over the same weights that disagree are the symptom.

**A wrong `backward`.** The gradient check of section 6, on seed 0, with three wrong versions of the method: one that ignores the mask, one that applies the 0 and 1 pattern without the scale, and one that draws a new mask. Each cell is the largest relative error with the largest absolute gap in brackets.

```text
gradient         correct              no mask               mask without the scale   a new mask
dense2.dweights  2.9e-09 (2.8e-11)   2.9e-09 (2.8e-11)   2.9e-09 (2.8e-11)   2.9e-09 (2.8e-11)
dense2.dbiases   2.4e-10 (2.0e-11)   2.4e-10 (2.0e-11)   2.4e-10 (2.0e-11)   2.4e-10 (2.0e-11)
dense1.dweights  4.3e-09 (3.1e-11)   8.5e-01 (6.6e-02)   5.0e-01 (5.2e-02)   1.1e+00 (1.1e-01)
dense1.dbiases   2.9e-10 (4.5e-11)   8.4e-01 (5.4e-01)   5.0e-01 (3.2e-01)   8.3e-01 (5.4e-01)
```

The gradients of `dense2` pass in every column, because they are computed before the gradient reaches the dropout layer. Only the layers behind it are wrong, and none of the three raises an error. The middle column is wrong by exactly the missing factor: at $p = 0.5$ the scale is 2, and the relative error is 0.50.

**`backward` after an evaluation forward.**

```text
a layer that has never run in training mode: AttributeError: 'Layer_Dropout' object has no attribute 'binary_mask'
training forward, mask:           [2. 2. 2. 2. 0. 2.]
evaluation forward, then backward: [2. 2. 2. 2. 0. 2.]  (the output was [1. 1. 1. 1. 1. 1.] )
```

The first case fails loudly. The second is silent: the evaluation pass was the identity, and `backward` applied the mask of the training pass before it.

**The seed set inside the loop.** A `np.random.seed(0)` before each forward pass, put there to make a run repeatable, restarts the stream: the script prints the same mask, `[2. 2. 2. 2. 0. 2.]`, for three epochs in a row. The seed is set once, before the data and the weights are drawn.

**A drop rate of 1.** The class stores a keep probability of 0, draws a mask of zeros and divides it by 0: every output is `nan`. NumPy warns and does not stop. The class of `nn-p01` rejects a rate outside $[0, 1)$ in its constructor; the class of this post is the shorter one of the cheatsheet and does not.

**Taking a drop rate of 0 for the baseline.** `Layer_Dropout(0.0)` keeps every entry and scales by 1, so it should be the network without the layer. Under `nnfs.init()` it is not, bit for bit:

```text
dtypes with Layer_Dropout: ReLU output float32, mask float64, dropout output float64, dense1.dbiases float64, dense1.dweights float32
no dropout layer  : dense1.dbiases float32, training accuracy 0.9633, test accuracy 0.8233
Layer_Dropout(0.0): dense1.dbiases float64, training accuracy 0.9733, test accuracy 0.7967
```

`np.random.binomial` returns integers and the division makes the mask float64, so the gradient behind the layer is float64 and the bias gradient of `dense1`, a plain `np.sum`, stays float64, where the weight gradient passes through the float32 `np.dot` of `nnfs.init()`. The arithmetic is the same and the rounding is not, and after 10,001 epochs the two runs of seed 0 differ by 1.00 point in training accuracy and 2.67 points in test accuracy. This is why the baseline of section 8 runs without the layer. It also shows how little a single run decides: a change in rounding alone moved 8 test points of 300.

---

## 12. Summary

| Concept | Takeaway |
|---|---|
| Mechanism | each activation is set to zero with probability $p$ on every training forward pass; each sample gets its own mask |
| Two readings | no neuron can rely on a particular other one; $2^n$ thinned networks share one set of weights |
| Inverted dropout | kept activations are divided by $1 - p$; the mean is unchanged and the variance is $a^2 p / (1 - p)$ |
| `Layer_Dropout` | `rate` is the drop probability, `self.rate` the keep probability; `backward` multiplies by the stored mask |
| The switch | `training=True` in passes followed by an update, `training=False` in every reported pass |
| Reading the figures | training accuracy through the mask was 3 to 11 points below the mask-off figure; only the mask-off figure is comparable with a test figure |
| Measured, 64 neurons | test accuracy lower on ten of ten seeds at $p$ = 0.1, 0.2 and 0.5; test above training only through the mask |

---

## Common pitfalls

1. **Leaving the mask on at evaluation.** `forward` defaults to `training=True`. Every pass whose figure is reported passes `training=False`.
2. **Comparing a masked training figure with a test figure.** The loop's accuracy scores thinned networks. The gap is read between two mask-off figures on the same weights.
3. **Reading a small gap as success.** Dropout narrowed the gap on the spiral network by making the training fit worse. The test figure is the one that counts.
4. **Reading `layer.rate` as the drop rate.** The attribute holds the keep probability $1 - p$.
5. **Choosing the rate from one run, or on the test set.** Runs differ by more than neighbouring rates do; the rate is compared over several seeds on validation data.

---

## Further reading

- Gal, Y. and Ghahramani, Z., *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"* (ICML, 2016). The mask kept on at test time, read as uncertainty.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapter 7 (MIT Press, 2016). Dropout among the other regularisers, with the ensemble view.
- Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R., *"Improving Neural Networks by Preventing Co-adaptation of Feature Detectors"* (arXiv:1207.0580, 2012). The report that introduced dropout.
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 15 (2020).
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R., *"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"* (Journal of Machine Learning Research, 2014). The full account, with thinned networks and the choice of rates.
- Wager, S., Wang, S., and Liang, P., *"Dropout Training as Adaptive Regularization"* (NeurIPS, 2013). Dropout related to an L2 penalty for generalised linear models.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 32 - Mini-batching](../32-mini-batching/index.md):** the training loop split into batches, the form in which the released projects use this layer.
- **[Post 30 - L1 and L2 regularisation](../30-l1-and-l2-regularisation/index.md):** the regulariser that acts on the weights, which did raise the test accuracy of the spiral network.
