# 28 - Generalization and testing

> **TL;DR.** The spiral classifier of post 27 is run forward-only, with no `backward` call and no optimiser call, on 300 points it was not trained on. On the documented seed it classifies 96.33 percent of its training points and 82.33 percent of the new ones, and over five seeds the gap is 10.67 to 14.00 points. The test loss turns upward after 400 to 2,800 epochs while the training loss keeps falling, although the test accuracy ends higher than at that turning point in all five runs. Of the four levers against overfitting, the two measured here do not follow the usual advice on this problem: 8 neurons in place of 64 shrink the gap and cost 28 to 42 points of test accuracy, and 128 neurons raise the test accuracy in all five runs.
>
> **Prerequisites:** [Post 27](../27-adam-optimiser/index.md).
> **Safe to skip?** Skip it if the reader can already evaluate a trained network on held-out data without touching its parameters, and can say what a training accuracy, a test accuracy and their difference each measure.
>
> **After reading, you will be able to:**
>
> - Run a forward-only test pass on held-out data and read its loss and accuracy against the training figures.
> - Distinguish good generalisation from overfitting by the shape of the loss curves and the geometry of the decision boundary.
> - Map an overfitting symptom to one of four levers: capacity, epoch budget, weight penalties and dropout.

![A dot plot with one row per seed, 0 to 4, for the network of post 27: a green circle marks its accuracy on its 300 training points and a hollow blue diamond its accuracy on 300 new points, with the same weights. Training and test are 96.33 and 82.33 percent on seed 0, 83.33 and 72.00 on seed 1, 78.00 and 67.33 on seed 2, 88.00 and 77.00 on seed 3, and 96.33 and 82.67 on seed 4, gaps of 14.00, 11.33, 10.67, 11.00 and 13.67 points.](diagrams/01-gap-over-seeds.svg)

*The same weights on two sets of 300 points, seeds 0 to 4: the test figure is lower on every seed, by 10.67 to 14.00 points.*

---

## 1. The question: what does a training accuracy leave out?

Training a neural network resembles studying for an exam. A student who memorises every practice question word for word can score 100 percent on the homework and still fail the exam, because the exam asks different questions about the same material. A student who has learned the material can answer questions never seen before. On the practice set the two students look identical.

Every loss and every accuracy of posts 22 to 27 was measured on the 300 points the network was trained on. Those figures are the practice set. The question of this post is: **how well does the trained network do on points it has never seen, and what does the difference from its training figures say?**

**Generalisation** is how well a trained model performs on data it was not trained on. The data kept aside to measure it is the **test set**, and the difference between training and test performance is the **generalisation gap**. A model that fits its training data much better than new data is **overfitting**; a model that does poorly even on its training data is **underfitting**.

Reaching a low training loss is therefore the first of two acts. The second is measuring the same loss on data the model never trained on. Part VII, this post and the three after it, is about the second act.

---

## 2. Where the framing comes from

The distinction between fitting a sample and fitting the process that produced it is older than deep learning. Three pieces of statistical learning theory are used here as framing; none is derived.

**Two kinds of risk.** In the vocabulary of Vapnik (1998), the average loss over the training set is the empirical risk, and the expected loss over the whole distribution of the data is the risk. Training minimises the first. The second is what matters, and it cannot be computed, only estimated, from points that played no part in training. A test set is that estimate.

**Bias and variance.** For a squared-error loss, the expected test error of a predictor splits into three terms: a noise floor that no model can remove, a bias term (how wrong the model is on average over training sets; this is the statistical term, not the bias of a neuron) and a variance term (how much its prediction changes when the training set is drawn again). Geman, Bienenstock and Doursat (1992) brought this decomposition to neural networks. Overfitting is the regime in which variance dominates: the model depends so much on its particular training points that it cannot be trusted on new ones.

**Model assessment.** Hastie, Tibshirani and Friedman (2009, chapter 7) describe the protocol used today: a training set to fit the model, a validation set to choose between models, and a test set for the final figure. This post uses two of the three, and the words "test" and "validation" both name the same forward-only pass on held-out data. [Post 29](../29-validation-and-hyperparameter-tuning/index.md) draws the line between them.

---

## 3. Running the test pass

The network is the documented run of [post 27](../27-adam-optimiser/index.md), unchanged, and `snippets/network.py` prints its setup: `nnfs.init()`, which sets seed 0 and float32 and replaces `np.dot` with a float32 version; `spiral_data(samples=100, classes=3)`; `Layer_Dense(2, 64)`, ReLU, `Layer_Dense(64, 3)`, softmax and cross-entropy; weights of `0.01 * randn`; `Optimizer_Adam(learning_rate=0.02, decay=1e-5)`; 10,001 full-batch epochs.

**When the test data is drawn decides what it is.** `spiral_data` takes no seed; it draws from NumPy's global random stream. With `samples=100, classes=3` it places 100 points per class at the fixed radii `np.linspace(0, 1, 100)` and adds normal noise to each point's angle (standard deviation 0.2 before the generator's factor of 2.5, so 0.5 radians), 300 normal draws in all. After `np.random.seed(0)` the training data takes draws 1 to 300 and the two weight arrays take the next $128 + 192 = 320$. The training loop draws nothing. A call to `spiral_data` after the loop, with no new seed, therefore takes draws 621 to 920: the same radii and labels as the training set, new angles. The top row of the figure below draws this order; section 8 measures the other two.

![Three rows of blocks, each block as wide as the normal draws it takes after the seed is set. The documented order: training data draws 1 to 300, the weights of the two layers 301 to 620, the training loop none, the test data 621 to 920, a gap of 14.00 points. The seed set again before the test draw: the test data takes draws 1 to 300 again and is the training set, a gap of 0.00 points. The test data drawn before the layers: test data 301 to 600, weights 601 to 920, a gap of 18.67 points.](diagrams/02-draw-order.svg)

*Which draws of the stream each array takes in three orders of the calls; only the first gives a test set that is new and a network that is the documented one.*

```python
# Fresh test data: the same generator, new points. Drawn after training and without a new
# seed, so these are the next 300 normal draws of the stream that np.random.seed(0) started.
X_test, y_test = spiral_data(samples=100, classes=3)

# Forward pass only. No backward call, no optimiser call.
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
accuracy = np.mean(predictions == y_test)
```

The four `forward` calls are those of the training loop with `X_test` and `y_test` in place of `X` and `y`. `snippets/network.py` wraps them in `evaluate(X, y, dense1, activation1, dense2, loss_activation)`, which returns the loss and the accuracy, and `snippets/test_pass.py` uses it to measure the training figures with the same final weights:

```text
last epoch of the loop, before its update: loss 0.0806  acc 0.9633
training data  loss 0.0806  acc 0.9633  (289 of 300)
test data      loss 1.1168  acc 0.8233  (247 of 300)
gap            loss +1.0362  acc 0.1400  (14.00 percentage points)
distinct points the two sets share: 1 (the centre (0, 0), where every arm starts)
```

The network classifies 289 of its 300 training points and 247 of the 300 new ones: 96.33 against 82.33 percent, a gap of 14.00 points. The loss moves much further than the accuracy, from 0.0806 to 1.1168; section 4 takes that figure apart.

| Quantity | Value on seed 0 | Meaning |
|---|:---:|---|
| Training accuracy | 0.9633 | the share of the training points classified correctly |
| Test accuracy | 0.8233 | the share of 300 unseen points classified correctly |
| Generalisation gap | 0.1400 | training accuracy minus test accuracy |

The first printed line is the figure of post 27. The training loop measures its loss and accuracy before each update, so its last line describes the weights one update before the final ones. On seed 0 the two readings agree. A gap must compare two figures measured with the same weights, so every training figure below is read by `evaluate` after the last update.

One seed is one draw of the data and of the weights. `snippets/seed_spread.py` calls `np.random.seed(s)` for five seeds after `nnfs.init()`, so that the training data, the weights and the test data are all drawn again:

```text
seed  train loss  train acc  test loss  test acc  gap in points
   0      0.0806     0.9633     1.1168    0.8233          14.00
   1      0.3915     0.8333     0.8949    0.7200          11.33
   2      0.4721     0.7800     0.8288    0.6733          10.67
   3      0.3070     0.8800     0.6568    0.7700          11.00
   4      0.1129     0.9633     1.1503    0.8267          13.67
loop's last accuracy, before its update: 0.9633  0.8233  0.8133  0.8867  0.9633
```

The training accuracy runs from 78.00 to 96.33 percent and the test accuracy from 67.33 to 82.67 percent. The gap is the steadier quantity: 10.67 to 14.00 points, and positive in every one of the five runs; the figure at the top of the post draws the five rows. These figures are those of seeds 0 to 4; no other seed is run here, and every statement below about "all five runs" is scoped to them. The 96.33 percent of seed 0 is the upper end of what this setup reaches, not its typical result. The last line shows why the same weights matter: on seed 2 the loop's last line reads 81.33 percent and the weights it then produces score 78.00.

### 3.1. Why the test pass is forward-only

- **An update on the test data turns it into training data.** The figure a test set reports is an estimate of the risk only as long as it has no influence on the parameters. Section 8 measures what a single update does.
- **Nothing in the test pass needs a gradient.** A loss and an accuracy come out of the forward pass. The test pass makes four of the eight calls of a training step and none of the optimiser's three kinds of call.
- **The test set is read, not used.** The same rule holds for the validation set of post 29, with one addition: the test set is opened once, at the end.

That the pass changes nothing can be checked, and `test_pass.py` does it. Before the pass it copies everything a training step would write:

```python
# What a test pass must leave alone: the parameters, the gradients and the optimiser.
parameters_before = [a.copy() for a in (dense1.weights, dense1.biases, dense2.weights, dense2.biases)]
gradients_before = [a.copy() for a in (dense1.dweights, dense1.dbiases, dense2.dweights, dense2.dbiases)]
iterations_before = optimizer.iterations
```

```text
parameters changed: 0 of 387
gradient arrays changed: 0 of 4
optimizer.iterations: 10001 before, 10001 after
dense1.inputs is X_test: True
```

None of the 387 parameters moved, the four gradient arrays are still those of the last training epoch, and the optimiser's step counter stands at 10,001. One thing did change: each `forward` stores its input for a later `backward` ([post 16](../16-coding-backpropagation/index.md)), so the cache of `dense1` now holds the test points. That is harmless as long as every `backward` follows a `forward` on the training data, which the loop guarantees.

---

## 4. Reading the gap

**The loss gap and the accuracy gap say different things.** The test loss of seed 0, 1.1168, is above $\ln 3 = 1.0986$, the loss of a uniform guess over three classes ([post 08](../08-loss-categorical-cross-entropy/index.md)), although the network is right on 82.33 percent of the test points. Splitting the loss by sample resolves it:

```text
misclassified test points: 53 of 300, mean loss 6.0916, share of the summed loss 0.9637
correct test points: 247 of 300, mean loss 0.0493
misclassified training points: 11 of 300, mean loss 1.1715
```

The 53 misclassified test points carry 96.37 percent of the test loss. Their mean loss of 6.09 is the loss of a probability of $e^{-6.09} \approx 0.002$ on the true class: the network is not unsure about these points, it is confidently wrong. On the training set the 11 misses have a mean loss of 1.17. Accuracy counts mistakes; the loss also weighs how confident each mistake was, and it is the confidence that did not carry over to new points.

**One test set of 300 points is itself a noisy reading.** The same script draws twenty more test sets, one after the other, and evaluates the same weights on each:

```text
test accuracy: lowest 0.7767, highest 0.8467, mean 0.8180
sets that reach the training accuracy of 0.9633: 0 of 20
binomial standard error at p = 0.818: 0.0223 for 300 points, 0.0039 for 10,000
```

The test accuracy of one network moves between 77.67 and 84.67 percent with the draw of the test set. A count of successes among $n$ independent points has the standard error $\sqrt{p(1 - p)/n}$, which is 2.2 points here. The gap of 14.00 points is more than six of those and none of the twenty sets reaches the training accuracy, so the gap is not an accident of one draw; a difference of two points between two networks would be. Nothing forces a test accuracy to lie below the training accuracy, and a single test set can land above it by chance when the two are close. The remedy for the noise is a larger test set: the released projects `nn-p01` and `nn-p03` each test on 10,000 images, where the same formula at the same $p$ gives 0.39 points, and `nn-p03` reports 87.12 percent with a spread of 1.19 points over three seeds, which is larger than its test-set noise.

**A gap is read together with the two figures it separates.** Four regimes are usually told apart:

| Regime | Train | Test | Gap | Measured in this post | Usual next step |
|---|:---:|:---:|:---:|---|---|
| **Underfitting** | low | low | small | 8 neurons: train 41.33 to 50.67, test 38.00 to 46.67, gap 0.33 to 7.00 | more neurons, more layers, more epochs |
| **Good fit** | high | high | small | no run of this post | keep the model and watch for a change in the data |
| **Overfitting** | high | lower | large | 64 neurons: gap 10.67 to 14.00 in all five runs, with train 78.00 to 96.33 and test 67.33 to 82.67; "train high" holds on two seeds only | the levers of section 6 |
| **Distribution shift** | high | low | very large | not measured: the test points come from the training generator | a test set from the right distribution |

The underfitting row comes from `snippets/narrower.py`, the same five seeds with 8 hidden neurons (the left panel of the figure in section 6). Its gaps are small, 0.33 to 7.00 points, and its test accuracy is 38.00 to 46.67 percent, little better than the 33.33 percent of a guess. A small gap is therefore no evidence of a good model. The script also counts the hidden neurons whose output is zero on all 300 training points after the last update:

```text
hidden neurons whose output is zero on all 300 training points, after the last update, by seed: 3  2  4  1  4
neurons still alive: 4 to 7 of 8
```

The label "underfitting" stands, and part of the missing capacity is neurons that died, the dead ReLU of [post 17](../17-backpropagation-through-activation-functions/index.md): the network that was trained has 4 to 7 working neurons, not 8.

The 64-neuron runs share one thing: a gap of more than ten points in each of the five runs, with test data from the same generator. Their training accuracy does not fit one label. It is 96.33 percent on seeds 0 and 4 and between 78.00 and 88.00 percent on seeds 1, 2 and 3, where the network is under-fitted and over-fitted at once: it has not fitted its training points and is still more than ten points worse on new ones. No run of this post reaches the second row.

---

## 5. What overfitting looks like

### 5.1. The shape of the loss curves

Loss curves work for any model, whatever the dimension of its input. Two curves are followed over the epochs: the loss on the training data, and the loss on held-out data.

`seed_spread.py` draws the test points before the loop and reads both sets forward-only every 100 epochs. Because the loop draws no random numbers, these are the test points that `test_pass.py` draws after it. Seed 0:

```text
epoch  train loss  train acc  test loss  test acc
    0      1.0986     0.3600     1.0986    0.3567
  100      0.7692     0.6733     0.8433    0.6200
  300      0.4453     0.8400     0.5727    0.7733
  700      0.2872     0.8900     0.4900    0.8133
 1000      0.2382     0.9133     0.4968    0.8200
 2000      0.1621     0.9333     0.5347    0.8400
 5000      0.1097     0.9467     0.8000    0.8467
10000      0.0806     0.9633     1.1165    0.8233
```

Three phases show in these rows and in the figure below, which draws all 101 checks of the run. Up to epoch 700 both losses fall together. Between epochs 700 and 2,000 the test loss stays near 0.5 while the training loss falls from 0.29 to 0.16. After that the test loss climbs to 1.12, more than twice its lowest value, while the training loss goes on falling. The training loss alone shows none of this, which is why it says that the optimiser works and nothing about generalisation.

![Two charts against epochs 0 to 10,000 for seed 0, both sets read every 100 epochs. Top, the loss: both start at ln 3, 1.0986; the training loss falls to 0.0806; the test loss is lowest, 0.4900, at epoch 700, then climbs to 0.8000 at epoch 5,000 and 1.1165 at epoch 10,000. Bottom, the accuracy: the training accuracy rises to 96.33 percent; the test accuracy is 81.33 percent at epoch 700, 84.67 at epoch 5,000 and 82.33 at epoch 10,000. A dotted line marks epoch 700 in both.](diagrams/03-loss-curves.svg)

*Seed 0 every 100 epochs: the test loss is lowest at epoch 700 and more than twice as high at the end, while the test accuracy ends above its value at epoch 700.*

The accuracy column tells a different story. The test accuracy is 81.33 percent at the lowest test loss, rises to 84.67 percent at epoch 5,000 and ends at 82.33. The other seeds agree:

```text
seed  epoch  test loss -> end     test acc -> end     gap in points -> end  boundary -> end  weight norm -> end
   0    700  0.4900 -> 1.1168     0.8133 -> 0.8233      7.67 -> 14.00            2161 ->  2346      54.75 -> 187.46
   1   1100  0.7635 -> 0.8949     0.6567 -> 0.7200     10.00 -> 11.33            2030 ->  2286      36.11 ->  66.14
   2   2400  0.7201 -> 0.8288     0.6267 -> 0.6733     11.67 -> 10.67            2277 ->  2402      52.89 -> 122.63
   3   2800  0.5719 -> 0.6568     0.7367 -> 0.7700     11.33 -> 11.00            2214 ->  2234      63.63 -> 104.34
   4    400  0.7419 -> 1.1503     0.7033 -> 0.8267     11.00 -> 13.67            1916 ->  2246      35.28 -> 135.46
runs whose test accuracy is lower at the check of lowest test loss than at the end: 5 of 5
```

Each row compares the check with the lowest test loss, between epochs 400 and 2,800, with the end of the run. The test loss is 0.08 to 0.63 higher at the end. The test accuracy is higher at the end too, in all five runs, by 1.00 to 12.33 points. The accuracy gap is wider at the end in three runs and narrower in two.

Training past the lowest test loss therefore damaged one thing and not the other in these five runs. The predictions got more confident, the confident mistakes of section 4 got more expensive, and the count of mistakes did not rise. Which reading matters depends on what the model is for: a model whose probabilities are used needs the low loss; a model of which only the predicted class is used is judged by its accuracy. Stopping where the held-out loss is lowest is called **early stopping**, and on these five runs it buys a lower test loss at the price of test accuracy.

Two cautions belong to this reading. The check of lowest test loss was found by looking at the test data, which is harmless in a diagnosis and not allowed for a figure that is reported; post 29 introduces the validation set for that purpose. And a held-out curve is not smooth: on seed 0 the test loss moves by 0.007 between the checks at 700 and 1,000, so a turn is read from the trend over many checks and not from the first rise. Prechelt (1998) studies stopping rules for such curves.

### 5.2. The geometry of the decision boundary

For a problem with two inputs the classifier can be drawn. The usual picture is this: a well-generalising classifier draws smooth curves between the classes and accepts a few misclassified training points, and an overfitted one bends its boundary around single points, noise included.

The last two column pairs of the table measure this on the spiral. `boundary_length` in `network.py` classifies a grid of $201 \times 201$ points over the square $[-1, 1]^2$ and counts the pairs of neighbouring grid points that are given different classes, a count that grows with the length of the boundary. The weight norm is the square root of the sum of the squares of all 320 weights.

```text
boundary length, end over check: 1.01 to 1.17; weight norm, end over check: 1.64 to 3.84
```

Between the check of lowest test loss and the end, the boundary gets 1 to 17 percent longer and the weights 1.64 to 3.84 times larger. The picture of a boundary that grows pockets is therefore only a small part of what happens here: the boundary moves a little, and the network never captures every training point (11 of 300 are still wrong on seed 0). What grows is the size of the weights, and with it the steepness of the probabilities on either side of the boundary. That is the confident-mistake reading of section 4 seen from the parameters, and it is the quantity the weight penalties of [post 30](../30-l1-and-l2-regularisation/index.md) act on. The figure below draws seed 0 at both checks.

![Two plots of the plane from minus 1 to 1 for seed 0, each tinted by the class the network answers on a 201 by 201 grid, with the 300 training points on top and the misclassified ones ringed. Left, epoch 700, the check of lowest test loss: 33 training points wrong, train 89.00 and test 81.33 percent, boundary length 2,161, weight norm 54.75. Right, epoch 10,000: 11 wrong, 96.33 and 82.33 percent, boundary length 2,346, weight norm 187.46.](diagrams/04-boundary-check-vs-end.svg)

*Seed 0 at the check of lowest test loss and at the end: the regions change in a few places, while the weight norm grows from 54.75 to 187.46.*

A common piece of advice is to prefer the simpler of two models that fit equally well, a preference often called Occam's razor. For a network, "simpler" is not the same as "fewer parameters": two networks with the same parameter count can fit very different sets of functions, depending on the depth, the activation function and the size of the weights. The measurement above gives a working sense for this series: of two networks with the same architecture, the one with the smaller weights is the simpler function.

---

## 6. Four levers against overfitting

| Lever | What it changes | Measured in this post | Where it is built |
|---|---|---|---|
| **Capacity** | the architecture: fewer neurons or layers | 8 neurons: smaller gap and lower test accuracy in all five runs | post 29 chooses a width on validation data |
| **Epoch budget** | the schedule: stop before the held-out loss climbs | lower test loss and lower test accuracy in all five runs | post 29 supplies the validation set the choice needs |
| **Weight penalties (L1, L2)** | the loss: a term that grows with the size of the weights | not measured here | [post 30](../30-l1-and-l2-regularisation/index.md) |
| **Dropout** | the forward pass in training: random activations set to zero | not measured here | [post 31](../31-dropout/index.md) |

**Capacity.** `narrower.py` and `snippets/wider.py` repeat the five seeds of section 3 with 8 and with 128 hidden neurons; nothing else changes. The figure after the listing sets their runs beside the 64-neuron runs.

```text
== 8 neurons, after the last update
train accuracy 41.33 to 50.67 percent, test accuracy 38.00 to 46.67 percent, gap 0.33 to 7.00 points
== 128 neurons, after the last update
train accuracy 94.33 to 98.00 percent, test accuracy 80.67 to 86.67 percent, gap 7.67 to 16.67 points
```

![Three dot plots for 8, 64 and 128 hidden neurons, 51, 387 and 771 parameters, one row per seed from 0 to 4, each with the training accuracy as a green circle and the test accuracy as a hollow blue diamond. 8 neurons: test 38.00 to 46.67 percent, gap 0.33 to 7.00 points. 64 neurons: test 67.33 to 82.67 percent, gap 10.67 to 14.00 points. 128 neurons: test 80.67 to 86.67 percent, gap 7.67 to 16.67 points.](diagrams/05-three-widths.svg)

*Five seeds at three widths: the narrow network has the smallest gaps and the lowest test accuracy, the wide one the highest test accuracy on every seed.*

Taking this much capacity away shrinks the gap and destroys the model: seed by seed, the 8-neuron network has a smaller gap than the 64-neuron one in all five runs and a test accuracy that is 28.33 to 41.67 points lower. Adding capacity does the opposite of what the lever predicts: seed by seed, the 128-neuron network has a higher test accuracy than the 64-neuron one in all five runs, by 0.33 to 19.33 points (86.67 against 67.33 percent on seed 2), and its gap is wider in three runs and narrower in two. A width changes how many draws the weights take, so each width is tested on its own 300 points, and the two smallest differences, 0.33 and 1.00 points, are inside the test-set noise of section 4. The wider network also fits its training data better, 94.33 to 98.00 percent against 78.00 to 96.33: on seeds 1, 2 and 3 the 64-neuron network stays below 90 percent on its own training points. On this problem the 64-neuron network is not too large. A smaller gap is not the goal; a higher test figure is, and the two can move in opposite directions.

**Epoch budget.** Section 5.1 measured it: stopping at the lowest test loss lowers the test loss in all five runs and the test accuracy in all five.

**Weight penalties and dropout.** Both leave the architecture and the schedule alone. A weight penalty adds a term to the loss that grows with the size of the weights, the quantity that section 5.2 saw grow by a factor of 1.64 to 3.84 while the test loss climbed. Dropout sets randomly chosen activations to zero during training and leaves every neuron on during evaluation, so the forward-only pass of this post gains a switch there. Whether either raises the test accuracy of the spiral network is measured in posts 30 and 31, not here.

Every lever has a setting that training does not learn: the number of neurons, the number of epochs, the regularisation strength $\lambda$, the dropout rate $p$, and beside them the learning rate $\alpha$ and its decay. These are hyperparameters, and choosing them by looking at the test figures of this section would spend the test set. Post 29 builds the procedure that avoids it.

---

## 7. Make it run: the test pass, the seeds and the widths

Six scripts hold every code block and every printed number of this post. They need NumPy and the `nnfs` package, set their seeds, and run from the series root:

```text
python posts/28-generalization-and-testing/snippets/network.py
python posts/28-generalization-and-testing/snippets/test_pass.py
python posts/28-generalization-and-testing/snippets/seed_spread.py
python posts/28-generalization-and-testing/snippets/narrower.py
python posts/28-generalization-and-testing/snippets/wider.py
python posts/28-generalization-and-testing/snippets/what_can_go_wrong.py
```

`network.py` holds the classes of posts 16, 19 and 27, unchanged, and the functions `build`, `train`, `evaluate` and `boundary_length`; run on its own it prints the setup and the starting loss in about a second. `test_pass.py` (about 10 seconds) is the documented run and sections 3 and 4. `seed_spread.py` (about 45 seconds) trains the five seeds and prints the tables of sections 3 and 5. `narrower.py` (about 25 seconds) and `wider.py` (about 70 seconds) change the width. `what_can_go_wrong.py` (about 20 seconds) runs the mistakes of section 8.

`evaluate` is the only function this post adds to the pipeline of Part VI:

```python
def evaluate(X, y, dense1, activation1, dense2, loss_activation):
    """Loss and accuracy on (X, y): four forward calls, no backward call, no optimiser."""
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    return float(loss), float(accuracy)
```

---

## 8. What can go wrong?

None of the mistakes below raises an error. Each prints a plausible figure.

**The seed set again before the test data is drawn.** A script that calls `np.random.seed(0)` before every `spiral_data` call, for the sake of reproducibility, gets the training set back:

```text
identical to the training data: True
'test' loss 0.0806  acc 0.9633  gap 0.00 points
```

A gap of exactly zero is the symptom. A test set needs draws the training set did not use: a later position in the same stream, as here, or a different seed.

**Figures compared across two orders of the draws.** Drawing the test set before the layers are created, next to the training set, is a legitimate order, and it is not the documented one. It gives the test set draws 301 to 600, and the weights the draws after them (the bottom row of the figure in section 3):

```text
same training data: True; same test data: False
training loss 0.0951  acc 0.9733;  test loss 0.9650  acc 0.7867;  gap 18.67 points
```

The training data is unchanged, and still a different network was trained: 97.33 percent on the training points, 78.67 on its test points, a gap of 18.67 points where the documented order gives 14.00. Neither figure is wrong and neither order is. The mistake is to set one against the other, or to believe that the documented run was reproduced: the two belong to different runs, and a result is reproducible only together with the order of every call that draws from the stream.

**A backward pass and an optimiser step inside the test pass.** A test pass copied from the training loop with its last eight lines left in:

```text
test pass   1 reports loss 1.1168  acc 0.8233; training acc after its update 0.8100
test pass   2 reports loss 1.3355  acc 0.7733; training acc after its update 0.7400
test pass 100 reports loss 0.2096  acc 0.9200; training acc after its update 0.8700
optimizer.iterations: 10101
```

The first pass still reports the honest 82.33 percent, because its figures are computed before its update. That one update, taken on the test points, lowers the training accuracy from 96.33 to 81.00 percent. After a hundred such passes the "test" accuracy reads 92.00 percent, higher than any of the twenty honest test sets of section 4, and it measures nothing, because the network has been trained on those points. The step counter gives the mistake away: 10,101 where a forward-only pass leaves 10,001.

**Two figures from different weights.** The gap of seed 2 is 10.67 points when both figures are read after the last update. Taking the training accuracy from the loop's last line, 81.33 percent, makes it 14.00.

---

## 9. Summary

| Concept | Takeaway |
|---|---|
| Test pass | four `forward` calls on held-out data; no `backward`, no optimiser; 0 of 387 parameters change |
| Test data | drawn after training from the same stream: draws 621 to 920 of the seed |
| Gap on seed 0 | 96.33 percent on training data, 82.33 on test data, 14.00 points |
| Gap over five seeds | 10.67 to 14.00 points, positive in every run |
| Loss against accuracy | 53 confident mistakes carry 96.37 percent of a test loss of 1.1168 |
| Curve cue | test loss lowest between epochs 400 and 2,800, then rising; test accuracy higher at the end in all five runs |
| Boundary cue | 1 to 17 percent longer at the end; weights 1.64 to 3.84 times larger |
| Four levers | capacity, epoch budget, weight penalties, dropout; the first two measured here, neither raised the test accuracy |

---

## Common pitfalls

1. **Reporting only the training accuracy.** It is measured on the points the network was fitted to. A model is described by both figures and by the gap between them.
2. **Setting the same seed before the test draw.** `spiral_data` then returns the training set, and the gap reads exactly zero.
3. **Leaving `backward` or the optimiser in the test pass.** A single update on the test points changes the network; after it the test set is training data.
4. **Reading a small gap as a good model.** The 8-neuron network has the smallest gap of the post and a test accuracy of 38.00 to 46.67 percent. Remedies for overfitting do not help a network that underfits.
5. **Reading one test set of 300 points to the last digit.** Its standard error is 2.2 points. A difference smaller than that between two networks is noise.
6. **Choosing a width or an epoch from the test figures.** Any choice made by looking at the test set makes its figure optimistic. That is what the validation set of post 29 is for.

---

## Further reading

- Geman, S., Bienenstock, E., and Doursat, R., *"Neural Networks and the Bias / Variance Dilemma"* (Neural Computation, 1992).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 5.2 (MIT Press, 2016).
- Hastie, T., Tibshirani, R., and Friedman, J., *The Elements of Statistical Learning*, chapter 7 (Springer, 2009).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 11 (2020).
- Prechelt, L., *"Early Stopping: But When?"* (Neural Networks: Tricks of the Trade, 1998).
- Vapnik, V., *Statistical Learning Theory* (Wiley, 1998).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 29 - Validation and hyperparameter tuning](../29-validation-and-hyperparameter-tuning/index.md):** a third set of data, so that widths, epochs and penalties can be chosen without spending the test set.
- **[Post 30 - L1 and L2 regularisation](../30-l1-and-l2-regularisation/index.md):** a penalty on the size of the weights, the quantity that grew here while the test loss climbed.
