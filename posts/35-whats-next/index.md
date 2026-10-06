# 35 - What to read after this series

> **TL;DR.** Everything on the usual path after this series is an addition to the skeleton it built: a forward pass, a backward pass and an optimiser step. Convolution, recurrence and attention are new layers that reuse one set of weights across positions; normalisation, residual connections and schedules are new training machinery; self-supervision, transfer, reinforcement learning and diffusion are new ways of posing the problem. This post places each on that map, says what it adds to the dense stack, and names a paper and a from-scratch tutorial to start from.
>
> **Prerequisites:** [Post 31](../31-dropout/index.md), [Post 34](../34-sigmoid-and-binary-cross-entropy/index.md).
> **Safe to skip?** Skip it if the reader can already say what convolution, recurrence and attention each add to a dense layer, and knows which of them the problem at hand calls for.
>
> **After reading, you will be able to:**
>
> - Place convolutional networks, recurrent networks and transformers on a map of what each adds to a dense network.
> - Name the canonical paper and a from-scratch tutorial for each of the three architectures.
> - Choose what to study next from the kind of problem to be solved.

![Three columns of cards. New layer types: convolutional layers, recurrent layers, and attention and transformers, each with what it adds, its paper and a from-scratch tutorial. New training infrastructure: normalisation, residual connections, learning-rate schedules, mixed precision and distributed training. New problem framings: self-supervised learning, transfer learning, reinforcement learning and diffusion models, closed by a note that the networks, losses and optimisers stay. Each card names what the addition adds and a source.](diagrams/01-whats-next-map.svg)

*Sections 2 to 4 on one page: each card says what the addition adds to the dense stack and where to start reading.*

---

## 1. The question: what does everything after this series add?

Posts 01 to 34 leave a set of classes written in NumPy and four projects that use them on real data. A stack of dense layers with a non-linearity between them is often called a multi-layer perceptron (MLP), and that is what the series built. The stock:

- **Layers:** `Layer_Dense` (post 04), ReLU and softmax (post 06), the sigmoid (posts 17 and 34), dropout (post 31).
- **Losses:** categorical cross-entropy (post 08) and binary cross-entropy (post 34). Mean squared error appears as a loss in the project `nn-p04` only.
- **Optimisers:** gradient descent, decay, momentum, AdaGrad, RMSProp and Adam (posts 22 to 27).
- **Generalisation:** the test pass (post 28), validation and k-fold cross-validation (post 29), L1 and L2 penalties (post 30), dropout (post 31).
- **Training practice:** mini-batches (post 32), weight initialisation (post 33), the gradient check (posts 10, 16 and 21).
- **Projects:** `nn-p01` (MNIST), `nn-p02` (two moons), `nn-p03` (Fashion-MNIST), `nn-p04` (California housing).

The series also left a habit: a comparison is read over several seeds and on held-out data. On the spiral, plain gradient descent ended anywhere between 43.0 and 87.0 percent training accuracy over ten seeds (post 22), no ordering of momentum, AdaGrad, RMSProp and Adam held across seeds (post 27, section 7), and L1 left no weight at exactly zero (post 30).

The question of this post: **what does each architecture and each training method met after this series add to that stock, and which should be studied first?**

The answer starts from what does not change. Every layer has a `forward` that stores its output and a `backward` that receives `dvalues` and stores `dinputs` (post 16). The loop that calls them in order and then hands every layer with parameters to an optimiser is [post 22](../22-gradient-descent-optimiser/index.md)'s. From post 23 on, every optimiser answers the same three calls: `pre_update_params`, `update_params(layer)` and `post_update_params` (post 23, section 8). The figure below draws the two contracts and one step of the loop that calls them.

![A card for any layer on the left with forward, which stores its output, and backward, which receives dvalues and stores dinputs. In the middle, one step of the loop as five boxes: forward of every layer in order, backward of every layer, pre_update_params, update_params for every layer with parameters, and post_update_params, with arrows from each step to the method it calls. On the right, a card for any optimiser with the three calls. Below, three cards for what does change: a layer with shared weights sums gradient contributions, training and evaluation modes, and carried state.](diagrams/02-the-skeleton-is-fixed.svg)

*A new layer brings a forward and a backward; a new optimiser brings three calls. The bottom row is section 8's list of what the loop does need.*

Sections 2 to 4 sort what is placed inside that skeleton into three kinds: new layers, new training machinery, and new ways of posing the problem. The first figure of this post maps all three.

---

## 2. New layer types

All three layers below answer one weakness of the dense layer: it has a separate weight for every input position, so it cannot use anything learned at one position at another. Each of the three reuses one set of weights across positions, and each does it differently. The figure below sets the three side by side with the counts of sections 2.1 to 2.3.

![Three panels. A 28 by 28 grid of pixels with one 5 by 5 filter shaded at two positions, and a table: a dense layer from 784 to 128 has 100,480 parameters, 8 filters of 5 by 5 have 208. Three recurrent steps, each input feeding its hidden state through W_x and each hidden state feeding the next through W_h, and a table: 4,800 parameters for 5 steps and for 500. A 6 by 6 grid of attention scores with one row shaded as one softmax, and a table: 100 positions give 10,000 scores, 1,000 give 1,000,000.](diagrams/03-weights-shared-across-positions.svg)

*One set of weights at every place, every timestep and every pair of positions; only the attention scores grow with the input.*

### 2.1. Convolutional layers: the same weights at every place

`Layer_Dense` treats its input as a flat row. To the first layer of `nn-p01`, the pixels at positions $(5, 5)$ and $(5, 6)$ of an image are two unrelated columns of $\mathbf{X}$, and an edge learned in one corner has to be learned again in every other.

A **convolutional layer** applies one small array of weights, a **filter**, at every position of the image. The output of a filter at one position is the weighted sum of a small patch plus a bias, the neuron of post 01 with a patch as its input. Sliding the filter gives a grid of such outputs, and a layer has several filters. It is a dense layer with two constraints: each output sees a patch only, and all outputs of one filter share their weights.

The parameter counts show what the sharing buys. The first layer of `nn-p01` maps 784 pixels to 128 neurons and has $784 \cdot 128 + 128 = 100{,}480$ parameters. A convolutional layer of 8 filters of size $5 \times 5$ on the same one-channel image has $8 \cdot 25 + 8 = 208$, whatever the size of the image.

The price is in the backward pass. A shared weight is used at every position, so its gradient is the sum of the contributions from all of them, the rule of post 11 for a variable that reaches the loss along several paths.

What to read:

- **The paper.** LeCun, Bottou, Bengio and Haffner, *"Gradient-Based Learning Applied to Document Recognition"* (Proceedings of the IEEE, 1998). It describes LeNet-5, a convolutional network that reads handwritten digits, trained by backpropagation.
- **The explainer.** The convolutional-network notes of Stanford's course CS231n.
- **From scratch.** The sibling series `cnn-from-scratch` builds the layer in NumPy on this series' classes: the forward pass with loops (`cnn-004`), the backward pass (`cnn-006`), and a LeNet-style network trained on MNIST against the dense baseline of `nn-p01` (`cnn-011`, `cnn-012`). The assignments of CS231n also have the reader write a convolution's forward and backward pass in NumPy.

### 2.2. Recurrent layers: the same weights at every timestep

A dense layer has a fixed number of inputs and treats every row of a batch as unrelated to the others. A sentence or a sensor trace has neither property: its length varies, and element $t$ depends on what came before.

A **recurrent layer** reads the sequence one element at a time and carries a **hidden state** from each step to the next. In this series' row convention,

$$\mathbf{h}_t = \tanh(\mathbf{x}_t \mathbf{W}_x + \mathbf{h}_{t-1} \mathbf{W}_h + \mathbf{b})$$

which is a dense layer whose input is the current element together with its own previous output, followed by tanh, the squashing activation met in post 06. The same $\mathbf{W}_x$, $\mathbf{W}_h$ and $\mathbf{b}$ are used at every step, so the number of parameters does not depend on the length of the sequence: with 10 features and 64 hidden units it is $10 \cdot 64 + 64 \cdot 64 + 64 = 4{,}800$ for 5 steps and for 500.

The backward pass runs through the steps in reverse and is called backpropagation through time. It multiplies by the recurrent weights and by a tanh slope once per step, so a gradient shrinks or grows roughly geometrically with the distance it travels (Hochreiter, 1991; Bengio, Simard and Frasconi, 1994). It is the same kind of compounding that post 04 measured on six stacked dense layers, here along time instead of depth. **LSTM** (Hochreiter and Schmidhuber, 1997) answers it with a second state that is carried forward by addition and guarded by gates built from sigmoids; the forget gate of the version used today was added by Gers, Schmidhuber and Cummins (2000). **GRU** (Cho et al., 2014) is a later gated cell with fewer parts. Gating reduces the vanishing of gradients over long spans and does not remove it.

What to read:

- **The paper.** Hochreiter and Schmidhuber, *"Long Short-Term Memory"* (Neural Computation, 1997).
- **The explainer.** Olah, *"Understanding LSTM Networks"* (2015), an essay that draws the cell gate by gate.
- **From scratch.** The sibling series `rnn-from-scratch` builds the cell (`rnn-004`), backpropagation through time (`rnn-007`), and an LSTM and a GRU in NumPy (`rnn-013`, `rnn-016`). Karpathy's essay *"The Unreasonable Effectiveness of Recurrent Neural Networks"* (2015) comes with `min-char-rnn`, a character-level recurrent network in about a hundred lines of NumPy; its parameter update is the AdaGrad of post 25.

### 2.3. Attention and transformers: every position reads every other

A recurrent layer passes information between distant positions through every step in between, and it cannot compute step $t$ before step $t - 1$. **Attention** connects positions directly. Each position issues a query, every position offers a key and a value, and the output at a position is an average of the values, weighted by how well each key matches the query:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

Every part is already in the series. $\mathbf{Q}$, $\mathbf{K}$ and $\mathbf{V}$ are each produced by a dense layer, with one row per position; $\mathbf{Q} \mathbf{K}^\top$ is the matrix product of post 02; the softmax is the row-wise softmax of post 06; $d_k$ is the width of a key. The weights are again shared across positions.

Bahdanau, Cho and Bengio (2015) added attention to a recurrent translation model. The **transformer** of Vaswani et al. (2017) removed the recurrence and built the model from attention, dense layers, residual connections and layer normalisation (section 3), with the order of the positions supplied as an extra input. All positions of a sequence are then computed at once. The cost is the score matrix, one score per pair of positions: 100 positions give 10,000 scores and 1,000 give 1,000,000. The large language models in wide use are transformers.

What to read:

- **The paper.** Vaswani et al., *"Attention Is All You Need"* (NeurIPS, 2017).
- **The explainer.** Alammar, *"The Illustrated Transformer"* (2018).
- **From scratch.** Karpathy's video lecture *"Let's build GPT: from scratch, in code, spelled out"* (2023) writes a small transformer language model in PyTorch, line by line. In the family, `rnn-030` and `rnn-032` reach attention from the recurrent side, and the sibling series `transformer` will take it from there.

---

## 3. New training infrastructure

These additions change how well a deep stack of layers trains. The sibling series `cnn-from-scratch` builds the first three.

### 3.1. Normalisation

Post 33 chooses the scale of the initial weights so that the spread of the activations survives the depth of the network. That holds at the start of training and nothing keeps it true afterwards. **Batch normalisation** (Ioffe and Szegedy, 2015) is a layer that, during training, subtracts the mean of each feature over the mini-batch and divides by its standard deviation, then applies a learned scale and shift. At evaluation it uses averages stored during training, so like dropout it behaves differently in the two modes.

The paper explained the benefit by "internal covariate shift", its name for the way a layer's input distribution moves as the layers below it learn. Santurkar et al. (2018) tested that explanation, found little support for it, and argued that the layer helps because it makes the loss surface smoother.

**Layer normalisation** (Ba, Kiros and Hinton, 2016) takes the mean and standard deviation over the features of each sample instead of over the batch, so it does not depend on the batch at all. It is the form transformers use. In the family, `cnn-014` builds batch normalisation with its backward pass.

- Ioffe and Szegedy, *"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"* (ICML, 2015).
- Santurkar, Tsipras, Ilyas and Madry, *"How Does Batch Normalization Help Optimization?"* (NeurIPS, 2018).

### 3.2. Residual connections

He, Zhang, Ren and Sun (2016) reported that a plain 56-layer convolutional network reached a higher *training* error than a 20-layer one on CIFAR-10. That is not overfitting, which would show in the test error only: the deeper network was harder to optimise. Their **residual connection** adds the input of a block to its output:

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

In the backward pass the block's `dinputs` is then the gradient through $F$ plus the upstream gradient itself, unchanged. However small the first term becomes, the second gives the gradient a direct path to the earlier layers. With such blocks the paper trained networks of 50, 101 and 152 layers on ImageNet. Transformers place a residual connection around every attention and dense sub-layer. In the family, `cnn-015` builds the block.

- He, Zhang, Ren and Sun, *"Deep Residual Learning for Image Recognition"* (CVPR, 2016).

### 3.3. Learning-rate schedules beyond $1/(1 + d \cdot t)$

Post 23 built one schedule, $\alpha_t = \alpha_0 / (1 + d \cdot t)$, inside `pre_update_params`. Any other schedule replaces that one line.

- **Cosine decay** follows half a cosine wave from $\alpha_0$ to a floor $\alpha_{\min}$ over $T$ updates fixed in advance: $\alpha_t = \alpha_{\min} + \tfrac{1}{2}(\alpha_0 - \alpha_{\min})(1 + \cos(\pi t / T))$. Loshchilov and Hutter (2017) introduced it with **warm restarts**, which return the rate to $\alpha_0$ at intervals.
- **Warmup** starts near zero and raises the rate over the first updates. Vaswani et al. (2017) warmed up over 4,000 updates and then decayed.
- **The one-cycle policy** (Smith, 2018) raises the rate to a peak and then lowers it below its starting value.

Cosine decay and post 23's schedule spend the same budget differently. With $\alpha_0 = 1$, halfway through 10,000 updates the cosine schedule (floor 0) is at 0.5, while post 23's schedule with $d = 10^{-3}$ is at $1/6 = 0.167$; at the end the cosine is at 0 and the other at $1/11 = 0.091$. Which is better on a given problem is a question for several seeds, as in Part VI. `cnn-017` builds these schedules.

### 3.4. Mixed precision

The spiral training runs of this series compute in float32, the precision `nnfs.init()` sets. Graphics processors can compute in 16-bit floats, which take half the memory and run faster on hardware built for them. **Mixed-precision training** (Micikevicius et al., 2018) runs the forward and backward passes in 16 bits, keeps a float32 copy of the weights for the update, and multiplies the loss by a constant so that small gradients do not round to zero. Post 21 showed how a gradient check fails in float32 although the code is right; halving the precision again makes that kind of care routine. Frameworks offer it as a switch, in PyTorch as `torch.autocast` with a gradient scaler.

### 3.5. Distributed training

When a model or a dataset outgrows one device, the work is spread over several. In **data-parallel** training every device holds a copy of the model, computes the gradient of its own part of the batch, and the gradients are combined before one update: the mini-batch of post 32, computed in pieces. PyTorch's `DistributedDataParallel` does this. In **model-parallel** training the model itself is split across devices.

---

## 4. New problem framings

Every post of this series trained on inputs with known correct outputs, which is **supervised learning**. The framings below change where the target comes from; the networks, losses and optimisers stay.

### 4.1. Self-supervised learning

A **self-supervised** task takes its labels from the input: predict the next word of a text, a word that was masked out, or a patch of an image that was hidden. The loss for the first two is the categorical cross-entropy of post 08, over a vocabulary instead of three classes. No labelling is needed, so the training set can be far larger than any labelled one.

- Devlin, Chang, Lee and Toutanova, *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* (NAACL, 2019). Masked-word prediction.
- Radford et al., *"Language Models are Unsupervised Multitask Learners"* (OpenAI, 2019). The GPT-2 report: next-word prediction.
- He et al., *"Masked Autoencoders Are Scalable Vision Learners"* (CVPR, 2022). Masked patches of images.

### 4.2. Transfer learning and fine-tuning

A network trained on one large task can be the starting point for another. **Fine-tuning** continues training a pretrained network on a small labelled set for the task of interest: in the terms of post 33 it is an initialisation, taken from another problem instead of from a random draw.

**LoRA** (Hu et al., 2022) makes fine-tuning cheap. It freezes a pretrained weight matrix $\mathbf{W}$ and trains only a low-rank correction, the product of two thin matrices, which is added to it. For a $1{,}000 \times 1{,}000$ matrix and rank 8 that is $8 \cdot (1{,}000 + 1{,}000) = 16{,}000$ trained numbers in place of 1,000,000, or 1.6 percent.

- Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"* (ICLR, 2022).

### 4.3. Reinforcement learning

In **reinforcement learning** an agent acts in an environment, receives a reward now and then, and has no label saying which action was right. Two families of method turn that into a gradient. Value-based methods train a network to predict the future reward of each action, a regression whose target is built from the network's own later predictions: Mnih et al. (2013) trained a convolutional network this way to play Atari games from the pixels of the screen. Policy-gradient methods (Williams, 1992) raise the log-probability of the actions that were followed by high reward, which in code is the cross-entropy gradient of post 19 with each sample weighted by its reward.

- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (MIT Press, 2018). The standard textbook.
- Mnih et al., *"Playing Atari with Deep Reinforcement Learning"* (NeurIPS Deep Learning Workshop, 2013).
- OpenAI's *Spinning Up in Deep RL* (2018), a short course with reference implementations.

The sibling series `rl-mastery` will cover this framing.

### 4.4. Diffusion models

A **diffusion model** generates images or audio. Noise is added to training examples in many small steps, and a network is trained to predict the noise that was added, which in the simplified objective of Ho, Jain and Abbeel (2020) is a mean squared error, the loss of `nn-p04`. Generation starts from pure noise and removes it step by step with the trained network. The idea goes back to Sohl-Dickstein et al. (2015).

- Ho, Jain and Abbeel, *"Denoising Diffusion Probabilistic Models"* (NeurIPS, 2020).
- Weng, *"What are Diffusion Models?"* (2021), an essay that derives the objective step by step.

The sibling series `diffusion` will cover this framing.

---

## 5. Three books

- **Goodfellow, Bengio and Courville**, *Deep Learning* (MIT Press, 2016). This series overlaps its chapters 6 to 8 (feedforward networks, regularisation, optimisation). Chapters 9 and 10, on convolutional networks and sequence modelling, are the natural continuation.
- **Zhang, Lipton, Li and Smola**, *Dive into Deep Learning* (Cambridge University Press, 2023). Its text is interleaved with runnable code, and it implements a convolution, a recurrent network and attention from a framework's array operations.
- **Bishop and Bishop**, *Deep Learning: Foundations and Concepts* (Springer, 2024). The successor to Bishop's *Pattern Recognition and Machine Learning*, with chapters on transformers and diffusion models.

Howard and Gugger, *Deep Learning for Coders with fastai and PyTorch* (O'Reilly, 2020), starts instead from working models in a framework and opens them up later.

---

## 6. Three frameworks

A framework supplies what this series wrote by hand, and one thing it did not write: **automatic differentiation**, an engine that records the operations of the forward pass and produces every `backward` from them. Karpathy's `micrograd` is such an engine for single numbers in a few short files, and his video lectures *"Neural Networks: Zero to Hero"* start from it.

- **PyTorch** (Paszke et al., 2019). Much published research code is written in it. The mapping is direct: `torch.nn.Linear` is `Layer_Dense`, with the weights stored one row per neuron as in posts 01 to 03; `torch.nn.CrossEntropyLoss` takes logits and is the combined softmax and loss of post 19; `loss.backward()` replaces the four `backward` calls; `optimizer.step()` replaces the update calls.
- **JAX.** NumPy's interface with function transformations: `grad` returns the gradient of a function, `jit` compiles it, and `vmap` adds a batch axis.
- **TensorFlow with Keras.** Widely deployed, with Keras as its layer interface.

One is enough to begin with, and PyTorch is the usual first choice.

---

## 7. Choosing what to study next

The choice follows from the data and the task.

| The problem | What it needs beyond this series | Start with |
|---|---|---|
| Rows of unrelated features (tables, regression) | nothing new in the layers | `nn-p04` |
| Images, spectrograms, other grids | weights shared over positions | `cnn-from-scratch` |
| Sequences read in order: text, time series | weights shared over time, a hidden state | `rnn-from-scratch` |
| Long text, language models | attention | section 2.2 first, then section 2.3 |
| Few labels, a related large dataset | a pretrained network | section 4.2, in a framework |
| Generating images or audio | a denoising objective | section 4.4 |
| Decisions with delayed reward | no labels; reward-weighted gradients | section 4.3 |

For a reader with no problem in mind, one order that works:

1. Re-implement `nn-p01` in PyTorch and compare its test accuracy over several seeds with the project's published figures.
2. Work through `cnn-from-scratch` as far as its MNIST network (`cnn-012`), which is measured against the same `nn-p01`.
3. Work through `rnn-from-scratch` as far as backpropagation through time and vanishing gradients (`rnn-007`, `rnn-008`), then its LSTM.
4. Read Alammar's essay, then Vaswani et al., then follow Karpathy's lecture and write the model.
5. Read the batch-normalisation and residual papers beside `cnn-014` and `cnn-015`.
6. Pick one framing from section 4 and build something small in it.

---

## 8. What can go wrong?

**The loop is assumed to need no change at all.** Three things do change, and all are small; they are the bottom row of the second figure. A layer that behaves differently in training and in evaluation, dropout in post 31 and batch normalisation here, needs a loop that knows which of the two it is running; the dropout layer of `nn-p01` takes a `training` argument for this, and the project's evaluation pass sets it to false. A recurrent layer carries state between calls, which has to be reset between unrelated sequences. And a layer with shared weights sums gradient contributions inside its `backward`, where `Layer_Dense` gets the sum from one matrix product.

**A re-implementation in a framework does not reproduce the series' numbers.** The defaults differ. `torch.nn.Linear` draws its weights and its biases from a uniform distribution whose width depends on the number of inputs, where this series began with `0.01 * randn` and zero biases; the `eps` of `torch.optim.Adam` defaults to $10^{-8}$, where the series' classes use $10^{-7}$; the random streams differ. A matching architecture with a different final accuracy is expected, and the comparison that means something is that of Part VI: several seeds on each side.

**A single run in a paper or a tutorial is read as a ranking.** Post 27 found Adam best of six optimisers on seed 0 and on no other seed of five. A table with one number per method deserves the same caution.

**The reading is done in the wrong order.** The LSTM and transformer papers were written for readers who knew the models before them, so the essays of section 2 come first. The transformer paper presents its model as a replacement for recurrent encoder-decoders, so recurrence comes before attention. One framework and one device carry everything in section 7.

---

## 9. Summary

| Addition | What it adds to the dense stack | First source |
|---|---|---|
| Convolution | one filter's weights reused at every position of a grid | LeCun et al. (1998); `cnn-from-scratch` |
| Recurrence | one layer's weights reused at every timestep, with a carried state | Hochreiter and Schmidhuber (1997); `rnn-from-scratch` |
| Attention | every position reads every other, weighted by a softmax of scores | Vaswani et al. (2017) |
| Normalisation | activations rescaled inside the forward pass | Ioffe and Szegedy (2015) |
| Residual connection | the upstream gradient passed back unchanged beside the block's own | He et al. (2016) |
| Schedules | another formula in `pre_update_params` | Loshchilov and Hutter (2017) |
| New framings | another source for the target; the same losses | section 4 |

The series began with a neuron as a weighted sum plus a bias (post 01), and the last optimiser it built is Adam, two moving averages each divided by $1 - \beta^t$ to undo its start at zero (post 27). Everything in the table is written against the same calls: `forward`, `backward` and the optimiser's update.

---

## Common pitfalls

1. **Treating a new layer as new mathematics.** Convolution, recurrence and attention are weighted sums, shared weights and a softmax; the chain rule of post 11 covers all three.
2. **Editing nothing in the loop.** Training and evaluation modes, and carried state, are the loop's business (section 8).
3. **Expecting a framework to match the from-scratch numbers.** Initialisation and defaults differ; compare over seeds.
4. **Skipping recurrence on the way to transformers.** The problem attention solves is defined there.
5. **Reading a paper's single-run table as an ordering.** Part VI is the counterexample.

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, chapters 9 and 10 (MIT Press, 2016).
- He, K., Zhang, X., Ren, S., and Sun, J., *"Deep Residual Learning for Image Recognition"* (CVPR, 2016).
- Hochreiter, S. and Schmidhuber, J., *"Long Short-Term Memory"* (Neural Computation, 1997).
- Ioffe, S. and Szegedy, C., *"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"* (ICML, 2015).
- LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P., *"Gradient-Based Learning Applied to Document Recognition"* (Proceedings of the IEEE, 1998).
- Vaswani, A., et al., *"Attention Is All You Need"* (NeurIPS, 2017).
- Zhang, A., Lipton, Z. C., Li, M., and Smola, A. J., *Dive into Deep Learning* (Cambridge University Press, 2023).

The other works are named in the sections that use them. Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

This is the last post of the series. The next reading is outside it, in the two sibling series that start from these classes: `cnn-from-scratch` at `cnn-001` and `rnn-from-scratch` at `rnn-001`. The two posts of this series that they lean on most are worth a second look first.

- **[Post 33 - Weight initialisation](../33-weight-initialisation/index.md):** the variance argument that `cnn-from-scratch` repeats for convolutional layers (`cnn-007`), and the question of the initial scale that `rnn-from-scratch` asks again of a recurrent matrix (`rnn-014`).
- **[Post 16 - Coding backpropagation](../16-coding-backpropagation/index.md):** the dense `backward` that `cnn-from-scratch` generalises to a convolution (`cnn-006`) and `rnn-from-scratch` to a cell unrolled through time (`rnn-007`).
