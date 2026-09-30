# Neural Networks from Scratch

Standard: 2.2.0 · Plan version: 1.0.0 · Status: draft · Date: 2026-10-03

Neural Networks from Scratch is a thirty-five-post series that builds a multi-layer perceptron by hand in Python and NumPy, with no framework, no autograd, and no hidden step: neurons and layers, the forward pass, categorical cross-entropy, the calculus that backpropagation needs, the backward pass of every component, six optimisers from vanilla gradient descent to Adam, generalisation, validation, L1 and L2 penalties, dropout, mini-batching, weight initialisation, and the sigmoid plus binary cross-entropy pair. Four projects (MNIST, two-moons, Fashion-MNIST, California housing) put the classes to work on real data. A reader who finishes it can write `Layer_Dense`, `Activation_ReLU`, `Activation_Softmax_Loss_CategoricalCrossentropy`, and `Optimizer_Adam` from memory, derive every gradient in them from the chain rule, train a dense network to about 98 percent on MNIST, read a train-test gap, and choose a regulariser for it.

# 1. Purpose and authority

This plan owns the curriculum of the series: which posts exist, in which order, what each one promises, what it requires and recommends, what it covers, which figures and labs carry it, and which primary sources anchor it. Layout, identifiers, the sidecar schema, the body template, the question bank, diagrams, tooling, and workflow are owned by series-standard/STANDARD.md at the version `series.yaml` pins, not by this document. `series.yaml` is the machine form of this lock: its `parts[].posts` lists the reading order, its ids are allocated from `next_post_ordinal` and never reused, and a dropped id goes to `retired`. The plan stage of `tools/validate.py` keeps the two in agreement.

All thirty-five posts are shipped, so this plan was written from the posts: the TL;DR, outcome bullets, section headings, "What to read next" lists, INDEX.md's dependency table, exercises.md, glossary.md, common_pitfalls.md, and `verify/RESULTS.md` are its sources. Where a post and this plan disagree the post wins until its migration commit, and Section 9 lists every such divergence.

# 2. Audience, entry, and exit levels

## Reader profiles

- The programmer with no machine-learning background: can write a Python list, call a function, and use a `for` loop, and has never used NumPy. This is the primary reader, and nn-001 to nn-005 assume nothing more.
- The framework user who wants the inside view: has called `nn.Linear` and `optim.Adam` and wants to know what each one computes in forward, backward, and update. This reader may start at nn-006 and should not skip Part V.
- The reader heading for cnn-from-scratch or rnn-from-scratch: both sibling series assume this one in full and name its ids in their entry requirements.

## Entry requirement

No other series is required. The calculus the series needs (derivatives, partial derivatives, the chain rule) is taught in nn-010 and nn-011 from zero; nothing beyond polynomial algebra is assumed before them. Readers who already have single-variable calculus can treat nn-010 as a review. The optional `nnfs` helper package (Kinsley and Kukieła) supplies `spiral_data` and a fixed seed; everything else is NumPy.

## Exit levels

- After nn-005 (beginner, end of Part I): the reader can code a neuron and a layer in plain Python and in NumPy, predict the shape of any `np.dot`, write `Layer_Dense`, and apply the `axis`, `keepdims`, and broadcasting rules without running the code.
- After nn-009 (beginner, end of Part III): the reader has a complete forward pass over the spiral data, computes categorical cross-entropy and accuracy for a batch, and can say why random search fails and gradient descent must exist.
- After nn-011 (beginner, end of Part IV): the reader differentiates polynomials, computes partial derivatives, reads a gradient component by component, and states the chain rule for a composition of any depth.
- After nn-021 (intermediate, end of Part V): the reader has `backward` on every component, can derive the three dense-layer gradients and the combined softmax plus cross-entropy gradient $(\hat{y} - y)/N$, and runs a fifteen-line forward and backward script that produces four gradient arrays.
- After nn-027 (intermediate, end of Part VI): the reader has six optimisers on the shared `pre_update_params` / `update_params` / `post_update_params` contract and can explain what each one fixes in the last; Adam reaches 96.3 percent on the spiral.
- After nn-031 (intermediate, end of Part VII): the reader runs a forward-only test pass, reads the gap, uses a three-way split or k-fold, and adds L1, L2, or dropout to the network with the train-versus-test switch in place.
- After nn-035 (intermediate): the reader trains with mini-batches, initialises with He or Glorot, solves a binary problem with sigmoid plus BCE, and can place convolution, recurrence, attention, and normalisation on a map of what each adds to this stack. This is the entry point of cnn-from-scratch and rnn-from-scratch.

## Gates

`series.yaml` declares no formal gates (`gates: []`). The de facto gates are the milestones INDEX.md lists at the end of each phase: writing `Layer_Dense` from memory after nn-005, hand-computing softmax and a cross-entropy loss after nn-009, writing `Layer_Dense.backward` from memory and stating $(\hat{y} - y)/N$ after nn-021, training the spiral classifier past 90 percent after nn-027, and implementing dropout with the correct train-test behaviour after nn-031. `gradient_checking.md` is the numerical gate for Part V.

# 3. Pedagogical rules

- Pure NumPy. Every class in the series and in `projects/*/nn.py` is NumPy only; `nnfs` is used for the spiral dataset and the seed, scikit-learn only to download the housing data in project 04. No framework appears in code until nn-035 names PyTorch as the next step.
- Implement before any library. Each operation is stated in one sentence, coded by hand in plain Python where that is instructive (nn-001, nn-012, nn-013), then collapsed into NumPy, then wrapped in a class on the `forward` and `backward` contract.
- Shapes first. Every post traces the shape of every intermediate array before it writes code and keeps a shape diary; the convention is a batch of $N$ rows, weights $(n_\text{inputs}, n_\text{neurons})$ from nn-004 onward, biases $(1, n_\text{neurons})$.
- Derive, then verify numerically. Every gradient is derived from the chain rule on a worked example with numbers (nn-012 to nn-015, nn-018, nn-019, nn-034) and the matrix form is checked against those numbers before it is coded. Proofs never go beyond what the chain rule and the power, sum, and product rules give.
- One dataset until the projects. The spiral dataset (three classes, 100 points per class) is the fixture from nn-004 to nn-031, so every optimiser and regulariser is compared on the same problem; real datasets appear only in the projects.
- Every number is produced by committed code. Accuracy and loss figures quoted in prose are reproduced by `verify/optimizer_results.py` and `verify/regularization_results.py`, and `verify/RESULTS.md` and `verify/projects_results.md` record the runs and mark the claims that did not hold.
- Each optimiser fixes the last one's flaw. Part VI is ordered so that every post names the failure of the previous optimiser on the spiral before it introduces the remedy, and nn-027 is presented as the synthesis of nn-024 and nn-026.
- One template. Each post has a TL;DR, three outcome bullets, a hero figure, numbered sections, Anticipated questions, Common pitfalls, Further reading pointing at REFERENCES.md, and "What to read next"; exercises and quizzes live in exercises.md and quizzes.md, never in the post.
- One notation. `notation_guide.md` is the arbiter for symbols and shapes and `glossary.md` for terms; AUDIT_REPORT.md records that the two disagree today and that the migration resolves it (Section 9).

# 4. Curriculum lock

35 posts in 8 parts. Post numbers are permanent; a post inserted later takes the next ordinal from `series.yaml` and the directories after it renumber (STANDARD.md Section 2.2).

| Part | Title | Posts | Ids |
|---|---|---|---|
| I | Part I - Foundations | 01-05 | nn-001 to nn-005 |
| II | Part II - Activations and forward pass | 06-07 | nn-006 to nn-007 |
| III | Part III - Loss and optimisation | 08-09 | nn-008 to nn-009 |
| IV | Part IV - Calculus for backpropagation | 10-11 | nn-010 to nn-011 |
| V | Part V - Backpropagation | 12-21 | nn-012 to nn-021 |
| VI | Part VI - Optimisers | 22-27 | nn-022 to nn-027 |
| VII | Part VII - Generalisation and regularisation | 28-31 | nn-028 to nn-031 |
| VIII | Part VIII - Practical training and extensions | 32-35 | nn-032 to nn-035 |

Retired ids: none. `next_post_ordinal` is 36.

# 5. Per-post specifications

Each spec is a drafting brief and the prose form of the lock; the heading id, title, part, level, scope, and dependencies must equal `series.yaml` and the sidecar (STANDARD.md Section 18.3). The sidecars carry no `level` or `prerequisites` today; the values below are the lock and the sidecars take them at migration (Section 9).

## Part I - Foundations

### nn-001 - Neurons and layers

- **Meta:** Part I; beginner; core; proof none; code included.
- **Central question:** What does a single neuron compute, and what changes when several neurons share the same inputs?
- **Promise:** The reader can state in one sentence what a neuron and a layer compute, code both in plain Python and then in two lines of NumPy, and predict every intermediate shape as one sample becomes a batch.
- **Coverage:** why from scratch, where the neuron came from, the weighted sum plus bias, a neuron with three and four inputs, a layer by hand, the same layer three ways, why NumPy wins, batches and the shape diary, the core formula $z = Wx + b$
- **Dependencies:** requires none; supports nn-002, nn-003.
- **Visuals:** hero `01-neuron-anatomy.svg` (inputs, weights, bias, and the sum); supporting figures for a layer of three neurons over four inputs and the batch shape diary.
- **Lab/gate:** Add a fourth neuron to the three-neuron layer with chosen weights, verify its output by hand and in code, and build a two-layer network with nested loops and no NumPy (exercises.md Part 1).
- **Anchors:** McCulloch and Pitts 1943, Rosenblatt 1958, Kinsley and Kukieła 2020 chapter 2, Goodfellow, Bengio and Courville chapter 6.

### nn-002 - NumPy and the dot product

- **Meta:** Part I; beginner; core; proof none; code included.
- **Central question:** Which of the three forms of `np.dot` runs for a given pair of shapes, and which neural-network object does each form represent?
- **Promise:** The reader can name the three forms of `np.dot`, match each to a neuron, a layer, or a batch, predict from shapes alone whether a call succeeds, and say when the transpose is needed and why a batch call is order-sensitive.
- **Coverage:** the dot product formally, vector by vector, matrix by vector and why order matters, matrix by matrix and the inner-dimension rule, the transpose, batching end to end, the shape diary for `np.dot`
- **Dependencies:** requires nn-001; supports nn-003, nn-005, nn-014.
- **Visuals:** hero `01-three-forms.svg` (the three forms of one call); supporting figures for the inner-dimension rule and the transpose.
- **Lab/gate:** Verify a dot product by hand against `np.dot`, pass a $(5, 4)$ batch through a three-neuron layer and confirm the $(5, 3)$ output, and time `np.dot` against a nested loop on $(1000, 100)$ by $(100, 64)$ (exercises.md Part 2).
- **Anchors:** Harris et al. 2020 (NumPy), the NumPy `np.dot` documentation, Strang, Introduction to Linear Algebra chapter 1, Kinsley and Kukieła 2020 chapter 2.

### nn-003 - Stacking layers and the forward pass

- **Meta:** Part I; beginner; core; proof idea; code included.
- **Central question:** How does one layer's output become the next layer's input, and why is a stack of linear layers still one linear function?
- **Promise:** The reader can implement a two-layer forward pass in NumPy and predict every intermediate shape, pick the weight-matrix shape for each layer from the sizes around it, and explain in one sentence why linear layers without activations do not add power.
- **Coverage:** from one layer to many, a recap of one layer, the architecture drawn out, the forward-pass chain, tracing the shapes, coding two layers and extending to more, the weight-shape rule restated, what the forward pass means
- **Dependencies:** requires nn-001, nn-002; supports nn-004, nn-006, nn-012.
- **Visuals:** hero `01-multi-layer-anatomy.svg` (two layers with shape badges at every stage); a supporting figure for the collapse of two linear maps into one.
- **Lab/gate:** Change layer sizes to five and two neurons and derive the weight shapes, add a third layer and print every intermediate shape, and set a layer to the identity to see the output unchanged (exercises.md Part 3).
- **Anchors:** Kinsley and Kukieła 2020 chapter 3, Goodfellow, Bengio and Courville section 6.1, Nielsen, Neural Networks and Deep Learning chapter 1.

### nn-004 - The Dense layer class and spiral data

- **Meta:** Part I; beginner; core; proof none; code included.
- **Central question:** How is the forward pass packaged into a reusable class, and what dataset will the rest of the series train on?
- **Promise:** The reader can implement `Layer_Dense` with weight initialisation and `forward`, generate the spiral dataset and say in one sentence why it is hard, and choose between the two weight-matrix conventions and name the one the series uses.
- **Coverage:** why a class, the spiral dataset and why not MNIST yet, a Python OOP refresher, the weight-matrix convention switch to $(n_\text{inputs}, n_\text{neurons})$, `__init__` and `forward`, using the class on one and two layers with a shape trace
- **Dependencies:** requires nn-003; supports nn-005, nn-007, nn-016, nn-033, rnn-001, rnn-004.
- **Visuals:** hero `01-spiral-data.svg` (300 points in three intertwined classes); a supporting figure for the class as state plus computation.
- **Lab/gate:** Vary `samples` and `classes` in `spiral_data` and plot the result, inspect the initial weights of `Layer_Dense(2, 64)` across two runs, and replace the `0.01` scale by `1.0` to see the outputs blow up (exercises.md Part 4).
- **Anchors:** Kinsley and Kukieła 2020 chapter 3 and the `nnfs` package, Stanford CS231n (the spiral classification demo), the Python data model documentation.

### nn-005 - Array summation, keepdims, and broadcasting

- **Meta:** Part I; beginner; core; proof none; code included.
- **Central question:** What do `axis` and `keepdims` do to a reduction, and which pairs of shapes can NumPy combine element-wise?
- **Promise:** The reader can predict the shape of `np.sum(a, axis=k, keepdims=...)` for any 2-D array, decide from two shapes whether they broadcast, and spot the silent bug where a $(n,)$ array broadcasts as a row when a column was intended.
- **Coverage:** `axis` as the dimension that disappears, `keepdims=True` and the per-row max subtraction bug, the broadcasting rules in the order NumPy applies them, broadcasting the bias in a forward pass, the shape diary for reductions
- **Dependencies:** requires nn-002; recommends nn-004; supports nn-006, nn-008, nn-031, nn-033.
- **Visuals:** hero `01-axis-summation.svg` (summing a 3 by 3 array along each axis); supporting figures for the keepdims bug and the broadcasting alignment rule.
- **Lab/gate:** Predict and then check the shapes of `np.sum` on a $(3, 4)$ array for every axis with and without keepdims, explain why $(5, 3) + (3,)$ works and $(3, 5) + (4,)$ fails, and fix the failing case (exercises.md Part 5).
- **Anchors:** The NumPy broadcasting documentation, Harris et al. 2020, Kinsley and Kukieła 2020 chapter 4.

## Part II - Activations and forward pass

### nn-006 - Activation functions: ReLU and Softmax

- **Meta:** Part II; beginner; core; proof idea; code included.
- **Central question:** Why does a deep network need a non-linearity between layers, and why do hidden layers get ReLU while the output layer gets softmax?
- **Promise:** The reader can state in one sentence why linear layers need a non-linearity between them, implement `Activation_ReLU` and `Activation_Softmax` on the `Layer_Dense` pattern, and explain why softmax subtracts the per-row maximum before exponentiating.
- **Coverage:** what activations buy, ReLU formally and why not sigmoid or tanh for hidden layers, ReLU's implementation, why ReLU fails at the output, softmax formally and where it comes from, the numerical-stability trick, the forward pass end to end with the default Dense-ReLU-Dense-Softmax pattern
- **Dependencies:** requires nn-003, nn-005; supports nn-007, nn-008, nn-019, nn-033.
- **Visuals:** hero `01-why-nonlinearity.svg` (linear stack against ReLU stack on a curved boundary); supporting figures for the ReLU and softmax curves and the max-subtraction trick.
- **Lab/gate:** Apply ReLU by hand, compute softmax of $[1, 2, 3]$ and $[1001, 1002, 1003]$ with and without max subtraction, verify the outputs sum to one, and sweep a temperature $T$ through the softmax (exercises.md Part 6).
- **Anchors:** Nair and Hinton 2010 (ReLU), Glorot, Bordes and Bengio 2011, Bridle 1990 (softmax), Goodfellow, Bengio and Courville section 6.2.

### nn-007 - Coding the complete forward pass

- **Meta:** Part II; beginner; core; proof none; code included.
- **Central question:** How do the four classes built so far snap together into a working classifier, and why is its untrained output uniform?
- **Promise:** The reader can assemble the complete forward-pass script from `Layer_Dense`, `Activation_ReLU`, and `Activation_Softmax`, audit the shape of every intermediate array against the architecture, and explain why an untrained network outputs about $[1/3, 1/3, 1/3]$ and why that is the right baseline.
- **Coverage:** what the post integrates, the architecture, the complete script, tracing the shapes, what the uniform output means, what the script is not, extending the depth
- **Dependencies:** requires nn-004, nn-006; supports nn-008.
- **Visuals:** hero `01-pipeline-anatomy.svg` (spiral data through Dense, ReLU, Dense, Softmax with shapes and roles).
- **Lab/gate:** Run the complete forward pass in a notebook and confirm the $(300, 3)$ output rows sum to one and sit near one third each; then add a third hidden layer and re-audit the shapes.
- **Anchors:** Kinsley and Kukieła 2020 chapter 4, Stanford CS231n (the minimal network case study).

## Part III - Loss and optimisation

### nn-008 - Loss: categorical cross-entropy

- **Meta:** Part III; beginner; core; proof guided; code included.
- **Central question:** How is "how wrong" turned into one number for a batch of softmax outputs, and why is the negative log the right function?
- **Promise:** The reader can compute categorical cross-entropy for a batch against integer or one-hot labels, explain in one sentence why $-\log$ turns a probability into a loss, and state why predictions are clipped to $[10^{-7}, 1 - 10^{-7}]$ before the log.
- **Coverage:** why a loss is needed, the formula and where it comes from, the $-\log$ curve, a worked example, integer labels by advanced indexing and one-hot labels by multiply and sum, clipping, the `Loss` and `Loss_CategoricalCrossentropy` classes, the forward pass with loss, accuracy as the complementary metric
- **Dependencies:** requires nn-006, nn-007; recommends nn-005; supports nn-009, nn-018.
- **Visuals:** hero `01-cross-entropy-curve.svg` (the negative-log curve); supporting figures for the two label formats and the clipping band.
- **Lab/gate:** Compute the loss for three predictions by hand and in code, confirm that the untrained spiral network gives a loss near $\ln 3 \approx 1.0986$ and accuracy near one third.
- **Anchors:** Shannon 1948, Goodfellow, Bengio and Courville sections 5.5 and 6.2.2, Bishop 2006 section 4.3, Kinsley and Kukieła 2020 chapter 5.

### nn-009 - Introduction to optimisation

- **Meta:** Part III; beginner; core; proof idea; code included.
- **Central question:** Once a loss exists, why is gradient descent, and not random search, the way to move the parameters?
- **Promise:** The reader can explain why random selection fails on a 21-parameter classifier and random perturbation fails on non-trivial data, state the gradient-descent update rule and name each symbol, and say which of nn-010 to nn-027 covers which part of the training story.
- **Coverage:** the state of the project, random selection, random perturbation, why random search cannot scale, the gradient as the missing piece, what gradient descent is not, why this post comes before the calculus, the hill-in-fog metaphor
- **Dependencies:** requires nn-008; supports nn-010, nn-022.
- **Visuals:** hero `01-strategies-compared.svg` (random selection, random perturbation, and gradient descent on the spiral).
- **Lab/gate:** Run the two random strategies for a fixed budget of iterations, record the best loss each reaches, and compare with the single gradient step the post previews.
- **Anchors:** Cauchy 1847, Ruder 2016 (an overview of gradient descent optimisation algorithms), Goodfellow, Bengio and Courville section 4.3, Kinsley and Kukieła 2020 chapter 6.

## Part IV - Calculus for backpropagation

### nn-010 - Derivatives, partial derivatives, and gradients

- **Meta:** Part IV; beginner; core; proof guided; code optional.
- **Central question:** What is a gradient, and how does it reduce to ordinary derivatives taken one variable at a time?
- **Promise:** The reader can differentiate any polynomial with the power and sum rules, compute partial derivatives by holding the other variables constant, and read a gradient vector component by component as the sensitivity of the loss to one weight.
- **Coverage:** why this calculus specifically, the derivative formally with the power and sum rules and slope as sensitivity, partial derivatives with worked examples and why variable-freezing works, the gradient and its two properties, reading a gradient per component, connecting the calculus to the network
- **Dependencies:** requires nn-009; supports nn-011, nn-012, nn-017.
- **Visuals:** hero `01-derivative-as-slope.svg` (three tangent lines on one curve); supporting figures for a partial derivative as a slice and the gradient as a vector.
- **Lab/gate:** Differentiate $f(x) = 5x^3 + 2x$, compute the partials of $f(x, y) = x^2 y + 3y$, and check one derivative numerically with a finite difference (INDEX.md Phase 3 milestones).
- **Anchors:** Spivak, Calculus chapters 9 and 10, Stewart, Calculus chapter 14, 3Blue1Brown, Essence of calculus, Kinsley and Kukieła 2020 chapter 7.

### nn-011 - The chain rule

- **Meta:** Part IV; beginner; core; proof guided; code optional.
- **Central question:** How is the derivative of a loss buried several functions deep turned into a product of small local derivatives?
- **Promise:** The reader can state the chain rule in one sentence and apply it to a polynomial composition, identify the local derivative of each function in a small network composition, and predict how many factors appear in the gradient for a weight in any layer.
- **Coverage:** why the chain rule matters most, the rule formally and what it is not, a distance-time-fuel example, a polynomial example, the extended chain rule and its partial-derivative form, the chain rule applied to a network, the pattern in three steps
- **Dependencies:** requires nn-010; supports nn-012, nn-019.
- **Visuals:** hero `01-chain-rule.svg` (a chain of boxes, each with its local derivative, multiplied into one gradient).
- **Lab/gate:** Apply the chain rule to $f(g(x))$ with $g(x) = 2x + 1$ and $f(g) = g^2$, then count the local factors for a weight in the first layer of a two-layer network (INDEX.md Phase 3 milestones).
- **Anchors:** Spivak, Calculus chapter 10, Rumelhart, Hinton and Williams 1986, Goodfellow, Bengio and Courville section 6.5.2, Kinsley and Kukieła 2020 chapter 7.

## Part V - Backpropagation

### nn-012 - Backpropagation through a single neuron

- **Meta:** Part V; intermediate; core; proof guided; code included.
- **Central question:** What does the chain rule look like on the smallest network whose chain has more than one factor?
- **Promise:** The reader can identify the four chain-rule factors in a single-neuron backward pass, compute the gradient of a squared-error loss with respect to each weight and the bias of a three-input ReLU neuron by hand, and run a 200-iteration gradient-descent loop that drives the loss toward zero.
- **Coverage:** the smallest interesting case, the setup and why squared error here, the forward pass by hand, the backward pass and its pattern, all four gradients, one gradient-descent step and why the loss never reaches exactly zero, the Python implementation, why "back" propagation
- **Dependencies:** requires nn-010, nn-011; recommends nn-003; supports nn-013.
- **Visuals:** hero `01-single-neuron-backprop.svg` (one neuron, four local derivatives, right to left).
- **Lab/gate:** Reproduce the four gradients by hand with the post's numbers, run the 200-iteration loop, and confirm the loss curve.
- **Anchors:** Rumelhart, Hinton and Williams 1986, Nielsen, Neural Networks and Deep Learning chapter 2, Kinsley and Kukieła 2020 chapter 9.

### nn-013 - Backpropagation through a layer of neurons

- **Meta:** Part V; intermediate; core; proof guided; code included.
- **Central question:** Does the single-neuron recipe change when several neurons share the same inputs?
- **Promise:** The reader can apply the chain rule across all fifteen parameters of a three-neuron layer with one upstream gradient, write the per-neuron backward pass as a Python loop and see where it becomes a matrix multiplication, and predict the gradient of any weight from the input vector and the upstream gradient.
- **Coverage:** from one neuron to a layer, the architecture, the chain rule for one weight and its generalisation, the forward pass with numbers, the backward pass, one gradient-descent step, the full implementation and where the matrix product hides, what this post adds beyond nn-012
- **Dependencies:** requires nn-012; supports nn-014.
- **Visuals:** hero `01-layer-backprop.svg` (three neurons over four inputs, one upstream gradient, twelve weight gradients).
- **Lab/gate:** Compute the twelve weight gradients and three bias gradients by hand for the post's layer and verify them against the loop implementation.
- **Anchors:** Rumelhart, Hinton and Williams 1986, LeCun et al. 1998 (Efficient BackProp), Kinsley and Kukieła 2020 chapter 9.

### nn-014 - Matrices in backpropagation

- **Meta:** Part V; intermediate; core; proof guided; code included.
- **Central question:** How do twelve per-weight gradients collapse into one matrix product, and why does that product sum over a batch for free?
- **Promise:** The reader can compute weight and bias gradients for a dense layer in two NumPy lines, read off the shape of each gradient array from the shapes of inputs and layer, and explain why the matrix form automatically sums contributions across a batch.
- **Coverage:** the case for matrices, the weight gradient as an outer product and why not a dot product, where $\partial L / \partial Z$ comes from, numerical verification against nn-013, bias gradients, batches with a worked example, the two-line backward pass, the convention reconciliation
- **Dependencies:** requires nn-013; recommends nn-002; supports nn-015, nn-016.
- **Visuals:** hero `01-matrix-weight-gradient.svg` (the outer product of the upstream column and the input row).
- **Lab/gate:** Verify that the matrix form reproduces nn-013's exact numbers for one sample, then extend to a three-sample batch and confirm the summed gradients by hand.
- **Anchors:** Petersen and Pedersen, The Matrix Cookbook, Goodfellow, Bengio and Courville section 6.5, Kinsley and Kukieła 2020 chapter 9.

### nn-015 - Gradients with respect to inputs

- **Meta:** Part V; intermediate; core; proof guided; code included.
- **Central question:** What must a layer pass back to the layer before it, and why is that gradient a sum?
- **Promise:** The reader can state why the input gradient is a sum while the weight gradient is not, derive $\partial L / \partial X = (\partial L / \partial Z) W^\top$ from the chain rule and verify it numerically, and list the three NumPy lines that complete a dense layer's backward pass.
- **Coverage:** why a third gradient is needed, why it is a sum and why the weight gradient was not, the matrix form, numerical verification and the convention note, batches, the complete backward toolkit for a dense layer
- **Dependencies:** requires nn-014; supports nn-016.
- **Visuals:** hero `01-input-gradients.svg` (each input feeding every neuron, its gradient summing the paths).
- **Lab/gate:** Verify the input gradient numerically for the post's layer with a finite difference on one input, then confirm the batched form on three samples.
- **Anchors:** Goodfellow, Bengio and Courville section 6.5.4, LeCun et al. 1998 (Efficient BackProp), Kinsley and Kukieła 2020 chapter 9.

### nn-016 - Coding backpropagation

- **Meta:** Part V; intermediate; core; proof none; code included.
- **Central question:** How do three posts of gradient theory become a `backward` method on `Layer_Dense` and `Activation_ReLU`?
- **Promise:** The reader can implement `Layer_Dense.backward(dvalues)` in three NumPy lines and explain each, implement `Activation_ReLU.backward(dvalues)` as a masked copy, and name what each layer caches in `forward` and why `backward` needs it.
- **Coverage:** the interface a layer needs, `Layer_Dense` with backward and the new weight convention, a numerical check, ReLU with backward and why a copy rather than in-place, caching, the full backward chain previewed
- **Dependencies:** requires nn-014, nn-015; recommends nn-004; supports nn-017, nn-018, nn-020, nn-030.
- **Visuals:** hero `01-dense-backward-class.svg` (one forward call, one backward call, three gradients out).
- **Lab/gate:** Write `Layer_Dense.backward` from memory, run the post's numerical check against the manual numbers of nn-013 to nn-015, and run `gradient_checking.md` on the layer.
- **Anchors:** Kinsley and Kukieła 2020 chapter 9, the PyTorch `torch.nn.Linear` backward as the production comparison, Nielsen chapter 2.

### nn-017 - Backpropagation through activation functions

- **Meta:** Part V; intermediate; core; proof guided; code included.
- **Central question:** Why is the backward step of an element-wise activation one multiply or mask, and why is softmax different?
- **Promise:** The reader can apply the element-wise activation backward in one line for ReLU, sigmoid, or tanh, say why ReLU's backward is cheaper than sigmoid's although the code shape is the same, and explain why softmax cannot be backpropagated element-wise.
- **Coverage:** where activations sit in the backward chain, ReLU's backward in detail with a worked example, the general element-wise pattern and the Jacobian view, why softmax is different, the toolkit status
- **Dependencies:** requires nn-016; recommends nn-010; supports nn-018, nn-019.
- **Visuals:** hero `01-elementwise-vs-coupled.svg` (diagonal Jacobian against full Jacobian).
- **Lab/gate:** Write the backward of sigmoid and tanh from their derivatives in the one-line pattern and check each against a finite difference.
- **Anchors:** Goodfellow, Bengio and Courville section 6.3, Glorot, Bordes and Bengio 2011, Kinsley and Kukieła 2020 chapter 9.

### nn-018 - Backpropagation through the loss function

- **Meta:** Part V; intermediate; core; proof guided; code included.
- **Central question:** Where does the first upstream gradient come from, and what is it for categorical cross-entropy?
- **Promise:** The reader can derive the cross-entropy gradient $-y / \hat{y}$ from the loss definition, implement `Loss_CategoricalCrossentropy.backward` for integer and one-hot labels, and explain why the gradient is divided by the batch size.
- **Coverage:** where backprop starts, the gradient from the definition with a single-sample example, batch behaviour with a worked example, integer labels, the complete `backward` method, why divide by the batch size
- **Dependencies:** requires nn-008, nn-016; recommends nn-017; supports nn-019.
- **Visuals:** hero `01-cross-entropy-backward.svg` (element-wise division by the prediction, one surviving entry per row).
- **Lab/gate:** Reproduce the worked batch example by hand, then confirm that scaling the batch size leaves the gradient magnitude per sample unchanged.
- **Anchors:** Goodfellow, Bengio and Courville section 6.2.2, Bishop 2006 section 4.3.4, Kinsley and Kukieła 2020 chapter 9.

### nn-019 - Softmax derivatives and the combined backward pass

- **Meta:** Part V; intermediate; core; proof guided; code included.
- **Central question:** Why does pairing softmax with cross-entropy collapse a full Jacobian into $\hat{y} - y$?
- **Promise:** The reader can explain why softmax's Jacobian is full rather than diagonal, apply the combined formula $\partial L / \partial Z = (\hat{y} - y) / N$ and recognise the cancellation that produces it, and implement `Activation_Softmax_Loss_CategoricalCrossentropy` in three lines.
- **Coverage:** why softmax cannot be backpropagated element-wise, the shortcut and why it is not magic, why every framework ships a combined version, a worked example, the combined class, label formats, the appendix with the full substitution derivation
- **Dependencies:** requires nn-017, nn-018; recommends nn-006, nn-011; supports nn-020, nn-034.
- **Visuals:** hero `01-combined-shortcut.svg` (the full Jacobian against the three-line shortcut).
- **Lab/gate:** Compute the explicit Jacobian route and the combined route on the post's example and confirm they agree to floating-point precision (appendix_softmax_combined_backward.md).
- **Anchors:** Bishop 2006 section 4.3.4, Goodfellow, Bengio and Courville section 6.2.2.3, Kinsley and Kukieła 2020 chapter 9, the PyTorch `CrossEntropyLoss` documentation.

### nn-020 - Assembling full backpropagation

- **Meta:** Part V; intermediate; core; proof none; code included.
- **Central question:** How do the `backward` methods of every component chain into one pass, and what carries the gradient between them?
- **Promise:** The reader can wire `Layer_Dense`, `Activation_ReLU`, and the combined softmax-loss class into a complete forward and backward pass, trace how one component's `dinputs` becomes the previous component's `dvalues`, and apply a single gradient-descent update to all four parameter arrays.
- **Coverage:** the pieces together, the forward pass, the backward pass, the gradient-descent update, why `dinputs` is the glue
- **Dependencies:** requires nn-016, nn-019; supports nn-021.
- **Visuals:** hero `01-full-backprop-pipeline.svg` (forward left to right, backward right to left, `dinputs` feeding `dvalues`).
- **Lab/gate:** Draw the pipeline from memory with every `forward` and `backward` call and the array that passes between each pair, then write one update step for all four parameter arrays.
- **Anchors:** Rumelhart, Hinton and Williams 1986, Kinsley and Kukieła 2020 chapter 9, Goodfellow, Bengio and Courville algorithm 6.4.

### nn-021 - Coding the full backpropagation

- **Meta:** Part V; intermediate; core; proof none; code included.
- **Central question:** What does the whole forward and backward pass look like as one short script, and what does it produce?
- **Promise:** The reader can read a fifteen-line script that runs a full forward and backward pass on the spiral dataset, inspect the four gradient arrays it produces, and sanity-check a gradient implementation by matching gradient shapes to parameter shapes.
- **Coverage:** the dataset, the three classes in one place, instantiating the network, the forward pass, the backward pass, inspecting the gradients, the full fifteen-line script
- **Dependencies:** requires nn-020; supports nn-022.
- **Visuals:** hero `01-forward-backward-script.svg` (the script with every shape annotated).
- **Lab/gate:** Run the script, confirm the four gradient shapes equal the parameter shapes, and run `gradient_checking.md` on the whole network (INDEX.md Phase 4 milestone).
- **Anchors:** Kinsley and Kukieła 2020 chapter 9, Karpathy, micrograd, Stanford CS231n (the minimal network case study).

## Part VI - Optimisers

### nn-022 - Gradient-descent optimiser

- **Meta:** Part VI; intermediate; core; proof idea; code included.
- **Central question:** What is the simplest rule that turns gradients into new weights, and how far does it get on the spiral?
- **Promise:** The reader can implement `Optimizer_SGD` and use it to update any layer, write a complete training loop (forward, backward, update, log) for the spiral classifier, and explain why vanilla gradient descent stalls near 65 percent as an efficiency limit rather than a dataset limit.
- **Coverage:** from gradients to weight updates and why the layer-optimiser split is clean, the `Optimizer_SGD` class, the training loop, what happens when it runs, the learning-rate trade-off, why vanilla SGD is slow
- **Dependencies:** requires nn-021; recommends nn-009; supports nn-023, nn-024, nn-032.
- **Visuals:** hero `01-sgd-update-and-lr.svg` (one update rule, three learning rates).
- **Lab/gate:** Run the training loop for 10 001 epochs at `lr=1.0` and reproduce the 64.7 percent final accuracy and 0.87 loss of `verify/RESULTS.md`.
- **Anchors:** Robbins and Monro 1951, Bottou 2010, Ruder 2016, Kinsley and Kukieła 2020 chapter 10.

### nn-023 - Learning-rate decay

- **Meta:** Part VI; intermediate; core; proof idea; code included.
- **Central question:** Why does one constant learning rate have to do two opposing jobs, and how does a decay schedule split them?
- **Promise:** The reader can derive and explain $\alpha(t) = \alpha_0 / (1 + d t)$ and choose a sensible $d$, extend `Optimizer_SGD` with `pre_update_params`, `post_update_params`, and an `iterations` counter, and diagnose whether a plateau is a learning-rate issue or a local-minimum issue.
- **Coverage:** the problem decay fixes, the decay formula and what different $d$ look like, why $1/(1 + d t)$, the updated optimiser class, the training loop with decay, what happens when it runs, what decay solves and does not, why three methods instead of one
- **Dependencies:** requires nn-022; supports nn-024.
- **Visuals:** hero `01-decay-schedule-and-result.svg` (three decay values and the loss curves they produce).
- **Lab/gate:** Reproduce the three decay runs of `verify/RESULTS.md` (decay $10^{-3}$ gives loss 0.76; $10^{-2}$ collapses to 39.7 percent; $10^{-4}$ barely moves) and explain each from the schedule.
- **Anchors:** Robbins and Monro 1951, Bottou 2012 (Stochastic gradient descent tricks), Kinsley and Kukieła 2020 chapter 10.

### nn-024 - Momentum

- **Meta:** Part VI; intermediate; core; proof idea; code included.
- **Central question:** How does a running velocity let the optimiser cancel oscillations and roll through shallow plateaus?
- **Promise:** The reader can explain momentum as vector cancellation across steps, implement it inside `Optimizer_SGD` with a per-layer `weight_momentums` buffer, and translate between the two sign conventions $v = \beta v - \alpha g$ and $v = \beta v + g$.
- **Coverage:** what goes wrong without memory, the vector-cancellation intuition, the momentum formula and its two sign conventions, where the velocity lives, the optimiser class with momentum, the training loop, what happens when it runs, choosing $\beta$
- **Dependencies:** requires nn-022, nn-023; supports nn-025, nn-027.
- **Visuals:** hero `01-momentum-trajectory.svg` (zig-zag against a smooth roll in a narrow valley).
- **Lab/gate:** Reproduce the momentum run of `verify/RESULTS.md` (`lr=1.0, decay=1e-3, momentum=0.9`: loss 0.12, accuracy 95.7 percent) and sweep $\beta$ through 0.5, 0.9, and 0.99.
- **Anchors:** Polyak 1964, Sutskever et al. 2013, Goh 2017 (Why momentum really works, Distill), Kinsley and Kukieła 2020 chapter 10.

### nn-025 - AdaGrad

- **Meta:** Part VI; intermediate; core; proof idea; code included.
- **Central question:** Why should every parameter have its own learning rate, and what goes wrong when its history only grows?
- **Promise:** The reader can explain why a single global learning rate is inadequate for parameters with different gradient scales, implement `Optimizer_Adagrad` with per-layer `weight_cache` and `bias_cache`, and predict when AdaGrad helps and when its dying rates dominate.
- **Coverage:** the remaining failure mode, the AdaGrad idea, a worked example of the cache growing forever, the structural flaw of dying rates, the optimiser class, the training loop, what happens when it runs, when AdaGrad is the right choice
- **Dependencies:** requires nn-024; supports nn-026.
- **Visuals:** hero `01-per-parameter-rates.svg` (two parameters with different gradient histories and different effective steps).
- **Lab/gate:** Reproduce the AdaGrad run of `verify/RESULTS.md` (`lr=1.0, decay=1e-4`: 84.0 percent) and plot the cache of one weight over training to see it never shrink.
- **Anchors:** Duchi, Hazan and Singer 2011, Ruder 2016, Kinsley and Kukieła 2020 chapter 10.

### nn-026 - RMSProp

- **Meta:** Part VI; intermediate; core; proof idea; code included.
- **Central question:** What one-line change bounds AdaGrad's cache, and how does the decay factor $\rho$ set its memory?
- **Promise:** The reader can explain why an exponential moving average over $g^2$ gives a bounded cache and how $\rho$ controls its horizon, implement `Optimizer_RMSprop` with sensible defaults for $\alpha$, $\rho$, and $\epsilon$, and read RMSProp as the bridge between AdaGrad and Adam.
- **Coverage:** what needs fixing, the exponential moving average, AdaGrad against RMSProp side by side, the update rule, the optimiser class, the training loop, results on the spiral, choosing $\rho$, RMSProp as the bridge to Adam
- **Dependencies:** requires nn-025; supports nn-027.
- **Visuals:** hero `01-rmsprop-vs-adagrad-cache.svg` (a cache that grows linearly against one that settles).
- **Lab/gate:** Reproduce the RMSProp run of `verify/RESULTS.md` (`lr=0.02, decay=1e-5, rho=0.999`: 90.0 percent) and compare the steady-state cache with AdaGrad's at the same step.
- **Anchors:** Tieleman and Hinton 2012 (Coursera lecture 6.5), Ruder 2016, Kinsley and Kukieła 2020 chapter 10.

### nn-027 - Adam

- **Meta:** Part VI; intermediate; core; proof idea; code included.
- **Central question:** How do momentum, an RMSProp cache, and a bias correction combine into the optimiser most networks ship with?
- **Promise:** The reader can decompose the Adam update into its first-moment EMA, second-moment EMA, and bias correction, implement `Optimizer_Adam` with two per-layer buffers and reproduce the spiral result, and reason about when Adam is the wrong default.
- **Coverage:** the synthesis, the Adam update rule, why bias correction is needed and why $\beta_2 = 0.999$, the optimiser class, the training loop, what happens when it runs, why Adam became the default, when it is wrong, variants worth knowing
- **Dependencies:** requires nn-024, nn-026; supports nn-028, nn-032.
- **Visuals:** hero `01-adam-pipeline.svg` (two EMAs, two corrections, one update).
- **Lab/gate:** Reproduce the Adam run of `verify/RESULTS.md` (`lr=0.02, decay=1e-5`: loss 0.08, 96.3 percent final, 97.7 percent peak) and state the Adam update rule from memory (INDEX.md Phase 5 milestone).
- **Anchors:** Kingma and Ba 2014, Loshchilov and Hutter 2019 (AdamW), Wilson et al. 2017, Reddi, Kale and Kumar 2018 (AMSGrad), Kinsley and Kukieła 2020 chapter 10.

## Part VII - Generalisation and regularisation

### nn-028 - Generalization and testing

- **Meta:** Part VII; intermediate; core; proof none; code included.
- **Central question:** Why is training accuracy only half the job, and what does the train-test gap tell you?
- **Promise:** The reader can run a forward-only validation pass and read its accuracy and loss against the training numbers, distinguish good generalisation from overfitting by decision-boundary geometry and loss-curve shape, and map an overfitting symptom to one of four levers.
- **Coverage:** the two acts of training, where the framing comes from, running the test and why it is forward-only, reading the gap, what overfitting looks like in the boundary and the loss curves, four levers to prevent it
- **Dependencies:** requires nn-027; supports nn-029, nn-030, nn-031, nn-032.
- **Visuals:** hero `01-train-vs-test.svg` (a small gap against a wide one); supporting figures for the overfit decision boundary and the diverging loss curves.
- **Lab/gate:** Generate fresh spiral data, run the forward-only pass on the nn-027 network, and record the gap; then overtrain a wider network and watch the gap open.
- **Anchors:** Vapnik 1998, Hastie, Tibshirani and Friedman chapter 7, Goodfellow, Bengio and Courville section 5.2, Kinsley and Kukieła 2020 chapter 11.

### nn-029 - Validation and hyperparameter tuning

- **Meta:** Part VII; intermediate; core; proof none; code included.
- **Central question:** How do you search a hyperparameter space and still have a test number you can trust?
- **Promise:** The reader can state why the test set is touched exactly once and the failure mode when it is reused, implement a clean `k_fold_split` in NumPy and use it to compare hyperparameter candidates, and recognise the common forms of data leakage and the preprocessing rules that prevent them.
- **Coverage:** the hyperparameter problem, the three-way split, k-fold cross-validation and a minimal implementation, using k-fold for hyperparameter search, data leakage, picking $k$ and other practical defaults, the grid-search loop
- **Dependencies:** requires nn-028; supports nn-030, nn-031.
- **Visuals:** hero `01-three-way-split-and-kfold.svg` (the three-way split beside a five-fold rotation).
- **Lab/gate:** Run the grid-search loop over two learning rates and two layer widths with 5-fold validation, pick the winner, and open the test set once.
- **Anchors:** Kohavi 1995, Kaufman et al. 2012 (leakage in data mining), Hastie, Tibshirani and Friedman section 7.10, Kinsley and Kukieła 2020 chapter 11.

### nn-030 - L1 and L2 regularisation

- **Meta:** Part VII; intermediate; core; proof guided; code included.
- **Central question:** How does a penalty on weight magnitude change the loss, the gradient, and the weights the optimiser prefers?
- **Promise:** The reader can write the L1 and L2 penalties and derive their gradients, explain why L1 induces sparsity and L2 shrinkage, extend `Layer_Dense` with `weight_regularizer_l1` and `weight_regularizer_l2` that hook into the existing forward and backward passes, and choose a sensible $\lambda$.
- **Coverage:** why large weights overfit, the two penalties and their behavioural difference, the penalty in the forward pass inside `Loss`, the extra gradient terms in `Layer_Dense.backward`, the updated class, choosing $\lambda$, results on the spiral, connections to weight decay and Bayesian priors
- **Dependencies:** requires nn-016, nn-028; recommends nn-029; supports none.
- **Visuals:** hero `01-l1-vs-l2-penalty.svg` (the V and the parabola); a supporting figure for the regularised decision boundary.
- **Lab/gate:** Reproduce the regularisation rows of `dashboards/Regularization_Comparison.md` from `verify/regularization_results.py` (L2 lifts test accuracy from 78.7 to 84.0 percent on the small spiral network) and explain why L1 plus L2 gives the smallest gap.
- **Anchors:** Tikhonov 1963, Tibshirani 1996 (lasso), Ng 2004 (L1 against L2), Krogh and Hertz 1992 (weight decay), Goodfellow, Bengio and Courville section 7.1, Kinsley and Kukieła 2020 chapter 14.

### nn-031 - Dropout

- **Meta:** Part VII; intermediate; core; proof idea; code included.
- **Central question:** What does zeroing random activations during training do to a network, and why must evaluation leave every neuron on?
- **Promise:** The reader can explain why random masking attacks co-adaptation and why the ensemble view makes the gain unsurprising, implement `Layer_Dropout` with the inverted convention (scale by $1/(1 - p)$ in training, identity at test time), and wire it into the pipeline with the train-versus-test switch.
- **Coverage:** two failure modes dropout attacks, co-adaptation and implicit ensembling, the dropout rate, the inverted-dropout trick, the `Layer_Dropout` class, plugging dropout into the network, the train-versus-test switch, results on the spiral, where dropout sits in the toolkit
- **Dependencies:** requires nn-028; recommends nn-005, nn-029; supports nn-035.
- **Visuals:** hero `01-dropout-train-vs-test.svg` (eight neurons in training and in evaluation).
- **Lab/gate:** Implement `Layer_Dropout`, train with 1000 samples per class, and reproduce the result that validation accuracy can exceed training accuracy; then show that dropout over-regularises the 64-neuron spiral network of the dashboard (INDEX.md Phase 6 milestone).
- **Anchors:** Srivastava et al. 2014, Hinton et al. 2012 (improving neural networks by preventing co-adaptation), Gal and Ghahramani 2016, Kinsley and Kukieła 2020 chapter 15.

## Part VIII - Practical training and extensions

### nn-032 - Mini-batching

- **Meta:** Part VIII; intermediate; core; proof idea; code included.
- **Central question:** Why does the full-batch training loop of nn-022 to nn-031 fail at scale, and what replaces it?
- **Promise:** The reader can distinguish full-batch gradient descent, true SGD, and mini-batch SGD by what each step consumes and the gradient noise that results, extend the single-loop training pattern into the two-loop epoch-by-batch structure with shuffling, and choose a batch size for a given dataset.
- **Coverage:** what the lectures glossed over, the three regimes, the two-loop training structure, the complete loop with Adam, choosing batch size and why powers of two, mini-batch noise as a feature, common variants and naming
- **Dependencies:** requires nn-022, nn-028; recommends nn-027; supports nn-035.
- **Visuals:** hero `01-batch-size-trajectories.svg` (three regimes through one loss landscape).
- **Lab/gate:** Read `projects/01-mnist-from-scratch/train.py` and `projects/03-fashion-mnist/train.py` against the post's pseudocode, then run project 01 and reproduce the 98.0 percent of `verify/projects_results.md`.
- **Anchors:** Bottou 2010, Keskar et al. 2017 (large-batch training and sharp minima), Masters and Luschi 2018, Goodfellow, Bengio and Courville section 8.1.3, Kinsley and Kukieła 2020 chapter 19.

### nn-033 - Weight initialisation

- **Meta:** Part VIII; intermediate; core; proof guided; code included.
- **Central question:** Why does `0.01 * randn` work for two hidden layers and fail silently at depth, and what scale preserves variance?
- **Promise:** The reader can derive the variance-preservation argument behind Glorot and He initialisation, implement both in `Layer_Dense` and say which one fits which activation, and recognise the symptoms of bad initialisation (zero gradients, NaN loss on step one, a plateau from the start).
- **Coverage:** the line the lectures glossed over, variance through one linear layer, Glorot initialisation, He initialisation as the ReLU fix, updating `Layer_Dense`, symptoms of bad initialisation, how initialisation interacts with normalisation and depth
- **Dependencies:** requires nn-004, nn-006; recommends nn-005; supports nn-034, nn-035.
- **Visuals:** hero `01-activation-variance-by-depth.svg` (three scales through ten layers).
- **Lab/gate:** Measure activation standard deviation and the ReLU zero-fraction per layer for a ten-layer stack under the three schemes, then re-train project 03 with `init="he"` and compare the first five epochs (INDEX.md Phase 7 milestone).
- **Anchors:** Glorot and Bengio 2010, He et al. 2015 (Delving deep into rectifiers), LeCun et al. 1998 (Efficient BackProp), Goodfellow, Bengio and Courville section 8.4.

### nn-034 - Sigmoid and binary cross-entropy

- **Meta:** Part VIII; intermediate; core; proof guided; code included.
- **Central question:** What replaces softmax plus categorical cross-entropy when there are exactly two classes, and does the combined-derivative shortcut survive?
- **Promise:** The reader can derive the combined gradient $\partial L / \partial z = (\sigma(z) - y) / N$ from first principles, implement a numerically stable `Activation_Sigmoid` and `Activation_Sigmoid_Loss_BinaryCrossentropy`, and choose between sigmoid plus BCE and softmax plus CCE for a binary problem.
- **Coverage:** why the binary case deserves its own treatment, the sigmoid and the overflow trap, binary cross-entropy, the combined backward trick, the combined class, when to use what
- **Dependencies:** requires nn-019; recommends nn-033; supports nn-035.
- **Visuals:** hero `01-sigmoid-bce-pipeline.svg` (logit to sigmoid to BCE forward, the shortcut backward).
- **Lab/gate:** Trace `projects/02-binary-classifier/nn.py` against the derivation, run the project, and reproduce the two-moons result of `verify/projects_results.md` (100 percent test accuracy with He init and noise 0.1).
- **Anchors:** Bishop 2006 section 4.3.2, Goodfellow, Bengio and Courville section 6.2.2.2, Kinsley and Kukieła 2020 chapter 16.

### nn-035 - What to read after this series

- **Meta:** Part VIII; intermediate; core; proof none; code none.
- **Central question:** What does each major architecture and training trick add to the forward, backward, and optimiser skeleton built here, and what should the reader study next?
- **Promise:** The reader can place CNNs, RNNs, and transformers on a map of what each adds to a vanilla MLP, name the canonical paper and a from-scratch tutorial for each, and choose what to study next from the kind of problem they care about.
- **Coverage:** what the reader already has, new layer types (convolution, recurrence, attention), new training infrastructure (batch normalisation, residual connections, schedulers, mixed precision, distributed training), new problem framings (self-supervised, transfer, reinforcement learning, diffusion), three books, three frameworks, a reading checklist
- **Dependencies:** requires nn-031, nn-034; recommends nn-032, nn-033; supports none.
- **Visuals:** hero `01-whats-next-map.svg` (three columns of additions on top of the from-scratch stack).
- **Lab/gate:** Re-implement one of the four projects in PyTorch, or build a one-conv-layer network on MNIST and compare with project 01; then pick one item from the reading list and start it.
- **Anchors:** LeCun et al. 1998 (LeNet), Hochreiter and Schmidhuber 1997, Vaswani et al. 2017, Ioffe and Szegedy 2015, He et al. 2016, Goodfellow, Bengio and Courville, Zhang et al., Dive into Deep Learning.

# 6. Series-specific contracts

## Notation

- `notation_guide.md` is the single source of truth once migrated; it uses bold matrices ($\mathbf{X}$, $\mathbf{W}$, $\mathbf{b}$, $\mathbf{Z}$, $\mathbf{A}$, $\hat{\mathbf{y}}$, $\mathbf{y}$) and the loss $L$, with shapes $(N, n_\text{features})$, $(n_\text{inputs}, n_\text{neurons})$, $(1, n_\text{neurons})$, $(N, n_\text{neurons})$.
- Code identifiers are `inputs`, `weights`, `biases`, `output`, `dvalues`, `dweights`, `dbiases`, `dinputs`; the maths uses the symbol, the code uses the identifier, and the guide is the arbiter where they differ.
- Optimiser variables: $\alpha$ is `learning_rate`, $d$ is `decay`, $\beta$ is `momentum`, $\rho$ is `rho`, $\beta_1$ and $\beta_2$ are `beta_1` and `beta_2`, $\epsilon$ is `epsilon`; buffers are `weight_momentums`, `weight_cache`, `bias_momentums`, `bias_cache`.
- The weight convention is $(n_\text{inputs}, n_\text{neurons})$ from nn-004 onward; nn-001 to nn-003 use the transposed layout and say so. The two conventions are reconciled in nn-014 section 8.
- Inline math is `$...$`, display math is `$$...$$` on its own line; no em-dashes in prose.

## Datasets and fixtures

- The spiral dataset (`nnfs.datasets.spiral_data`, 100 samples per class, 3 classes, seed fixed by `nnfs.init()`) is the fixture for nn-004 to nn-031 and for `verify/`.
- Projects use mnist (project 01), two-moons generated in NumPy (project 02), fashion-mnist (project 03), and california-housing via scikit-learn (project 04). Downloads are cached under `mnist_cache/` and `fashion_mnist_cache/` at the repository root; the migration moves them under the projects and registers the four datasets in `assets/registry/datasets.yaml` (Section 9).

## Compute tier

- CPU only. Every figure in the posts and in `verify/` is a NumPy run on a CPU: 10 001 epochs on the spiral in under a minute per optimiser, 20 epochs of MNIST in minutes. `vocab.compute` is empty until the sidecars carry the key; the migration sets it to `[cpu]`.

## Code

- There is no library under `code/`; `series.yaml` has `library: null`. The classes live in the posts, in `cumulative_notebook.ipynb` (every class in one runnable notebook), and in each project's `nn.py`. The projects import from `01-mnist-from-scratch/nn.py` unless they need a class it lacks.
- `verify/optimizer_results.py` and `verify/regularization_results.py` are the reproduction scripts; `verify/RESULTS.md` and `verify/projects_results.md` are the records and mark every claim that did not hold (project 03 at 87.1 percent against the README's 88.9, project 02 fixed by He init).
- `gradient_checking.md` is the numerical check for Part V; it is a document with code today and becomes a snippet at migration.

## Claim standards

- Every accuracy or loss quoted in a post is a number in `verify/RESULTS.md` or `verify/projects_results.md` with the config that produced it; AUDIT_REPORT.md lists the nine printed outputs that did not reproduce and the migration replaces each with the verified value.
- Optimiser comparisons are made on the same seed, network ($2 \to 64 \to 3$), data, and epoch budget (10 001), so rows differ only in the optimiser.

## Interactives

- None. Diagrams are hand-drawn SVGs produced by `assets/diagrams/`, one hero per post plus supporting figures, light and dark; the poster in `poster/` is the one-page summary. No interactive is planned.

## Series-specific checks

- The series has no `tools/lint_repo.py` yet; the migration adds one that runs the family validator. Candidate series checks: every "Part NN" reference in a body resolves to a directory (AUDIT_REPORT.md found 241 such references), and every `verify/` number quoted in prose matches the records.

# 7. Sibling handoffs and deliberate omissions

## Assumed from other series

- Nothing by id. The series is an entry point of the family: it teaches its own calculus (nn-010, nn-011) and its own NumPy (nn-002, nn-005). Readers who want the mathematics in depth are pointed in prose at the calculus and algebra series, which are not locked, so no id is written here.

## Handed to siblings

- cnn-from-scratch (locked) assumes nn-001 to nn-035 in full and cites nn-016 (dense backward), nn-023 (decay schedule and the three-method optimiser contract), nn-032 (training loop), nn-033 (He initialisation), and nn-035 in prose; its specs carry no `requires` into nnfs, so no nn spec lists a cnn id under supports. Convolution, pooling, batch normalisation, residual connections, augmentation, and modern schedulers are its topics; nn-035 only names them.
- rnn-from-scratch (locked) requires nn-004 on rnn-004 and recommends nn-004 on rnn-001; nn-004 lists both under supports. Its plan cites nn-006, nn-008, nn-011 to nn-017, nn-019 to nn-022, nn-027 to nn-029, and nn-031 to nn-033 in prose. Recurrence, backpropagation through time, LSTM, and GRU are its topics.
- transformer (not locked): attention and the transformer, which nn-035 names and leaves to that series.
- diffusion (not locked) and rl-mastery (not locked): the generative and reinforcement-learning framings nn-035 names in one paragraph each.
- machine-learning (not locked): the bias-variance decomposition, learning curves, and the theory of generalisation that nn-028 uses as framing only.

## Deliberate omissions

- Batch normalisation, layer normalisation, residual connections, warmup and cosine schedules, mixed precision: named in nn-035, owned by cnn-from-scratch and transformer.
- Regression and mean squared error as a loss: implemented in project 04, not in a post; nn-012 uses squared error for one neuron only.
- Convolutional, recurrent, and attention layers: nn-035 names them and the sibling series build them.
- Automatic differentiation and a computational-graph engine: the series derives every gradient by hand on purpose; dl-framework-from-scratch owns autograd.
- GPU and framework code: PyTorch appears only as the recommended next step in nn-035.

# 8. Projects

`series.yaml` lists `projects: []` and `next_project_ordinal: 1`: the four project directories exist with README, `nn.py`, `data.py`, `train.py`, `evaluate.py`, and `requirements.txt`, but none has a `project.yaml`, so no project id is allocated by this lock. The migration writes a `project.yaml` for each in directory order, allocating nn-p01 to nn-p04, and the fields below are the plan's record of what each will carry.

## nn-p01 MNIST from scratch (planned id)

- slug `mnist-from-scratch`; kind: model; status: complete; level: intermediate; after_post: nn-032; uses: nn-016, nn-019, nn-027, nn-030, nn-031, nn-032.
- problem: show that the from-scratch stack composes into a working model on real data. A two-hidden-layer network ($784 \to 128 \to 128 \to 10$) with Adam, L2, and dropout, trained with mini-batches of 128 for 20 epochs, reaches 98.0 percent test accuracy (`verify/projects_results.md`; the README claims about 97).
- users: readers finishing Part VIII who want the first real result; the dense baseline that cnn-012 measures against (118,282 parameters).

## nn-p02 Binary classifier on two-moons (planned id)

- slug `binary-classifier`; kind: model; status: complete; level: intermediate; after_post: nn-034; uses: nn-019, nn-027, nn-033, nn-034.
- problem: apply sigmoid plus binary cross-entropy to a 2-D dataset whose decision boundary can be plotted. A $2 \to 16 \to 16 \to 1$ network with He initialisation and Adam reaches 100 percent test accuracy at noise 0.1 and dumps a $200 \times 200$ decision grid for plotting. `verify/projects_results.md` records that the shipped `0.01 * randn` init underfit at 85 percent, the failure nn-033 describes.
- users: readers of nn-034 who want to see the combined-derivative trick on a problem they can draw.

## nn-p03 Fashion-MNIST (planned id)

- slug `fashion-mnist`; kind: model; status: complete; level: intermediate; after_post: nn-033; uses: nn-027, nn-028, nn-032, nn-033.
- problem: run the project 01 network unchanged on a harder dataset and read where it fails. Only `data.py` changes; accuracy falls to 87.1 percent and the confusion matrix concentrates the errors in the shirt cluster and the shoe cluster.
- users: readers who need to learn that MNIST is a deceptive benchmark; the dense baseline that cnn-013 measures against.

## nn-p04 California housing regression (planned id)

- slug `california-housing-regression`; kind: model; status: complete; level: intermediate; after_post: nn-032; uses: nn-016, nn-027, nn-029, nn-032.
- problem: the first regression in the series: no softmax, no sigmoid, mean squared error as the loss, target standardised with training-fold statistics and de-standardised at evaluation. A $8 \to 64 \to 64 \to 1$ network with Adam reaches $R^2 = 0.823$ and RMSE about 49 thousand dollars on the test fold; the top-coding of the target at 500 thousand dollars shows as a vertical cluster in the scatter.
- users: readers who want to see the forward and backward machinery survive a change of loss and the no-leakage rule of nn-029 made explicit.

# 9. Risks and decisions

## Risk register

- Notation drift. AUDIT_REPORT.md found the bold-matrix convention used in no forward-pass post and no SVG, and `notation_guide.md` and `glossary.md` disagreeing on all seven core symbols. Mitigation: the plan names `notation_guide.md` as the arbiter (Section 6); the migration commits of nn-001 to nn-035 align the bodies and the diagram generator, and the glossary is regenerated from the guide.
- Unreproduced numbers. Nine printed outputs and twenty-two internal contradictions (stale part counts, conflicting accuracy figures, a wrong momentum formula, dashboard numbers that contradict the posts) are listed in AUDIT_REPORT.md. Mitigation: `verify/RESULTS.md` and `verify/projects_results.md` are the records; each migration commit replaces the quoted number with the verified one.
- Em-dash overruns and long TL;DRs (nn-035 alone has 39 em-dashes). Mitigation: the family validator's body stage with `EMDASH_CAP = 0` at migration; TL;DRs trimmed to the family length.
- Exercises for nn-032 to nn-035 do not exist (exercises.md says so). Mitigation: the question bank migration writes them as `questions.json` for those four posts first.
- A reader skipping Part IV. The calculus posts are beginner and short, and framework users may skip them; Part V then fails. Mitigation: nn-012 requires nn-010 and nn-011 in the lock, and nn-010 is marked optional code so it is read as theory, not skipped as a code post.

## Decisions locked

- 35 posts, 8 parts, no planned entries, `next_post_ordinal` 36. The series is complete and the lock adds nothing; a new topic (batch normalisation, autograd, regression as a post) is a sibling's topic or a new id after nn-035, never an insertion.
- Parts follow the sidecar `part` strings, not INDEX.md's seven phases: the sidecars split INDEX.md's "Phase 2: Forward pass complete (6 to 9)" into Part II (nn-006, nn-007) and Part III (nn-008, nn-009), and INDEX.md's phases 1 to 7 are the plan's Parts I to VIII with that one split. INDEX.md and README.md keep their phase tables until migration, when they are regenerated from `series.yaml`.
- Levels. Sidecars carry no `level`; the plan sets nn-001 to nn-011 beginner and nn-012 to nn-035 intermediate (`levels_used: [beginner, intermediate]`). The series has no advanced post: nothing in it needs more than the chain rule, and the hardest material (Part V) is intermediate by its proofs, not advanced by its prerequisites. nn-035 is intermediate because it presumes the whole series.
- Dependencies. Sidecars carry no `prerequisites` or `recommended`; the plan's graph was built from the posts' "What to read next" lists, their TL;DR back-references, and INDEX.md's dependency lookup table. `requires` is the smallest set that makes the post readable (mostly the previous post and the posts whose classes it extends); `recommends` holds INDEX.md's "review these first" entries that are not strictly needed (nn-005 on nn-008, nn-031, nn-033; nn-002 on nn-014; nn-004 on nn-016; nn-006 and nn-011 on nn-019; nn-009 on nn-022; nn-027 on nn-032; nn-029 on nn-030 and nn-031; nn-033 on nn-034). `supports` is the exact inverse of requires plus recommends across the 35 specs, plus rnn-001 and rnn-004 on nn-004 from the rnn-from-scratch plan.
- Cross-series ids. Only locked siblings are named by id: rnn-from-scratch (nn-004 supports rnn-001 and rnn-004) and cnn-from-scratch (in prose only, since its specs require no nn id). The calculus, algebra, transformer, machine-learning, and diffusion series are described in prose in Section 7.
- Scope is `core` for every post; proof maturity is `none` for code-assembly posts, `idea` for motivation and optimiser posts, `guided` for every derivation; code is `included` everywhere except nn-010 and nn-011 (`optional`, a few lines of checking code) and nn-035 (`none`).
- Projects are not given ids by this lock (`projects: []`) because none has a `project.yaml`; Section 8 reserves nn-p01 to nn-p04 in directory order for the migration.
- The post 28 sidecar's part string read "Part VII" followed by "Generalization & Regularization" (American spelling and an ampersand) while nn-029 to nn-031 read "Generalisation and regularisation"; it was normalised to the latter so the series has eight parts rather than nine. No other sidecar was changed except the inserted `id:`.

## Recorded divergences between shipped posts and their specs

- Titles are kept verbatim from the sidecars minus the `Part NN · ` prefix. Three diverge from house style and are kept: nn-028 "Generalization and testing" uses American spelling where the rest of the series (including its own part title) uses "generalisation"; nn-006 "Activation functions: ReLU and Softmax" capitalises Softmax; nn-004 "The Dense layer class and spiral data" capitalises Dense. The migration commit of each post decides whether to rename (and re-slug nn-028 to `28-generalisation-and-testing`) or keep the proper-noun casing.
- Sidecar `part` strings separate the numeral from the title with an em-dash while `series.yaml` and this plan use the hyphen form the standard requires ("Part I - Foundations"); `assign_ids.py` and the validator accept both. Every sidecar migrates to the hyphen form.
- Sidecars carry only `slug`, `title`, `date`, `tags`, `hero`, `reading_time`, `part`, and now `id`. At migration each gains `level` (from the Meta line), `prerequisites` (from `requires`), `recommended` (from `recommends`), `code` (from the Meta line), and the remaining required keys (`summary`, `updated`, `status`, `outcomes`) from the post's TL;DR and outcome bullets.
- The posts have no prerequisites line; their "What to read next" lists are forward pointers, and nn-031's reads as a series closing ("The series is complete") although four posts follow. The migration adds the prerequisites line from the lock and rewrites nn-031's closing to point at nn-032.
- `exercises.md` and `quizzes.md` hold the assessment for nn-001 to nn-031 at the repository root; the standard puts it in `questions.json` per post. The migration imports them with `tools/import_exercises.py` and writes banks for nn-032 to nn-035.
- `GLOSSARY.md`, `CHEATSHEET.md`, `CONTRIBUTING.md`, and `LICENSE` are required at the root; the series has `glossary.md`, four files under `cheatsheets/`, and no CONTRIBUTING or LICENSE. The migration renames and adds them. `AUDIT_REPORT.md`, `appendix_softmax_combined_backward.md`, `gradient_checking.md`, `common_pitfalls.md`, `exercises.md`, `quizzes.md`, and the weight and cache files at the root move under `audits/`, `posts/*/snippets/`, or the projects.
- Project READMEs quote results that `verify/projects_results.md` contradicts (project 03 claims 88.9 percent, verified 87.1; project 04 claims $R^2$ about 0.78, verified 0.823; project 01 claims about 97, verified 98.0). The verified numbers are the ones this plan quotes; the READMEs are corrected when `project.yaml` is written.

# 10. Roadmap

All 35 posts are drafted and shipped; nothing is planned. The lock is the first step of the tier A2 migration (STANDARD.md Section 13), and the order of work after it is:

1. Lock commit: `Lock the curriculum`, tag `lock-1.0.0`. Contents: `series.yaml`, `plan.md`, the `id:` line in 35 sidecars, and the post 28 part string.
2. Sidecar migration, one commit per part in reading order (Part I first): `level`, `prerequisites`, `recommended`, `code`, `summary`, `updated`, `status`, `outcomes`, and the hyphen part string, with the title decisions of Section 9 taken per post.
3. Body migration in the same order: notation aligned to `notation_guide.md`, verified numbers from `verify/`, em-dashes removed, prerequisites line added, "What to read next" reduced to the house count, nn-031's closing rewritten.
4. Question banks: import `exercises.md` and `quizzes.md` into `questions.json` for nn-001 to nn-031, then write banks for nn-032 to nn-035.
5. Root layout: the required root files, `audits/`, `assets/registry/datasets.yaml`, `tools/lint_repo.py`, the four `project.yaml` files allocating nn-p01 to nn-p04, and INDEX.md and README.md regenerated from `series.yaml`.
6. `python tools/validate.py nnfs --strict --run` clean, then `status: complete` is confirmed and the plan moves to `Status: locked`.
