# Try It Yourself — Exercises for Every Lecture

*Hands-on experiments to deepen your understanding. Each exercise is tied to a specific lecture — complete it after reading the corresponding blog post.*

---

## Part 1 — Neurons and Layers

1. **Add a 4th neuron** to the 3-neuron layer. Choose your own weights and bias. Verify the output by hand and with code.
2. **Change the inputs** to `[0, 0, 0, 0]`. What output does every neuron produce? Why?
3. **Set all biases to 0.** How does this change the outputs? When might zero biases be useful?
4. **Create a 2-layer network manually** (no NumPy): feed the 3 outputs from layer 1 as inputs to a new 2-neuron layer. Write the code using nested loops.

---

## Part 2 — NumPy and the Dot Product

1. **Verify `np.dot` by hand.** Take `weights = [[0.2, 0.8, -0.5, 1.0]]` and `inputs = [1, 2, 3, 2.5]`. Compute the dot product manually, then verify with `np.dot`.
2. **Batch size experiment.** Create a batch of 5 different input samples (shape `(5, 4)`). Pass them through a 3-neuron layer using `np.dot`. Confirm the output shape is `(5, 3)`.
3. **Transpose investigation.** What happens if you forget to transpose the weight matrix? Try `np.dot(X, weights)` vs `np.dot(X, weights.T)` and explain the error or shape mismatch.
4. **Large-scale timing.** Create random inputs of shape `(1000, 100)` and weights `(100, 64)`. Time `np.dot` vs a nested Python loop. How much faster is NumPy?

---

## Part 3 — Stacking Layers

1. **Change layer sizes.** Modify the network so Layer 1 has 5 neurons and Layer 2 has 2 neurons. What must the weight shapes be?
2. **Add a 3rd layer** with 4 neurons. Chain it after Layer 2. Print intermediate shapes at every step.
3. **Identity experiment.** Set Layer 2's weights to the identity matrix (if the dimensions allow). What happens to the output?

---

## Part 4 — Dense Layer Class & Spiral Data

1. **Vary `samples` in `spiral_data`.** Try 10, 100, 1000. Plot each. How does the data density change?
2. **Increase to 5 classes.** Call `spiral_data(samples=100, classes=5)`. How many total points? Plot them.
3. **Inspect initial weights.** Create `Layer_Dense(2, 64)` and print `layer.weights.shape`, `layer.biases.shape`, and the first 5 weight values. Run it twice — are the weights the same?
4. **Weight scale experiment.** Change the `0.01 * np.random.randn(...)` to `1.0 * np.random.randn(...)`. Run the forward pass. Are the outputs reasonable or extremely large?

---

## Part 5 — Summation & Broadcasting

1. **Axis playground.** Create a `(3, 4)` array. Compute `np.sum` with `axis=0`, `axis=1`, and `axis=None`. Predict the shapes before running.
2. **keepdims experiment.** Repeat the above with `keepdims=True`. How do the shapes differ? Why does this matter for subtraction/division?
3. **Broadcasting challenge.** A matrix is shape `(5, 3)` and a vector is shape `(3,)`. What happens when you add them? What about `(5, 3)` + `(5, 1)`? Try both and explain.
4. **Find the bug.** Write code that tries to add a `(4,)` vector to a `(3, 5)` matrix. Why does it fail? Fix it.

---

## Part 6 — Activation Functions (ReLU & Softmax)

1. **ReLU by hand.** Apply ReLU to `[-2, 0, 3, -0.5, 7]`. Verify with `np.maximum(0, x)`.
2. **Softmax numerical stability.** Compute softmax of `[1, 2, 3]` and `[1001, 1002, 1003]`. Do you get the same output? Now subtract the max first and try again.
3. **Softmax properties.** Verify that softmax outputs always sum to 1. Try with 5 different input vectors.
4. **Temperature scaling.** Divide your inputs by a "temperature" $T$ before softmax. Try $T = 0.1, 1.0, 10.0$. What happens to the output distribution as $T$ changes?

---

## Part 7 — Complete Forward Pass

1. **Run the full forward pass** with the spiral data. Print the shape after every layer and activation.
2. **Inspect predictions.** Print the first 5 rows of the softmax output. Do they sum to 1? Which class has the highest probability for each?
3. **Random weights baseline.** Without any training, what accuracy do you get? (For 3 classes, random should be ~33%.) Verify this.

---

## Part 8 — Loss Functions (Categorical Cross-Entropy)

1. **Confidence vs loss.** Compute cross-entropy loss for predictions `[0.9, 0.05, 0.05]`, `[0.6, 0.2, 0.2]`, and `[0.33, 0.34, 0.33]` when the true class is 0. How does confidence affect loss?
2. **Perfect prediction.** What is the loss when the prediction is `[1.0, 0.0, 0.0]` and the true class is 0? What about `[0.0, 0.0, 1.0]` with true class 0?
3. **Clipping experiment.** Remove the `np.clip` in the loss calculation. Feed in a prediction of `[0.0, 1.0, 0.0]` with true class 0. What happens? Why is clipping necessary?
4. **Batch loss.** Compute the mean loss over 5 samples. Verify that `np.mean` of individual losses matches the batch calculation.

---

## Part 9 — Introduction to Optimization

1. **Random search.** Generate 1000 random weight sets for a single neuron. Compute the loss for each. Find the best one. How close is it to the gradient-descent solution?
2. **Step size exploration.** Using the simple `w = w - lr * gradient` rule, try learning rates of `0.001`, `0.01`, `0.1`, `1.0`, `10.0` on a simple function $f(w) = w^2$. Which converges? Which diverges?
3. **Visualize the loss landscape.** For $f(w) = (w - 3)^2$, plot loss vs $w$ for $w \in [-5, 10]$. Mark the minimum.

---

## Part 10 — Derivatives, Gradients & Partial Derivatives

1. **Numerical derivative.** Compute the derivative of $f(x) = x^3$ at $x = 2$ using the limit definition with $h = 0.001$. Compare to the analytical answer $3x^2 = 12$.
2. **Partial derivatives.** For $f(x, y) = x^2 + 3xy + y^2$, compute $\partial f / \partial x$ and $\partial f / \partial y$ at $(1, 2)$ both analytically and numerically.
3. **Gradient vector.** For $f(x, y, z) = x^2 + y^2 + z^2$, compute the gradient at $(1, 2, 3)$. Which direction does the negative gradient point? What does this mean geometrically?

---

## Part 11 — The Chain Rule

1. **Chain rule practice.** Given $y = (3x + 2)^4$, find $dy/dx$ using the chain rule. Verify numerically at $x = 1$.
2. **Two-step composition.** For $f(x) = \text{ReLU}(2x - 1)$, compute $df/dx$ at $x = 0, 0.5, 1.0, 2.0$. Where is the derivative zero?
3. **Three-layer chain.** If $L = g(f(h(x)))$, write out $dL/dx$ using the chain rule. Then pick simple functions for $g$, $f$, $h$ and verify numerically.

---

## Part 12 — Backprop Through a Single Neuron

1. **Single neuron gradient descent.** Start with random $w$ and $b$. Use target output = 1.0, input = 2.0. Run 100 iterations of gradient descent with $\alpha = 0.01$. Plot the loss curve.
2. **Learning rate sensitivity.** Repeat with $\alpha = 0.001, 0.1, 1.0$. Which converges fastest? Which oscillates?
3. **Two-input neuron.** Extend to $z = w_1 x_1 + w_2 x_2 + b$. Compute all three gradients and train for 100 steps.

---

## Part 13 — Backprop Through a Layer

1. **3-neuron layer gradients.** Create a layer with 2 inputs and 3 neurons. Compute the forward pass, then manually compute $\partial L / \partial w$ for each of the 6 weights (using a simple MSE loss). Verify with backprop code.
2. **Gradient shape check.** After calling `layer.backward(dvalues)`, print `layer.dweights.shape` and `layer.dbiases.shape`. Confirm they match `layer.weights.shape` and `layer.biases.shape`.

---

## Part 14 — Matrices in Backpropagation

1. **Manual matrix multiply.** For a `(3, 2)` input and `(2, 4)` weight matrix, compute the forward pass by hand. Then compute $\partial L / \partial W = X^T \cdot \text{dvalues}$ by hand.
2. **Transpose intuition.** Why is the transpose needed in `np.dot(self.inputs.T, dvalues)`? Try without the transpose and explain the shape error.

---

## Part 15 — Gradients w.r.t. Inputs

1. **Why input gradients?** Explain in your own words why we need $\partial L / \partial X$ even though $X$ is not a trainable parameter.
2. **Shape verification.** For a layer with input shape `(5, 3)` and weights `(3, 4)`, what shape is `dinputs`? Verify: it must match the input shape.
3. **Gradient flow.** Create a 3-layer network. Print `dinputs.shape` after each backward call. Verify that each layer's `dinputs` shape matches the previous layer's output shape.

---

## Part 16 — Coding Backpropagation

1. **End-to-end test.** Create a 2-layer network. Run forward, compute loss, run backward. Print all gradient shapes and verify they match parameter shapes.
2. **One step of training.** After backward pass, manually update: `layer.weights -= 0.01 * layer.dweights`. Run forward again. Did the loss decrease?
3. **Gradient accumulation bug.** What happens if you forget to reset gradients between iterations? Run 5 forward-backward passes without resetting. How do the gradients behave?

---

## Part 17 — Backprop Through Activation Functions

1. **ReLU backward by hand.** For inputs `[-1, 2, 0, 3, -0.5]` and incoming gradients `[0.5, 0.3, 0.1, 0.8, 0.2]`, compute the ReLU backward pass manually.
2. **Dead neuron detection.** After training for 100 iterations, count how many neurons in Layer 1 have all-zero outputs across the entire batch. These are "dead neurons."
3. **Leaky ReLU.** Implement a Leaky ReLU activation (slope 0.01 for negative inputs). Compare the number of dead neurons to standard ReLU after training.

---

## Part 18 — Backprop Through the Loss Function

1. **Gradient of cross-entropy.** For prediction `[0.7, 0.2, 0.1]` and true class 0, compute $-1/\hat{y}_0$ by hand. Then compute the combined softmax + cross-entropy gradient.
2. **Numerical gradient check.** Perturb one element of the softmax input by $+0.001$ and $-0.001$, recompute the loss both times, and estimate the gradient as $(L^+ - L^-) / 0.002$. Compare to your analytical gradient.

---

## Part 19 — Softmax Derivatives

1. **Jacobian computation.** For softmax output `[0.7, 0.2, 0.1]`, compute the full $3 \times 3$ Jacobian matrix by hand using $\partial S_i / \partial z_j$.
2. **Diagonal vs off-diagonal.** Verify that diagonal entries are $S_i(1 - S_i)$ and off-diagonal entries are $-S_i S_j$.
3. **Shortcut verification.** Confirm that the combined softmax + cross-entropy backward shortcut (`predictions - targets`) gives the same result as the full Jacobian approach.

---

## Part 20 — Assembling Full Backpropagation

1. **Trace the pipeline.** Draw (on paper or digitally) the complete forward and backward flow for a 2-layer network. Label every intermediate tensor and its shape.
2. **Single sample walkthrough.** Take one sample from spiral data. Run forward and backward by hand (with small layer sizes like 2 neurons each). Verify every gradient matches the code.
3. **Module checklist.** List all the classes you've built so far. For each, write one sentence about what `forward()` and `backward()` do.

---

## Part 21 — Coding Complete Backpropagation

1. **Shape audit.** After every forward and backward call, print the relevant tensor shapes. Make a table: layer → forward output shape → backward dinputs shape.
2. **Loss sanity check.** Run one forward pass with random weights. Is the loss approximately $-\ln(1/3) \approx 1.1$ (random guess for 3 classes)? If not, debug.
3. **Manual weight update.** After one backward pass, update all weights with `lr = 0.01`. Recompute forward. Does loss decrease?

---

## Part 22 — Gradient Descent

1. **Learning rate sweep.** Train for 1000 epochs with $\alpha \in \{0.001, 0.01, 0.1, 0.5, 1.0, 5.0\}$. Plot accuracy vs epoch for each. Which is best?
2. **Architecture experiment.** Try 16, 32, 64, and 128 neurons in the hidden layer. How does accuracy change for each?
3. **Loss landscape.** For a single weight $w$, perturb it in the range $[w-2, w+2]$ in 100 steps. Plot the loss at each value. Is the landscape smooth or bumpy?
4. **Epoch count.** Train with $\alpha = 1.0$ for 100, 1000, 10000, and 50000 epochs. Is there a point of diminishing returns?

---

## Part 23 — Learning Rate Decay

1. **Decay curve.** Plot $\alpha(t) = \frac{\alpha_0}{1 + \text{decay} \cdot t}$ for $\alpha_0 = 1.0$ and decay $\in \{0.001, 0.01, 0.1\}$ over 10000 steps. How fast does each shrink?
2. **Decay vs no decay.** Train two models: one with `decay=1e-3`, one without. Plot both loss curves on the same graph. When does decay help?
3. **Step decay.** Implement a step decay: cut $\alpha$ in half every 2500 epochs. Compare to the smooth $1/(1 + \text{decay} \cdot t)$ approach.

---

## Part 24 — Momentum

1. **Momentum sweep.** Train with $\beta \in \{0.0, 0.5, 0.9, 0.99\}$. Plot accuracy curves. What's the sweet spot?
2. **Oscillation visualization.** Train without momentum and print the first weight of Layer 1 every 100 epochs. Then do the same with $\beta = 0.9$. Is the path smoother with momentum?
3. **Momentum + decay.** Combine $\beta = 0.9$ with `decay=1e-3`. Does this beat either technique alone?

---

## Part 25 — AdaGrad

1. **Cache growth.** After training for 10000 epochs with AdaGrad, print the min, max, and mean of `layer.weight_cache`. How large has the cache grown?
2. **Learning rate freeze.** After 10000 epochs, compute the effective learning rate for the largest-cache weight: $\alpha / \sqrt{G + \epsilon}$. Is it essentially zero?
3. **AdaGrad vs SGD+Momentum.** Train both for 10000 epochs and compare final accuracy. Which wins and why?

---

## Part 26 — RMSProp

1. **Cache comparison.** Train with RMSProp and print `weight_cache` stats after 10000 epochs. Compare to AdaGrad's cache — which grows more?
2. **Rho sensitivity.** Try $\rho \in \{0.5, 0.9, 0.99, 0.999\}$. How does $\rho$ affect convergence speed?
3. **Long-run stability.** Train RMSProp and AdaGrad for 50000 epochs each. Does RMSProp maintain its effective learning rate better?

---

## Part 27 — Adam Optimizer

1. **Bias correction timing.** Print the bias-correction multiplier $1 / (1 - \beta_1^t)$ for $t = 1, 2, 5, 10, 50$. At what iteration does it drop below 1.1 (i.e., < 10% boost)?
2. **Adam vs everything.** Train with SGD, SGD+Momentum, AdaGrad, RMSProp, and Adam. Record final accuracy for each. Make a bar chart.
3. **Beta tuning.** Try `beta_1=0.8` and `beta_1=0.99` (keeping `beta_2=0.999`). How does this change convergence?
4. **Cold start test.** Train with and without bias correction (set correction denominator to 1.0). Compare the first 100 epochs. Is the uncorrected version slower to start?

---

## Part 28 — Generalization & Testing

1. **Overfitting detector.** After training, generate test data and compute both train and test accuracy. If the gap is > 5%, the model is likely overfitting.
2. **Model complexity sweep.** Train with 8, 16, 32, 64, 128, 256 neurons. Record train and test accuracy for each. Plot both curves. Which neuron count gives the best test accuracy?
3. **Epoch-based overfitting.** Record test accuracy every 500 epochs during training. Does it peak and then decline? At which epoch?

---

## Part 29 — Validation & Hyperparameter Tuning

1. **Manual k-fold.** Split 300 spiral data samples into 5 folds (60 each). Train on 4 folds, validate on the 5th. Rotate and compute mean validation error.
2. **Hyperparameter search.** Test 3 learning rates × 3 neuron counts = 9 combinations. Use the validation set to pick the best one. Then evaluate on the test set exactly once.
3. **Data leakage simulation.** Intentionally include 10% of test samples in training. Compare test accuracy to the clean case. How much does leakage inflate the score?

---

## Part 30 — L1 & L2 Regularization

1. **Lambda sweep.** Train with $\lambda \in \{0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$. Record train and test accuracy. Is there a "sweet spot"?
2. **Weight magnitude inspection.** After training, compute `np.mean(np.abs(layer.weights))` for each lambda value. Verify that higher lambda → smaller weights.
3. **L1 vs L2.** Train one model with L1 only, another with L2 only, and a third with both. Compare test accuracy and weight distributions (use a histogram).
4. **Sparsity check.** After L1 training, count how many weights are within 0.001 of zero. Compare to L2. L1 should push more weights to near-zero.

---

## Part 31 — Dropout

1. **Dropout rate sweep.** Train with rates `0.0, 0.1, 0.2, 0.5, 0.8`. Record train and test accuracy. Which rate gives the best test performance?
2. **Mask visualization.** During one forward pass, print the `binary_mask` for the dropout layer. Count the fraction of zeros — does it match the dropout rate?
3. **Dropout + no regularization.** Remove L2 regularization but keep dropout at 0.2. Does dropout alone prevent overfitting?
4. **Dropout only at test time (bug).** Accidentally leave dropout active during testing. How much does test accuracy drop compared to the correct implementation?
5. **Per-layer dropout.** Add dropout layers after both Layer 1 and Layer 2 with different rates (e.g., 0.2 and 0.1). Does this beat same-rate dropout?

---

*After completing these exercises, check the [Comparison Dashboards](dashboards/Optimizer_Comparison.md) to see how all optimizers and regularization methods stack up side-by-side.*
