# Neural Networks from Scratch — Learning Pathway

*A guided route through the 31-part series. Use the checkboxes to track your progress.*

---

## How to Use This Guide

1. **Work through phases in order** — each builds on the last
2. **Check off** each item as you complete it: blog → quiz → exercise
3. **Don't skip the exercises** — reading alone won't build intuition
4. **Use the prerequisite check** at the start of each phase
5. **Revisit** earlier phases if a later concept feels unclear

---

## Phase 1: Foundation (Parts 1–5)

> **Goal:** Understand what a neuron and layer compute, and become fluent in NumPy.

**Prerequisite check:** Can you write a Python list, call a function, and use a `for` loop? If yes, you're ready.

| # | Task | Status |
|---|------|--------|
| 1 | Read [Part 1 — Neurons and Layers](Lecture1/blog/Part1_Neurons_and_Layers.md) | ☐ |
| 2 | Complete Part 1 Quiz (3 min) | ☐ |
| 3 | Do Part 1 Exercises (code by hand) | ☐ |
| 4 | Read [Part 2 — NumPy Dot Product](Lecture2/blog/Part2_NumPy_Dot_Product.md) | ☐ |
| 5 | Complete Part 2 Quiz | ☐ |
| 6 | Do Part 2 Exercises (predict shapes before running) | ☐ |
| 7 | Read [Part 3 — Stacking Layers](Lecture3/blog/Part3_Stacking_Layers_Forward_Pass.md) | ☐ |
| 8 | Complete Part 3 Quiz | ☐ |
| 9 | Read [Part 4 — Dense Layer Class](Lecture4/blog/Part4_Dense_Layer_Class_Spiral_Data.md) | ☐ |
| 10 | Complete Part 4 Quiz + Exercises | ☐ |
| 11 | Read [Part 5 — Summation & Broadcasting](Lecture5/blog/Part5_Array_Summation_Broadcasting.md) | ☐ |
| 12 | Complete Part 5 Quiz + Exercises | ☐ |

**Phase 1 checkpoint:** You should be able to:
- [ ] Code a neuron from scratch without looking at notes
- [ ] Predict the output shape of `np.dot(A, B)` for any A, B shapes
- [ ] Explain why `keepdims=True` matters in one sentence
- [ ] Build a `Layer_Dense` class from memory

---

## Phase 2: Forward Pass Complete (Parts 6–9)

> **Goal:** Build and run an end-to-end classification network.

**Prerequisite check:** Can you explain what `np.dot(X, W) + b` computes and predict its output shape? If yes, you're ready.

| # | Task | Status |
|---|------|--------|
| 1 | Read [Part 6 — Activation Functions](Lecture6/blog/Part6_Activation_Functions_ReLU_Softmax.md) | ☐ |
| 2 | Complete Part 6 Quiz + Exercises | ☐ |
| 3 | Read [Part 7 — Complete Forward Pass](Lecture7/blog/Part7_Complete_Forward_Pass.md) | ☐ |
| 4 | **Milestone:** Run the complete forward pass yourself in a notebook | ☐ |
| 5 | Read [Part 8 — Loss Functions](Lecture8/blog/Part8_Loss_Categorical_CrossEntropy.md) | ☐ |
| 6 | Complete Part 8 Quiz + Exercises | ☐ |
| 7 | Read [Part 9 — Introduction to Optimization](Lecture9/blog/Part9_Introduction_to_Optimization.md) | ☐ |
| 8 | Complete Part 9 Quiz + Exercises | ☐ |

**Phase 2 checkpoint:** You should be able to:
- [ ] Explain why stacking linear layers without activation is useless
- [ ] Hand-compute softmax for a 3-element vector
- [ ] Compute cross-entropy loss for a single prediction
- [ ] Explain why random search fails for optimization

---

## Phase 3: Calculus Foundation (Parts 10–11)

> **Goal:** Understand derivatives, partial derivatives, and the chain rule.

**Prerequisite check:** Can you compute `f(x) = 3x²` at x=2? If yes, you're ready. No calculus background needed — we start from zero.

| # | Task | Status |
|---|------|--------|
| 1 | Read [Part 10 — Derivatives & Gradients](Lecture10/blog/Part10_Calculus_Derivatives_Gradients.md) | ☐ |
| 2 | Complete Part 10 Quiz + Exercises | ☐ |
| 3 | Read [Part 11 — The Chain Rule](Lecture11/blog/Part11_Chain_Rule.md) | ☐ |
| 4 | Complete Part 11 Quiz + Exercises | ☐ |

**Phase 3 checkpoint:** You should be able to:
- [ ] Differentiate $f(x) = 5x^3 + 2x$ using the power rule
- [ ] Compute partial derivatives of $f(x,y) = x^2 y + 3y$
- [ ] Apply the chain rule to $f(g(x))$ where $g(x) = 2x+1$ and $f(g) = g^2$

---

## Phase 4: Backpropagation (Parts 12–21)

> **Goal:** Understand and code the complete backward pass.

**Prerequisite check:** Can you state the chain rule in symbols? If yes, you're ready. This is the hardest phase — take your time.

| # | Task | Status |
|---|------|--------|
| 1 | Read [Part 12 — Backprop Single Neuron](Lecture12/blog/Part12_Backprop_Single_Neuron.md) | ☐ |
| 2 | Complete Part 12 Exercises (trace all 4 gradients by hand) | ☐ |
| 3 | Read [Part 13 — Backprop Through a Layer](Lecture13/blog/Part13_Backprop_Through_a_Layer.md) | ☐ |
| 4 | Complete Part 13 Quiz + Exercises | ☐ |
| 5 | Read [Part 14 — Matrices in Backprop](Lecture14/blog/Part14_Matrices_in_Backpropagation.md) | ☐ |
| 6 | Read [Part 15 — Gradients wrt Inputs](Lecture15/blog/Part15_Gradients_wrt_Inputs.md) | ☐ |
| 7 | **Milestone:** Write down the 3 backprop formulas from memory | ☐ |
| 8 | Read [Part 16 — Coding Backprop](Lecture16/blog/Part16_Coding_Backpropagation.md) | ☐ |
| 9 | Complete Part 16 Exercises | ☐ |
| 10 | Read [Part 17 — Backprop Activation Functions](Lecture17/blog/Part17_Backprop_Activation_Functions.md) | ☐ |
| 11 | Read [Part 18 — Backprop Through Loss](Lecture18/blog/Part18_Backprop_Through_Loss.md) | ☐ |
| 12 | Read [Part 19 — Softmax Derivatives](Lecture19/blog/Part19_Softmax_Derivatives.md) | ☐ |
| 13 | *(Optional)* Read the [Softmax Backward Appendix](appendix_softmax_combined_backward.md) for full derivation | ☐ |
| 14 | Read [Part 20 — Assembling Full Backprop](Lecture20/blog/Part20_Assembling_Full_Backpropagation.md) | ☐ |
| 15 | Read [Part 21 — Coding Full Backprop](Lecture21/blog/Part21_Coding_Full_Backpropagation.md) | ☐ |
| 16 | **Milestone:** Run full forward + backward pass in notebook | ☐ |
| 17 | Run the [Gradient Checker](gradient_checking.md) to verify your implementation | ☐ |

**Phase 4 checkpoint:** You should be able to:
- [ ] Explain backprop in one paragraph without using code
- [ ] Write `backward()` for `Layer_Dense` from memory
- [ ] Explain why `dinputs` is the "glue" between layers
- [ ] State the combined softmax + cross-entropy gradient: $\hat{y} - y$

---

## Phase 5: Optimizers (Parts 22–27)

> **Goal:** Build 5 progressively better optimizers and see training actually work.

**Prerequisite check:** Can you run a forward+backward pass and inspect gradients? If yes, you're ready. This phase is fun — you finally see the network learn.

| # | Task | Status |
|---|------|--------|
| 1 | Read [Part 22 — Gradient Descent](Lecture22/blog/Part22_Gradient_Descent_Optimizer.md) | ☐ |
| 2 | **Milestone:** Train a network and watch loss decrease! | ☐ |
| 3 | Read [Part 23 — Learning Rate Decay](Lecture23/blog/Part23_Learning_Rate_Decay.md) | ☐ |
| 4 | Read [Part 24 — Momentum](Lecture24/blog/Part24_Momentum.md) | ☐ |
| 5 | Complete Parts 22–24 Quiz + Exercises | ☐ |
| 6 | Read [Part 25 — AdaGrad](Lecture25/blog/Part25_AdaGrad.md) | ☐ |
| 7 | Read [Part 26 — RMSProp](Lecture26/blog/Part26_RMSProp.md) | ☐ |
| 8 | Read [Part 27 — Adam](Lecture27/blog/Part27_Adam_Optimizer.md) | ☐ |
| 9 | Complete Parts 25–27 Quiz + Exercises | ☐ |
| 10 | Review the [Optimizer Comparison Dashboard](dashboards/Optimizer_Comparison.md) | ☐ |

**Phase 5 checkpoint:** You should be able to:
- [ ] Explain what problem each optimizer solves over the previous one
- [ ] State the Adam update rule from memory
- [ ] Train a network to >90% accuracy on spiral data

---

## Phase 6: Generalization & Regularization (Parts 28–31)

> **Goal:** Make models that work on unseen data.

**Prerequisite check:** Can you train a model with Adam and get >90% accuracy? If yes, you're ready.

| # | Task | Status |
|---|------|--------|
| 1 | Read [Part 28 — Generalization](Lecture28/blog/Part28_Generalization_Testing.md) | ☐ |
| 2 | Read [Part 29 — Validation & K-Fold](Lecture29/blog/Part29_Validation_Hyperparameter_Tuning.md) | ☐ |
| 3 | Complete Parts 28–29 Quiz + Exercises | ☐ |
| 4 | Read [Part 30 — L1 & L2 Regularization](Lecture30/blog/Part30_L1_L2_Regularization.md) | ☐ |
| 5 | Read [Part 31 — Dropout](Lecture31/blog/Part31_Dropout.md) | ☐ |
| 6 | Complete Parts 30–31 Quiz + Exercises | ☐ |
| 7 | Review the [Regularization Comparison Dashboard](dashboards/Regularization_Comparison.md) | ☐ |

**Phase 6 checkpoint:** You should be able to:
- [ ] Explain overfitting to a non-technical person
- [ ] Add L2 regularization to your network's forward and backward passes
- [ ] Implement dropout with correct train/test behavior

---

## Capstone: MNIST

> **Goal:** Apply everything to a real-world problem.

| # | Task | Status |
|---|------|--------|
| 1 | Complete the [MNIST Capstone](capstone_mnist.md) | ☐ |
| 2 | Achieve ≥95% test accuracy | ☐ |
| 3 | Experiment: try different architectures, optimizers, regularization | ☐ |

---

## Reference Materials

Use these alongside the main series:

| Resource | When to Use |
|----------|------------|
| [Glossary](glossary.md) | When you encounter an unfamiliar term |
| [Notation Guide](notation_guide.md) | When a symbol is unclear |
| [Concept Dependency Map](concept_dependency_map.md) | When you're unsure what to review |
| [Cheatsheets](cheatsheets/) | Quick refresher on a topic section |
| [Cumulative Notebook](cumulative_notebook.ipynb) | Running code from any lecture |
| [Quizzes](quizzes.md) | Testing your understanding |
| [Exercises](exercises.md) | Hands-on practice |

---

## Estimated Time

| Phase | Parts | Estimated Time |
|-------|:---:|----------------|
| 1. Foundation | 1–5 | One focused weekend |
| 2. Forward Pass | 6–9 | 1–2 days |
| 3. Calculus | 10–11 | 1 day |
| 4. Backprop | 12–21 | 1–2 weeks (the core) |
| 5. Optimizers | 22–27 | 3–4 days |
| 6. Regularization | 28–31 | 2–3 days |
| Capstone | MNIST | 1–2 days |

**Total:** ~3–4 weeks at a steady pace, or ~2 weeks intensive.

---

*Remember: understanding every step is more valuable than rushing through. If something doesn't click, re-read the SVG diagrams and work through the numerical examples by hand.*
