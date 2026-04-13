# Concept Dependency Map

*Which concepts depend on which? Use this when a topic feels confusing — trace back to its prerequisites.*

---

## How to Read This Map

- **→** means "is required by" (read left to right)
- If concept B depends on concept A, master A first before tackling B
- Each concept links to its blog post

---

## Visual Map

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    PHASE 1: FOUNDATION                  │
                    │                                                         │
                    │   [Part 1: Neurons]──→[Part 2: NumPy Dot Product]       │
                    │        │                       │                        │
                    │        └───────┬───────────────┘                        │
                    │                ↓                                        │
                    │   [Part 3: Stacking Layers]                             │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 4: Dense Layer Class] ←── Python OOP basics     │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 5: Broadcasting & keepdims]                     │
                    └────────────────┼────────────────────────────────────────┘
                                     │
                    ┌────────────────↓────────────────────────────────────────┐
                    │              PHASE 2: FORWARD PASS                      │
                    │                                                         │
                    │   [Part 6: ReLU & Softmax] ←── Part 5 (keepdims)       │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 7: Complete Forward Pass]                       │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 8: Cross-Entropy Loss] ←── Part 5 (axis, log)  │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 9: Intro to Optimization]                      │
                    └────────────────┼────────────────────────────────────────┘
                                     │
                    ┌────────────────↓────────────────────────────────────────┐
                    │           PHASE 3: CALCULUS                             │
                    │                                                         │
                    │   [Part 10: Derivatives & Gradients]                    │
                    │                │                                        │
                    │                ↓                                        │
                    │   [Part 11: Chain Rule]                                 │
                    └────────────────┼────────────────────────────────────────┘
                                     │
          ┌──────────────────────────↓──────────────────────────────────────────────┐
          │                    PHASE 4: BACKPROPAGATION                              │
          │                                                                          │
          │   [Part 12: Backprop Single Neuron] ←── Parts 10-11 (chain rule)        │
          │                │                                                         │
          │                ↓                                                         │
          │   [Part 13: Backprop Through a Layer]                                    │
          │                │                                                         │
          │         ┌──────┴──────┐                                                  │
          │         ↓             ↓                                                  │
          │   [Part 14: dW]  [Part 15: dX]   ← Matrix forms of gradients            │
          │         │             │                                                  │
          │         └──────┬──────┘                                                  │
          │                ↓                                                         │
          │   [Part 16: Coding Backprop]                                             │
          │                │                                                         │
          │         ┌──────┴──────────────────┐                                      │
          │         ↓                         ↓                                      │
          │   [Part 17: Activation Backprop] [Part 18: Loss Backprop]                │
          │         │                         │                                      │
          │         └─────────┬───────────────┘                                      │
          │                   ↓                                                      │
          │   [Part 19: Softmax + CE Combined] ←── Part 6 (softmax forward)         │
          │                   │                                                      │
          │                   ↓                                                      │
          │   [Part 20: Assembling Full Backprop]                                    │
          │                   │                                                      │
          │                   ↓                                                      │
          │   [Part 21: Coding Full Backprop]                                        │
          └───────────────────┼──────────────────────────────────────────────────────┘
                              │
          ┌───────────────────↓──────────────────────────────────────────────────────┐
          │                    PHASE 5: OPTIMIZERS                                    │
          │                                                                           │
          │   [Part 22: Gradient Descent] ←── Part 21 (full backprop code)           │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 23: Learning Rate Decay]                                          │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 24: Momentum]                                                     │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 25: AdaGrad] ←── new idea (per-parameter rates)                  │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 26: RMSProp] ←── fixes AdaGrad's problem                        │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 27: Adam] ←── combines Parts 24 + 26                             │
          └───────────────────┼──────────────────────────────────────────────────────┘
                              │
          ┌───────────────────↓──────────────────────────────────────────────────────┐
          │              PHASE 6: GENERALIZATION & REGULARIZATION                     │
          │                                                                           │
          │   [Part 28: Generalization & Testing]                                     │
          │                │                                                          │
          │                ↓                                                          │
          │   [Part 29: Validation & K-Fold]                                          │
          │                │                                                          │
          │         ┌──────┴──────┐                                                   │
          │         ↓             ↓                                                   │
          │   [Part 30: L1/L2]  [Part 31: Dropout]                                   │
          └──────────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Lookup Table

Use this when stuck on a specific Part — find what to review.

| If You're Stuck On… | Review These First |
|---------------------|--------------------|
| [Part 2](Lecture2/blog/Part2_NumPy_Dot_Product.md) — Dot products | Part 1 (neurons, weighted sums) |
| [Part 3](Lecture3/blog/Part3_Stacking_Layers_Forward_Pass.md) — Stacking layers | Parts 1-2 (single layer, transpose) |
| [Part 5](Lecture5/blog/Part5_Array_Summation_Broadcasting.md) — Broadcasting | Part 2 (array shapes) |
| [Part 6](Lecture6/blog/Part6_Activation_Functions_ReLU_Softmax.md) — Softmax | Part 5 (axis, keepdims, broadcasting) |
| [Part 8](Lecture8/blog/Part8_Loss_Categorical_CrossEntropy.md) — Cross-entropy | Part 6 (softmax outputs), Part 5 (indexing) |
| [Part 12](Lecture12/blog/Part12_Backprop_Single_Neuron.md) — Backprop | Parts 10-11 (derivatives, chain rule) |
| [Part 14](Lecture14/blog/Part14_Matrices_in_Backpropagation.md) — Matrix backprop | Parts 2, 12-13 (matrix math, single-neuron backprop) |
| [Part 16](Lecture16/blog/Part16_Coding_Backpropagation.md) — Coding backward | Parts 14-15 (formulas), Part 4 (class structure) |
| [Part 19](Lecture19/blog/Part19_Softmax_Derivatives.md) — Softmax backward | Part 6 (softmax forward), Part 11 (chain rule) |
| [Part 22](Lecture22/blog/Part22_Gradient_Descent_Optimizer.md) — Gradient descent | Part 21 (full backprop code) |
| [Part 24](Lecture24/blog/Part24_Momentum.md) — Momentum | Part 22 (basic SGD, learning rate), Part 23 (decay) |
| [Part 27](Lecture27/blog/Part27_Adam_Optimizer.md) — Adam | Parts 24 (momentum) + 26 (RMSProp) |
| [Part 30](Lecture30/blog/Part30_L1_L2_Regularization.md) — L1/L2 | Part 16 (backward method), Part 28 (overfitting) |
| [Part 31](Lecture31/blog/Part31_Dropout.md) — Dropout | Part 28 (overfitting), Part 5 (broadcasting) |

---

## Three Pillars

The 31 parts teach three pillars that converge:

```
   FORWARD PASS          BACKWARD PASS           TRAINING
   (Parts 1-8)           (Parts 10-21)          (Parts 22-31)
        │                      │                      │
   Build the              Understand how         Make weights
   computation            to compute              better over
   pipeline               gradients               time
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                        Working Neural
                         Network ✓
```

---

*Print this page and keep it next to your notebook as you work through the series.*
