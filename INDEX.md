# Neural Networks from Scratch — Complete Blog Series

*A 31-part series building neural networks entirely from scratch.*

---

## Foundation: Neurons & Layers
| # | Blog | Topic |
|---|---|---|
| 1 | [Part 1 — Neurons and Layers](Lecture1/blog/Part1_Neurons_and_Layers.md) | Coding a neuron, a layer, loops, and the first NumPy version |
| 2 | [Part 2 — NumPy and the Dot Product](Lecture2/blog/Part2_NumPy_Dot_Product.md) | Dot products, matrix multiplication, and batch processing |
| 3 | [Part 3 — Stacking Layers & the Forward Pass](Lecture3/blog/Part3_Stacking_Layers_Forward_Pass.md) | Chaining multiple layers together |
| 4 | [Part 4 — The Dense Layer Class & Spiral Data](Lecture4/blog/Part4_Dense_Layer_Class_Spiral_Data.md) | OOP structure, reusable layers, and non-linear data |
| 5 | [Part 5 — Array Summation, keepdims & Broadcasting](Lecture5/blog/Part5_Array_Summation_Broadcasting.md) | Shapes, axes, and broadcasting rules |

## Activation Functions & Forward Pass
| # | Blog | Topic |
|---|---|---|
| 6 | [Part 6 — Activation Functions (ReLU & Softmax)](Lecture6/blog/Part6_Activation_Functions_ReLU_Softmax.md) | Why non-linearity matters and how logits become probabilities |
| 7 | [Part 7 — Coding the Complete Forward Pass](Lecture7/blog/Part7_Complete_Forward_Pass.md) | End-to-end forward pass with classes |

## Loss Functions & Optimization Intro
| # | Blog | Topic |
|---|---|---|
| 8 | [Part 8 — Loss Functions: Categorical Cross-Entropy](Lecture8/blog/Part8_Loss_Categorical_CrossEntropy.md) | Measuring how wrong the network is |
| 9 | [Part 9 — Introduction to Optimization](Lecture9/blog/Part9_Introduction_to_Optimization.md) | Why random search fails and why gradients matter |

## Calculus for Neural Networks
| # | Blog | Topic |
|---|---|---|
| 10 | [Part 10 — Calculus: Derivatives, Partial Derivatives, and Gradients](Lecture10/blog/Part10_Calculus_Derivatives_Gradients.md) | Calculus foundations for backpropagation |
| 11 | [Part 11 — The Chain Rule](Lecture11/blog/Part11_Chain_Rule.md) | The core rule that makes backprop work |

## Backpropagation
| # | Blog | Topic |
|---|---|---|
| 12 | [Part 12 — Backpropagation Through a Single Neuron](Lecture12/blog/Part12_Backprop_Single_Neuron.md) | The first complete backward pass |
| 13 | [Part 13 — Backpropagation Through a Layer of Neurons](Lecture13/blog/Part13_Backprop_Through_a_Layer.md) | Extending gradients to a full layer |
| 14 | [Part 14 — Matrices in Backpropagation](Lecture14/blog/Part14_Matrices_in_Backpropagation.md) | Matrix-form gradients and batching |
| 15 | [Part 15 — Gradients with Respect to Inputs](Lecture15/blog/Part15_Gradients_wrt_Inputs.md) | Why `dinputs` matters for chaining layers |
| 16 | [Part 16 — Coding Backpropagation in Python](Lecture16/blog/Part16_Coding_Backpropagation.md) | Implementing `backward()` for dense layers |
| 17 | [Part 17 — Backpropagation Through Activation Functions](Lecture17/blog/Part17_Backprop_Activation_Functions.md) | ReLU backward pass and softmax preview |
| 18 | [Part 18 — Backpropagation Through the Loss Function](Lecture18/blog/Part18_Backprop_Through_Loss.md) | Cross-entropy backward pass |
| 19 | [Part 19 — Softmax Derivatives and the Combined Backward Pass](Lecture19/blog/Part19_Softmax_Derivatives.md) | The softmax + cross-entropy shortcut |
| 20 | [Part 20 — Assembling Full Backpropagation](Lecture20/blog/Part20_Assembling_Full_Backpropagation.md) | Connecting every backward block |
| 21 | [Part 21 — Coding Full Backpropagation](Lecture21/blog/Part21_Coding_Full_Backpropagation.md) | End-to-end code for the full backward pipeline |

## Optimizers
| # | Blog | Topic |
|---|---|---|
| 22 | [Part 22 — Gradient Descent Optimizer](Lecture22/blog/Part22_Gradient_Descent_Optimizer.md) | Vanilla gradient descent |
| 23 | [Part 23 — Learning Rate Decay](Lecture23/blog/Part23_Learning_Rate_Decay.md) | Reducing step size over time |
| 24 | [Part 24 — Momentum](Lecture24/blog/Part24_Momentum.md) | Smoother updates and faster convergence |
| 25 | [Part 25 — AdaGrad](Lecture25/blog/Part25_AdaGrad.md) | Per-parameter adaptive learning rates |
| 26 | [Part 26 — RMSProp](Lecture26/blog/Part26_RMSProp.md) | Fixing AdaGrad's shrinking learning rate |
| 27 | [Part 27 — The Adam Optimizer](Lecture27/blog/Part27_Adam_Optimizer.md) | Momentum + RMSProp + bias correction |

## Generalization & Regularization
| # | Blog | Topic |
|---|---|---|
| 28 | [Part 28 — Generalization & Testing](Lecture28/blog/Part28_Generalization_Testing.md) | Train vs test performance and overfitting |
| 29 | [Part 29 — Validation, Hyperparameter Tuning & K-Fold Cross-Validation](Lecture29/blog/Part29_Validation_Hyperparameter_Tuning.md) | Cleaner evaluation and better model selection |
| 30 | [Part 30 — L1 & L2 Regularization](Lecture30/blog/Part30_L1_L2_Regularization.md) | Penalizing large weights |
| 31 | [Part 31 — Dropout](Lecture31/blog/Part31_Dropout.md) | Randomly disabling neurons during training |

---

## Getting Started

| Resource | Description |
|---|---|
| [Prerequisites Self-Check](prerequisites.md) | What you need (and don't need) before starting |
| [Learning Pathway](learning_pathway.md) | Guided route through all 31 parts with progress tracking |
| [Concept Dependency Map](concept_dependency_map.md) | Which concepts depend on which — trace back when stuck |
| [Glossary](glossary.md) | Plain-English definitions for every term in the series |

## Student Resources

| Resource | Description |
|---|---|
| [Cumulative Notebook](cumulative_notebook.ipynb) | All code from the series in one runnable notebook |
| [Exercises](exercises.md) | Hands-on experiments for every lecture |
| [Quizzes](quizzes.md) | Multiple-choice checks with explanations |
| [Common Pitfalls & Debugging](common_pitfalls.md) | Most frequent mistakes and how to fix them |
| [Notation Guide](notation_guide.md) | A single place to decode symbols, tensor shapes, and optimizer notation |
| [Gradient Checking](gradient_checking.md) | Numerically verify your backprop implementation |
| [Softmax Backward Appendix](appendix_softmax_combined_backward.md) | Full derivation of the combined softmax + cross-entropy backward pass |
| [MNIST Capstone](capstone_mnist.md) | Apply the from-scratch network to handwritten digits |

### Cheat Sheets
| Sheet | Covers |
|---|---|
| [Foundation](cheatsheets/01_Foundation.md) | Neurons, layers, NumPy, broadcasting (Parts 1–5) |
| [Activations & Loss](cheatsheets/02_Activations_Loss_Forward.md) | ReLU, Softmax, cross-entropy, forward pass (Parts 6–9) |
| [Calculus & Backprop](cheatsheets/03_Calculus_Backpropagation.md) | Derivatives, chain rule, and full backward pass (Parts 10–21) |
| [Optimizers & Regularization](cheatsheets/04_Optimizers_Regularization.md) | SGD through Adam, L1/L2, and dropout (Parts 22–31) |

### Dashboards
| Dashboard | Description |
|---|---|
| [Optimizer Comparison](dashboards/Optimizer_Comparison.md) | SGD vs Decay vs Momentum vs AdaGrad vs RMSProp vs Adam |
| [Regularization Comparison](dashboards/Regularization_Comparison.md) | None vs L1 vs L2 vs Dropout vs combined |

### Production Notes
| Resource | Description |
|---|---|
| [Animation Storyboards](animation_storyboards.md) | Short-loop animation ideas for the hardest concepts |

---

*Each blog is designed to stand alone while still fitting into a coherent learning path built from first principles.*
