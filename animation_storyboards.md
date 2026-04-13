# Animation Storyboards — Short Concept Loops for Hard Topics

*A production guide for adding short, repeatable motion assets that make the hardest ideas easier to grasp.*

---

## Why Motion Helps This Series

The written blogs and SVGs already do a good job with static structure. Motion helps most when the learner must understand:

- how values move through a network,
- how shapes change during matrix operations,
- how gradients split and recombine,
- how optimizer paths differ over time,
- and when training-time behavior differs from test-time behavior.

For this series, the best format is **short looping clips** rather than long explainer videos.

### Suggested format

- Duration: 12–25 seconds
- Export: MP4 and WebM
- Optional fallback: GIF for markdown previews
- Style: clean labels, large arrows, minimal text, one concept per clip
- Reuse: base frames can be built from the existing SVG assets

---

## Priority Order

If you only build five animations first, make these:

1. Batch matrix multiply + transpose
2. Broadcasting + `keepdims=True`
3. Softmax stabilization and probability conversion
4. Gradient flow through the chain rule
5. Optimizer path comparison in a narrow valley

## Prototype Assets Added In This Repo

The first five prototype assets are now available as animated SVGs, and the initial layouts have been refined for better spacing and label fit:

- [Batch transpose animation](Lecture2/blog/assets/batch-transpose-animation.svg)
- [Broadcasting + keepdims animation](Lecture5/blog/assets/broadcasting-keepdims-animation.svg)
- [Softmax stabilization animation](Lecture6/blog/assets/softmax-stabilization-animation.svg)
- [Chain rule flow animation](Lecture11/blog/assets/chain-rule-flow-animation.svg)
- [Optimizer valley animation](Lecture22/blog/assets/optimizer-valley-animation.svg)

Additional motion companions now cover later backpropagation and regularization topics:

- [Single-neuron backprop flow](Lecture12/blog/assets/single-neuron-backprop-flow-animation.svg)
- [Layer backprop flow](Lecture13/blog/assets/layer-backprop-flow-animation.svg)
- [Matrix weight gradient animation](Lecture14/blog/assets/matrix-weight-gradient-animation.svg)
- [Input-gradient accumulation animation](Lecture15/blog/assets/input-gradient-accumulation-animation.svg)
- [Softmax shortcut animation](Lecture19/blog/assets/softmax-shortcut-animation.svg)
- [Dropout training vs testing](Lecture31/blog/assets/dropout-train-test-animation.svg)
- [Overfitting training vs validation divergence](Lecture28/blog/assets/overfitting-animation.svg)
- [Adam optimizer five-step pipeline](Lecture27/blog/assets/adam-breakdown-animation.svg)

---

## 1. Batch Matrix Multiply + Transpose

**Lectures:** 2–3  
**Learner question:** Why do we need `weights.T` for batches?  
**Duration:** 15 seconds

### Visual sequence

1. Show input matrix `(3 x 4)` sliding in from the left.
2. Show weight matrix `(3 x 4)` and flash a red mismatch on inner dimensions.
3. Rotate/flip the weight matrix into `(4 x 3)`.
4. Highlight matching inner dimensions.
5. Animate the multiplication result appearing as `(3 x 3)`.

### On-screen text

- "Inputs: 3 samples x 4 features"
- "Weights: 3 neurons x 4 weights"
- "Transpose -> 4 x 3"
- "Now (3 x 4) · (4 x 3) works"

### Voiceover script

"For one input vector, each neuron can dot its weights with the input directly. For a batch, the input becomes a matrix. Now the inner dimensions must match, so we transpose the weight matrix and get one output row per sample."

---

## 2. Broadcasting + `keepdims=True`

**Lecture:** 5  
**Learner question:** Why does `(3,)` behave differently from `(3,1)`?  
**Duration:** 18 seconds

### Visual sequence

1. Show a `(3 x 4)` matrix.
2. Sum along `axis=1` and display the result as `(3,)`.
3. Try to subtract it from the original matrix and show the alignment failure.
4. Re-run with `keepdims=True` to get `(3 x 1)`.
5. Animate the column vector stretching across 4 columns through broadcasting.

### On-screen text

- "Without keepdims -> shape (3,)"
- "With keepdims -> shape (3 x 1)"
- "Broadcast down each row"

### Voiceover script

"The values are the same, but the shape changes everything. A flat `(3,)` vector cannot align the way we want here. Keeping the reduced dimension gives a `(3 x 1)` column, and now NumPy can broadcast it across every row correctly."

---

## 3. Softmax Stabilization and Probability Conversion

**Lecture:** 6  
**Learner question:** Why subtract the max before exponentials?  
**Duration:** 20 seconds

### Visual sequence

1. Start with logits `[1001, 1002, 1003]`.
2. Animate `exp(logit)` exploding off-screen.
3. Reset and subtract max to get `[-2, -1, 0]`.
4. Animate stable exponentials `[0.135, 0.368, 1.0]`.
5. Normalize to probabilities summing to `1.0`.

### On-screen text

- "Raw logits"
- "Exponentials overflow"
- "Subtract max -> same probabilities, safer numbers"
- "Normalize -> probabilities"

### Voiceover script

"Softmax needs exponentials, but exponentials explode fast. Subtracting the maximum shifts every value by the same amount, so the final probabilities stay the same while the computation becomes numerically stable."

---

## 4. Chain Rule as Gradient Flow

**Lectures:** 11–12  
**Learner question:** What does the chain rule mean operationally?  
**Duration:** 16 seconds

### Visual sequence

1. Show a tiny graph: `x -> z -> a -> L`.
2. Forward arrows move left to right in one color.
3. Backward arrows move right to left in another color.
4. At each edge, multiply the incoming gradient by the local derivative.
5. End with a boxed `dL/dx`.

### On-screen text

- "Forward: compute values"
- "Backward: multiply local derivatives"
- "Each step passes one gradient to the previous node"

### Voiceover script

"The chain rule is not just a formula to memorize. In code, it is a routing rule. Each block receives a gradient from the right, multiplies by its local derivative, and sends a new gradient left."

---

## 5. Single-Neuron Backprop Sign Check

**Lecture:** 12  
**Learner question:** How do I know whether a weight should go up or down?  
**Duration:** 12 seconds

### Visual sequence

1. Show a neuron with one weight and one bias.
2. Display current output above target.
3. Flash positive and negative weight tweaks.
4. Show one tweak lowering loss and the other raising loss.
5. Highlight the gradient sign and the update direction.

### Voiceover script

"A gradient is directional advice. If increasing a weight increases the loss, the gradient is positive and gradient descent moves the weight down. If increasing the weight lowers the loss, the sign flips and the update goes up instead."

---

## 6. Layer Backprop Path Accumulation

**Lectures:** 13–15  
**Learner question:** Why do input gradients collect contributions from multiple neurons?  
**Duration:** 18 seconds

### Visual sequence

1. Show one input node connected to three neurons.
2. Animate three separate gradient paths flowing backward.
3. Sum the three contributions into one `dinput` value.
4. Contrast this with a single weight that only belongs to one path.

### On-screen text

- "One input affects multiple neurons"
- "Backward: contributions add"
- "One weight belongs to one connection"

### Voiceover script

"A weight lives on one edge, so its gradient comes from one path. But an input fans out to many neurons, so its backward signal must add the contribution from every downstream path."

---

## 7. Softmax + Cross-Entropy Shortcut

**Lecture:** 19  
**Learner question:** Why does the backward pass collapse to predicted minus true?  
**Duration:** 20 seconds

### Visual sequence

1. Show the long path: logits -> softmax -> probabilities -> cross-entropy.
2. Overlay a dense Jacobian block and mark it as "messy."
3. Fade into the simplified formula `y_hat - y`.
4. Show one worked row: `[0.7, 0.1, 0.2] - [1, 0, 0]`.

### Voiceover script

"Softmax alone couples all outputs, which makes its backward pass messy. But once you combine it with cross-entropy, the algebra collapses into a clean result: predicted minus true, normalized by batch size."

---

## 8. Optimizer Paths in a Narrow Valley

**Lectures:** 22–27  
**Learner question:** Why do SGD, momentum, RMSProp, and Adam behave differently?  
**Duration:** 20 seconds

### Visual sequence

1. Draw a narrow curved valley.
2. Animate SGD zig-zagging.
3. Animate momentum smoothing the zig-zag.
4. Animate RMSProp taking uneven step sizes.
5. Animate Adam reaching the minimum fastest.

### On-screen text

- "SGD: noisy path"
- "Momentum: smoother direction"
- "RMSProp: adaptive step sizes"
- "Adam: both"

### Voiceover script

"All optimizers follow gradients, but they use that information differently. SGD follows the current slope only. Momentum remembers past directions. RMSProp scales steps per parameter. Adam combines both ideas."

---

## 9. Overfitting vs Generalization

**Lectures:** 28–29  
**Learner question:** Why is high training accuracy not enough?  
**Duration:** 16 seconds

### Visual sequence

1. Show a training curve climbing steadily.
2. Show a validation curve peaking and then drifting downward.
3. Highlight the widening gap.
4. Label the gap "overfitting."

### Voiceover script

"The model is not judged by how well it memorizes training data. It is judged by how well it handles new data. When training keeps improving but validation gets worse, the model is learning the training set too specifically."

---

## 10. Dropout During Training vs Testing

**Lecture:** 31  
**Learner question:** Why is dropout on during training but off during testing?  
**Duration:** 15 seconds

### Visual sequence

1. Show a hidden layer with some neurons randomly dimmed during training.
2. Show the scaled surviving outputs.
3. Switch to testing mode with all neurons active.
4. Keep the overall activation magnitude visually similar.

### Voiceover script

"During training, dropout removes random neurons so the network cannot over-rely on any one path. During testing, we turn dropout off and use the full network. The training-time scaling keeps the expected magnitude consistent between the two modes."

---

## Production Guidelines

### Visual language

- Keep the same color mapping across clips.
- Use one dominant concept per animation.
- Show shapes directly on matrices and tensors.
- Prefer arrows and highlights over dense text blocks.

### Export guidelines

- 16:9 for YouTube and embedded pages
- 1:1 or 4:5 variants for shorts and social snippets
- Subtitles burned in for silent autoplay
- First 2 seconds must communicate the concept without audio

### Reuse strategy

- Start from existing SVG diagrams as base frames.
- Animate in HTML/CSS/JS or After Effects, then export.
- Reuse common elements: arrows, labels, color keys, tensor boxes.

---

## Recommended Build Order

1. Batch transpose
2. Broadcasting + keepdims
3. Softmax stabilization
4. Chain rule flow
5. Optimizer valley comparison
6. Layer backprop accumulation
7. Dropout training vs testing
8. Overfitting vs validation
9. Softmax + cross-entropy shortcut
10. Single-neuron sign intuition

---

*These clips should be treated as companions to the blogs, not replacements. The blog explains the idea. The animation should make the same idea obvious in one glance.*