# Appendix — Why Softmax + Cross-Entropy Backward Simplifies to Predicted Minus True

*This appendix expands Part 19's combined-backward derivation with extra algebraic steps, a class-index implementation, and a numerical check.*

---

## Goal

Show why the backward pass for **softmax + categorical cross-entropy** becomes:

$$
\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k
$$

That result looks magical if you only see the final formula. It is not magic. It comes from rewriting the loss in a cleaner form before differentiating.

---

## 1. Start with Softmax

For class $k$:

$$
\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

where:

- $z_k$ is the logit for class $k$,
- $\hat{y}_k$ is the predicted probability for class $k$.

---

## 2. Write Cross-Entropy for One Sample

For one-hot targets:

$$
L = -\sum_i y_i \log(\hat{y}_i)
$$

Because one-hot labels have exactly one `1`, only the true class survives the sum.

If the true class is $t$, then:

$$
L = -\log(\hat{y}_t)
$$

---

## 3. Substitute the Softmax Expression

Replace $\hat{y}_t$ with the softmax formula:

$$
L = -\log\left(\frac{e^{z_t}}{\sum_j e^{z_j}}\right)
$$

Use the log rule $\log(a/b) = \log a - \log b$:

$$
L = -\left(\log(e^{z_t}) - \log\left(\sum_j e^{z_j}\right)\right)
$$

Since $\log(e^{z_t}) = z_t$:

$$
L = -z_t + \log\left(\sum_j e^{z_j}\right)
$$

This is the key simplification.

---

## 4. Differentiate with Respect to One Logit $z_k$

Now compute:

$$
\frac{\partial L}{\partial z_k}
$$

Differentiate the two terms separately.

### First term: $-z_t$

$$
\frac{\partial (-z_t)}{\partial z_k} =
\begin{cases}
-1 & k = t \\
0 & k \ne t
\end{cases}
$$

That is exactly $-y_k$ in one-hot form.

### Second term: $\log\left(\sum_j e^{z_j}\right)$

Apply the chain rule:

$$
\frac{\partial}{\partial z_k} \log\left(\sum_j e^{z_j}\right)
= \frac{1}{\sum_j e^{z_j}} \cdot \frac{\partial}{\partial z_k}\left(\sum_j e^{z_j}\right)
$$

Only the $k$th exponential depends on $z_k$, so:

$$
\frac{\partial}{\partial z_k}\left(\sum_j e^{z_j}\right) = e^{z_k}
$$

Therefore:

$$
\frac{\partial}{\partial z_k} \log\left(\sum_j e^{z_j}\right)
= \frac{e^{z_k}}{\sum_j e^{z_j}}
= \hat{y}_k
$$

---

## 5. Combine the Two Pieces

$$
\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k
$$

That is the full result.

In vector form:

$$
\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}
$$

---

## 6. Why This Avoids the Full Jacobian

If you differentiate softmax alone, you get a **Jacobian matrix** because each output depends on all logits.

If you differentiate cross-entropy alone, you get a gradient with respect to the softmax outputs.

Multiplying those two pieces is valid, but messy.

The trick above avoids that mess because:

1. combine softmax and cross-entropy first,
2. rewrite the loss using `log(softmax)`,
3. differentiate the simplified expression directly.

The full Jacobian has not disappeared. Its effect is already folded into the simplified expression.

---

## 7. Class-Index Version

In code, labels are often stored as class indices rather than one-hot rows.

If the true class is `t`, then the one-hot vector is implicit. The combined backward step becomes:

```python
dinputs = predictions.copy()
dinputs[range(samples), y_true] -= 1
dinputs /= samples
```

That line:

- keeps the predicted probabilities for every class,
- subtracts `1` only at the true-class position,
- divides by batch size to convert the sum into a mean gradient.

---

## 8. Tiny Numerical Check

Suppose:

$$
\hat{\mathbf{y}} = [0.7, 0.1, 0.2], \quad \mathbf{y} = [1, 0, 0]
$$

Then:

$$
\hat{\mathbf{y}} - \mathbf{y} = [0.7 - 1, 0.1 - 0, 0.2 - 0] = [-0.3, 0.1, 0.2]
$$

If this is one sample in a batch of size 3, divide by 3:

$$
\left[-0.1, 0.0333, 0.0667\right]
$$

This matches the code and the result shown in Part 19.

---

## 9. Intuition

The combined gradient is easy to read:

- if the model assigns **too much** probability to a class, the gradient is positive there,
- if the model assigns **too little** probability to the true class, the gradient is negative there.

So the update pushes:

- the true class logit **up**,
- the incorrect class logits **down**.

That is exactly what we want the model to learn.

---

## 10. Summary

The shortcut works because:

1. cross-entropy with one-hot labels becomes $-\log(\hat{y}_t)$,
2. `log(softmax)` becomes `logit of true class minus log-sum-exp`,
3. differentiating that simplified form gives:

$$
\boxed{\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}}
$$

---

*Best companion reading:* [Part 19 — Softmax Derivatives](posts/19-softmax-derivatives-and-the-combined-backward-pass/index.md) and [Gradient Checking](gradient_checking.md).