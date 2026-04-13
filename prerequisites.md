# Before You Start — Prerequisites Self-Check

*Everything you need (and don't need) before diving into the series.*

---

## What You DO Need

### 1. Basic Python (30 minutes to verify)

Can you do all of these? If yes, you're ready for Part 1.

```python
# Variables and arithmetic
x = 5
y = x * 2 + 1

# Lists
my_list = [1, 2, 3, 4]
print(my_list[0])     # → 1
print(len(my_list))   # → 4

# Loops
for item in my_list:
    print(item)

# Functions
def add(a, b):
    return a + b

result = add(3, 4)    # → 7

# zip()
names = ["Alice", "Bob"]
scores = [90, 85]
for name, score in zip(names, scores):
    print(name, score)
```

**Not comfortable with these?** Spend 1-2 hours on any Python basics tutorial first. The series assumes this level.

---

### 2. Basic Math (Algebra Level)

Can you evaluate these?

| Expression | Answer |
|-----------|--------|
| $3 \times 4 + 2$ | 14 |
| $0.5 \times (-2) + 1$ | 0 |
| $e^0$ | 1 |
| $\ln(1)$ | 0 |

If you know what multiplication, addition, and exponents are, you have enough math. **No calculus needed** — we teach it from scratch in Part 10.

---

### 3. Python Environment

You need Python 3.7+ with NumPy installed:

```bash
pip install numpy
pip install nnfs      # for spiral dataset helper
```

**Optional but recommended:**
- Jupyter Notebook or VS Code with the Python extension
- The [Cumulative Notebook](cumulative_notebook.ipynb) open alongside the blog posts

---

## What You DON'T Need

| NOT Required | Why |
|-------------|-----|
| Linear algebra course | We teach matrix operations (dot product, transpose) from scratch in Parts 2-3 |
| Calculus course | We teach derivatives and chain rule from scratch in Parts 10-11 |
| Prior ML/DL experience | We start from zero — if you've used TensorFlow/PyTorch, great, but not assumed |
| GPU | Everything runs on CPU. No CUDA needed |
| Advanced Python (decorators, generators, metaclasses) | Only basic OOP (classes, `__init__`, `self`) — taught in Part 4 |

---

## Recommended Study Setup

1. **Split screen:** Blog post on one side, Jupyter notebook on the other
2. **Type the code yourself** — don't copy-paste. Typing builds muscle memory
3. **Predict before running:** Before executing a code cell, guess the output. If wrong, figure out why
4. **Use the SVG diagrams:** They're not decorative — they show the exact computation happening at each step
5. **Do the exercises:** Reading is passive; exercises make knowledge stick

---

## Quick Start

Ready? Start here: **[Part 1 — Neurons and Layers](Lecture1/blog/Part1_Neurons_and_Layers.md)**

Or if you want the guided route: **[Learning Pathway](learning_pathway.md)**

---

*This series assumes you learn best by building things. Every concept is followed by code you can run.*
