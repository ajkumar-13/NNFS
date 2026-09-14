# Hand-drawn companions to the post 02 diagrams

Five scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

Four of the five are **alternates** to the clean vector figures one directory up. The
fifth is a **published figure**: section 3.1 draws a boundary around `np.dot` that the
post had no figure for, and the pitfalls list names reaching for `*` where `np.dot`
belongs as a recurring error.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-three-forms` | 960 × 540 | alternate | One name, three behaviours, chosen silently by the argument shapes. |
| `02-order-matters` | 960 × 500 | alternate | Same matrix, same vector, swapped order, two different answers. |
| `03-shape-rule` | 960 × 420 | alternate | Two axes have to agree and vanish; the other two are the answer. |
| `04-batch-transpose` | 960 × 500 | alternate | The failing call above the working one, so the one character between them is visible. |
| `05-not-the-same-call` | 960 × 460 | published | Section 3.1: `np.dot`, `*`, and `@` separated by what each does to the shape. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:02
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/02-numpy-and-the-dot-product/diagrams/excalidraw/03-shape-rule.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:02` will fight over the same file, so fold anything worth keeping
back into `post02.js`.

## Notes on these

- **`01-three-forms`.** The operand grids are drawn empty. This figure is about
  which shapes are legal and what comes out, and numbers in forty-odd cells
  would pull it towards a different argument. The grids centre on 234 rather
  than the panel's midpoint, because the band they sit in is bounded by the two
  rules at 152 and 316, not by the card.
- **`02-order-matters`.** Both panels carry the same matrix and the same vector,
  and both results are the post's own printed output, so the figure can be
  checked against the code block. The walked axis is outlined on the operand it
  is walked over — column one on the left, row one on the right — and the worked
  sum for that one slice is written underneath.
- **`03-shape-rule`.** Each shape is drawn as its two axes in separate boxes
  rather than typeset as `(m, n)`. Tinting one character inside a string means
  stepping x by a guessed font advance, and the guess is wrong the moment the
  reader's monospace font is not the one it was measured against: Consolas
  advances at 0.55em where JetBrains Mono is 0.60, which over a short expression
  pulls the pieces visibly apart. Boxes are geometry the renderer controls.
- **`04-batch-transpose`.** The failing call sits directly above the working one
  at equal weight. One character separates them, and stacking is what makes that
  character visible; side by side, the eye compares layouts instead.
- **`05-not-the-same-call`.** The lasso around the middle card is stroke-only and
  sits inside the card, around the two shape lines alone. Drawn any wider it
  crosses into the cards on either side, which is what a first pass did.
