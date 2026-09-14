# Hand-drawn companions to the post 14 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 6 observes that the gradient shapes do not depend on the
batch size and section 9 explains why the sum over samples needs no code, and neither
had a figure.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-matrix-weight-gradient` | 960 × 480 | alternate | The shape arithmetic, then the same twelve numbers Part 13 wrote by hand. |
| `02-batch-axis-contracts` | 960 × 480 | published | Sections 6 and 9: N sits in the contracted position, so it vanishes. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:14
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/14-matrices-in-backpropagation/diagrams/excalidraw/02-batch-axis-contracts.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:14` will fight over the same file, so fold anything worth keeping
back into `post14.js`.

## Notes on these

- **`01-matrix-weight-gradient`.** This covers similar ground to post 13's
  `02-outer-product`, which is the posts' own doing: Part 13 reaches the matrix
  by broadcasting a reshaped column and Part 14 reaches it by a matrix product.
  The alternate leans on what is actually different here — the shape arithmetic
  drawn at full size, and the observation that the result already carries the
  shape the weights are stored in, so `weights -= lr * dL_dW` needs no reshaping.
- **`02-batch-axis-contracts`.** Both rows end on an identical `(m, n)` pair of
  boxes. That repetition is the argument: the only thing that changes between
  one sample and a batch of five hundred is the length of the axis being
  contracted, and the contracted axis is by definition the one that does not
  survive. The band puts the two lines of the backward pass side by side so the
  asymmetry is visible — the weight sum hides inside the product, and the bias
  sum has nowhere to hide.
