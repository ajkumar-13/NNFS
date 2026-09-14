# Hand-drawn companions to the post 17 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 3 generalises ReLU into a family that shares one backward
line and differs only in `f′`, and a table of three formulas is exactly the thing a
figure can say better than prose.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-elementwise-vs-coupled` | 960 × 480 | alternate | Two Jacobians at the same size; only the count of non-zero cells differs. |
| `02-elementwise-family` | 960 × 480 | published | Section 3: three activations, three derivative shapes, one line of code. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:17
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/17-backpropagation-through-activation-functions/diagrams/excalidraw/02-elementwise-family.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:17` will fight over the same file, so fold anything worth keeping
back into `post17.js`.

## Notes on these

- **`01-elementwise-vs-coupled`.** The zeros stay on the canvas in the diagonal
  Jacobian rather than being left blank. The two panels have to be the same
  object at the same size for the comparison to land; an empty grid beside a
  full one reads as two different kinds of thing rather than as one matrix with
  fewer live entries.
- **`02-elementwise-family`.** Each derivative is plotted on its own vertical
  scale, with its maximum labelled. Sharing one scale across the three would
  draw sigmoid's derivative — which peaks at 0.25 — as a flat smear next to
  ReLU's and tanh's, and the point of the panel is the *shape* of each curve,
  not their relative heights. The 0.25 label is what carries the magnitude
  instead.
