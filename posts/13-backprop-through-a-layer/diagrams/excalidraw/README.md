# Hand-drawn companions to the post 13 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 7.1 calls one line the structural heart of the post and
names it an outer product, and that line is the bridge into Part 14, but nothing drew it.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-layer-backprop` | 960 × 540 | alternate | Fifteen parameters, one upstream number, and a table of where it lands. |
| `02-outer-product` | 960 × 450 | published | Section 7.1: a column against a row, with one cell traced to its pair. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:13
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/13-backprop-through-a-layer/diagrams/excalidraw/02-outer-product.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:13` will fight over the same file, so fold anything worth keeping
back into `post13.js`.

## Notes on these

- **`01-layer-backprop`.** The backward row is one chain, not three. The claim of
  the section is that the upstream gradient is computed once at the loss and
  shared by all fifteen parameters, and drawing three parallel chains would
  argue the opposite. The gradient table is laid out with inputs across the
  columns so the repeated rows are visible: every neuron gets the same numbers,
  because only the input index varies.
- **`02-outer-product`.** One cell of the result is outlined and traced by
  dotted leaders back to the two numbers that made it, with the arithmetic
  spelled out beneath. Without that trace the figure is three grids that happen
  to sit next to each other; with it, the reshape in the code line stops looking
  like a NumPy trick and starts looking like the shape of the operation.
