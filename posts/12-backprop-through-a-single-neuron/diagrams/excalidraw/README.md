# Hand-drawn companions to the post 12 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 6 spends the four gradients on one step and section 6.1
works out what happens when the step is repeated, and neither had a figure.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-single-neuron-backprop` | 960 × 540 | alternate | Forward and backward in the same five columns, with the running gradient at every stage. |
| `02-one-step-and-after` | 960 × 470 | published | Sections 6 and 6.1: the step the gradients buy, and the collapse it starts. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:12
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/12-backprop-through-a-single-neuron/diagrams/excalidraw/01-single-neuron-backprop.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:12` will fight over the same file, so fold anything worth keeping
back into `post12.js`.

## Notes on these

- **`01-single-neuron-backprop`.** The running gradient is written at every
  stage, not only at the two ends. The claim of the post is that one number is
  carried leftwards and reused across all four parameters, and a figure that
  showed only the final four gradients would let a reader believe they were
  computed independently. The table beneath makes the same point in columns: the
  upstream row is constant and only the factor row varies.
- **`02-one-step-and-after`.** The single step sits beside the trajectory it
  starts, because the section is really making two claims — that the gradients
  are correct, and that repeating the step converges geometrically. The curve is
  36 · 0.49ᵗ, which is section 6.1's own derivation rather than a sketch of a
  decay; its first two points, 36 and 17.6, are the numbers the post prints.
