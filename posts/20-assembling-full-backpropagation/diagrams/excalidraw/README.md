# Hand-drawn companions to the post 20 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 1 opens by counting what ten posts of backpropagation
actually produced, and the existing figure is about the wiring rather than the inventory.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-full-backprop-pipeline` | 960 × 540 | alternate | Forward and backward in the same columns, with `dinputs` named on every link. |
| `02-the-toolkit` | 960 × 470 | published | Section 1: three classes, and only one of them holds anything to learn. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:20
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/20-assembling-full-backpropagation/diagrams/excalidraw/02-the-toolkit.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:20` will fight over the same file, so fold anything worth keeping
back into `post20.js`.

## Notes on these

- **`01-full-backprop-pipeline`.** `dinputs` is written on each link between the
  backward cards rather than labelling the row once. Section 5's claim is that
  one variable is the entire glue between components, and naming it three times
  in three places is what makes that concrete. The backward cards sit under the
  forward objects they belong to, joined by dotted droppers, so the reversal is
  carried by the arrows instead of by re-ordering the columns.
- **`02-the-toolkit`.** The three cards are sorted by whether they hold
  parameters, not by the order they run in. That ordering turns the inventory
  into an argument: the two parameterless classes route gradients and never
  learn, which is exactly why the optimiser two posts later only ever reaches
  into `Layer_Dense`. The 21-parameter count is the same one Part 09 used to
  argue that random search was hopeless.
