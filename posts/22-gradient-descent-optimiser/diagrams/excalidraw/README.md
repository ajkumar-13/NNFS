# Hand-drawn companions to the post 22 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 4 reports what ten thousand epochs of plain SGD actually
achieve and draws the conclusion the next five posts exist for.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-sgd-update-and-lr` | 960 × 540 | alternate | One rule, one knob, three settings on a shared vertical scale. |
| `02-sgd-is-slow` | 960 × 470 | published | Section 4: the curve is still falling when the budget runs out. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:22
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/22-gradient-descent-optimiser/diagrams/excalidraw/02-sgd-is-slow.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:22` will fight over the same file, so fold anything worth keeping
back into `post22.js`.

## Notes on these

- **`01-sgd-update-and-lr`.** The three learning-rate panels share one vertical
  scale, and the figure says so in a caption. Given its own scale, the diverging
  run would be drawn as a tidy wave the same height as the working descent, and
  the panel would understate the failure it exists to show.
- **`02-sgd-is-slow`.** The curve is drawn so that it is visibly still falling at
  the right-hand edge, and the axis stops at the budget rather than at
  convergence. The post's claim is not that SGD fails but that it has not
  finished, and a chart that ran to a flat tail would say the opposite. The
  numbers are the post's own measured runs, and the bar panel puts Adam's ten
  thousand epochs against SGD's fifty thousand so the gap is a length rather
  than a sentence.
