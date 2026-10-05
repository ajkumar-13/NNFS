# Hand-drawn companions to the post 23 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 8 introduces the three-hook contract every optimiser in
the series obeys, which is the reason none of the loop code changes again.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-decay-schedule-and-result` | 960 × 540 | alternate | Three schedules beside the four results they produce, including the one that is worse than no decay. |
| `02-three-hook-contract` | 960 × 480 | published | Section 8: six optimisers, and only one of the three columns ever moves. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:23
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/23-learning-rate-decay/diagrams/excalidraw/02-three-hook-contract.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:23` will fight over the same file, so fold anything worth keeping
back into `post23.js`.

## Notes on these

- **`01-decay-schedule-and-result`.** The schedule panel and the results panel
  sit side by side so a curve and the accuracy it produced are read together.
  The `d = 1e−2` run is included even though it is a failure, because the
  section's actual finding is that too much decay is *worse than none* — a
  figure showing only the working schedule would make decay look free.
- **`02-three-hook-contract`.** The rows are laid out so the `pre` and `post`
  columns visibly stop changing after the second one, with dashed rules
  separating them from the `update` column and a header that names it as the
  only column that moves. That is the whole claim: the four optimisers still to
  come are each a change to one cell, and the training loop is written once.
