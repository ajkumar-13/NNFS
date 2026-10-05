# Hand-drawn companions to the post 29 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 5 names data leakage as the single most common source of
inflated benchmarks, and it had no figure.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-three-way-split-and-kfold` | 960 × 540 | alternate | A fixed slice beside a rotating one, in the same colours. |
| `02-data-leakage` | 960 × 490 | published | Section 5: the same three operations, in two orders. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:29
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/29-validation-and-hyperparameter-tuning/diagrams/excalidraw/02-data-leakage.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:29` will fight over the same file, so fold anything worth keeping
back into `post29.js`.

## Notes on these

- **`01-three-way-split-and-kfold`.** The validation slot is the same tan in both
  panels, so the rotating slot on the right is recognisably the same object as
  the fixed slice on the left. The test slice appears only on the left, which is
  correct and worth noticing: k-fold rotates the *validation* slot and never
  touches the test set at all.
- **`02-data-leakage`.** Both pipelines hold the same three boxes; only their
  order differs. Leakage is an ordering mistake rather than a conceptual one, so
  a figure that showed two *different* pipelines would misdescribe it — what
  makes the second wrong is that a step everyone agrees is necessary happens one
  position too early.
