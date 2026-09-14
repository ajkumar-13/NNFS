# Hand-drawn companions to the post 09 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 1 counts the twenty-one parameters and section 4 argues
that guessing them cannot scale, and neither had a figure.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-strategies-compared` | 960 × 540 | alternate | Three algorithms on one valley, each above its own loss curve. |
| `02-parameter-count` | 960 × 470 | published | Sections 1 and 4: every parameter drawn, and the scale that makes luck useless. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:09
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/09-introduction-to-optimisation/diagrams/excalidraw/01-strategies-compared.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:09` will fight over the same file, so fold anything worth keeping
back into `post09.js`.

## Notes on these

- **`01-strategies-compared`.** All three panels share one valley cross-section,
  drawn from the same function, so the sample paths can be compared directly.
  The perturbation walk has to stop visibly short of the floor and jitter in
  place: a first pass let it descend nearly as far as gradient descent, and the
  two panels then said the same thing. The random-selection dots stay high on
  the walls for the same reason, since the claim is that random draws almost
  never land in the valley at all.
- **`02-parameter-count`.** Every one of the twenty-one parameters is drawn as
  an individual cell. The count is small enough to show in full, and showing it
  in full is what makes ten million read as a change of kind rather than as a
  bigger number in a table.
