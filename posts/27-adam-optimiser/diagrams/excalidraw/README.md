# Hand-drawn companions to the post 27 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: bias correction is the only part of Adam that neither parent
supplies, and the existing figure draws the correction boxes without saying what they do.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-adam-pipeline` | 960 × 540 | alternate | Two lanes at equal weight, each credited to the post it came from. |
| `02-bias-correction` | 960 × 470 | published | Section 2.1: why both moments start too small, and why the fix removes itself. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:27
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/27-adam-optimiser/diagrams/excalidraw/02-bias-correction.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:27` will fight over the same file, so fold anything worth keeping
back into `post27.js`.

## Notes on these

- **`01-adam-pipeline`.** The two lanes are drawn at identical size and each
  carries a label naming the post it came from. Adam is a composition, not an
  invention, and those two labels are what make the figure say so; without them
  it is just a diagram of a formula. The lane colours are the ones Parts 24 and
  26 used for momentum and RMSProp respectively.
- **`02-bias-correction`.** The amplifier curve does two jobs at once, which is
  why it is worth plotting rather than tabulating: it shows that the correction
  is *large* at the start — 10× for β₁, 1000× for β₂ — and that it *removes
  itself*, reaching 1 without any schedule. Log axes are what let both curves,
  three orders of magnitude apart in starting value and two in duration, sit on
  one chart.
