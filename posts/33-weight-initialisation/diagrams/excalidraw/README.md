# Hand-drawn companions to the post 33 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 4's factor of 2 comes from one fact about ReLU that a
variance-by-depth chart has no way to show.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-activation-variance-by-depth` | 960 × 540 | alternate | Ten layers, three schemes, and the recursion that explains all of them. |
| `02-the-factor-of-two` | 960 × 470 | published | Section 4: half a distribution mapped onto one point, and the 2 that undoes it. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:33
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/33-weight-initialisation/diagrams/excalidraw/02-the-factor-of-two.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:33` will fight over the same file, so fold anything worth keeping
back into `post33.js`.

## Notes on these

- **`01-activation-variance-by-depth`.** The failing curve is allowed to run off
  the bottom of the chart rather than being squeezed onto it. `0.01 · randn` at
  64 inputs reaches 10⁻²² by layer ten, and rescaling the axis to fit that would
  flatten the two working schemes into a single line at the top and destroy the
  comparison. Running off the edge, with a label saying so, is the honest
  drawing.
- **`02-the-factor-of-two`.** The argument is about what happens to a
  *distribution*, so the figure draws a distribution: a bell with its negative
  half shaded and an arrow sweeping that half onto a single spike at zero.
  Neither the depth chart nor the two formulas can show why the constant is 2
  rather than any other number, and "half the mass lands on one point" is the
  only thing that does.
