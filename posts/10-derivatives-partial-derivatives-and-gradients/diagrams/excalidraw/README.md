# Hand-drawn companions to the post 10 diagrams

Three scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

Two of the three are **alternates** to the clean vector figures one directory up. The
third is a **published figure**: section 2.3 closes by turning the three readings of a
slope into three update decisions, and that step had no figure.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-derivative-as-slope` | 960 × 500 | alternate | Three tangents on one curve, so the slopes can be compared. |
| `02-partial-to-gradient` | 960 × 460 | alternate | Three branches differing only in which variable is left free. |
| `03-slope-to-update` | 960 × 460 | published | Section 2.3: sign decides the direction, size decides the distance. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:10
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/10-derivatives-partial-derivatives-and-gradients/diagrams/excalidraw/03-slope-to-update.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:10` will fight over the same file, so fold anything worth keeping
back into `post10.js`.

## Notes on these

- **`01-derivative-as-slope`.** All three tangents sit on one parabola. On three
  separate axes they would be three unrelated straight lines, and the thing
  worth seeing is that the same curve produces a slope of minus three, one and
  four depending only on where you stand.
- **`02-partial-to-gradient`.** The three branch cards are drawn identically
  apart from which variable is named. That sameness is the argument: a partial
  derivative is an ordinary derivative with the other variables held still, not
  a different operation.
- **`03-slope-to-update`.** The curve is lifted ten pixels clear of the baseline.
  With the minimum sitting exactly on the axis, the zero-slope tangent in the
  middle card merged into it and stopped reading as a tangent at all. All three
  cards run the same arithmetic with the same learning rate, so the only thing
  that varies across them is the number the derivative supplied.
