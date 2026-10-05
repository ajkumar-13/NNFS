# Hand-drawn companions to the post 28 diagrams

Three scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

Two of the three are **alternates** to the clean vector figures one directory up. The
third is a **published figure**: section 4 classifies a model into four regimes from its
train–test gap, and both existing figures only ever show one of those four.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-train-vs-test` | 960 × 580 | alternate | One dataset, two boundaries, and three points that decide everything. |
| `02-loss-divergence` | 960 × 540 | alternate | Two curves, three phases, and one stopping point that belongs to only one of them. |
| `03-four-regimes` | 960 × 470 | published | Section 4: four diagnoses from the same two numbers. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:28
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/28-generalization-and-testing/diagrams/excalidraw/03-four-regimes.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:28` will fight over the same file, so fold anything worth keeping
back into `post28.js`.

## Notes on these

- **`01-train-vs-test`.** Both panels carry the identical twenty-three points and
  only the boundary moves, so "the overfit model bent to reach these three" is
  something the eye can verify rather than a claim in a caption. Which points a
  boundary gets wrong is computed from the boundary function itself, so the
  rings cannot drift out of agreement with the curve. The shaded regions are
  drawn as vertical strips because the renderer has no filled free-form polygon;
  at this scale a 12px strip is indistinguishable from a true region fill.
- **`02-loss-divergence`.** The stopping marker sits *on* the validation curve,
  not between the two curves. The section's claim is that the training curve
  contains no information about when to stop, so the marker has to visibly
  belong to the other one.
- **`03-four-regimes`.** All four cards share one bar scale, which is what makes
  the gaps comparable across them — underfitting's two-point gap and
  distribution shift's fifty-three-point gap have to be measured against the
  same ruler for the figure to mean anything. The band names the one distinction
  the numbers cannot make: overfitting and distribution shift differ only in
  where the test set came from.
