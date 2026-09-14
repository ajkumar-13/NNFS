# Hand-drawn companions to the post 05 diagrams

Four scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

Three of the four are **alternates** to the clean vector figures one directory up. The
fourth is a **published figure**: sections 2.5 and 6 both say that `(3,)`, `(1, 3)` and
`(3, 1)` print alike and broadcast differently, which is the root of every bug in the
post, and no figure showed the three side by side.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-axis-summation` | 960 × 460 | alternate | The same nine numbers reduced three ways, with the collapsing slices outlined on the source. |
| `02-keepdims-matters` | 960 × 500 | alternate | One keyword between a right answer and a wrong one, with both results shown in full. |
| `03-broadcasting-rules` | 960 × 520 | alternate | Four cases, with a verdict under every aligned pair of axes. |
| `04-three-shapes` | 960 × 440 | published | Sections 2.5 and 6: which of the three stretches sideways, and which two do not. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:05
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/05-array-summation-keepdims-and-broadcasting/diagrams/excalidraw/04-three-shapes.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:05` will fight over the same file, so fold anything worth keeping
back into `post05.js`.

## Notes on these

- **`01-axis-summation`.** The slices being collapsed are outlined on the source
  grid rather than named in a caption, so "the axis you name is the axis that
  disappears" becomes something the eye can check. Both reductions are drawn as
  horizontal strips because both really do return shape `(3,)`; drawing the
  `axis=1` result as a column would quietly assume the `keepdims` the next figure
  is about.
- **`02-keepdims-matters`.** The wrong result is written out in full rather than
  labelled wrong. Its numbers are plausible and its shape is correct, and that is
  precisely why the bug survives to training time; a figure that only stamped it
  as an error would lose the thing worth showing.
- **`03-broadcasting-rules`.** Each aligned pair of axes gets its own verdict —
  equal, stretch, or a mismatch — because the rule is applied per axis. Judging
  only the whole pair would hide *where* the failing case fails, which in the
  fourth card is the leading axis, not the trailing one. The padded axis is drawn
  as a dashed box so the rule that creates it is visible.
- **`04-three-shapes`.** The stretch direction is drawn on a target grid with the
  replicated slice shaded, so the two row cases look alike and the column case
  looks different at a glance. The middle card is deliberately neutral grey: its
  whole point is that it behaves identically to the first despite having a
  different shape written on it.
