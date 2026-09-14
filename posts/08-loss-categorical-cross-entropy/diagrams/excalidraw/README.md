# Hand-drawn companions to the post 08 diagrams

Three scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

Two of the three are **alternates** to the clean vector figures one directory up. The
third is a **published figure**: section 2.2 claims two networks can share an accuracy
without sharing a loss, and section 9 returns to it, but nothing showed the two numbers
disagreeing.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-cross-entropy-curve` | 960 × 500 | alternate | The shape of −log(p), and why its steep end is the training signal. |
| `02-indexing-methods` | 960 × 460 | alternate | Two label formats reaching the same three numbers. |
| `03-loss-vs-accuracy` | 960 × 470 | published | Sections 2.2 and 9: one accuracy, two losses, nine times apart. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:08
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/08-loss-categorical-cross-entropy/diagrams/excalidraw/01-cross-entropy-curve.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:08` will fight over the same file, so fold anything worth keeping
back into `post08.js`.

## Notes on these

- **`01-cross-entropy-curve`.** The curve is sampled at sixty points rather than
  drawn as a handful of segments. The entire argument of the section is the
  steepness below p = 0.1, and a coarse polyline cuts that stretch off with a
  straight chord — flattening exactly the part worth seeing.
- **`02-indexing-methods`.** Both panels print `[0.7, 0.5, 0.9]` at the foot in
  the same position and the same size. The repetition is the argument: the label
  format changes the ergonomics of the lookup and nothing about the arithmetic.
- **`03-loss-vs-accuracy`.** Both batches are correct on all three samples, so
  accuracy is pinned at 3/3 in both panels and cannot be the thing separating
  them. Choosing a batch with an error in it would have let a reader attribute
  the difference to the mistake rather than to the confidence, which is the
  opposite of what the section says.
