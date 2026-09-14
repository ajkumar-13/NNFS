# Hand-drawn companions to the post 03 diagrams

Three scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

Two of the three are **alternates** to the clean vector figures one directory up. The
third is a **published figure**: section 4 substitutes `Z₁` away and lands on a single
equivalent layer, which is the claim the whole post turns on and the reason Part 06
exists, and the post had no figure for it.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-multi-layer-anatomy` | 960 × 540 | alternate | One sample through two layers, with both layer cards drawn identically. |
| `02-dimension-flow` | 960 × 420 | alternate | A batch as a chain of shapes, and the two places an inner dimension has to agree. |
| `03-linear-collapse` | 960 × 460 | published | Section 4: the two-layer chain and the one-layer chain that computes the same function. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:03
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/03-stacking-layers-and-the-forward-pass/diagrams/excalidraw/03-linear-collapse.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:03` will fight over the same file, so fold anything worth keeping
back into `post03.js`.

## Notes on these

- **`01-multi-layer-anatomy`.** The two layer cards are drawn identically, down
  to the three neuron circles and the position of every label. That sameness is
  the argument: the second layer is not a new kind of object, it is the first
  layer again with the previous output standing in for the input. Only the
  indices in the formulas differ.
- **`02-dimension-flow`.** The inner-dimension checks are centred under the pair
  of chips they compare, not under the gap between them, so it is unambiguous
  which two shapes are being held against each other. Shape continuity is the
  only thing stacking actually adds, so it is the only thing this figure tracks;
  the bias broadcasts are noted but not drawn.
- **`03-linear-collapse`.** Both chains are drawn at the same chip size on the
  same canvas. Rendering the collapsed form smaller would suggest it is a
  simplification or an approximation, when the point is that the two compute the
  same function exactly. The arrow drops from `Z₁`, which is the term the
  substitution removes.
