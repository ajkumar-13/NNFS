# Hand-drawn companions to the post 04 diagrams

Four scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

Three of the four are **alternates** to the clean vector figures one directory up. The
fourth is a **published figure**: section 2.1 defends the choice of spiral data over
MNIST on the grounds of scale, and that argument had no figure.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-spiral-data` | 960 × 540 | alternate | Three hundred points, three wound classes, and one straight line that fails on all of them. |
| `02-weight-convention` | 960 × 460 | alternate | The same weights laid out two ways, and which axis a neuron runs along. |
| `03-dense-layer-class` | 960 × 540 | alternate | One blueprint above, two instances below, sharing the class and nothing else. |
| `04-why-spirals-not-mnist` | 960 × 440 | published | Section 2.1: two features against seven hundred and eighty-four. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:04
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/04-dense-layer-class-and-spiral-data/diagrams/excalidraw/01-spiral-data.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:04` will fight over the same file, so fold anything worth keeping
back into `post04.js`.

## Notes on these

- **`01-spiral-data`.** `spiralPoints()` in `post04.js` is `nnfs.datasets.spiral_data`
  reimplemented, so the sketch plots the curve the reader's own
  `spiral_data(samples=100, classes=3)` call produces rather than a decorative
  swirl. The one part that cannot be copied is the noise: nnfs draws it from a
  seeded normal, and a random draw here would put a diff in every rebuild, so a
  hashed sine stands in. Its half-width is 0.12 rather than nnfs's 0.2, because a
  uniform hash spends none of its mass near zero where a normal spends most of
  its own, and at the matching half-width the arms stop reading as arms.
- **`02-weight-convention`.** Each neuron is drawn as one outlined block rather
  than as a tint over a shared grid. The block *is* the neuron, and its
  orientation — lying flat across the columns, or standing up down the rows — is
  the entire difference between the two conventions. Both panels use the same
  three colours for the same three neurons.
- **`03-dense-layer-class`.** The two instance cards spell out their own arrays
  with their own shapes. What confuses first readers is not what the class does
  but that each instance owns a private copy of everything in it, so the copies
  are drawn rather than described.
- **`04-why-spirals-not-mnist`.** The MNIST panel draws a coarse grid rather than
  a real twenty-eight-square one: the argument is the count, not the pixels, and
  784 cells at this size is a grey rectangle. The spiral panel plots forty points
  per class rather than the two dozen a first pass used, because below about
  thirty the arms stop resolving and the panel ends up arguing for a blob.
