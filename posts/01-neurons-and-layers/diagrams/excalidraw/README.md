# Hand-drawn companions to the post 01 diagrams

Six scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

Five of the six are **alternates** to the clean vector figures one directory up. The
sixth is a **published figure**: section 11 makes the structural claim of the whole
series and the post had no figure for it, so it was drawn here and promoted into
`diagrams/`.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-neuron-anatomy` | 960 × 540 | alternate | Four inputs, four weights, one bias on its own axis, one number out. |
| `02-layer-as-stacked-neurons` | 960 × 500 | alternate | One neuron beside three, fed by the same four dots: 5 parameters against 15. |
| `03-three-implementations` | 960 × 520 | alternate | By hand, two loops, one NumPy call, with the identical result under all three. |
| `04-shape-diary` | 960 × 440 | alternate | Three settings, five columns, and the one row where the transpose appears. |
| `05-batch-broadcasting` | 960 × 500 | alternate | The batched pass as a shape equation, then the same pass written out in numbers. |
| `06-what-gets-added` | 960 × 460 | published | Section 11: the Part 01 core, and the three things the remaining posts add to it. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:01
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/01-neurons-and-layers/diagrams/excalidraw/01-neuron-anatomy.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:01` will fight over the same file, so fold anything worth keeping
back into `post01.js`.

## Notes on these

- **`01-neuron-anatomy`.** The bias arrives from above on a dashed line rather
  than from the left with the inputs. It is the one term with no input to
  multiply, and a diagram that lets it enter alongside the others loses exactly
  that. The four annotation cards are the rows of the component table in section
  3, in the same order, so the figure and the table can be read against each
  other.
- **`02-layer-as-stacked-neurons`.** Each neuron's fan of four connections is
  drawn in its own colour. That is what turns "each neuron carries its own
  weights" from a caption into something visible: three colours leaving the same
  four input dots. The nodes are radius 24 on a 53px pitch rather than 26 on 59,
  because at the wider setting the bottom node sat on the parameter count
  beneath it.
- **`03-three-implementations`.** The identical result `[4.8, 1.21, 2.385]` is
  repeated at the foot of all three columns. The repetition is the argument: the
  arithmetic is invariant and only the representation changes, so the columns
  have to agree digit for digit. The empty space under the NumPy column is left
  in rather than closed up, because the emptiness is the point.
- **`04-shape-diary`.** The batch row is marked with a spine rather than by
  recolouring its text. Accent ink on an accent wash measures under 3:1 in both
  themes, and the two cells worth emphasising — the transposed call and its
  output shape — are the two that most need to stay readable.
- **`05-batch-broadcasting`.** The grids carry no numbers. The top half is about
  shapes flowing and the band below is where the values are checked, so neither
  half has to do both jobs. Every number in the band is the post's own printed
  output, with the raw column back-computed by subtracting the bias, so it can
  be checked against the code block rather than taken on trust.
- **`06-what-gets-added`.** The core sits at the same visual weight as the three
  additions beside it, because the claim of the section is that the core is not
  the small part. It is the only scene here with no clean vector counterpart.
