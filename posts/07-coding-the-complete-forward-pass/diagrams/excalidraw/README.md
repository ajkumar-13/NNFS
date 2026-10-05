# Hand-drawn companions to the post 07 diagrams

Three scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

Two of the three are **alternates** to the clean vector figures one directory up. The
third is a **published figure**: section 5 pulls two separate readings out of one printed
row, and neither had a figure.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-pipeline-anatomy` | 960 × 540 | alternate | The four objects above, the code that builds them below, in the same order. |
| `02-shape-audit` | 960 × 420 | alternate | Five shapes, with the batch axis tied across all of them. |
| `03-uniform-baseline` | 960 × 470 | published | Section 5: the first row the network prints, and the two different things it proves. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:07
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/07-coding-the-complete-forward-pass/diagrams/excalidraw/02-shape-audit.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:07` will fight over the same file, so fold anything worth keeping
back into `post07.js`.

## Notes on these

- **`01-pipeline-anatomy`.** The three code cards sit under the pipeline in the
  same left-to-right order as the objects they describe, so an object in the
  diagram and the line that creates it are found at roughly the same place on
  the page.
- **`02-shape-audit`.** Each shape is two boxes rather than one label, which is
  what lets the batch axis carry a different colour from the feature axis and be
  tied across all five shapes with a single dashed run. The `X` and
  `probabilities` labels sit *below* that tie: placed beside the boxes they
  crossed the dotted droppers, which a first pass did.
- **`03-uniform-baseline`.** Both panels share one bar scale. Rescaling each to
  its own maximum would make the uniform row look as decisive as the trained
  one, which is the exact confusion the section exists to prevent. The scale is
  130 rather than 150 because at 150 the 0.95 bar pushed its value label into
  the panel rule.
