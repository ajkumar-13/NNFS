# Hand-drawn companions to the post 31 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 2.2 gives dropout a second, statistical reading, and the
existing figure only ever draws one mask.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-dropout-train-vs-test` | 960 × 540 | alternate | Two layers, with the arithmetic that makes them interchangeable downstream. |
| `02-implicit-ensemble` | 960 × 470 | published | Section 2.2: four masks over one layer, and why that is an ensemble. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:31
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/31-dropout/diagrams/excalidraw/02-implicit-ensemble.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:31` will fight over the same file, so fold anything worth keeping
back into `post31.js`.

## Notes on these

- **`01-dropout-train-vs-test`.** The per-unit outputs are written on the figure
  and the two sums are stated, because inverted dropout is an arithmetic claim:
  8 × 0.8 × 1.25 = 8 is the reason the layer after this one does not need to know
  which mode it is in. A figure that only showed crossed-out units would leave
  the 1.25 unexplained.
- **`02-implicit-ensemble`.** Four masks over the same five units, not one. With
  a single mask on the page, dropout looks like a layer with holes in it; the
  word "subnetwork" only becomes concrete once several of them are visible at
  once and it is clear they are drawn from the same five circles.
- In both scenes a dropped unit keeps its outline and gains a cross rather than
  disappearing. It is switched off for one step, not removed from the
  architecture, and a vanishing unit would say the wrong thing.
