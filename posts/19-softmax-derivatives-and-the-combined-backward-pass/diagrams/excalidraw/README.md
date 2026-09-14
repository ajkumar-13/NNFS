# Hand-drawn companions to the post 19 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 2.2 gives two decisive reasons every framework ships the
fused operation, and the cost argument only lands once the numbers are written down.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-combined-shortcut` | 960 × 500 | alternate | Two routes to one gradient, with the cost of each route made visible. |
| `02-why-fused` | 960 × 460 | published | Section 2.2: quadratic against linear, at the class counts of real tasks. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:19
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/19-softmax-derivatives-and-the-combined-backward-pass/diagrams/excalidraw/02-why-fused.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:19` will fight over the same file, so fold anything worth keeping
back into `post19.js`.

## Notes on these

- **`01-combined-shortcut`.** Both routes reach the same gradient, so the figure
  has to make the *cost of getting there* the visible difference. The left panel
  draws the dense Jacobian at full size and the right panel draws nothing at
  all — five lines of algebra and no matrix — which is the honest picture of
  what each option allocates.
- **`02-why-fused`.** The class counts are written out in full rather than given
  as orders of magnitude. "Quadratic against linear" reads as a footnote until
  2 500 000 000 is sitting next to 50 000 in the same row; the ratio column then
  says the same thing a third way. The three rows are named after real tasks so
  the numbers are not abstract.
