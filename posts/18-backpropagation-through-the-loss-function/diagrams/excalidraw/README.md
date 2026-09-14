# Hand-drawn companions to the post 18 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 5 flags that `dvalues` means something different in this
class than in every other one, which is a naming trap rather than an arithmetic one.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-cross-entropy-backward` | 960 × 480 | alternate | Four grids, with the entries the one-hot mask kills drawn faint throughout. |
| `02-where-backprop-starts` | 960 × 460 | published | Section 5: the loss is the only class with nothing to its right. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:18
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/18-backpropagation-through-the-loss-function/diagrams/excalidraw/01-cross-entropy-backward.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:18` will fight over the same file, so fold anything worth keeping
back into `post18.js`.

## Notes on these

- **`01-cross-entropy-backward`.** The one-hot mask is carried through all four
  grids as colour, not just applied at the end: entries the mask kills are drawn
  subtle in the predictions too. That makes "only one number per row survives"
  something the eye reads before any arithmetic, which is the shape of the whole
  gradient. The footer names the `0.0` entries in row three, because the post
  itself flags that they would make the division read `0/0` and explains that
  `forward` clips for exactly that reason.
- **`02-where-backprop-starts`.** The loss panel's second box is dashed and
  labelled *nothing*. Leaving the space empty would read as an unfinished
  drawing; drawing the absence is what makes "there is no layer to the right"
  the thing the panel is about, and it is the entire reason `dvalues` holds
  predictions here and gradients everywhere else.
