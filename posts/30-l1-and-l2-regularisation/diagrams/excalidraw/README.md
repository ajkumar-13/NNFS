# Hand-drawn companions to the post 30 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 2.3's table runs from |w| = 0.01 to 100, and a plot over
w ∈ [−2, 2] cannot show what happens across that range.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-l1-vs-l2-penalty` | 960 × 540 | alternate | Two penalty shapes, with the arrows carrying the difference. |
| `02-gradient-pressure` | 960 × 470 | published | Section 2.3: a flat line and a line of slope 1, and where they cross. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:30
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/30-l1-and-l2-regularisation/diagrams/excalidraw/02-gradient-pressure.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:30` will fight over the same file, so fold anything worth keeping
back into `post30.js`.

## Notes on these

- **`01-l1-vs-l2-penalty`.** The arrows are the subject, not the curves. They sit
  at the same three weight values in both panels so their lengths can be compared
  directly: equal on the left, growing on the right. Anyone can draw a V beside a
  parabola; what the figure has to say is what the *gradient* does at each point.
- **`02-gradient-pressure`.** Log-log across five decades, because that is the
  range section 2.3's table actually covers and the range on which "constant"
  and "proportional" separate. The crossing at |w| = 0.5 is computed from
  λ = 2λw rather than eyeballed, and the card beside it says explicitly that the
  crossing moves with λ — the post is careful that this is not a universal
  threshold, and a marked point on a chart invites exactly that misreading.
