# Hand-drawn companions to the post 24 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 2 explains *why* momentum works by decomposing two
consecutive steps, and the existing figure shows only the trajectory that results.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-momentum-trajectory` | 960 × 540 | alternate | One valley, one starting point, two paths. |
| `02-vector-cancellation` | 960 × 470 | published | Section 2: the components that cancel and the ones that add. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:24
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/24-momentum/diagrams/excalidraw/02-vector-cancellation.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:24` will fight over the same file, so fold anything worth keeping
back into `post24.js`.

## Notes on these

- **`01-momentum-trajectory`.** Both panels draw the same contours from the same
  function and start from the same point, so the path is the only variable. The
  minimum marker is drawn *after* the path and sits on a small opaque patch: in
  the left panel the zig-zag crosses that spot a dozen times and buried it
  entirely on the first pass.
- **`02-vector-cancellation`.** The two steps are drawn with their components as
  dashed legs, and the numbers on the right are the arithmetic those legs
  perform. Section 2's claim is a statement about vector addition, so the figure
  has to show the addition rather than its consequence — the consequence is
  already what figure 01 is for. The β strip reads as three horizontal groups
  rather than a table, because the card is one row tall and column headers over
  side-by-side groups line up with nothing.
