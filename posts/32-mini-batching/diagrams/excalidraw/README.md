# Hand-drawn companions to the post 32 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 2's table hides its own punchline — the per-epoch cost is
identical in all three columns — and a trajectory plot cannot say that.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-batch-size-trajectories` | 960 × 540 | alternate | One landscape, three paths, and what the noise is actually for. |
| `02-same-cost-more-steps` | 960 × 470 | published | Section 2: same work per epoch, one step against sixty thousand. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:32
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/32-mini-batching/diagrams/excalidraw/02-same-cost-more-steps.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:32` will fight over the same file, so fold anything worth keeping
back into `post32.js`.

## Notes on these

- **`01-batch-size-trajectories`.** The three paths share a start, a landscape
  and a destination; only the per-step wobble differs, which is the only thing
  batch size changes about a trajectory. The wobble comes from a hashed sine
  rather than a random draw, so the jagged path is jagged the same way on every
  rebuild.
- **`02-same-cost-more-steps`.** All three data bars are the identical width,
  and each card repeats "60 000 samples, every card" under it. That repetition
  is the argument: the section's table lists cost-per-step and gradient quality
  and buries the fact that *cost per epoch does not change at all*. With the
  bars drawn to different widths the figure would say the opposite of what the
  post says.
