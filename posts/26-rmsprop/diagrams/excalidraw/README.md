# Hand-drawn companions to the post 26 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 2 explains the fix as a memory horizon, and the existing
figure shows the consequence — a cache that converges — rather than the mechanism.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-rmsprop-vs-adagrad-cache` | 960 × 540 | alternate | One cache grows without limit; the other has a fixed point, derived beside it. |
| `02-memory-horizon` | 960 × 470 | published | Section 2: how much a gradient still counts, and where each ρ's cliff falls. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:26
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/26-rmsprop/diagrams/excalidraw/02-memory-horizon.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:26` will fight over the same file, so fold anything worth keeping
back into `post26.js`.

## Notes on these

- **`01-rmsprop-vs-adagrad-cache`.** Log-log axes again, for the same reason as
  post 25: AdaGrad's cache is linear in `t` and RMSProp's is flat, so on linear
  axes the flat one is pinned to the floor and there is nothing to compare. The
  side card derives the fixed point rather than asserting the convergence — the
  claim is that `G = ρG + (1−ρ)ḡ²` *has* a solution, and one line of algebra is
  what makes that a fact instead of an observation about a curve.
- **`02-memory-horizon`.** Age is on a log axis so that all three horizons fit on
  one chart with their cliffs an even distance apart; on a linear axis ρ = 0.9
  would collapse into the y-axis and ρ = 0.999 would be a flat line. AdaGrad
  appears on the same chart as a line at 1.0 that never falls, which is the
  whole comparison: it is not a different kind of rule, it is this rule with an
  infinite horizon.
