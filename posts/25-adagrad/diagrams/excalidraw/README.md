# Hand-drawn companions to the post 25 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 1 sets up the problem AdaGrad exists to solve, and the
existing figure draws the mechanism rather than the problem.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-per-parameter-rates` | 960 × 540 | alternate | Two effective rates on log-log, and the flaw visible in the same picture. |
| `02-one-rate-two-params` | 960 × 470 | published | Section 1: both candidate learning rates, both failing. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:25
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/25-adagrad/diagrams/excalidraw/02-one-rate-two-params.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:25` will fight over the same file, so fold anything worth keeping
back into `post25.js`.

## Notes on these

- **`01-per-parameter-rates`.** Log-log axes are the right frame here because for
  a constant gradient magnitude the effective rate is exactly α/(|g|·√t), which
  is a straight line of slope −½. Both curves being straight and parallel is the
  finding: the ratio between the two rates never changes, and neither line ever
  flattens. Each is labelled at its own right-hand end — a legend block in the
  upper left sat directly on the upper curve.
- **`02-one-rate-two-params`.** Both candidate learning rates are worked, not
  just the one that fails. Section 1's argument is a proof by exhaustion — every
  α is either sized for the large gradient or the small one — and showing only
  one side would leave it as an assertion. The band closes with the arithmetic
  that makes AdaGrad's answer work: a gradient 100× larger accumulates a cache
  10 000× larger, whose root is 100×, so the two steps come out equal.
