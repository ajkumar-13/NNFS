# Hand-drawn companions to the post 16 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 4.1 warns that dropping `.copy()` silently mutates the
caller's array, which is a fault in memory rather than in arithmetic, and nothing in the
series draws what aliasing actually looks like.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-dense-backward-class` | 960 × 500 | alternate | The two methods at equal size, with the cache line and the line that reads it both marked. |
| `02-copy-not-alias` | 960 × 470 | published | Section 4.1: one array with two names, against two arrays with two names. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:16
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/16-coding-backpropagation/diagrams/excalidraw/02-copy-not-alias.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:16` will fight over the same file, so fold anything worth keeping
back into `post16.js`.

## Notes on these

- **`01-dense-backward-class`.** The two methods are drawn at the same size in
  the same card. `forward` looks trivial next to `backward` and it is tempting
  to shrink it, but the one line it gained over Part 04 is the line `backward`
  depends on, and the spine marks it in both places. Splitting them into two
  figures would lose that pairing entirely.
- **`02-copy-not-alias`.** The names and the arrays are drawn as separate things,
  with leaders between them, because the fault is in what the names point at
  rather than in any number. Both panels run the identical masking operation and
  produce the identical `dinputs`; the only difference the figure has to carry
  is how many boxes there were to begin with.
