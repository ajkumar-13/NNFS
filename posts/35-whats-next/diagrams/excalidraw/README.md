# Hand-drawn companions to the post 35 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: the map says what is out there, and section 1 makes a second claim
the map cannot — that none of it requires starting again.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-whats-next-map` | 960 × 540 | alternate | Four directions out of a working MLP, sorted by what each adds. |
| `02-the-skeleton-is-fixed` | 960 × 470 | published | Section 1: everything on that map plugs into the loop you already wrote. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:35
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/35-whats-next/diagrams/excalidraw/02-the-skeleton-is-fixed.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:35` will fight over the same file, so fold anything worth keeping
back into `post35.js`.

## Notes on these

- **`01-whats-next-map`.** All four columns are drawn to the same height, and the
  band under them says so out loud, because the obvious reading of a "what's
  next" figure is that the first column was the warm-up. The first column is a
  flat inventory; the other three are glossed pairs, which keeps a sub-label
  from being mistaken for a caption on the entry below it.
- **`02-the-skeleton-is-fixed`.** The loop sits in the middle and the two
  extension points face inwards, because the claim is not "here are three
  things" but "these two plug into that one". The three optimiser hook lines are
  tinted separately from the rest of the loop so the right-hand card has
  something to point at. Both side cards close on the same sentence shape — the
  loop calls those by name and asks nothing else — since the interface being
  *small* is what makes it survive a transformer.
