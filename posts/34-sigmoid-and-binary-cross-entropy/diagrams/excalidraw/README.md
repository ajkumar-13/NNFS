# Hand-drawn companions to the post 34 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 6 collapses everything from Part 06 onward into one choice
about the last layer, and one of its four cases is a genuine trap.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-sigmoid-bce-pipeline` | 960 × 540 | alternate | The binary head, the cancellation, and the overflow it has to avoid. |
| `02-choosing-the-head` | 960 × 470 | published | Section 6: four tasks, four heads, and the one that looks like another. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:34
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/34-sigmoid-and-binary-cross-entropy/diagrams/excalidraw/02-choosing-the-head.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:34` will fight over the same file, so fold anything worth keeping
back into `post34.js`.

## Notes on these

- **`01-sigmoid-bce-pipeline`.** The cancellation is stated in words as well as
  drawn: BCE's backward divides by p(1 − p) and sigmoid's multiplies by it. That
  is the whole reason the binary head earns its own post, and a pipeline of
  boxes alone would leave `(p − y)/N` looking like a definition rather than a
  result.
- **`02-choosing-the-head`.** The four cards are laid out so multi-label sits
  next to multi-class, because that adjacency is where the mistake happens: they
  have the same number of output neurons and different activations. The
  multi-label card is the only one whose note starts with the word "trap", and
  the band spells out why — softmax makes the outputs sum to 1, which asserts
  exactly one positive label.
