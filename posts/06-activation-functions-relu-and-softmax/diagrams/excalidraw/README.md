# Hand-drawn companions to the post 06 diagrams

Four scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

Three of the four are **alternates** to the clean vector figures one directory up. The
fourth is a **published figure**: section 2 rests its whole argument on "that kink is
everything" and never draws the kink, and section 2.2's four boundaries had nowhere
to sit.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-why-nonlinearity` | 960 × 480 | alternate | Two layers drawn twice on one axis frame, differing only by the call between them. |
| `02-softmax-stability` | 960 × 480 | alternate | The same four steps down both columns, and the one row where they diverge. |
| `03-forward-pass-pipeline` | 960 × 420 | alternate | Four objects in a row, and which of them change the shape. |
| `04-relu-anatomy` | 960 × 460 | published | Section 2: the kink itself, with section 2.2's four boundaries beside it. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:06
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/06-activation-functions-relu-and-softmax/diagrams/excalidraw/04-relu-anatomy.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:06` will fight over the same file, so fold anything worth keeping
back into `post06.js`.

## Notes on these

- **`01-why-nonlinearity`.** Both panels draw the same axis frame at the same
  origin and the same extent, so the only difference the eye can find between
  them is the shape of the curve. The vertices on the right are marked
  individually because each one is a hidden unit switching on or off, and the
  count is the argument.
- **`02-softmax-stability`.** Both panels run four steps at the same four
  vertical positions, including a "shifted" row the naive panel does not use.
  Drawing that row greyed out rather than omitting it means the row where the
  two diverge is found by scanning straight down instead of by reading captions.
- **`03-forward-pass-pipeline`.** The shape badge under every stage stays at
  `(N, 3)` from the first dense layer onward, and the annotations name which
  stages change the shape and which change only the values. That distinction is
  the thing a reader most often has backwards about activations.
- **`04-relu-anatomy`.** The "negative in → 0 out" label sits above the flat arm
  rather than below it, because the kink's leader arrives from underneath and
  the two collided in a first pass. The four boundary cards share the canvas
  with the plot rather than living in their own figure, since each of them is a
  property of the shape drawn beside it.
