# Hand-drawn companions to the post 15 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--nn-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 1 justifies the whole post in one sentence and section 6
collects the three gradients a dense layer emits, and the existing figure is about the
sum inside a single layer rather than the handoff between two.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-input-gradients` | 960 × 460 | alternate | The same layer twice, with one path lit and then all of them. |
| `02-gradient-handoff` | 960 × 470 | published | Sections 1 and 6: two gradients stay, one travels. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:15
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/15-gradients-with-respect-to-inputs/diagrams/excalidraw/02-gradient-handoff.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--nn-ink)` and friends. A hand edit and a later
`npm run build:15` will fight over the same file, so fold anything worth keeping
back into `post15.js`.

## Notes on these

- **`01-input-gradients`.** Both panels draw the identical three-neuron layer,
  including the routes that are not taken; the unlit paths stay on the canvas in
  border grey rather than being removed. Deleting them would make the two panels
  different diagrams, when the point is that they are the same diagram with a
  different number of live paths.
- **`02-gradient-handoff`.** The three outputs are sorted by destination rather
  than by the order the code computes them. Two are consumed by the optimiser
  and vanish; one leaves the layer entirely, and the arrow that carries it off
  the bottom-left of the card is the whole reason the post exists. Listing them
  in code order would have buried that distinction in the middle of the list.
