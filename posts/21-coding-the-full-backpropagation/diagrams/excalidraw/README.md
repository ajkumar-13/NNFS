# Hand-drawn companions to the post 21 diagrams

Two scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

One is an **alternate** to the clean vector figure one directory up. The other is a
**published figure**: section 4 checks the loss against −log(1/3) and section 6 checks
every gradient's shape against its parameter, and neither check had a figure.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-forward-backward-script` | 960 × 540 | alternate | The script in three blocks, beside the twenty-one gradients it leaves behind. |
| `02-two-sanity-checks` | 960 × 460 | published | Sections 4 and 6: two prints that catch a broken backward pass on step zero. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:21
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/21-coding-the-full-backpropagation/diagrams/excalidraw/02-two-sanity-checks.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:21` will fight over the same file, so fold anything worth keeping
back into `post21.js`.

## Notes on these

- **`01-forward-backward-script`.** The three script blocks carry coloured
  spines in the same colours the pipeline figures use for forward and backward,
  so the blocks are identifiable before a line is read. The gradient panel ends
  on the number 21, which is deliberately the same figure Part 09 used to argue
  that random search was hopeless: the post is closing a loop opened twelve
  posts earlier.
- **`02-two-sanity-checks`.** Both checks are drawn as things that happen *before*
  training, which is the property that makes them worth a figure. A backward pass
  can be wrong in a way that still runs, still returns correctly-shaped arrays,
  and still lets the loss drift down — so a check that depends on the loss
  falling proves nothing. These two do not.
