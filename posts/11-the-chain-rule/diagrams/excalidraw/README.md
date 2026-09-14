# Hand-drawn companions to the post 11 diagrams

Three scenes, each in two forms:

| File | What it is |
|---|---|
| `NN-name.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `NN-name.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

Two of the three are **alternates** to the clean vector figures one directory up. The
third is a **published figure**: sections 4 and 7 give a mechanical procedure and work
it on a polynomial, and both existing figures draw the shape of the rule rather than
the act of applying it.

| Scene | Canvas | Status | What it argues |
|---|---|---|---|
| `01-chain-rule` | 960 × 460 | alternate | Two functions in a row, with each local slope under the box that owns it. |
| `02-chain-in-a-network` | 960 × 480 | alternate | Four factors, each tied to the class that computes it. |
| `03-three-step-pattern` | 960 × 470 | published | Sections 4 and 7: the procedure, worked on 3(2x²)⁵. |

A *published* figure is the one the post embeds: its `.svg` was promoted into `diagrams/`
one directory up. An *alternate* sits beside a clean vector figure that already exists,
for slides, talks, and anywhere a sketch reads better than a diagram.

## Regenerating

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:11
```

Every element carries a seed derived from its index, so a rebuild is
byte-identical. Nothing here depends on a random number, which is what keeps a
regeneration from showing up as a diff.

## Editing one by hand

Open the `.excalidraw`, move things, save over it, then re-render:

```bash
cd assets/diagrams/excalidraw-generator
node render-scene.js ../../../posts/11-the-chain-rule/diagrams/excalidraw/02-chain-in-a-network.excalidraw
```

Use Excalidraw's own "Export to SVG" instead and you lose the thing that makes
these files work in dark mode: it bakes literal hex into every shape, where this
renderer emits `var(--ce-ink)` and friends. A hand edit and a later
`npm run build:11` will fight over the same file, so fold anything worth keeping
back into `post11.js`.

## Notes on these

- **`01-chain-rule`.** The backward row sits in the same columns as the forward
  row, so each local derivative is directly beneath the box that owns it. That
  vertical alignment is the argument: a factor of the product belongs to exactly
  one function, not to the chain as a whole.
- **`02-chain-in-a-network`.** The four factor cards are left in forward order
  and joined to their functions by dotted droppers, rather than being re-ordered
  right-to-left to match the way the product is written. Keeping the columns is
  what makes "reading the chain rule is reading the architecture backwards"
  visible; the reversal is carried by the arrow instead.
- **`03-three-step-pattern`.** The polynomial is worked in full rather than
  gestured at, because the step a reader has to reproduce is the one the other
  two figures skip. The band underneath maps the same three steps onto the
  network and names the middle one as the only real work, which is what the
  remaining nine posts are.
