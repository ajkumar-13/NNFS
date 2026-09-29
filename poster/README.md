# One page of neural networks from scratch

One scene, in two forms:

| File | What it is |
|---|---|
| `one-page-of-neural-networks-from-scratch.excalidraw` | An Excalidraw scene. Open it at [excalidraw.com](https://excalidraw.com) or in the VS Code Excalidraw extension and edit it directly. |
| `one-page-of-neural-networks-from-scratch.svg` | The same scene rendered to a self-contained SVG with [roughjs](https://roughjs.com), themed with the repo's `--ce-*` tokens so it works in light and dark mode. |

Thirty-five posts and four projects on one canvas, in eight panels: the forward pass in
order with a shape against every stage, the four object kinds and the one contract they
share, the three gradients a Dense layer computes, the optimiser lineage from SGD to Adam
with measured results, the training loop in its two lanes, six ways training fails and
the first fix for each, what closes the train-test gap, and the numerical floor
underneath all of it. At 1200 by 1700 it sits three pixels off the A-series ratio, which
is to say it prints at A2 without a crop worth noticing.

It shares its grid with the Context Engineering series' poster — two 530-wide columns, a
full-width lane at panel five, another at panel eight — so the two hang next to each
other. The per-post companions in `posts/*/diagrams/excalidraw/` are the same hand at
figure scale; this is the sheet you pull a panel out of for a talk or a whiteboard
session.

Four things about it are worth knowing before you edit it.

- **Every number on it is measured, not quoted.** The optimiser figures come from
  `verify/RESULTS.md` and the project figures from `verify/projects_results.md`, both
  produced by running the series' own classes under `nnfs.init()` (seed 0). Where a
  post's prose and the measured value disagree — and in a few places they do — the
  measured value is the one printed here. Vanilla SGD is 64.7 %, not the 57.3 % some
  early drafts claimed.

- **The type is smaller than the per-post figures.** A poster holds four times as much
  text in the same column, and Excalifont runs wide, so the body sits at 12–13 where a
  figure would use 13–14, and table glosses drop to 11.

- **Text on a colored fill uses `--ce-on-accent`,** which is dark ink in both themes, not
  `--ce-on-fill`. In this palette dark ink wins on every fill: against `--ce-primary`, the
  lightest of the four, it measures 4.34:1 where on-fill measures 3.95, and against
  `--ce-accent` it is 6.62 against 2.59. The badge glyphs are set at 20, which counts as
  large text, so 4.34 clears the 3:1 bar with room left.

- **Panel two's badges are numerals, not initials.** The four object kinds are Layer,
  Activation, Loss and Optimiser. An initial badge would put `L` on two of the four,
  which is worse than carrying no letter at all.

## Regenerating

The generator lives at `assets/diagrams/excalidraw-generator/`, alongside the thirty-five
that draw the per-post companions:

```bash
cd assets/diagrams/excalidraw-generator
npm install
npm run build:poster
```

Every element carries a seed derived from its index, so a rebuild is byte-identical.
Nothing here depends on a random number, which is what keeps a regeneration from showing
up as a diff.

## Editing it by hand instead

The `.excalidraw` file is a real scene, so you can open and edit it. How it saves depends
on where you open it:

- **The VS Code Excalidraw extension** edits the file in place. Ctrl-S and the scene on
  disk is your edit. This is the one to use.
- **excalidraw.com** works on a copy in browser storage. Your change is not on disk until
  you use *File → Save to...* and overwrite the original.

Then re-render, from the generator directory:

```bash
cd assets/diagrams/excalidraw-generator
npm run render -- ../../../poster/one-page-of-neural-networks-from-scratch.excalidraw
```

It writes the SVG next to the scene, inheriting the canvas size, `<title>`, and `<desc>`
from the file already there, so an edit does not cost you the accessible description.

**Do not use Excalidraw's own Export to SVG.** It bakes literal hex into every shape, and
the whole point of the renderer here is that it emits `var(--ce-ink)` and friends so one
file serves light and dark mode. An exported SVG looks right in whichever mode you
exported from and wrong in the other.

Two more things worth knowing before you edit:

- **`npm run build:poster` overwrites hand edits.** The generator is the source of truth
  and it does not read the scene, it replaces it. Once you start editing by hand, either
  stop running the build script or fold the change back into `poster.js`. For anything
  structural, `poster.js` is the better place to make it.
- **Colours picked from Excalidraw's palette will not follow the theme.** Only the `--ce-*`
  values map back to CSS variables; anything else is baked in as a literal and will be
  wrong in one of the two modes. The renderer warns you and names the offending colours.

Coordinates in `poster.js` are text *baselines*, the way a typeset sheet's own source
reads. `tb()` converts a baseline into the top-left origin an Excalidraw text element
wants, so every number in the generator can be checked against a rendered line. The build
also runs the shared `fit()` checker, so an overrun is reported as "this line needs 237px
and has 228" rather than being left for someone to notice in a render.
