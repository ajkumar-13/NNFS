# 0004. He initialisation by default, the small initialisation kept as a flag

- Status: accepted
- Date: 2026-10-05, recording a change made on 2026-06-10

## Context

As first written, `Layer_Dense` drew its weights as 0.01 times a standard normal, the initialisation of the series' early posts, and the default noise was 0.2. With that configuration the network scored 85 percent on the test set, 170 of 200, the figure that section 4 of `docs/EVALUATION.md` reproduces. On 2026-06-10 the initialisation was changed to He, $\sqrt{2/n_\text{in}}$ times a standard normal, and the default noise to 0.1, and the score became 200 of 200.

That history is exactly the lesson of `nn-033`, and it was visible only in the commit log. A reader who meets the project today sees a network that works and has no way to see the one that did not.

## Decision

`Layer_Dense` takes the `init` argument that `nn-033` gives it, with the values `he` (the default), `xavier` (also accepted as `glorot`), and `small`, and raises `ValueError` for anything else. `Model` and the `--init` flag of the training command offer `he`, `xavier`, and `small`. The weights file records which one was used.

He is the default, because every hidden activation is ReLU. The small initialisation stays available so that the failure is reproduced with the same command and measured in `docs/EVALUATION.md`, not merely described.

All three draw the same standard normal numbers for a given seed and differ only in the factor that multiplies them, so a comparison at one seed compares scales and nothing else.

## Consequences

- The documented run is unchanged: `he` multiplies the draw by the same factor the code used before the argument existed.
- The comparison is part of the evaluation: 200 of 200 against 174 of 200 at seed 0, and at least 197 against at most 181 over ten seeds.
- The noise level was changed in the same commit as the initialisation, so the history alone does not separate the two effects. The evaluation does: it reports both initialisations at both noise levels.
- `xavier` is offered because the post's class has it. It is measured in the seed sweep and behaves like `he` on this network.
