# NNFC Series — Audit Report

*Pre-fix audit of the Neural Networks from Scratch series. Generated 2026-06-09 from a 45-agent review (35 posts, 54 post SVGs, 8 supporting-doc groups, 2 cross-cutting passes). This report is findings + a prioritized fix plan; no source files were changed.*

Full machine-readable detail: `C:\Temp\audit_digest.md` (per-item, with line numbers). This document is the human summary.

---

## 1. Executive summary

The series is **pedagogically strong** (depth 4–5 / 5, clarity 4–5 / 5 on every post) and structurally sound: all 35 posts exist and link correctly, 241 cross-post "Part NN" references resolve with zero number/topic mismatches, all 35 hero paths resolve, the bibliography is complete in both directions, and the projects/notebook code runs end-to-end.

The defects cluster into four buckets:

| Bucket | Severity | Count | Headline |
|---|---|---|---|
| **Notation** | systemic | ~25 posts + all docs + all 54 SVGs | The canonical bold-matrix convention is used in **zero** forward-pass posts and **zero** SVGs. The two reference docs (notation guide vs glossary) contradict each other on 7/7 core symbols. |
| **Correctness** | critical | **9** | Fabricated/incorrect printed outputs and numbers that a reader who runs the code will not reproduce. |
| **Correctness** | major | **22** | Internal contradictions: stale part counts, conflicting accuracy figures, a wrong momentum formula, dashboard numbers that contradict the posts they cite. |
| **Polish** | minor | many | Em-dash overruns (post 35 = 39), over-long TL;DRs (most posts), a few `you`/`I` voice slips, SVG stroke-width drift. |

**The single most important decision before any fixing:** which notation convention is canonical. You chose bold-matrix; §2 shows the true churn that implies and a lower-cost alternative. Please confirm or switch — every notation fix depends on it.

---

## 2. Notation — your main concern (the central finding)

### 2.1 The two "source of truth" docs disagree with each other

`notation_guide.md` and `glossary.md` are both presented as canonical, but they conflict on **all seven** core forward-pass symbols, and the glossary invents an eighth that the guide never defines:

| Quantity | notation_guide.md | glossary.md |
|---|---|---|
| Input | $\mathbf{X}$ (bold) | $X$ (plain) |
| Weights | $\mathbf{W}$ (bold) | $W$ (plain) |
| Bias | $\mathbf{b}$ (bold) | $b$ (plain) |
| Pre-activation | $\mathbf{Z}$ (bold capital) | $z$ (plain lowercase) |
| Activation / prediction | $\mathbf{A}$ and $\hat{\mathbf{y}}$ (separate, bold) | one row `a or ŷ` (plain) |
| Labels | $\mathbf{y}$ (bold) | $y$ (plain) |
| Loss | `L or 𝓛` (hedged) | $L$ |
| Layer output | (uses Z/A) | **invents $F_n$**: $F_n = f(F_{n-1}\cdot W_n + b_n)$ |

The glossary also writes the Bias formula as $z = Xw + b$ — mixing a lowercase scalar $z,w$ with a capital matrix $X$ in one expression.

### 2.2 The canon exists only on paper

A repo-wide scan for `\mathbf`:

- **0 matches** across posts 01–10 (the *entire* forward-pass arc) and 0 in the optimizer posts 22–32, 34, 35.
- **0 matches** across all 54 SVGs (their math labels are plain `X`, `W`, `Y`, `F₁` — `font-weight:bold` is used only for titles).
- `\mathbf` appears **only** in the backprop cluster (posts 11, 14, 15, 16, 17, 19, 20, 33) — and even there it is mixed with non-bold forms.

So conformance today is roughly **8 partially-bold backprop posts vs ~25 fully-plain posts + the glossary + all 4 cheatsheets + all 54 SVGs.** The series was never actually standardized to its own stated convention.

### 2.3 Decision — LOCKED: bold-matrix (confirmed 2026-06-09, after seeing the churn)

The series standardizes on **bold-matrix**, with `notation_guide.md` as the single source of truth. The honest cost of each direction is below for the record; bold was reaffirmed knowingly. **Both options fully fix the "same quantity, two symbols in one blog" complaint** — that is a consistency fix, independent of letter style.

| | **Bold-matrix** (your current choice) | **Plain italic** (lower-churn alternative) |
|---|---|---|
| Touches | notation_guide stays; rewrite ~25 posts + glossary + 4 cheatsheets + **all 54 SVGs** (font-weight on individual symbols, each needs re-render-and-verify) | revert notation_guide to plain; fix only ~8 backprop posts + reconcile glossary; **SVGs already conform** |
| Beginner-friendliness | bold matrices are a more advanced-textbook look | plain $X,W,z,a,\hat y$ is what 25 posts + every diagram already teach |
| Effort | **high** (SVG re-rendering is the expensive part) | **low–moderate** |

**Chosen: bold-matrix**, for the textbook-correct look, accepting the 54-SVG re-render pass and the ~25-post rewrite. Target forms (from `notation_guide.md`): $\mathbf{X},\mathbf{W},\mathbf{b},\mathbf{Z},\mathbf{A},\hat{\mathbf{y}},\mathbf{y}$; scalars stay italic lowercase ($z,w,b,a$ for a single neuron/element); $L$ (drop the $\mathcal{L}$ hedge); $\alpha,\beta,\rho,\lambda,\epsilon$.

Everything below in this section is a *drift* that must be unified up to that target.

### 2.4 Series-wide drift map (per quantity)

| Quantity | Conforms | Drifts (and how) |
|---|---|---|
| **Input X** | posts 14–17 (`\mathbf{X}`) | posts 01 (`\vec{x}`+`X_i`), 02 (`X`/`\vec{x}`), 03/06/07 (`X`), 33 (`\mathbf{x}`); glossary, cheatsheets, all SVGs (`X`) |
| **Weights W** | 11,14,15,16,17,20,33 (`\mathbf{W}`) | 01,02,03,06,07,13,25 (`W`); glossary; cheatsheets; all SVGs |
| **Bias b** | post 11 (`\mathbf{b}`) | **Capital-B cluster (wrong letter):** 13,14,15,16,17,20 use `B_k`/`\mathbf{B}`. Plain: 01/02 (`\vec b`), 03/06/07, glossary, cheatsheets, SVGs |
| **Pre-activation Z** | 14,15,17,19 (`\mathbf{Z}`) | forward arc uses `F`/`F_n`/`Y` not Z at all; post 20 plain `Z`; lowercase scalar `z` (correct) in 11,12,33,34 |
| **Activation A** | post 17 (`\mathbf{A}`) | forward arc uses `F`/`Y`; element `A_k` in 13,14,17,19; glossary invents `F_n` |
| **Prediction ŷ** | post 18 (`\hat{\mathbf{y}}`) | plain `\hat y` everywhere else; **post 19 uses both forms** |
| **Loss L** | most posts, cheatsheets, glossary (`L`) | post 30 (`𝓛_total/data/reg`), post 34 mixes `𝓛` and `L_i`; guide still hedges `L or 𝓛` |
| **Learning rate α** | 22–29, 35 (`\alpha`) | **wrong letter:** posts 09,10,12,13,30 use `\eta` |
| **Stability ε** | guide, glossary, cheatsheet, post 10 (`\epsilon`) | posts 23,25,26,27 use `\varepsilon` (glyph drift) |
| Momentum β, sq-grad ρ, reg λ | consistent everywhere | — none — |

### 2.5 Within-post inconsistencies — the exact thing you flagged

Same quantity, two-or-more symbols **inside a single post**:

- **Post 01** — layer output is $\hat y$ (§3) / $\vec y$ (§6) / `Y` (SVG 05) / `ŷ` (SVGs 01–02): four forms, one quantity.
- **Post 02** — single sample is `X` (line 206) and `\vec x` (line 249) in the same post.
- **Post 07** — pre-activation is "logits" / `o` / `F1·W2+b2`; activation output is unnamed / `F1` / `F₁`.
- **Post 11** — `z` scalar (§2–4) vs `z_1` indexed with bold matrix factors (§6), no bridge.
- **Post 13** — bias is `B_k`/`B_1` in prose but `b₁` in the SVG glyph and `B₁` in the SVG gradient panel (case clash).
- **Post 18** — prediction `\hat y` (lines 13,31,160,…) vs `\hat{\mathbf y}` (lines 41,59); label `-y` vs `\mathbf y_i` likewise.
- **Post 19** — softmax output is `A_k` (§1) then `\hat y_k` (everywhere after); and `\hat y - y` (plain) vs `\hat{\mathbf y} - \mathbf y` (bold) for the same batch result.
- **Post 20** — line 33 `Z = X\mathbf{W} + \mathbf{B}` mixes **four** notations in one equation (plain Z, plain X, bold W, capital-bold B).
- **Post 21** — `\hat y` (table) vs Unicode `ŷ` (code comment).
- **Post 22** — gradient as `∂L/∂θ` (body) vs `∇_θ L` (summary table).
- **Post 33** — `z` left non-bold while `x`, `W` are bold in `z = \mathbf{W}\mathbf{x}`.
- **Post 34** — loss `L` vs `𝓛` vs `L_i`; probability `\hat y` vs `p` vs `\sigma(z)`.

---

## 3. Critical correctness errors (9) — fix before anything ships

These are wrong numbers, fabricated outputs, or false statements a reader *will* hit by running the code. They undermine the series' own "verify by hand" ethos.

1. **Post 03** — printed Layer-2 output block is fabricated. Running the post's own code yields different columns 2–3; only column 1 matches. *(Replace with real output.)*
2. **Post 08** — claims 0.01 is "twenty times more costly" than 0.5; the actual `-log` ratio is ~6.6×, not 20×. Repeated in the prose and SVG 01. *(Change to "about seven times".)*
3. **Post 12** — printed losses (1.2543 / 0.4568 / 0.2762) are fabricated; the real loop hits ~1e-14 by iter ~30. The numbers are also mutually inconsistent. *(Re-run and paste, or redesign the example to decay gradually.)*
4. **Post 12** — claims loss "converges to about 0.20" after 200 iters; it actually goes to ~0. Uncitable. *(Same root cause as #3.)*
5. **Post 13** — stated $Z=[3.0,7.2,11.4]$ is wrong; true is $[3.1,7.2,11.3]$ (hand-check: $0.1+0.4+0.9+1.6+0.1=3.1$). The SVG inherits the same error. *(Fix prose, table, and SVG.)*
6. **Post 13** — training-loop "expected output" doesn't match the code at lr=0.001 (claims iter-20 loss 9.843; real is 5.330). *(Regenerate the whole block.)*
7. **Post 23** — Part 22 baseline accuracy is "67%" in Part 22 but "57.3%"/"57%" in Part 23's table and SVG for the *same* config; final loss is 0.68 vs 0.768. A "harder configuration" is referenced but never defined. *(Reconcile to one baseline across Parts 22–23 + SVG.)*
8. **Post 28** — training accuracy is "95.7%" once (line 57) but "93%" everywhere else in the post and both SVGs. *(Pick one; 93% matches the diagrams.)*
9. **Glossary** — two critical notation contradictions vs the notation guide (the plain `X/W/b` symbols table, and the invented `F_n` forward formula). *(Reconcile per §2.3 decision.)*

---

## 4. Major correctness & consistency (22 in posts + many in docs)

Grouped by theme. Full per-item detail with line numbers in the digest.

### 4.1 Stale "31 parts" framing (series is now 35)
- **INDEX.md** "The 31 parts teach three pillars" and "TRAINING (Parts 22–31)"; the concept-dependency map + lookup table stop at Part 31 (Parts 32–35 have no prerequisite guidance).
- **quizzes.md** title "All 31 Parts"; **exercises.md** "every lecture" — no exercises/quizzes exist for Parts 32–35.
- **Cheatsheets** cover only Parts 1–31 (legitimately labelled, but nothing covers 32–35).
- **Projects 01 & 02** README claims mini-batching "is the only thing the lectures skipped" (Post 32 now covers it) and "the series only covered multi-class classification" (Post 34 is sigmoid+BCE). Related-lecture tables omit Posts 32 and 34.

### 4.2 Broken / dead reference
- **`concept_dependency_map.md`** is linked from README (×2) and INDEX (×1) but **does not exist**. The map content lives inline in INDEX. *(Create the file or repoint the links.)*

### 4.3 Wrong formula / wrong numbers in reference assets
- **Cheatsheet 04** — SGD+Momentum update rule carries a spurious `(1-β)` factor (`v ← βv + (1-β)∇L`). Classical momentum has **no** `(1-β)` term (that belongs to Adam's β₁). It even contradicts the cheatsheet's own code two lines below. **Wrong math.**
- **Dashboards/Optimizer_Comparison.md** — headline accuracy/loss numbers **contradict the posts they cite**: RMSProp shown ~95% (canon ~88%), plain SGD ~85–88% (canon 57.3%), Adam ~97% (canon 95.7%), AdaGrad drawn as worst/"stalls" (canon ~89.3%, beats SGD+decay). Adam decay 5e-7 vs canonical 1e-5. The ranking is essentially inverted.
- **Dashboards/Regularization_Comparison.md** — presents L2+Dropout(0.1) as a ~87% winner; Part 31 says that exact config *over-regularizes* the spiral to ~66.6%. Plus a dead-code bug: `train_preds` computed from test outputs, never used; no train accuracy ever printed.

### 4.4 Cross-artifact contradictions within posts
- **Post 18** — hero SVG uses prediction row `[0.2,0.3,0.5]` → gradient −0.667, but the worked example uses `[0,0,1]` → −0.333; figure and text disagree. Also the shown forward returns un-averaged loss while the prose claims it returns the mean (the `1/N` source is never shown). The displayed "Output" is idealized (real code yields `nan`).
- **Post 14** — TL;DR "15 per-weight gradients" — there are 12 weight gradients (15 = 12 weights + 3 biases); biases use a separate `np.sum`, not the matrix product.
- **Post 17** — TL;DR says softmax "gets two whole posts (19 and 20)"; Part 20 is "Assembling full backpropagation", not softmax-specific.
- **Post 20** — "Seven posts of backprop theory" (TL;DR) vs "After ten posts" (§1); mini-batching linked to "Part 22" but it's Part 32; optimizer ranges "22–27" vs "24–27".
- **Post 25** — AdaGrad ε shown **inside** the root in the primary rule/caption/summary but **outside** in the code/SVG/§5. Pick the outside form (matches code).
- **Post 31** — BatchNorm cite "Li et al., 2018" vs "2019" in Further reading + REFERENCES; Part 27 baseline "93%" (table) vs "95%" (prose, ×2).
- **Post 32** — Smith et al. "2017" (body) vs "2018" (Further reading + REFERENCES); worked example uses `decay=1e-4` while the pitfalls section says use `1e-5 or smaller`.
- **Post 33** — per-layer shrink "150×" (§3) vs "~32×" (pitfalls); neither traces cleanly (n_in=64 → ~156× linear / ~312× ReLU).
- **Notebook** — markdown "6 optimizer variants" vs code "4 optimizer classes"; a "decreased by 0.0000" demo that shows no visible decrease; a "different seed" comment where no seed is set.

### 4.5 Smaller factual slips
- **Post 05** — a bullet header says "Broadcasting is not interchangeable across operand order" but its own body says order doesn't matter (broadcasting *is* commutative). Header contradicts body.
- **Post 10** — "they coincide only for the trivial function $f(x)=e^x$" is false (any $Ce^x$ works; $e^x$ isn't "trivial").
- **Post 04** — printed spiral output looks synthetic (row 2 = 2×row 1, an exact-zero row at index 3 is impossible for nnfs spiral data).
- **Quizzes** — Part 3 note "59 = 56 + 11 biases" — arithmetic wrong (56+11 = 67); Part 27 Adam year 2014 vs the post's ICLR 2015; Part 25 ε inside-root contradicts Post 25.
- **Appendix** — overclaims it "fills the main mathematical gap in Part 19" (Part 19 already has that derivation); promises a side-by-side Jacobian route it never shows; uses `\mathbf{z}` (lowercase) where Part 19 uses `\mathbf{Z}`.

---

## 5. SVG conformance (54 post diagrams)

Notation drift in SVGs mirrors the prose drift in §2 (plain `X/W/Y/F₁`, never bold `Z/A`); fixing those is part of the §2.3 decision. Beyond notation, the recurring **style-token** issues are:

- **Stroke widths off the 1.5 / 1.0 / 0.75 set** — many SVGs use 1.2, 1.8, 2.0, 2.5 (e.g. post 01 neuron node 1.8, out-line 2.0; post 02 leader 1.2; post 03 conn 1.2 / node 1.8). Systemic, minor.
- **Hardcoded `#FFFFFF`** appears in all 54 SVGs (white badge text) instead of a `--ce-*` token. Mostly cosmetic but technically off-token.
- **Cross-artifact value errors** (already in §3/§4): post 13 SVG (Z/A = 3.0/11.4), post 18 SVG (−0.667), post 03 SVG output nodes labelled `ŷ` on a linear no-softmax stack.
- A handful flagged `visual_verify_recommended` — these should be rendered and eyeballed during the fix pass (the diagram standard requires render-and-view).

What's **good**: every SVG has `viewBox` with no root width/height, `<title>`+`<desc>`, the dark-mode `@media` block, Inter/JetBrains fonts via CSS vars, the `--ce-*` palette for shapes, and no emoji glyphs (mismatch X-marks are drawn as `<line>`, correctly).

---

## 6. Pedagogy / teaching quality (the depth check)

**Overall: strong.** Every post rates depth 4–5 and clarity 4–5. The series consistently does the things good teaching needs: examples before formalism, per-term arithmetic tables, explicit shape diaries, "what X is not" boundary-setting, and anticipated-question blocks. No post is poorly taught.

The improvements are **recurring patterns**, not one-off failures:

1. **Terms used before defined** (most common). Acronyms/terms appear ahead of their gloss: `broadcasting` (01), `FLOPs`/`representation` (03), `logits`↔pre-activation link (06, 07), `local minimum` (09, 22), `EMA`/`convex combination`/`moment` (23, 26), `implicit regularisation`/`online learning` (32), `ReLU` unexpanded in several backprop posts, `MLP`/`autograd`/`FSDP` (35). Fix: one-clause gloss on first use.
2. **Claims asserted, not shown.** The "why" is often deferred or skipped: the linear-collapse substitution (03), why `−grad` is steepest descent (10), the EWMA `1/(1-β)` horizon (24), the EMA fixed point (26), the `1/B` variance drop (32), the geometric-mean dropout argument (31). Fix: a 1–3 line derivation or worked micro-example each.
3. **Fabricated / idealized outputs** (overlaps §3). Posts 03, 04, 12, 13, 18, 21 show numbers that don't come from the shown code. This is the highest-impact pedagogy problem because it breaks trust exactly where the post says "verify it yourself."
4. **Shape-tracking gaps.** Posts 11, 19, 20 drop the inline shape annotations the series uses elsewhere; a small `(N,2)→(N,3)` table (as Post 07 does) would restore the discipline.
5. **Voice slips** — body `you`/`I` (against the third-person standard) in posts 10, 21, 30, 32, 33 (mostly FAQ headers).

Lowest-depth post: **35-whats-next** (depth 3 — appropriate, it's a forward-pointing reading list, but it lacks a Common-pitfalls block and expands few acronyms).

---

## 7. Style-standard conformance

- **Em-dash cap (~8–10/post): exceeded by** post **35 (39!)**, **32 (13)**, **01 (11)**; borderline **19 & 21 (10)**. Most others are fine.
- **TL;DR "exactly two sentences": violated widely** — the audit flagged over-long TL;DRs in posts 02, 05, 09, 10, 11, 15, 16, 18, 19, 21, 23, 24, 26, 30, 33, 34, 35. This is the most common style miss.
- **Voice** — see §6.5 (posts 10, 21, 30, 32, 33).
- **Missing bottom block** — post 35 has no "Common pitfalls" section.
- Spelling drift: `common_pitfalls.md` uses American "Regularization"/short phase titles vs the series' British "Generalisation and Regularisation".

---

## 8. Verified clean (no action)

- All 35 posts present, contiguous, correctly linked; **241 "Part NN" cross-refs correct** (0 number/topic mismatches).
- **All 35 hero paths resolve** (`diagrams/NN-*.svg`; the old `assets/` worry is moot — no `assets/` refs remain).
- **No stale `LectureNN/` layout** anywhere; remaining "Lecture" hits are real citations/column headers.
- **Bibliography complete** both directions (no cited-but-missing, no orphans).
- **Optimizer scalars consistent** (β/β₁, ρ/β₂, λ) across guide, glossary, cheatsheet, posts.
- **Projects & notebook code runs** end-to-end with correct shapes; their math notation is correctly scalar (no bold-matrix violations).
- **Appendix math is correct** (softmax+CCE = ŷ−y verified: `[-0.3,0.1,0.2]/3 = [-0.1,0.033,0.067]`).

---

## 9. Recommended fix order

**Step 0 — notation decided: bold-matrix (§2.3).** `notation_guide.md` is the target; everything conforms up to it.

**Step 1 — critical correctness (§3), highest priority.** Re-run every code block flagged in posts 03, 04, 12, 13, 18, 21 and paste real output; fix the post 08 "20×", post 13 hand-arithmetic, and post 23/28 conflicting figures. These are trust-breakers.

**Step 2 — reference-asset errors (§4.3).** Cheatsheet-04 momentum formula; both dashboards' numbers vs the canonical posts; the dashboard dead-code bug. These actively mislead.

**Step 3 — stale "31 parts" sweep (§4.1) + dead link (§4.2).** INDEX pillars/dependency map, quizzes/exercises titles + Parts 32–35 coverage, projects 01/02 framing, `concept_dependency_map.md`.

**Step 4 — notation standardization (§2.4–2.5)** under the Step-0 decision. Unify the within-post offenders first (posts 01, 07, 18, 19, 20), then the series-wide drifts (capital-B→b, η→α, ε glyph, 𝓛→L), then SVGs if bold is chosen.

**Step 5 — remaining major contradictions (§4.4) + pedagogy passes (§6).** One-clause glosses, the "show don't assert" micro-derivations, shape tables.

**Step 6 — polish (§7).** TL;DR trims, em-dash reductions (start with posts 35, 32, 01), voice fixes, SVG stroke-width normalization, render-verify flagged diagrams.

---

*Counts: 35 posts · 54 post SVGs · 9 critical + 22 major post-level correctness findings · ~120 notation drift instances · 8 doc groups audited. Reviewers: 45 agents, 2.58M tokens, 964 tool calls.*
