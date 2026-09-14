// The post 32 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 2 makes the point the whole choice turns on — every regime touches
// every sample once per epoch, so the cost is identical and only the number of
// update steps changes — and a trajectory plot cannot say that.
//
//   01-batch-size-trajectories  960 x 540
//   02-same-cost-more-steps     960 x 470   (new)

const { T, rect, circle, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// A deterministic wobble, so a "noisy" path is reproducible across rebuilds.
const jit = (i) => {
  const s = Math.sin(i * 12.9898) * 43758.5453;
  return (s - Math.floor(s)) * 2 - 1;
};

// ---------------------------------------------------------------- diagram 1
// One landscape, three paths from the same start. Only the noise per step
// differs, which is the only thing batch size changes about the trajectory.
function batchSizeTrajectories() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('The same descent, at three batch sizes',
    'Every path starts at the same point and every one gets one pass over the data. What changes is the step count.')];

  els.push(...card(M, 86, 560, 330, { spine: T.primary }));
  els.push(text(62, 98, 'One loss landscape, three trajectories', { size: 14, stroke: T.primary }));

  const CX = 330, CY = 268;
  [1, 2, 3, 4].forEach((k) => {
    els.push(rect(CX - k * 54, CY - k * 26, k * 108, k * 52, {
      stroke: T.border, strokeWidth: 1, roughness: 0.5,
    }));
  });

  const PATHS = [
    { c: T.inkMuted, n: 5, noise: 0, lab: 'full-batch' },
    { c: T.success, n: 16, noise: 7, lab: 'mini-batch' },
    { c: T.alert, n: 42, noise: 20, lab: 'pure SGD' },
  ];
  PATHS.forEach((p, pi) => {
    const pts = [];
    for (let i = 0; i <= p.n; i++) {
      const t = i / p.n;
      const bx = 540 - t * 208, by = 158 + t * 110;
      pts.push([bx + jit(pi * 97 + i) * p.noise, by + jit(pi * 53 + i * 7) * p.noise * 0.5]);
    }
    els.push(line(pts, { stroke: p.c, strokeWidth: pi === 0 ? 2.2 : 1.6, roughness: 0.35 }));
    els.push(circle(540, 158, 4, { stroke: T.ink, fill: T.ink, strokeWidth: 1 }));
  });
  els.push(circle(CX, CY, 4.5, { stroke: T.ink, fill: T.ink, strokeWidth: 1.2 }));
  els.push(text(CX + 8, CY + 4, 'minimum', { size: 9, stroke: T.inkSubtle }));
  els.push(text(500, 138, 'start', { size: 9, stroke: T.inkSubtle }));

  PATHS.forEach((p, i) => {
    els.push(line([[70, 348 + i * 20], [104, 348 + i * 20]],
      { stroke: p.c, strokeWidth: 2.2, roughness: 0.3 }));
    els.push(text(114, 342 + i * 20, p.lab, { size: 11, stroke: p.c }));
  });

  els.push(...card(620, 86, 300, 330, { spine: T.success }));
  els.push(text(642, 98, 'What the noise is for', { size: 14, stroke: T.success }));
  els.push(rule(636, 904, 122));
  els.push(text(642, 136, 'A full-batch optimiser sitting in a', { size: 10, stroke: T.ink }));
  els.push(text(642, 152, 'shallow minimum has gradient zero', { size: 10, stroke: T.ink }));
  els.push(text(642, 168, 'and stops there.', { size: 10, stroke: T.ink }));
  els.push(text(642, 194, 'A mini-batch optimiser at the same', { size: 10, stroke: T.ink }));
  els.push(text(642, 210, 'point has gradient zero in', { size: 10, stroke: T.ink }));
  els.push(text(642, 226, 'expectation and non-zero on any', { size: 10, stroke: T.ink }));
  els.push(text(642, 242, 'given batch, so it gets kicked out.', { size: 10, stroke: T.ink }));
  els.push(rule(636, 904, 266));
  els.push(text(642, 278, 'The same escape momentum buys', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 294, 'with a velocity buffer, mini-batching', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 310, 'gets for free from the sampling.', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 338, 'It also biases training towards flat', { size: 10, stroke: T.inkSubtle }));
  els.push(text(642, 354, 'minima, which is why very large', { size: 10, stroke: T.inkSubtle }));
  els.push(text(642, 370, 'batches can generalise worse.', { size: 10, stroke: T.inkSubtle }));

  els.push(rect(M, 442, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 454, 'Mini-batching is not a compromise that trades accuracy for memory.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 478, 'It takes more steps, escapes shallow minima the exact gradient cannot, and fits in memory. Three wins, one change to the loop.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Three batch-size regimes: full-batch, mini-batch and pure SGD trajectories',
    desc: 'A hand-drawn figure. The main panel shows one elliptical loss landscape with three '
      + 'trajectories starting from the same point in the upper right and heading for the minimum at '
      + 'the centre. The full-batch path is smooth with only five points; the mini-batch path is '
      + 'mildly wobbly with sixteen; the pure SGD path is jagged with forty-two. A side card '
      + 'explains what the noise is for: a full-batch optimiser sitting in a shallow minimum has '
      + 'gradient zero and stops there, whereas a mini-batch optimiser at the same point has '
      + 'gradient zero only in expectation and non-zero on any given batch, so it gets kicked out; '
      + 'the same escape that momentum buys with a velocity buffer, mini-batching gets free from the '
      + 'sampling. It adds that mini-batch noise biases training towards flat minima, which is why '
      + 'very large batches can generalise worse. A band states that mini-batching is not a '
      + 'compromise trading accuracy for memory but takes more steps, escapes shallow minima the '
      + 'exact gradient cannot, and fits in memory.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 2's table hides its own punchline: the per-epoch cost is the
// same in all three columns. Putting that row first, identical across the three
// cards, is what makes the rest of the comparison mean anything.
function sameCostMoreSteps() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Same work per epoch, wildly different step counts',
    'All three touch every sample exactly once. The only thing batch size buys is how often the weights move.')];

  const REG = [
    {
      c: T.inkMuted, name: 'Full-batch', b: 'B = 60 000', segs: 1,
      updates: '1', noise: '0.004×',
      note: 'One exact gradient, one step.\nThe intermediates do not fit\nin memory past toy sizes.',
    },
    {
      c: T.success, name: 'Mini-batch', b: 'B = 128', segs: 14,
      updates: '469', noise: '0.088×',
      note: 'The production default.\nEnough noise to escape a\nshallow minimum, not enough\nto destabilise training.',
    },
    {
      c: T.alert, name: 'Pure SGD', b: 'B = 1', segs: 48,
      updates: '60 000', noise: '1.000×',
      note: 'Every step is a single sample,\nso every direction is a very\nhigh-variance guess.',
    },
  ];

  const CWD = 280, TOP = 90, CHT = 250;
  REG.forEach((r, i) => {
    const x = M + i * 300, inner = CWD - 40;
    els.push(...card(x, TOP, CWD, CHT, { spine: r.c }));
    els.push(text(x + 20, TOP + 12, r.name, { size: 14, stroke: r.c }));
    els.push(text(x + 160, TOP + 15, r.b, { size: 11, family: 3, stroke: T.ink }));

    // The dataset, cut into batches. Every card's bar is the same width,
    // because every card processes the same 60 000 samples per epoch.
    els.push(text(x + 20, TOP + 42, 'one epoch of data', { size: 9, stroke: T.inkSubtle }));
    const BX = x + 20, BW = 240, sw = BW / r.segs;
    for (let s = 0; s < r.segs; s++) {
      els.push(rect(BX + s * sw, TOP + 58, sw, 22, {
        stroke: r.c, fill: r.c, strokeWidth: 1, opacity: 22, roundness: null, roughness: 0.2,
      }));
    }
    els.push(text(x + 20, TOP + 86, '60 000 samples, every card', { size: 9, stroke: T.inkSubtle }));

    els.push(rule(x + 16, x + CWD - 16, TOP + 108));
    els.push(text(x + 20, TOP + 118, 'weight updates per epoch', { size: 9, stroke: T.inkSubtle }));
    els.push(text(x + 20, TOP + 134, r.updates, { size: 22, family: 3, stroke: r.c }));
    els.push(text(x + 160, TOP + 122, 'gradient noise', { size: 9, stroke: T.inkSubtle }));
    els.push(text(x + 160, TOP + 140, r.noise, { size: 14, family: 3, stroke: T.ink }));

    els.push(rule(x + 16, x + CWD - 16, TOP + 172));
    els.push(text(x + 20, TOP + 182, fit(r.note, 10, inner, 'regime note'), { size: 10, stroke: T.inkMuted }));
  });

  els.push(rect(M, 366, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 378, 'A batch gradient averages B per-sample gradients, so its variance falls as 1/B and its noise as 1/√B.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 402, 'Going from B = 1 to B = 128 cuts the noise by a factor of about eleven and still leaves 469 steps per epoch instead of one.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 444, 'Nothing in any optimiser class changes. Only the training loop gains an inner loop over batches.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Batch size changes the number of update steps, not the work per epoch',
    desc: 'Three hand-drawn cards comparing batch-size regimes on a sixty-thousand-sample dataset. '
      + 'Each card draws the same-width bar of one epoch of data, cut into a different number of '
      + 'batches: full-batch at B equals sixty thousand is one segment giving one weight update per '
      + 'epoch and a relative gradient noise of nought point nought nought four; mini-batch at B '
      + 'equals one hundred and twenty-eight is many segments giving four hundred and sixty-nine '
      + 'updates and noise of nought point nought eight eight; pure SGD at B equals one is very many '
      + 'segments giving sixty thousand updates and noise of one. Notes record that full-batch '
      + 'intermediates do not fit in memory past toy sizes, that mini-batch is the production '
      + 'default with enough noise to escape a shallow minimum but not enough to destabilise '
      + 'training, and that pure SGD makes every direction a high-variance guess. A band explains '
      + 'that a batch gradient averages B per-sample gradients so its variance falls as one over B '
      + 'and its noise as one over the root of B, and that moving from one to a hundred and '
      + 'twenty-eight cuts the noise elevenfold while still leaving four hundred and sixty-nine '
      + 'steps per epoch. A footer notes that no optimiser class changes; only the training loop '
      + 'gains an inner loop.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { batchSizeTrajectories, sameCostMoreSteps };

if (require.main === module) {
  emit('01-batch-size-trajectories', batchSizeTrajectories());
  emit('02-same-cost-more-steps', sameCostMoreSteps());
}
