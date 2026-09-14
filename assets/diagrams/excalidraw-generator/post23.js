// The post 23 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 8 introduces the three-hook contract and tabulates how all six
// optimisers fit it. That table is the architectural spine of Parts 22 to 27 and
// the reason none of the loop code has to change again.
//
//   01-decay-schedule-and-result  960 x 540
//   02-three-hook-contract        960 x 480   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// The schedules on the left and what they produce on the right, so a curve and
// its result are read together. Both extremes are shown, because the section's
// finding is that too much decay is worse than none.
function decayScheduleAndResult() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Making the learning rate a function of time',
    'Large steps early, small steps late. The only question is how fast to move between the two.')];

  els.push(...card(M, 86, 880, 66, { stroke: T.primary, strokeWidth: 1.6 }));
  els.push(text(62, 100, 'α(t)  =  α₀ / (1 + d · t)', { size: 19, family: 3, stroke: T.ink }));
  els.push(text(400, 100, 'monotonic · slows its own decay · tends to 0 without reaching it',
    { size: 10, stroke: T.inkSubtle }));
  els.push(text(400, 120, 'recomputed once per step, before any layer is updated',
    { size: 10, stroke: T.inkSubtle }));

  els.push(...card(M, 176, 420, 244, { spine: T.primary }));
  els.push(text(62, 188, 'Three schedules', { size: 14, stroke: T.primary }));
  els.push(text(62, 210, 'α₀ = 1.0, over 10 000 iterations', { size: 10, stroke: T.inkSubtle }));
  const X0 = 96, X1 = 430, Y0 = 384, Y1 = 240;
  const px = (t) => X0 + (t / 10000) * (X1 - X0);
  const py = (a) => Y0 - (a / 1.05) * (Y0 - Y1);
  els.push(line([[X0, Y1 - 6], [X0, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0, Y0], [X1 + 8, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  [[1.0, '1.0'], [0.5, '0.5'], [0, '0']].forEach(([a, lab]) => {
    els.push(text(56, py(a) - 6, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'right', width: 34 }));
  });
  const SCHED = [
    { d: 0, c: T.inkSubtle, lab: 'd = 0', end: '1.000' },
    { d: 1e-3, c: T.success, lab: 'd = 1e−3', end: '0.091' },
    { d: 1e-2, c: T.alert, lab: 'd = 1e−2', end: '0.010' },
  ];
  SCHED.forEach((s) => {
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const t = (i / 60) * 10000;
      pts.push([px(t), py(1 / (1 + s.d * t))]);
    }
    els.push(line(pts, { stroke: s.c, strokeWidth: 2.2, roughness: 0.25 }));
  });
  els.push(text(96, 392, '0', { size: 9, stroke: T.inkSubtle }));
  els.push(text(390, 392, '10 000', { size: 9, stroke: T.inkSubtle }));
  SCHED.forEach((s, i) => {
    els.push(text(250, 232 + i * 18, `${s.lab}  →  ${s.end}`,
      { size: 10, family: 3, stroke: s.c }));
  });

  els.push(...card(500, 176, 420, 244, { spine: T.accent }));
  els.push(text(522, 188, 'What each one produces', { size: 14, stroke: T.accent }));
  els.push(text(522, 210, 'spiral set, 10 000 epochs, seed 0', { size: 10, stroke: T.inkSubtle }));
  els.push(rule(516, 904, 232));
  els.push(text(522, 242, 'configuration', { size: 9, stroke: T.inkSubtle }));
  els.push(text(760, 242, 'loss', { size: 9, stroke: T.inkSubtle }));
  els.push(text(844, 242, 'accuracy', { size: 9, stroke: T.inkSubtle }));
  const RES = [
    ['fixed α = 1.0', '0.87', '64.7%', T.inkMuted],
    ['decay 1e−3', '0.76', '64.7%', T.success],
    ['decay 1e−4', '0.97', '59.0%', T.inkMuted],
    ['decay 1e−2', '1.07', '39.7%', T.alert],
  ];
  RES.forEach((r, i) => {
    const y = 266 + i * 30;
    els.push(text(522, y, r[0], { size: 11, family: 3, stroke: r[3] }));
    els.push(text(760, y, r[1], { size: 11, family: 3, stroke: r[3] }));
    els.push(text(844, y, r[2], { size: 11, family: 3, stroke: r[3] }));
  });
  els.push(text(522, 392, 'Too much decay is worse than none at all.', { size: 10, stroke: T.alert }));

  els.push(rect(M, 446, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 458, 'Decay lowers the loss and stops the late-training oscillation. It does not raise the accuracy.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 482, 'Smaller late steps settle better; they do not travel further. Getting past 65% needs bigger, better-aimed steps, which is Part 24.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Learning-rate decay schedule and the loss it produces',
    desc: 'A hand-drawn figure with a formula band above two panels. The formula is alpha of t equals '
      + 'alpha nought over one plus d times t, annotated as monotonic, slowing its own decay, and '
      + 'tending to zero without reaching it, recomputed once per step before any layer is updated. '
      + 'The left panel plots three schedules from an initial rate of one over ten thousand '
      + 'iterations: no decay holding at one, a decay of one in a thousand tapering to nought point '
      + 'nought nine one, and a decay of one in a hundred falling to nought point nought one. The '
      + 'right panel tabulates what each produces on the spiral set: a fixed rate gives a loss of '
      + 'nought point eight seven at sixty-four point seven per cent, decay of one in a thousand '
      + 'gives nought point seven six at the same accuracy, one in ten thousand gives nought point '
      + 'nine seven at fifty-nine per cent, and one in a hundred gives one point nought seven at '
      + 'thirty-nine point seven, worse than no decay at all. A band notes that decay lowers the '
      + 'loss and stops late oscillation but does not raise the accuracy, because smaller late steps '
      + 'settle better without travelling further.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 8 sets up a contract that the next four posts all obey. Laying
// the six optimisers out as rows makes the actual claim visible: two of the
// three columns stop changing after this post, and every later optimiser is a
// change to the middle one alone.
function threeHookContract() {
  resetSeq();
  const W = 960, H = 480;

  const els = [...heading('One contract, six optimisers',
    'Three methods, called in the same order every step. From here on only the middle one ever changes.')];

  els.push(...card(M, 86, 880, 76, { stroke: T.primary, strokeWidth: 1.6 }));
  const CALLS = [
    { x: 70, w: 230, t: 'pre_update_params()', s: 'once per step' },
    { x: 340, w: 260, t: 'update_params(layer)', s: 'once per trainable layer' },
    { x: 640, w: 240, t: 'post_update_params()', s: 'once per step' },
  ];
  CALLS.forEach((c, i) => {
    els.push(text(c.x, 100, c.t, { size: 13, family: 3, stroke: T.primary, align: 'center', width: c.w }));
    els.push(text(c.x, 124, c.s, { size: 9, stroke: T.inkSubtle, align: 'center', width: c.w }));
    if (i < 2) {
      els.push(arrow([[c.x + c.w + 4, 108], [CALLS[i + 1].x - 4, 108]],
        { stroke: T.inkMuted, strokeWidth: 1.2, roughness: 0.4 }));
    }
  });

  const CX = [56, 240, 430, 800];
  els.push(text(CX[0], 190, 'optimiser', { size: 9, stroke: T.inkSubtle }));
  els.push(text(CX[1], 190, 'pre', { size: 9, stroke: T.inkSubtle }));
  els.push(text(CX[2], 190, 'update  ·  the only column that moves', { size: 9, stroke: T.accent }));
  els.push(text(CX[3], 190, 'post', { size: 9, stroke: T.inkSubtle }));

  const ROWS = [
    ['SGD · Part 22', '—', 'θ −= α · g', '—', T.inkMuted],
    ['+ decay · Part 23', 'recompute α', 'θ −= α · g', 't += 1', T.primary],
    ['Momentum · Part 24', 'recompute α', 'update v, then θ −= α · v', 't += 1', T.accent],
    ['AdaGrad · Part 25', 'recompute α', 'G += g², then θ −= α · g / √(G+ε)', 't += 1', T.accent],
    ['RMSProp · Part 26', 'recompute α', 'EMA of g², then θ −= α · g / √(E+ε)', 't += 1', T.accent],
    ['Adam · Part 27', 'recompute α', 'EMA of g and g², bias-correct, update', 't += 1', T.success],
  ];
  ROWS.forEach((r, i) => {
    const y = 212 + i * 34;
    if (i > 0) els.push(rule(50, 906, y - 8));
    els.push(text(CX[0], y, r[0], { size: 11, stroke: r[4] }));
    els.push(text(CX[1], y, r[1], { size: 10, family: 3, stroke: i >= 1 ? T.inkSubtle : T.inkSubtle }));
    els.push(text(CX[2], y, fit(r[2], 11, 360 * 0.94, 'update cell'),
      { size: 11, family: 3, stroke: r[4] }));
    els.push(text(CX[3], y, r[3], { size: 10, family: 3, stroke: T.inkSubtle }));
  });

  // The two columns that stop changing, bracketed.
  els.push(line([[232, 204], [232, 396]],
    { stroke: T.border, strokeWidth: 1, strokeStyle: 'dashed', roughness: 0.25 }));
  els.push(line([[792, 204], [792, 396]],
    { stroke: T.border, strokeWidth: 1, strokeStyle: 'dashed', roughness: 0.25 }));

  els.push(rect(M, 408, 880, 56, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 420, 'The training loop is written once and never edited again. Swapping the optimiser is swapping one object.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 444, 'PyTorch, TensorFlow and Optax all separate a scheduler step, a per-parameter update, and a counter bump. The names differ; the shape does not.',
    { size: 10, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The three-hook optimiser contract, shared by every optimiser in the series',
    desc: 'A hand-drawn figure. Along the top, the three methods in call order: pre_update_params '
      + 'once per step, update_params once per trainable layer, and post_update_params once per '
      + 'step. Below, a six-row table with one row per optimiser. Vanilla SGD from Part 22 does '
      + 'nothing in pre or post and subtracts alpha times the gradient in update. SGD with decay '
      + 'from Part 23 recomputes alpha in pre, keeps the same update, and increments the counter in '
      + 'post. Momentum, AdaGrad, RMSProp and Adam all repeat that same pre and post, and differ '
      + 'only in the update column: momentum updates a velocity, AdaGrad accumulates squared '
      + 'gradients and divides by their root, RMSProp uses an exponential moving average of the '
      + 'same, and Adam keeps moving averages of both the gradient and its square with a bias '
      + 'correction. The update column is headed as the only column that moves, and dashed rules '
      + 'separate it from the two that do not. A band notes that the training loop is written once '
      + 'and never edited again, and that PyTorch, TensorFlow and Optax all separate the same three '
      + 'concerns.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { decayScheduleAndResult, threeHookContract };

if (require.main === module) {
  emit('01-decay-schedule-and-result', decayScheduleAndResult());
  emit('02-three-hook-contract', threeHookContract());
}
