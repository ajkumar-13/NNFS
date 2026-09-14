// The post 24 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 2 explains *why* momentum works by decomposing two consecutive steps
// into components that cancel and components that reinforce. The existing figure
// shows the trajectory that results; nothing showed the arithmetic behind it.
//
//   01-momentum-trajectory  960 x 540
//   02-vector-cancellation  960 x 470   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// Both panels use the same valley and the same starting point, so the only
// thing that differs between them is the path taken.
function momentumTrajectory() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('The same valley, walked two ways',
    'A long narrow ravine: steep across, gentle along. The gradient points mostly at the walls, not at the floor.')];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 320, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 120, fit(o.sub, 11, 380, 'panel sub'), { size: 11, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + 404, 146));

    // Elongated contours, identical in both panels.
    [1, 2, 3, 4].forEach((k) => {
      els.push(rect(x + 210 - k * 42, 262 - k * 17, k * 84, k * 34, {
        stroke: T.border, strokeWidth: 1, roughness: 0.5,
      }));
    });
    els.push(line(o.path.map(([px, py]) => [x + px, py]),
      { stroke: o.c, strokeWidth: 2, roughness: 0.4 }));
    o.path.forEach(([px, py], i) => {
      if (i % 2 === 0) {
        els.push(circle(x + px, py, 3, { stroke: o.c, fill: T.surface, strokeWidth: 1.2 }));
      }
    });

    // The target, drawn last and on a small opaque patch. In the left panel the
    // zig-zag crosses this spot a dozen times and buries anything under it.
    els.push(rect(x + 178, 286, 64, 16, { stroke: T.surface, fill: T.surface, strokeWidth: 1, roundness: null }));
    els.push(circle(x + 210, 262, 4.5, { stroke: T.ink, fill: T.ink, strokeWidth: 1.2 }));
    els.push(text(x + 178, 288, 'minimum', { size: 9, stroke: T.inkSubtle, align: 'center', width: 64 }));
    els.push(text(x + 20, 336, fit(o.note, 11, 380, 'panel note'), { size: 11, stroke: T.inkMuted }));
    els.push(text(x + 20, 376, o.result, { size: 12, family: 3, stroke: o.c }));
  };

  panel(M, {
    c: T.alert, title: 'No momentum', sub: 'each step is the current gradient and nothing else',
    path: [[350, 190], [92, 214], [332, 232], [116, 248], [312, 260], [136, 268],
      [292, 274], [156, 278], [274, 280]],
    note: 'The across-valley component dominates every step, so\nthe walk spends its budget crossing, not descending.',
    result: '10 000 epochs  →  64.7%',
  });

  panel(500, {
    c: T.success, title: 'With momentum, β = 0.9', sub: 'each step adds the remembered ones',
    path: [[350, 190], [300, 212], [268, 230], [246, 244], [232, 252], [222, 257],
      [216, 260], [212, 261], [210, 262]],
    note: 'Opposite across-valley components cancel as they\naccumulate; the consistent downhill ones add up.',
    result: '10 000 epochs  →  95.7%',
  });

  els.push(rect(M, 432, 880, 74, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 444, 'v  =  β · v  −  α · g                    θ  =  θ  +  v',
    { size: 15, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 472, 'One velocity buffer per parameter array, one new hyperparameter, and a thirty-one point gain over plain SGD.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Vanilla SGD against SGD with momentum through a narrow valley',
    desc: 'Two hand-drawn panels over identical elongated contour rings with the minimum at the '
      + 'centre, both starting from the same point in the upper right. The left panel, in alert red '
      + 'and labelled no momentum, shows a path that zig-zags violently from wall to wall, barely '
      + 'advancing along the valley floor, annotated as spending its budget crossing rather than '
      + 'descending and reaching sixty-four point seven per cent in ten thousand epochs. The right '
      + 'panel, in success green and labelled momentum with beta of nought point nine, shows a '
      + 'smooth path that curves almost directly into the minimum, annotated as having opposite '
      + 'across-valley components cancel while the consistent downhill ones add up, reaching '
      + 'ninety-five point seven per cent in the same budget. A band gives the update rule, velocity '
      + 'equals beta times velocity minus alpha times the gradient with the parameter then moved by '
      + 'the velocity, and notes that this costs one velocity buffer per parameter array and one new '
      + 'hyperparameter.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 2 is a two-vector argument: the across-valley components of
// consecutive steps disagree and the downhill components agree, so summing them
// dampens one and reinforces the other. Drawing the components separately is
// what turns that from a sentence into something checkable.
function vectorCancellation() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Why adding two steps helps',
    'Consecutive steps in a valley disagree about which way is sideways and agree about which way is down.')];

  els.push(...card(M, 86, 420, 240, { spine: T.accent }));
  els.push(text(62, 98, 'Two consecutive steps', { size: 15, stroke: T.accent }));
  els.push(text(62, 120, 'taken separately, as vanilla SGD does', { size: 10, stroke: T.inkSubtle }));
  els.push(rule(56, 444, 144));

  els.push(text(96, 158, 'step 1', { size: 11, stroke: T.accent }));
  els.push(arrow([[100, 178], [250, 212]], { stroke: T.accent, strokeWidth: 2.2, roughness: 0.4 }));
  els.push(line([[100, 178], [250, 178]], { stroke: T.inkSubtle, strokeWidth: 1, strokeStyle: 'dashed', roughness: 0.3 }));
  els.push(line([[250, 178], [250, 212]], { stroke: T.inkSubtle, strokeWidth: 1, strokeStyle: 'dashed', roughness: 0.3 }));
  els.push(text(268, 172, 'across  +150', { size: 10, family: 3, stroke: T.inkMuted }));
  els.push(text(268, 190, 'down    +34', { size: 10, family: 3, stroke: T.inkMuted }));

  els.push(text(96, 242, 'step 2', { size: 11, stroke: T.accent }));
  els.push(arrow([[250, 262], [100, 296]], { stroke: T.accent, strokeWidth: 2.2, roughness: 0.4 }));
  els.push(line([[250, 262], [100, 262]], { stroke: T.inkSubtle, strokeWidth: 1, strokeStyle: 'dashed', roughness: 0.3 }));
  els.push(line([[100, 262], [100, 296]], { stroke: T.inkSubtle, strokeWidth: 1, strokeStyle: 'dashed', roughness: 0.3 }));
  els.push(text(268, 256, 'across  −150', { size: 10, family: 3, stroke: T.inkMuted }));
  els.push(text(268, 274, 'down    +34', { size: 10, family: 3, stroke: T.inkMuted }));

  els.push(...card(500, 86, 420, 240, { spine: T.success }));
  els.push(text(522, 98, 'The same two, added first', { size: 15, stroke: T.success }));
  els.push(text(522, 120, 'which is what a velocity buffer does', { size: 10, stroke: T.inkSubtle }));
  els.push(rule(516, 904, 144));
  els.push(arrow([[640, 172], [640, 240]], { stroke: T.success, strokeWidth: 2.6, roughness: 0.35 }));
  els.push(text(560, 196, 'sum', { size: 11, stroke: T.success }));
  els.push(text(676, 168, 'across   +150 − 150  =  0', { size: 11, family: 3, stroke: T.alert }));
  els.push(text(676, 192, 'down     +34 + 34   =  +68', { size: 11, family: 3, stroke: T.success }));
  els.push(text(676, 226, 'The bounce cancels itself.', { size: 10, stroke: T.inkSubtle }));
  els.push(text(676, 242, 'The descent doubles.', { size: 10, stroke: T.inkSubtle }));
  els.push(text(522, 284, 'A plateau works the same way: a tiny current gradient still moves,', { size: 10, stroke: T.inkMuted }));
  els.push(text(522, 300, 'because the velocity it inherits from a thousand steps is not tiny.', { size: 10, stroke: T.inkMuted }));

  // Three groups reading left to right, not a table: the card is one row tall
  // and column headers over horizontal groups do not line up with anything.
  els.push(...card(M, 346, 880, 66, { spine: T.primary }));
  els.push(text(62, 356, 'momentum factor  ·  effective horizon of 1 / (1 − β)  ·  accuracy after 10 000 epochs',
    { size: 10, stroke: T.inkSubtle }));
  const BETAS = [
    ['β = 0.0', '1 gradient', '64.7%', T.inkMuted],
    ['β = 0.5', '2 gradients', '78.0%', T.warn],
    ['β = 0.9', '10 gradients', '95.7%', T.success],
  ];
  BETAS.forEach((b, i) => {
    const x = 62 + i * 292;
    els.push(text(x, 380, b[0], { size: 13, family: 3, stroke: b[3] }));
    els.push(text(x + 76, 382, `·  ${b[1]}`, { size: 10, family: 3, stroke: T.inkSubtle }));
    els.push(text(x + 186, 380, b[2], { size: 13, family: 3, stroke: b[3] }));
  });

  els.push(text(M, 432, 'β = 0.9 averages roughly the last ten gradients: long enough to carry real inertia, short enough to still notice the current one.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'How momentum cancels the bounce and reinforces the descent',
    desc: 'A hand-drawn figure in two cards. The left shows two consecutive gradient steps taken '
      + 'separately, as vanilla SGD does: step one points down and to the right with an across '
      + 'component of plus one hundred and fifty and a down component of plus thirty-four, and step '
      + 'two points down and to the left with an across component of minus one hundred and fifty and '
      + 'the same down component of plus thirty-four, each decomposed with dashed component lines. '
      + 'The right card adds them first, which is what a velocity buffer does: the across components '
      + 'cancel to zero and the down components sum to plus sixty-eight, so the bounce cancels itself '
      + 'and the descent doubles, with a note that a plateau works the same way because the velocity '
      + 'inherited from a thousand steps is not tiny even when the current gradient is. A strip '
      + 'beneath gives three momentum factors with their effective horizons and measured accuracy: '
      + 'nought averages one gradient for sixty-four point seven per cent, nought point five '
      + 'averages two for seventy-eight, and nought point nine averages ten for ninety-five point '
      + 'seven.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { momentumTrajectory, vectorCancellation };

if (require.main === module) {
  emit('01-momentum-trajectory', momentumTrajectory());
  emit('02-vector-cancellation', vectorCancellation());
}
