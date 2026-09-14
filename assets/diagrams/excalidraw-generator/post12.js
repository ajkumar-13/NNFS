// The post 12 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 6 takes one gradient-descent step with the four gradients just
// derived, and section 6.1 works out that the loss then collapses geometrically.
// That is the post proving its own arithmetic, and it had no figure.
//
//   01-single-neuron-backprop  960 x 540
//   02-one-step-and-after      960 x 470   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// Forward across the top, backward across the bottom, in the same five columns.
// The running gradient is written at every stage rather than only at the ends,
// because the claim of the post is that one number is carried leftwards and
// reused, not that four separate gradients are computed.
function singleNeuronBackprop() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Backpropagation through one neuron',
    'Three inputs, one bias, a ReLU and a squared error. Every number below is worked by hand in the post.')];

  const STAGES = [
    { x: 60, w: 150, name: 'xᵢ · wᵢ', val: '−3,  2,  6' },
    { x: 250, w: 130, name: 'Σ  + b', val: 'z = 6' },
    { x: 420, w: 130, name: 'ReLU', val: 'ŷ = 6' },
    { x: 590, w: 130, name: '( ŷ − y )²', val: 'y = 0' },
    { x: 760, w: 110, name: 'L', val: 'L = 36' },
  ];
  STAGES.forEach((s, i) => {
    els.push(...card(s.x, 110, s.w, 60, { stroke: T.primary, strokeWidth: 1.5 }));
    els.push(text(s.x, 128, s.name, { size: 15, family: 3, stroke: T.primary, align: 'center', width: s.w }));
    els.push(text(s.x, 178, s.val, { size: 12, family: 3, stroke: T.ink, align: 'center', width: s.w }));
    if (i < STAGES.length - 1) {
      els.push(arrow([[s.x + s.w + 2, 140], [STAGES[i + 1].x - 2, 140]],
        { stroke: T.primary, strokeWidth: 1.3, roughness: 0.4 }));
    }
  });
  els.push(text(60, 88, 'forward', { size: 10, stroke: T.primary }));

  // Backward: the same columns, the running gradient at each one.
  const BACK = [
    { i: 4, top: '∂L/∂L', val: '1' },
    { i: 3, top: '∂L/∂ŷ  =  2ŷ', val: '12' },
    { i: 2, top: '× ReLU′(6) = 1', val: '12' },
    { i: 1, top: '× ∂z/∂(xw) = 1', val: '12' },
    { i: 0, top: '× xᵢ', val: 'fans out' },
  ];
  BACK.forEach((b) => {
    const s = STAGES[b.i];
    els.push(...card(s.x, 250, s.w, 60, { spine: T.alert }));
    els.push(text(s.x, 262, fit(b.top, 10, s.w - 22, 'back top'),
      { size: 10, family: 3, stroke: T.inkMuted, align: 'center', width: s.w }));
    els.push(text(s.x, 282, b.val, { size: 15, family: 3, stroke: T.alert, align: 'center', width: s.w }));
    els.push(line([[s.x + s.w / 2, 176], [s.x + s.w / 2, 246]],
      { stroke: T.border, strokeWidth: 1, strokeStyle: 'dotted', roughness: 0.3 }));
  });
  [[250, 214], [420, 384], [590, 554], [760, 724]].forEach(([a, b]) => {
    els.push(arrow([[a - 2, 280], [b + 2, 280]], { stroke: T.alert, strokeWidth: 1.3, roughness: 0.4 }));
  });
  els.push(text(806, 228, 'backward', { size: 10, stroke: T.alert }));

  els.push(...card(M, 342, 880, 118, { spine: T.accent }));
  els.push(text(62, 354, 'The upstream gradient is 12 for every parameter. Only the last factor changes.',
    { size: 12, stroke: T.ink }));
  els.push(rule(56, 904, 374));
  const COLS = [
    ['parameter', 'w₀', 'w₁', 'w₂', 'b'],
    ['upstream', '12', '12', '12', '12'],
    ['× last factor', 'x₀ = 1', 'x₁ = −2', 'x₂ = 3', '1'],
    ['gradient', '12', '−24', '36', '12'],
  ];
  COLS.forEach((row, j) => {
    const y = 384 + j * 16;
    els.push(text(62, y, row[0], { size: 10, stroke: T.inkSubtle }));
    row.slice(1).forEach((v, i) => {
      els.push(text(200 + i * 170, y, v, {
        size: j === 3 ? 12 : 10, family: 3,
        stroke: j === 3 ? T.accent : T.ink, align: 'center', width: 150,
      }));
    });
  });

  els.push(text(M, 480, 'Every weight’s gradient is the upstream gradient times the input connected to that weight. That one sentence is most of backpropagation,',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 502, 'and it survives unchanged when the scalars below become the vectors and matrices of Parts 13 and 14.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Backpropagation through a single neuron with three inputs',
    desc: 'A hand-drawn two-row diagram. The forward row runs left to right in blue through five '
      + 'stages: the products x times w giving minus three, two and six; the sum plus bias giving z '
      + 'equals six; ReLU giving y-hat equals six; the squared error against a target of zero; and '
      + 'the loss L equals thirty-six. The backward row runs right to left in red in the same five '
      + 'columns, each joined to the stage above by a dotted line: the derivative of L by itself is '
      + 'one, the derivative of L by y-hat is two y-hat or twelve, multiplying by ReLU prime of six '
      + 'which is one leaves twelve, multiplying by the sum derivative of one leaves twelve, and at '
      + 'the products it fans out by multiplying by each input. A table beneath shows that the '
      + 'upstream gradient is twelve for every parameter and only the last factor changes: times x '
      + 'nought of one gives twelve, times x one of minus two gives minus twenty-four, times x two '
      + 'of three gives thirty-six, and the bias factor of one gives twelve. A footer states that '
      + 'every weight’s gradient is the upstream gradient times the input connected to that weight.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 6 spends the four gradients on one step and section 6.1 works
// out what happens if the step is repeated. Putting the single step beside the
// trajectory it starts is what turns "the gradients are correct" from an
// assertion into something the reader watches happen.
function oneStepAndAfter() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Spending the gradients',
    'One update with α = 0.01, and what the same update does when it is repeated.')];

  els.push(...card(M, 86, 440, 280, { spine: T.accent }));
  els.push(text(62, 98, 'One step', { size: 15, stroke: T.accent }));
  els.push(text(62, 120, 'w ← w − 0.01 · ∂L/∂w', { size: 11, family: 3, stroke: T.inkSubtle }));
  els.push(rule(56, 464, 146));

  const HEAD = ['', 'old', '∂L/∂w', 'new'];
  const XS = [62, 190, 280, 380];
  HEAD.forEach((h, i) => els.push(text(XS[i], 158, h, { size: 10, stroke: T.inkSubtle })));
  const ROWS = [
    ['w₀', '−3', '12', '−3.12'],
    ['w₁', '−1', '−24', '−0.76'],
    ['w₂', '2', '36', '1.64'],
    ['b', '1', '12', '0.88'],
  ];
  ROWS.forEach((r, j) => {
    const y = 182 + j * 30;
    els.push(text(XS[0], y, r[0], { size: 12, family: 3, stroke: T.ink }));
    els.push(text(XS[1], y, r[1], { size: 12, family: 3, stroke: T.inkMuted }));
    els.push(text(XS[2], y, r[2], { size: 12, family: 3, stroke: T.accent }));
    els.push(text(XS[3], y, r[3], { size: 12, family: 3, stroke: T.ink }));
  });
  els.push(rule(56, 464, 312));
  els.push(text(62, 322, 'z: 6 → 4.20        L: 36 → 17.64', { size: 13, family: 3, stroke: T.success }));
  els.push(text(62, 344, 'w₁ had a negative gradient, so the rule raised it.',
    { size: 10, stroke: T.inkSubtle }));

  els.push(...card(520, 86, 400, 280, { spine: T.success }));
  els.push(text(542, 98, 'And then it collapses', { size: 15, stroke: T.success }));
  els.push(text(542, 120, 'the same step, repeated', { size: 11, stroke: T.inkSubtle }));
  els.push(rule(536, 904, 146));

  // L_t = 36 · 0.49^t, from z_{t+1} = 0.7 z_t.
  const X0 = 566, X1 = 886, Y0 = 316, YT = 168;
  els.push(line([[X0 - 8, Y0], [X1 + 8, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0, Y0 + 8], [X0, YT - 8]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  const pxT = (t) => X0 + (t / 20) * (X1 - X0);
  const pyL = (L) => Y0 - (L / 36) * (Y0 - YT);
  const pts = [];
  for (let i = 0; i <= 60; i++) {
    const t = (i / 60) * 20;
    pts.push([pxT(t), pyL(36 * Math.pow(0.49, t))]);
  }
  els.push(line(pts, { stroke: T.success, strokeWidth: 2.2, roughness: 0.3 }));
  [[0, '36'], [1, '17.6'], [2, '8.6'], [5, '1.0']].forEach(([t, lab]) => {
    const L = 36 * Math.pow(0.49, t);
    els.push(circle(pxT(t), pyL(L), 4, { stroke: T.success, fill: T.surface, strokeWidth: 1.4 }));
    els.push(text(pxT(t) - 24, pyL(L) - 20, lab,
      { size: 10, family: 3, stroke: T.success, align: 'center', width: 48 }));
  });
  els.push(text(X0 - 30, Y0 + 8, '0', { size: 10, stroke: T.inkSubtle, align: 'center', width: 30 }));
  els.push(text(X1 - 20, Y0 + 8, '20', { size: 10, stroke: T.inkSubtle, align: 'center', width: 40 }));
  els.push(text(542, 344, 'iterations · by 20 the squared loss has underflowed to 0.0000',
    { size: 10, stroke: T.inkSubtle }));

  els.push(rect(M, 394, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 406, 'Each step scales the pre-activation by 0.7, so the squared loss is multiplied by 0.49 every iteration.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 430, 'It approaches zero without reaching it: the gradient shrinks in lock-step with z, so the steps shrink too.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'One gradient-descent step on the single neuron, and the trajectory it starts',
    desc: 'A hand-drawn figure in two halves. The left card takes one update with a learning rate of '
      + 'nought point nought one: w nought goes from minus three to minus three point one two on a '
      + 'gradient of twelve, w one from minus one to minus nought point seven six on a gradient of '
      + 'minus twenty-four, w two from two to one point six four on a gradient of thirty-six, and '
      + 'the bias from one to nought point eight eight on a gradient of twelve. The pre-activation z '
      + 'falls from six to four point two and the loss from thirty-six to seventeen point six four, '
      + 'with a note that w one had a negative gradient so the rule raised it. The right card plots '
      + 'the loss over twenty iterations of the same step, marking thirty-six at the start, then '
      + 'seventeen point six, eight point six and one point nought, decaying geometrically towards '
      + 'zero and underflowing by iteration twenty. A band explains that each step scales the '
      + 'pre-activation by nought point seven so the squared loss is multiplied by nought point four '
      + 'nine each time, and that it approaches zero without reaching it because the gradient '
      + 'shrinks in lock-step with z.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { singleNeuronBackprop, oneStepAndAfter };

if (require.main === module) {
  emit('01-single-neuron-backprop', singleNeuronBackprop());
  emit('02-one-step-and-after', oneStepAndAfter());
}
