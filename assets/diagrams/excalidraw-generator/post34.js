// The post 34 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 6 turns the whole of Parts 06 to 34 into one decision about the last
// layer, and it contains a genuine trap — multi-label is sigmoid per output, not
// softmax — that a pipeline diagram of the binary case cannot warn about.
//
//   01-sigmoid-bce-pipeline  960 x 540
//   02-choosing-the-head     960 x 470   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// The forward chain across the top and the combined backward beneath it, with
// the cancellation stated rather than implied: the whole reason this head is
// worth its own post is that the division disappears.
function sigmoidBcePipeline() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('One neuron, one probability, one gradient',
    'The binary head. Same shape as softmax with cross-entropy, and half the arithmetic.')];

  const STAGES = [
    { x: 50, w: 120, t: 'z', s: 'one logit', c: T.primary },
    { x: 210, w: 200, t: 'σ(z)', s: '1 / (1 + e⁻ᶻ)', c: T.accent },
    { x: 450, w: 200, t: 'p', s: 'in (0, 1)', c: T.accent },
    { x: 690, w: 230, t: 'BCE(p, y)', s: 'y is 0 or 1', c: T.alert },
  ];
  STAGES.forEach((s, i) => {
    els.push(...card(s.x, 110, s.w, 76, { stroke: s.c, spine: s.c }));
    els.push(text(s.x, 126, s.t, { size: 17, family: 3, stroke: s.c, align: 'center', width: s.w }));
    els.push(text(s.x, 156, s.s, { size: 10, family: 3, stroke: T.inkMuted, align: 'center', width: s.w }));
    if (i < 3) {
      els.push(arrow([[s.x + s.w + 4, 148], [STAGES[i + 1].x - 4, 148]],
        { stroke: T.primary, strokeWidth: 1.3, roughness: 0.4 }));
    }
  });
  els.push(text(50, 90, 'forward', { size: 10, stroke: T.primary }));

  els.push(...card(M, 216, 880, 96, { spine: T.success }));
  els.push(text(62, 228, 'The combined backward', { size: 14, stroke: T.success }));
  els.push(text(62, 254, '∂L/∂z  =  (p − y) / N', { size: 18, family: 3, stroke: T.ink }));
  els.push(text(360, 250, 'Taken separately, BCE’s backward divides by p(1 − p) and sigmoid’s',
    { size: 10, stroke: T.inkMuted }));
  els.push(text(360, 268, 'multiplies by it. Fused, the two cancel and no division survives.',
    { size: 10, stroke: T.inkMuted }));
  els.push(text(360, 290, 'The same cancellation as softmax with cross-entropy in Part 19.',
    { size: 10, stroke: T.inkSubtle }));

  els.push(...card(M, 336, 420, 128, { spine: T.alert }));
  els.push(text(62, 348, 'The overflow trap', { size: 13, stroke: T.alert }));
  els.push(text(62, 372, '1 / (1 + np.exp(-z))', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(62, 392, 'overflows for z below about −700.', { size: 10, stroke: T.inkMuted }));
  els.push(text(62, 416, 'Split on the sign: use eᶻ / (1 + eᶻ) when z < 0,', { size: 10, stroke: T.inkMuted }));
  els.push(text(62, 432, 'so exp never sees a large positive argument.', { size: 10, stroke: T.inkMuted }));

  els.push(...card(500, 336, 420, 128, { spine: T.primary }));
  els.push(text(522, 348, 'Against 2-class softmax', { size: 13, stroke: T.primary }));
  [
    ['output neurons', '2', '1'],
    ['target format', 'one-hot (N, 2)', 'scalar (N,)'],
    ['forward cost', '2 exp, 2 div, 1 log', '1 exp, 1 div, 1 log'],
  ].forEach((r, i) => {
    const y = 374 + i * 26;
    els.push(text(522, y, r[0], { size: 9, stroke: T.inkSubtle }));
    els.push(text(650, y, r[1], { size: 9, family: 3, stroke: T.inkMuted }));
    els.push(text(790, y, r[2], { size: 9, family: 3, stroke: T.primary }));
  });
  els.push(text(650, 356, 'softmax', { size: 9, stroke: T.inkSubtle }));
  els.push(text(790, 356, 'sigmoid', { size: 9, stroke: T.primary }));

  els.push(text(M, 492, 'The two are mathematically identical at two classes. The sigmoid form is half the output and one line less algebra.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Sigmoid and binary cross-entropy, with the combined backward shortcut',
    desc: 'A hand-drawn figure. The forward chain runs left to right: a single logit z enters a '
      + 'sigmoid computing one over one plus e to the minus z, producing a probability p in the open '
      + 'interval nought to one, which binary cross-entropy compares against a target y of nought or '
      + 'one. A card beneath gives the combined backward, the derivative of the loss with respect to '
      + 'z being p minus y over N, and explains that BCE’s backward divides by p times one minus p '
      + 'while sigmoid’s multiplies by it, so fused the two cancel and no division survives, the '
      + 'same cancellation as softmax with cross-entropy in Part 19. Two further cards cover the '
      + 'overflow trap, where the naive one over one plus exp of minus z overflows below about minus '
      + 'seven hundred and the fix is to split on the sign of z, and a comparison against two-class '
      + 'softmax showing one output neuron instead of two, scalar targets instead of one-hot, and '
      + 'half the forward cost.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 6 collapses everything from Part 06 onward into one choice, and
// the third column is a real trap: multi-label looks like multi-class and needs
// the binary head, because softmax forces the outputs to sum to one.
function choosingTheHead() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Choosing the last layer',
    'Everything from Part 06 onward comes down to this. One of the four columns is a trap.')];

  const HEADS = [
    {
      c: T.success, name: 'Binary', classes: '2 classes',
      out: '1 neuron', act: 'sigmoid', loss: 'BCE', tgt: 'scalar 0 or 1',
      note: 'One number is enough:\n1 − p is the other class,\nimplicitly.', part: 'Part 34',
    },
    {
      c: T.primary, name: 'Multi-class', classes: '3+ classes',
      out: 'K neurons', act: 'softmax', loss: 'CCE', tgt: 'one-hot or index',
      note: 'Softmax forces the K\noutputs to sum to 1, which\nis exactly right here.', part: 'Part 19',
    },
    {
      c: T.alert, name: 'Multi-label', classes: 'K labels at once',
      out: 'K neurons', act: 'sigmoid ×K', loss: 'BCE, summed', tgt: 'K independent 0/1',
      note: 'The trap. Softmax would\nforce the labels to compete\nfor one unit of probability.', part: 'Part 34, K times',
    },
    {
      c: T.warn, name: 'Regression', classes: 'continuous',
      out: '1 or more', act: 'none', loss: 'MSE', tgt: 'a real number',
      note: 'No activation at all. The\nnetwork output is already\nthe prediction.', part: 'project 04',
    },
  ];

  const CWD = 205, TOP = 90, CHT = 258;
  HEADS.forEach((h, i) => {
    const x = M + i * 225, inner = CWD - 32;
    els.push(...card(x, TOP, CWD, CHT, { spine: h.c }));
    els.push(text(x + 16, TOP + 12, h.name, { size: 14, stroke: h.c }));
    els.push(text(x + 16, TOP + 34, h.classes, { size: 10, stroke: T.inkSubtle }));
    els.push(rule(x + 14, x + CWD - 14, TOP + 56));
    [['output', h.out], ['activation', h.act], ['loss', h.loss], ['target', h.tgt]].forEach((r, j) => {
      const y = TOP + 68 + j * 30;
      els.push(text(x + 16, y, r[0], { size: 9, stroke: T.inkSubtle }));
      els.push(text(x + 16, y + 12, fit(r[1], 11, inner * 0.94, 'head field'),
        { size: 11, family: 3, stroke: j === 1 ? h.c : T.ink }));
    });
    els.push(rule(x + 14, x + CWD - 14, TOP + 192));
    els.push(text(x + 16, TOP + 202, fit(h.note, 9, inner, 'head note'), { size: 9, stroke: T.inkMuted }));
    els.push(text(x + 16, TOP + 240, h.part, { size: 9, stroke: h.c }));
  });

  els.push(rect(M, 374, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 386, 'Multi-label is the one that catches people: K classes, but a sample can carry several of them at once.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 410, 'Softmax makes the outputs sum to 1, which asserts exactly one positive label. K independent sigmoids assert nothing of the kind.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Choosing the last-layer activation and loss for the task',
    desc: 'Four hand-drawn cards, one per task type. Binary, for two classes, uses one output neuron '
      + 'with a sigmoid and binary cross-entropy against a scalar target of nought or one, because '
      + 'one number is enough and one minus p is the other class implicitly; this is Part 34. '
      + 'Multi-class, for three or more classes, uses K neurons with softmax and categorical '
      + 'cross-entropy against a one-hot or index target, because softmax forces the K outputs to '
      + 'sum to one which is exactly right there; this is Part 19. Multi-label, where K labels can '
      + 'apply at once, is marked as the trap: it uses K neurons but K independent sigmoids and a '
      + 'summed binary cross-entropy against K independent nought-or-one targets, because softmax '
      + 'would force the labels to compete for one unit of probability. Regression uses no '
      + 'activation at all and mean squared error against a real number, since the network output is '
      + 'already the prediction. A band restates the trap: softmax asserts exactly one positive '
      + 'label, and K independent sigmoids assert nothing of the kind.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { sigmoidBcePipeline, choosingTheHead };

if (require.main === module) {
  emit('01-sigmoid-bce-pipeline', sigmoidBcePipeline());
  emit('02-choosing-the-head', choosingTheHead());
}
