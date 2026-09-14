// The post 19 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 2.2 gives two decisive reasons every framework ships the fused op, and
// the cost argument in particular only lands once the numbers are written down —
// at fifty thousand classes the Jacobian route is fifty thousand times the work.
//
//   01-combined-shortcut  960 x 500
//   02-why-fused          960 x 460   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

function grid(els, x, y, vals, cw, ch, o = {}) {
  const rows = vals.length, cols = vals[0].length;
  els.push(rect(x, y, cols * cw, rows * ch, {
    stroke: o.c ?? T.ink, fill: T.surface, strokeWidth: 1.5, roundness: null,
  }));
  for (let i = 1; i < cols; i++) {
    els.push(line([[x + i * cw, y], [x + i * cw, y + rows * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  for (let j = 1; j < rows; j++) {
    els.push(line([[x, y + j * ch], [x + cols * cw, y + j * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  const size = o.size ?? 9;
  vals.forEach((row, j) => row.forEach((v, i) => {
    els.push(text(x + i * cw, y + j * ch + (ch - size) / 2 - 1, String(v), {
      size, family: 3, stroke: o.ink ?? T.ink, align: 'center', width: cw,
    }));
  }));
}

// ---------------------------------------------------------------- diagram 1
// The two routes end on the same numbers, so the figure has to make the cost of
// getting there the visible difference: a full matrix on the left, a single
// subtraction on the right, at the same scale.
function combinedShortcut() {
  resetSeq();
  const W = 960, H = 500;

  const els = [...heading('Two routes to the same gradient',
    'Both are correct. One builds a dense Jacobian per sample; the other never builds anything.')];

  els.push(...card(M, 86, 420, 300, { spine: T.alert }));
  els.push(text(62, 98, 'Option 1 · differentiate each op', { size: 14, stroke: T.alert }));
  els.push(text(62, 122, 'softmax backward, then loss backward', { size: 10, stroke: T.inkSubtle }));
  els.push(rule(56, 444, 148));
  els.push(text(62, 158, 'build ∂A/∂Z, one C×C matrix per sample', { size: 10, stroke: T.ink }));
  grid(els, 140, 180, [
    ['A₁(1−A₁)', '−A₁A₂', '−A₁A₃'],
    ['−A₂A₁', 'A₂(1−A₂)', '−A₂A₃'],
    ['−A₃A₁', '−A₃A₂', 'A₃(1−A₃)'],
  ], 60, 34, { c: T.alert });
  els.push(text(62, 292, 'then multiply it by  −y / ŷ', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(62, 318, 'A matrix product per sample.', { size: 10, stroke: T.inkMuted }));
  els.push(text(62, 336, 'O(N C²) time, O(C²) memory.', { size: 10, stroke: T.alert }));
  els.push(text(62, 360, 'Correct, and the way nobody does it.', { size: 10, stroke: T.inkSubtle }));

  els.push(...card(500, 86, 420, 300, { spine: T.success }));
  els.push(text(522, 98, 'Option 2 · substitute, then differentiate', { size: 14, stroke: T.success }));
  els.push(text(522, 122, 'one expression, differentiated once', { size: 10, stroke: T.inkSubtle }));
  els.push(rule(516, 904, 148));
  const STEPS = [
    'L  =  −log( ŷₜ )',
    '   =  −Zₜ  +  log Σⱼ eᶻʲ',
    '',
    '∂L/∂Zₖ  =  −yₖ  +  ŷₖ',
    '        =  ŷₖ − yₖ',
  ];
  STEPS.forEach((s, i) => els.push(text(544, 162 + i * 24, s,
    { size: 12, family: 3, stroke: i >= 3 ? T.success : T.ink })));
  els.push(rule(516, 904, 292));
  els.push(text(522, 302, 'The exponentials cancel before anything', { size: 10, stroke: T.inkMuted }));
  els.push(text(522, 318, 'is computed. No Jacobian is ever built.', { size: 10, stroke: T.inkMuted }));
  els.push(text(522, 342, 'O(N C) time, no extra memory.', { size: 10, stroke: T.success }));
  els.push(text(522, 366, 'Correct, and what every framework ships.', { size: 10, stroke: T.inkSubtle }));

  els.push(rect(M, 412, 880, 74, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(56, 424, 'self.dinputs = dvalues.copy()', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(56, 442, 'self.dinputs[range(samples), y_true] -= 1', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(56, 460, 'self.dinputs /= samples', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(520, 432, 'Three lines, and the second one is the whole', { size: 11, stroke: T.inkMuted }));
  els.push(text(520, 452, 'derivation: subtract 1 where the label is 1.', { size: 11, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Softmax and cross-entropy backward: the full Jacobian against the clean shortcut',
    desc: 'Two hand-drawn panels reaching the same gradient. The left, in alert red and labelled '
      + 'option one, differentiates each operation in turn: it builds the softmax Jacobian, a dense '
      + 'C by C matrix per sample with A k times one minus A k on the diagonal and minus A k A j off '
      + 'it, then multiplies it by minus y over y-hat. It is annotated as a matrix product per '
      + 'sample, order N C squared in time and C squared in memory, correct and the way nobody does '
      + 'it. The right, in success green and labelled option two, substitutes first: the loss minus '
      + 'log of y-hat at the true class becomes minus Z at the true class plus the log of the sum of '
      + 'exponentials, whose derivative is minus y k plus y-hat k, which is y-hat k minus y k. It is '
      + 'annotated as having the exponentials cancel before anything is computed, order N C in time '
      + 'with no extra memory. A band gives the three-line implementation and notes that the second '
      + 'line is the whole derivation: subtract one where the label is one.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 2.2 gives two reasons the fused op exists. The speed one is a
// numeric argument that prose flattens — "quadratic against linear" sounds like
// a detail until the class counts of real tasks are written next to it.
function whyFused() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('Why every framework fuses these two operations',
    'The shortcut is not a tidiness argument. At realistic class counts the explicit route stops being runnable.')];

  els.push(...card(M, 86, 520, 280, { spine: T.alert }));
  els.push(text(62, 98, 'Numbers per sample, per backward step', { size: 14, stroke: T.alert }));
  els.push(rule(56, 544, 128));

  const HEAD = ['classes C', 'Jacobian · C²', 'shortcut · C', 'ratio'];
  const XS = [62, 190, 350, 470];
  HEAD.forEach((h, i) => els.push(text(XS[i], 140, h, { size: 10, stroke: T.inkSubtle })));
  const ROWS = [
    ['3', '9', '3', '3×', 'the spiral set'],
    ['1 000', '1 000 000', '1 000', '1 000×', 'ImageNet'],
    ['50 000', '2 500 000 000', '50 000', '50 000×', 'a language model vocabulary'],
  ];
  ROWS.forEach((r, j) => {
    const y = 168 + j * 56;
    els.push(text(XS[0], y, r[0], { size: 13, family: 3, stroke: T.ink }));
    els.push(text(XS[1], y, r[1], { size: 13, family: 3, stroke: T.alert }));
    els.push(text(XS[2], y, r[2], { size: 13, family: 3, stroke: T.success }));
    els.push(text(XS[3], y, r[3], { size: 13, family: 3, stroke: T.ink }));
    els.push(text(XS[0], y + 20, fit(r[4], 10, 460, 'row note'), { size: 10, stroke: T.inkSubtle }));
    if (j < 2) els.push(rule(56, 544, y + 40));
  });
  els.push(text(62, 336, 'O(N C²) against O(N C). The gap is the class count itself.',
    { size: 11, stroke: T.inkMuted }));

  els.push(...card(600, 86, 320, 280, { spine: T.success }));
  els.push(text(622, 98, 'And it is more stable', { size: 14, stroke: T.success }));
  els.push(rule(614, 904, 128));
  els.push(text(622, 142, 'Computed separately, the two ops', { size: 11, stroke: T.ink }));
  els.push(text(622, 160, 'materialise an intermediate that can', { size: 11, stroke: T.ink }));
  els.push(text(622, 178, 'underflow before the log sees it.', { size: 11, stroke: T.ink }));
  els.push(rule(614, 904, 204));
  els.push(text(622, 216, 'log( softmax(Z) )', { size: 12, family: 3, stroke: T.inkMuted }));
  els.push(text(622, 240, '=  Zₖ  −  log Σⱼ eᶻʲ', { size: 12, family: 3, stroke: T.success }));
  els.push(text(622, 268, 'The right-hand side is the log-sum-exp', { size: 10, stroke: T.inkSubtle }));
  els.push(text(622, 284, 'trick, which is Part 06’s max subtraction', { size: 10, stroke: T.inkSubtle }));
  els.push(text(622, 300, 'wearing a different hat.', { size: 10, stroke: T.inkSubtle }));
  els.push(text(622, 328, 'The unstable middle step never exists.', { size: 11, stroke: T.success }));

  els.push(rect(M, 392, 880, 56, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 404, 'PyTorch ships it as nn.CrossEntropyLoss; TensorFlow as softmax_cross_entropy_with_logits.',
    { size: 12, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 428, 'Both take raw logits rather than probabilities, for exactly the two reasons above.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Why softmax and cross-entropy are shipped as one fused operation',
    desc: 'A hand-drawn figure in two cards. The left tabulates the numbers each route computes per '
      + 'sample per backward step at three class counts: at three classes, as in the spiral set, the '
      + 'Jacobian needs nine and the shortcut three, a ratio of three; at a thousand classes, as in '
      + 'ImageNet, a million against a thousand; and at fifty thousand, a language-model vocabulary, '
      + 'two and a half billion against fifty thousand, a ratio of fifty thousand. It notes that the '
      + 'gap between order N C squared and order N C is the class count itself. The right card gives '
      + 'the stability reason: computed separately the two operations materialise an intermediate '
      + 'that can underflow before the logarithm sees it, whereas the log of softmax rewritten as Z '
      + 'k minus the log of the sum of exponentials is the log-sum-exp trick, which is Part 06’s max '
      + 'subtraction in a different form, and the unstable middle step never exists. A band notes '
      + 'that PyTorch ships this as nn.CrossEntropyLoss and TensorFlow as '
      + 'softmax_cross_entropy_with_logits, both taking raw logits rather than probabilities for '
      + 'exactly these two reasons.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { combinedShortcut, whyFused };

if (require.main === module) {
  emit('01-combined-shortcut', combinedShortcut());
  emit('02-why-fused', whyFused());
}
