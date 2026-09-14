// The post 08 diagrams, hand-drawn.
//
// Scenes 01 and 02 mirror the clean vector figures one directory up. Scene 03 is
// new: section 2.2 claims two networks can share an accuracy and not a loss, and
// section 9 returns to it, but nothing showed the two numbers disagreeing.
//
//   01-cross-entropy-curve  960 x 500
//   02-indexing-methods     960 x 460
//   03-loss-vs-accuracy     960 x 470   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// The post's own worked batch: three samples, true classes 0, 1, 1.
const SOFTMAX = [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]];
const LABELS = [0, 1, 1];

function grid(els, x, y, vals, cw, ch, o = {}) {
  const rows = vals.length, cols = vals[0].length;
  els.push(rect(x, y, cols * cw, rows * ch, {
    stroke: o.c ?? T.ink, fill: T.surface, strokeWidth: o.strokeWidth ?? 1.4, roundness: null,
  }));
  for (let i = 1; i < cols; i++) {
    els.push(line([[x + i * cw, y], [x + i * cw, y + rows * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  for (let j = 1; j < rows; j++) {
    els.push(line([[x, y + j * ch], [x + cols * cw, y + j * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  const size = o.size ?? 11;
  vals.forEach((row, j) => row.forEach((v, i) => {
    els.push(text(x + i * cw, y + j * ch + (ch - size) / 2 - 1, String(v), {
      size, family: 3, stroke: o.ink ?? T.ink, align: 'center', width: cw,
    }));
  }));
  return { w: cols * cw, h: rows * ch, bottom: y + rows * ch };
}

// ---------------------------------------------------------------- diagram 1
// The curve is sampled densely rather than drawn as a few line segments,
// because the whole argument is the steepness near zero and a coarse polyline
// flattens exactly the part worth seeing.
function crossEntropyCurve() {
  resetSeq();
  const W = 960, H = 500;

  const els = [...heading('The −log curve',
    'Loss against the probability the network put on the right answer. Nothing else enters the formula.')];

  const X0 = 90, X1 = 560, Y0 = 380, LMAX = 5, YSPAN = 270;
  const px = (p) => X0 + p * (X1 - X0);
  const py = (l) => Y0 - Math.min(l, LMAX) / LMAX * YSPAN;

  els.push(line([[X0 - 12, Y0], [X1 + 16, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  els.push(line([[X0, Y0 + 12], [X0, 106]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  els.push(text(X1 - 30, Y0 + 12, 'p', { size: 12, stroke: T.inkSubtle }));
  els.push(text(X0 - 66, 104, '−log(p)', { size: 12, stroke: T.inkSubtle }));

  [0, 1, 2, 3, 4].forEach((l) => {
    els.push(text(X0 - 40, py(l) - 6, String(l), { size: 10, stroke: T.inkSubtle, align: 'right', width: 30 }));
    els.push(line([[X0 - 5, py(l)], [X0 + 5, py(l)]], { stroke: T.border, strokeWidth: 1, roughness: 0.2 }));
  });
  [0.25, 0.5, 0.75, 1].forEach((p) => {
    els.push(text(px(p) - 20, Y0 + 10, String(p), { size: 10, stroke: T.inkSubtle, align: 'center', width: 40 }));
    els.push(line([[px(p), Y0 - 5], [px(p), Y0 + 5]], { stroke: T.border, strokeWidth: 1, roughness: 0.2 }));
  });

  // 60 samples from 0.01 to 1: enough that the near-vertical stretch below
  // p = 0.1 stays vertical instead of being cut off by a straight chord.
  const pts = [];
  for (let i = 0; i <= 60; i++) {
    const p = 0.01 + (i / 60) * 0.99;
    pts.push([px(p), py(-Math.log(p))]);
  }
  els.push(line(pts, { stroke: T.primary, strokeWidth: 2.4, roughness: 0.3 }));

  const REF = [[0.1, 2.303], [0.3, 1.204], [0.5, 0.693], [0.7, 0.357], [0.9, 0.105]];
  REF.forEach(([p, l]) => {
    els.push(circle(px(p), py(l), 4.5, { stroke: T.accent, fill: T.surface, strokeWidth: 1.5 }));
    els.push(text(px(p) - 34, py(l) - 24, l.toFixed(3),
      { size: 10, family: 3, stroke: T.accent, align: 'center', width: 68 }));
  });

  els.push(...card(600, 110, 320, 270, { spine: T.primary }));
  els.push(text(622, 122, 'Two properties', { size: 15, stroke: T.primary }));
  els.push(rule(614, 906, 150));
  els.push(text(622, 160, 'Loss is never negative.', { size: 12, stroke: T.ink }));
  els.push(text(622, 180, 'p lies in (0, 1], so log(p) ≤ 0.', { size: 10, stroke: T.inkMuted }));
  els.push(text(622, 214, 'Loss is zero only at p = 1.', { size: 12, stroke: T.ink }));
  els.push(text(622, 234, 'Any doubt at all costs something.', { size: 10, stroke: T.inkMuted }));
  els.push(rule(614, 906, 264));
  els.push(text(622, 274, 'p = 0.50   →   0.693', { size: 11, family: 3, stroke: T.inkMuted }));
  els.push(text(622, 296, 'p = 0.10   →   2.303', { size: 11, family: 3, stroke: T.inkMuted }));
  els.push(text(622, 318, 'p = 0.01   →   4.605', { size: 11, family: 3, stroke: T.alert }));
  els.push(text(622, 344, 'Confident and wrong costs most.', { size: 10, stroke: T.inkSubtle }));

  els.push(text(M, 420, 'The steepness near zero is the training signal: a confident wrong answer produces a far larger gradient than an unsure one,',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 442, 'so the weights are pulled hardest towards fixing the mistakes the network was most certain about.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'The negative-log curve: cross-entropy loss against predicted probability',
    desc: 'A hand-drawn line chart of minus log p against the probability p the network assigned to '
      + 'the correct class, for p from nought point nought one to one. The curve is very high near '
      + 'zero, passing above four and a half at p equals nought point nought one, and falls smoothly '
      + 'to exactly zero at p equals one. Five reference dots mark nought point one at two point '
      + 'three nought three, nought point three at one point two nought four, nought point five at '
      + 'nought point six nine three, nought point seven at nought point three five seven, and '
      + 'nought point nine at nought point one nought five. A side panel gives two properties: loss '
      + 'is never negative because p lies in nought to one so its log is at most zero, and loss is '
      + 'zero only at p equals one so any doubt at all costs something. A footer explains that the '
      + 'steepness near zero is the training signal, since a confident wrong answer produces a far '
      + 'larger gradient than an unsure one.',
  };
}

// ---------------------------------------------------------------- diagram 2
// Both routes end on the same three numbers, printed identically at the foot of
// each panel. That repetition is the point: the label format changes the
// ergonomics of the lookup and nothing about the arithmetic.
function indexingMethods() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('Two label formats, one lookup',
    'Pulling the correct-class probability out of a softmax matrix. The format decides the syntax, not the answer.')];

  const CW = 52, CH = 30;

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 290, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 122, fit(o.code, 11, 380 * 0.94, 'code'),
      { size: 11, family: 3, stroke: T.ink }));
    els.push(rule(x + 16, x + 404, 148));
  };

  // --- integer labels: pick one cell per row
  panel(M, {
    c: T.primary, title: 'Integer labels', code: 'softmax[range(N), labels]',
  });
  els.push(text(62, 164, 'labels', { size: 10, stroke: T.inkSubtle }));
  grid(els, 62, 182, LABELS.map((l) => [l]), 44, CH, { c: T.primary });
  els.push(text(130, 164, 'softmax', { size: 10, stroke: T.inkSubtle }));
  grid(els, 130, 182, SOFTMAX, CW, CH);
  LABELS.forEach((l, j) => {
    els.push(rect(130 + l * CW, 182 + j * CH, CW, CH,
      { stroke: T.primary, strokeWidth: 2, roundness: null }));
  });
  els.push(arrow([[300, 227], [332, 227]], { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
  els.push(text(344, 164, 'result', { size: 10, stroke: T.inkSubtle }));
  grid(els, 344, 182, [[0.7], [0.5], [0.9]], 52, CH, { c: T.primary });
  els.push(text(62, 292, 'One cell per row, chosen by its column number.',
    { size: 11, stroke: T.inkMuted }));
  els.push(text(62, 316, 'range(N) walks the rows; labels walks the columns.',
    { size: 10, stroke: T.inkSubtle }));
  els.push(text(62, 344, '[0.7, 0.5, 0.9]', { size: 14, family: 3, stroke: T.primary }));

  // --- one-hot labels: zero the wrong columns, then sum the row
  panel(500, {
    c: T.accent, title: 'One-hot labels', code: 'np.sum(softmax * onehot, axis=1)',
  });
  const ONEHOT = LABELS.map((l) => [0, 1, 2].map((i) => (i === l ? 1 : 0)));
  const MASKED = SOFTMAX.map((row, j) => row.map((v, i) => (ONEHOT[j][i] ? v : 0)));
  els.push(text(522, 164, 'softmax', { size: 10, stroke: T.inkSubtle }));
  grid(els, 522, 182, SOFTMAX, 40, CH, { size: 9 });
  els.push(text(646, 218, '×', { size: 15, stroke: T.inkMuted, align: 'center', width: 20 }));
  els.push(text(670, 164, 'one-hot', { size: 10, stroke: T.inkSubtle }));
  grid(els, 670, 182, ONEHOT, 34, CH, { c: T.accent, size: 10 });
  els.push(text(778, 218, '=', { size: 15, stroke: T.inkMuted, align: 'center', width: 20 }));
  els.push(text(802, 164, 'masked', { size: 10, stroke: T.inkSubtle }));
  grid(els, 802, 182, MASKED, 34, CH, { c: T.accent, size: 9 });
  els.push(text(522, 292, 'Wrong-class columns are multiplied to zero,',
    { size: 11, stroke: T.inkMuted }));
  els.push(text(522, 312, 'then each row is summed, leaving one number per row.',
    { size: 11, stroke: T.inkMuted }));
  els.push(text(522, 344, '[0.7, 0.5, 0.9]', { size: 14, family: 3, stroke: T.accent }));

  els.push(rect(M, 400, 880, 52, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 414, 'Identical output. The label format changes the ergonomics of the lookup and nothing about the arithmetic.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Two label formats, two ways to extract the correct-class probability',
    desc: 'Two hand-drawn panels pulling the same three numbers out of the same softmax matrix. The '
      + 'left panel uses integer labels nought, one, one with advanced indexing, softmax indexed by '
      + 'range N and labels: one cell per row is outlined, row nought column nought at nought point '
      + 'seven, row one column one at nought point five, and row two column one at nought point '
      + 'nine, with range N walking the rows and labels walking the columns. The right panel uses '
      + 'one-hot labels and an element-wise multiply that zeroes the wrong-class columns, followed '
      + 'by a per-row sum; the softmax, one-hot and masked grids are shown in turn. Both panels end '
      + 'on the identical result nought point seven, nought point five, nought point nine, and a '
      + 'band notes that the label format changes the ergonomics of the lookup and nothing about the '
      + 'arithmetic.',
  };
}

// ---------------------------------------------------------------- diagram 3
// New. Section 2.2 asserts that two networks can share an accuracy without
// sharing a loss, and section 9 returns to it. Both batches here are right on
// every sample, so accuracy cannot separate them and the whole difference lands
// in the loss, which is what the section is claiming.
function lossVsAccuracy() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Same accuracy, different loss',
    'Two batches, both right on all three samples. Accuracy cannot tell them apart and the loss can.')];

  const CW = 58, CH = 32;

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 288, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 120, fit(o.sub, 11, 380, 'panel sub'), { size: 11, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + 404, 146));

    els.push(text(x + 24, 158, 'softmax output', { size: 10, stroke: T.inkSubtle }));
    grid(els, x + 24, 176, o.probs, CW, CH, { size: 11 });
    // The cell the loss actually reads, outlined on every row.
    LABELS.forEach((l, j) => {
      els.push(rect(x + 24 + l * CW, 176 + j * CH, CW, CH,
        { stroke: o.c, strokeWidth: 2, roundness: null }));
    });
    LABELS.forEach((l, j) => {
      els.push(text(x + 208, 176 + j * CH + 9, `true ${l}`, { size: 10, stroke: T.inkSubtle }));
    });

    els.push(text(x + 280, 158, 'accuracy', { size: 10, stroke: T.inkSubtle }));
    els.push(text(x + 280, 176, '3 / 3', { size: 17, family: 3, stroke: T.inkMuted }));
    els.push(text(x + 280, 202, '100%', { size: 13, family: 3, stroke: T.inkMuted }));
    els.push(text(x + 280, 236, 'mean loss', { size: 10, stroke: T.inkSubtle }));
    els.push(text(x + 280, 254, o.loss, { size: 20, family: 3, stroke: o.c }));

    els.push(rule(x + 16, x + 404, 296));
    els.push(text(x + 24, 306, fit(o.per, 10, 380 * 0.94, 'per-sample'),
      { size: 10, family: 3, stroke: T.inkMuted }));
    els.push(text(x + 24, 344, fit(o.verdict, 11, 380, 'verdict'), { size: 11, stroke: T.inkMuted }));
  };

  panel(M, {
    c: T.warn, title: 'Barely right', sub: 'the worked batch from section 4',
    probs: SOFTMAX, loss: '0.385',
    per: '−log(0.70)=0.357  −log(0.50)=0.693  −log(0.90)=0.105',
    verdict: 'Sample 2 is correct on a coin flip and costs the most.',
  });

  panel(500, {
    c: T.success, title: 'Confidently right', sub: 'the same three answers, held firmly',
    probs: [[0.95, 0.03, 0.02], [0.02, 0.96, 0.02], [0.01, 0.97, 0.02]], loss: '0.041',
    per: '−log(0.95)=0.051  −log(0.96)=0.041  −log(0.97)=0.030',
    verdict: 'Same predictions, nine times less loss to work with.',
  });

  els.push(rect(M, 396, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 408, 'Accuracy counts how often the argmax is right. Loss measures how much probability mass sat on the right answer.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 432, 'Accuracy is the number a reader understands; loss is the one that can be differentiated, which is why training optimises it instead.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Loss and accuracy are not the same measurement',
    desc: 'Two hand-drawn panels holding two batches of three softmax rows, with true classes nought, '
      + 'one and one, and the cell the loss reads outlined on every row. The left panel, the worked '
      + 'batch from section 4, has correct-class probabilities of nought point seven, nought point '
      + 'five and nought point nine, giving per-sample losses of nought point three five seven, '
      + 'nought point six nine three and nought point one nought five, and a mean loss of nought '
      + 'point three eight five; its second sample is correct only on a coin flip and costs the '
      + 'most. The right panel holds the same three predictions far more confidently, at nought '
      + 'point nine five, nought point nine six and nought point nine seven, for a mean loss of '
      + 'nought point nought four one. Both panels report an accuracy of three out of three, one '
      + 'hundred per cent. A band notes that accuracy counts how often the argmax is right while '
      + 'loss measures how much probability mass sat on the right answer, and that loss is the one '
      + 'that can be differentiated, which is why training optimises it instead.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { crossEntropyCurve, indexingMethods, lossVsAccuracy };

if (require.main === module) {
  emit('01-cross-entropy-curve', crossEntropyCurve());
  emit('02-indexing-methods', indexingMethods());
  emit('03-loss-vs-accuracy', lossVsAccuracy());
}
