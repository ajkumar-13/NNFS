// The post 31 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 2.2 gives dropout a second, statistical reading — every step samples
// one of 2ⁿ subnetworks that all share one set of weights — and the existing
// figure only ever draws one mask.
//
//   01-dropout-train-vs-test  960 x 540
//   02-implicit-ensemble      960 x 470   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// A neuron, live or dropped. A dropped one keeps its outline and gains a cross,
// because it is still part of the layer — it is switched off for this step, not
// removed from the architecture.
function unit(els, cx, cy, r, live, c) {
  els.push(circle(cx, cy, r, {
    stroke: live ? c : T.border, fill: T.surface, strokeWidth: live ? 1.6 : 1.1,
  }));
  if (!live) {
    const d = r * 0.62;
    els.push(line([[cx - d, cy - d], [cx + d, cy + d]], { stroke: T.alert, strokeWidth: 1.4, roughness: 0.5 }));
    els.push(line([[cx - d, cy + d], [cx + d, cy - d]], { stroke: T.alert, strokeWidth: 1.4, roughness: 0.5 }));
  }
}

// ---------------------------------------------------------------- diagram 1
// The scale factor is written on the surviving units, not just named in a
// caption: inverted dropout is the reason the two panels can be different at
// all while the layer after them stays the same.
function dropoutTrainVsTest() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Dropout is two different layers',
    'One during training, one during evaluation. Inverted dropout is what lets the layer after them not care which.')];

  const MASK = [1, 0, 1, 1, 0, 1, 1, 1];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 300, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 122, o.sub, { size: 10, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + 404, 148));

    els.push(text(x + 20, 162, 'incoming', { size: 9, stroke: T.inkSubtle }));
    MASK.forEach((m, i) => {
      const cx = x + 70 + i * 42;
      els.push(text(cx - 20, 178, '1.0', { size: 9, family: 3, stroke: T.inkMuted, align: 'center', width: 40 }));
      unit(els, cx, 218, 15, o.train ? m === 1 : true, o.c);
      els.push(text(cx - 21, 246, o.out(m), { size: 9, family: 3, stroke: o.c, align: 'center', width: 42 }));
    });
    els.push(text(x + 20, 246, 'out', { size: 9, stroke: T.inkSubtle }));

    els.push(rule(x + 16, x + 404, 276));
    els.push(text(x + 20, 288, o.sum, { size: 12, family: 3, stroke: T.ink }));
    els.push(text(x + 20, 316, fit(o.note, 10, 380, 'panel note'), { size: 10, stroke: T.inkMuted }));
  };

  panel(M, {
    c: T.accent, train: true, title: 'Training  ·  mask and rescale', sub: 'rate p = 0.2, so 1 / (1 − p) = 1.25',
    out: (m) => (m ? '1.25' : '0'),
    sum: 'sum  =  6 × 1.25  =  7.5',
    note: 'Two units are switched off this step, and the survivors are\nscaled up so the layer’s expected output is unchanged.',
  });

  panel(500, {
    c: T.success, train: false, title: 'Evaluation  ·  nothing at all', sub: 'the mask is not applied',
    out: () => '1.0',
    sum: 'sum  =  8 × 1.0  =  8.0',
    note: 'No mask, no rescale, no branch in the arithmetic.\nThe layer is used exactly as it is.',
  });

  els.push(rect(M, 412, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 424, 'Expected training output is 8 × 0.8 × 1.25 = 8, which is the evaluation output. That equality is the whole trick.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 448, 'Vanilla dropout instead scales at test time. Inverted dropout puts the cost in training, where it is already being paid.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 498, 'Dropped units keep their outlines here. They are switched off for one step, not removed from the architecture.',
    { size: 11, stroke: T.inkSubtle }));

  return {
    W, H, els,
    title: 'Dropout in training mode against testing mode',
    desc: 'Two hand-drawn panels over the same row of eight neurons, each receiving an incoming '
      + 'activation of one. The left panel, training mode at a dropout rate of nought point two so '
      + 'the scale factor is one point two five, crosses out two of the eight and marks the six '
      + 'survivors as outputting one point two five each, for a sum of seven point five; a note '
      + 'records that the survivors are scaled up so the layer’s expected output is unchanged. The '
      + 'right panel, evaluation mode, applies no mask at all: all eight output one point nought for '
      + 'a sum of eight, with no rescale and no branch in the arithmetic. A band notes that the '
      + 'expected training output, eight times nought point eight times one point two five, equals '
      + 'the evaluation output of eight, and that vanilla dropout instead scales at test time where '
      + 'inverted dropout puts the cost in training. A footer observes that dropped units keep their '
      + 'outlines because they are switched off for one step rather than removed from the '
      + 'architecture.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 2.2 reads dropout as sampling from an ensemble. Drawing four
// different masks over the same five units is what makes "subnetwork" concrete;
// with one mask on the page it is just a layer with holes in it.
function implicitEnsemble() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Every step trains a different network',
    'The mask is not damage to one network. It is a choice of which of 2ⁿ networks gets this step.')];

  els.push(...card(M, 86, 520, 290, { spine: T.primary }));
  els.push(text(62, 98, 'One set of weights, many subnetworks', { size: 14, stroke: T.primary }));

  const UX = (k) => 150 + k * 62;
  [0, 1, 2, 3, 4].forEach((k) => unit(els, UX(k), 152, 15, true, T.primary));
  els.push(text(62, 146, 'the layer', { size: 10, stroke: T.inkSubtle }));
  els.push(text(460, 138, 'n = 5 units', { size: 11, stroke: T.ink }));
  els.push(text(460, 158, '2⁵ = 32 subsets', { size: 11, family: 3, stroke: T.primary }));

  els.push(rule(56, 544, 184));
  els.push(text(62, 194, 'four steps, four samples', { size: 10, stroke: T.inkSubtle }));
  const MASKS = [
    [1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [0, 0, 1, 0, 1],
  ];
  MASKS.forEach((m, r) => {
    const y = 224 + r * 36;
    els.push(text(62, y - 6, `step ${r + 1}`, { size: 9, family: 3, stroke: T.inkSubtle }));
    m.forEach((v, k) => unit(els, UX(k), y, 11, v === 1, T.accent));
    els.push(text(460, y - 6, `${m.filter(Boolean).length} of 5 active`, { size: 9, stroke: T.inkSubtle }));
  });

  els.push(...card(620, 86, 300, 290, { spine: T.success }));
  els.push(text(642, 98, 'Why that is an ensemble', { size: 14, stroke: T.success }));
  els.push(rule(636, 904, 122));
  els.push(text(642, 134, 'Each step takes one SGD step on', { size: 10, stroke: T.ink }));
  els.push(text(642, 150, 'whichever subnetwork it sampled.', { size: 10, stroke: T.ink }));
  els.push(text(642, 176, 'All of them share the same weights,', { size: 10, stroke: T.ink }));
  els.push(text(642, 192, 'so that update reaches every other', { size: 10, stroke: T.ink }));
  els.push(text(642, 208, 'subnetwork at the same time.', { size: 10, stroke: T.ink }));
  els.push(rule(636, 904, 232));
  els.push(text(642, 244, 'With dropout off, the full network', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 260, 'approximates an average over all', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 276, 'of them: an ensemble with no extra', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 292, 'training, storage or inference cost.', { size: 10, stroke: T.inkMuted }));
  els.push(rule(636, 904, 316));
  els.push(text(642, 328, '2ⁿ counts on/off combinations and', { size: 9, stroke: T.inkSubtle }));
  els.push(text(642, 344, 'does not depend on p. The rate only', { size: 9, stroke: T.inkSubtle }));
  els.push(text(642, 360, 'sets how often each is sampled.', { size: 9, stroke: T.inkSubtle }));

  els.push(rect(M, 400, 880, 56, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 412, 'A 64-unit hidden layer has 2⁶⁴ subnetworks. Training visits a vanishing fraction of them, and that is not the point.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 436, 'The point is that no weight can become critical to the output, because the step that needed it might not include it.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Dropout as sampling from an ensemble of subnetworks',
    desc: 'A hand-drawn figure. The left card shows a layer of five units, noting that n equals five '
      + 'gives two to the fifth, or thirty-two, possible subsets. Beneath it, four training steps '
      + 'each sample a different mask over the same five units, with three, three, four and two '
      + 'units active respectively; dropped units keep their outlines and gain a cross. The right '
      + 'card explains why this is an ensemble: each step takes one SGD step on whichever '
      + 'subnetwork it sampled, all of them share the same weights so that update reaches every '
      + 'other subnetwork at once, and with dropout off the full network approximates an average '
      + 'over all of them, giving an ensemble with no extra training, storage or inference cost. It '
      + 'adds that two to the n counts on-off combinations and does not depend on the rate p, which '
      + 'only sets how often each subnetwork is sampled. A band notes that a sixty-four-unit layer '
      + 'has two to the sixty-fourth subnetworks, that training visits a vanishing fraction of them, '
      + 'and that the point is instead that no weight can become critical to the output.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { dropoutTrainVsTest, implicitEnsemble };

if (require.main === module) {
  emit('01-dropout-train-vs-test', dropoutTrainVsTest());
  emit('02-implicit-ensemble', implicitEnsemble());
}
