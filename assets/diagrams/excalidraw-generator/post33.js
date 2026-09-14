// The post 33 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 4's factor of 2 is the entire difference between Glorot and He, and it
// comes from one fact about ReLU that the variance-by-depth chart cannot show —
// half the distribution is mapped onto a single point.
//
//   01-activation-variance-by-depth  960 x 540
//   02-the-factor-of-two             960 x 470   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// The failing curve is allowed to run off the bottom of the chart rather than
// being squeezed onto it. Rescaling to fit 10^-22 would flatten the two working
// schemes into one line at the top and lose the comparison entirely.
function activationVarianceByDepth() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('What the initial scale does over ten layers',
    'One multiplier per layer, compounded. Below one it vanishes, above one it explodes, and only exactly one survives.')];

  els.push(...card(M, 86, 560, 330, { spine: T.primary }));
  els.push(text(62, 98, 'Activation variance by depth,  log₁₀ scale', { size: 14, stroke: T.primary }));
  els.push(text(62, 120, '10 layers of 64 units', { size: 10, stroke: T.inkSubtle }));

  const X0 = 116, X1 = 560, Y0 = 372, Y1 = 152;
  const px = (d) => X0 + (d / 10) * (X1 - X0);
  const py = (lv) => Y0 - ((lv + 10) / 11) * (Y0 - Y1);
  els.push(line([[X0, Y1 - 6], [X0, Y0 + 6]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0 - 6, py(0)], [X1 + 8, py(0)]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  [[0, '1'], [-4, '10⁻⁴'], [-8, '10⁻⁸']].forEach(([lv, lab]) => {
    els.push(text(52, py(lv) - 6, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'right', width: 56 }));
  });
  [0, 2, 4, 6, 8, 10].forEach((d) => {
    els.push(text(px(d) - 20, Y0 + 10, String(d), { size: 9, family: 3, stroke: T.inkSubtle, align: 'center', width: 40 }));
  });
  els.push(text(px(5) - 60, 390, 'layer depth', { size: 9, stroke: T.inkSubtle, align: 'center', width: 120 }));

  // 0.01 * randn at n_in = 64 multiplies the variance by 64 * 1e-4 = 0.0064
  // per layer, which is log10(0.0064) = -2.19 per layer.
  const SCHEMES = [
    { c: T.alert, per: Math.log10(0.0064), lab: '0.01 · randn' },
    { c: T.success, per: 0, lab: 'He  ·  √(2/n_in), ReLU' },
    { c: T.primary, per: -0.04, lab: 'Xavier  ·  tanh' },
  ];
  SCHEMES.forEach((s) => {
    const pts = [];
    for (let d = 0; d <= 10; d += 0.25) {
      const lv = s.per * d;
      if (lv < -10) break;
      pts.push([px(d), py(lv)]);
    }
    els.push(line(pts, { stroke: s.c, strokeWidth: 2.4, roughness: 0.3 }));
  });
  els.push(text(px(6), py(-0.6), 'He  ·  √(2/n_in),  ReLU', { size: 10, family: 3, stroke: T.success }));
  els.push(text(px(6), py(-1.9), 'Xavier  ·  tanh', { size: 10, family: 3, stroke: T.primary }));
  els.push(text(px(1.2), py(-6.4), '0.01 · randn', { size: 10, family: 3, stroke: T.alert }));
  els.push(text(px(2.4), py(-9.2), 'off the chart by layer 5', { size: 9, stroke: T.alert }));

  els.push(...card(620, 86, 300, 330, { spine: T.accent }));
  els.push(text(642, 98, 'The whole story, in one line', { size: 14, stroke: T.accent }));
  els.push(rule(636, 904, 122));
  els.push(text(642, 140, 'Var(z)  =  n_in · Var(W) · Var(x)', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(642, 168, 'so after L layers the variance is', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 190, '( n_in · Var(W) )ᴸ', { size: 13, family: 3, stroke: T.accent }));
  els.push(rule(636, 904, 220));
  [
    ['> 1', 'explodes exponentially', T.alert],
    ['< 1', 'vanishes exponentially', T.alert],
    ['= 1', 'holds', T.success],
  ].forEach((r, i) => {
    const y = 236 + i * 38;
    els.push(text(642, y, `n_in · Var(W)  ${r[0]}`, { size: 10, family: 3, stroke: T.ink }));
    els.push(text(642, y + 16, r[1], { size: 10, stroke: r[2] }));
  });
  els.push(rule(636, 904, 352));
  els.push(text(642, 364, 'Every scheme below is a way of', { size: 10, stroke: T.inkSubtle }));
  els.push(text(642, 380, 'making that product equal one.', { size: 10, stroke: T.inkSubtle }));

  els.push(rect(M, 442, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 454, '0.01 · randn at 64 inputs multiplies the variance by 0.0064 each layer, a shrink of about 150× per layer.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 478, 'Two hidden layers survive that. Five do not, and the flat loss curve that follows looks exactly like a broken optimiser.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Activation variance through ten layers under three initialisation schemes',
    desc: 'A hand-drawn chart of activation variance on a base-ten log scale against layer depth from '
      + 'nought to ten, for ten layers of sixty-four units. The series default of nought point '
      + 'nought one times a standard normal draw falls steeply, losing about two orders of magnitude '
      + 'per layer and running off the bottom of the chart by layer five. He initialisation, the '
      + 'square root of two over n_in used with ReLU, holds flat across all ten layers, as does '
      + 'Xavier with tanh. A side card gives the reason: the variance of a layer output is n_in '
      + 'times the weight variance times the input variance, so after L layers it is that product '
      + 'raised to the power L; above one it explodes exponentially, below one it vanishes, and only '
      + 'at exactly one does it hold. A band notes that nought point nought one times randn at '
      + 'sixty-four inputs multiplies the variance by nought point nought nought six four each '
      + 'layer, that two hidden layers survive it and five do not, and that the resulting flat loss '
      + 'curve looks exactly like a broken optimiser.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. The factor of 2 that separates He from Glorot comes from one fact about
// ReLU: half of a zero-centred distribution is mapped onto a single point. That
// is a statement about a distribution, and the variance-by-depth chart has no
// way to show it.
function theFactorOfTwo() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Where He’s factor of two comes from',
    'ReLU discards half of a zero-centred distribution. The 2 is not a tuning constant; it is that halving, undone.')];

  els.push(...card(M, 86, 420, 280, { spine: T.alert }));
  els.push(text(62, 98, 'What ReLU does to a distribution', { size: 14, stroke: T.alert }));
  els.push(text(62, 120, 'inputs centred on zero, as initialisation intends', { size: 10, stroke: T.inkSubtle }));

  const AX = 250, AY = 290, SD = 62;
  els.push(line([[80, AY], [430, AY]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[AX, AY + 8], [AX, 160]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(text(AX - 24, AY + 10, '0', { size: 9, stroke: T.inkSubtle }));

  // The negative half, shaded as strips, is the part that lands on zero.
  for (let x = 96; x < AX; x += 10) {
    const z = (x + 5 - AX) / SD;
    const h = 108 * Math.exp(-0.5 * z * z);
    els.push(rect(x, AY - h, 10, h, {
      stroke: T.alert, fill: T.alert, strokeWidth: 0.5, opacity: 20, roundness: null, roughness: 0.2,
    }));
  }
  const bell = [];
  for (let x = 90; x <= 410; x += 6) {
    const z = (x - AX) / SD;
    bell.push([x, AY - 108 * Math.exp(-0.5 * z * z)]);
  }
  els.push(line(bell, { stroke: T.inkMuted, strokeWidth: 2, roughness: 0.3 }));

  els.push(arrow([[168, AY + 14], [AX - 8, AY + 14]], { stroke: T.alert, strokeWidth: 1.4, roughness: 0.4 }));
  els.push(line([[AX, AY], [AX, AY - 120]], { stroke: T.alert, strokeWidth: 3, roughness: 0.3 }));
  els.push(text(96, AY + 22, 'half the mass, all mapped onto this one point',
    { size: 10, stroke: T.alert }));
  els.push(text(62, 336, 'Var( ReLU(z) )  ≈  ½ · Var(z)', { size: 13, family: 3, stroke: T.alert }));

  els.push(...card(500, 86, 420, 280, { spine: T.success }));
  els.push(text(522, 98, 'So the weights carry twice as much', { size: 14, stroke: T.success }));
  els.push(rule(516, 904, 126));
  const SCH = [
    {
      n: 'Glorot · Xavier', v: 'Var(W)  =  2 / (n_in + n_out)',
      code: 'randn(n_in, n_out) * sqrt(2 / (n_in + n_out))',
      when: 'tanh, sigmoid, softsign — activations that\nkeep roughly the variance they were given', c: T.primary,
    },
    {
      n: 'He', v: 'Var(W)  =  2 / n_in',
      code: 'randn(n_in, n_out) * sqrt(2 / n_in)',
      when: 'ReLU and its relatives — the only ones that\nthrow half the distribution away', c: T.success,
    },
  ];
  SCH.forEach((s, i) => {
    const y = 142 + i * 108;
    els.push(text(522, y, s.n, { size: 12, stroke: s.c }));
    els.push(text(522, y + 20, s.v, { size: 11, family: 3, stroke: T.ink }));
    els.push(text(522, y + 40, fit(s.code, 9, 380 * 0.94, 'code'), { size: 9, family: 3, stroke: T.inkMuted }));
    els.push(text(522, y + 60, fit(s.when, 9, 380, 'when'), { size: 9, stroke: T.inkSubtle }));
    if (i === 0) els.push(rule(516, 904, y + 92));
  });

  els.push(rect(M, 392, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 404, 'Glorot preserves variance for an activation that preserves variance. ReLU is not one of those.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 428, 'Using Glorot with ReLU halves the variance at every layer, which is the vanishing case with a gentler slope.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Why He initialisation doubles the weight variance for ReLU',
    desc: 'A hand-drawn figure in two cards. The left shows a zero-centred bell curve with its entire '
      + 'negative half shaded and an arrow sweeping that half onto a single vertical spike at zero, '
      + 'annotated as half the mass all mapped onto one point, and concluding that the variance of '
      + 'ReLU of z is about half the variance of z. The right card gives the two schemes: Glorot, '
      + 'also called Xavier, with a weight variance of two over the sum of n_in and n_out, for tanh, '
      + 'sigmoid and softsign — activations that keep roughly the variance they were given; and He, '
      + 'with a weight variance of two over n_in, for ReLU and its relatives, the only ones that '
      + 'throw half the distribution away. Both give the corresponding line of NumPy. A band states '
      + 'that Glorot preserves variance for an activation that preserves variance and ReLU is not '
      + 'one of those, and that using Glorot with ReLU halves the variance at every layer, which is '
      + 'the vanishing case with a gentler slope.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { activationVarianceByDepth, theFactorOfTwo };

if (require.main === module) {
  emit('01-activation-variance-by-depth', activationVarianceByDepth());
  emit('02-the-factor-of-two', theFactorOfTwo());
}
