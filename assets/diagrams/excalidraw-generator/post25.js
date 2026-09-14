// The post 25 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 1 sets up the problem AdaGrad exists to solve with a two-parameter
// loss whose gradients differ by a factor of a hundred, and shows that no single
// learning rate serves both. The existing figure draws the mechanism; nothing
// drew the problem.
//
//   01-per-parameter-rates  960 x 540
//   02-one-rate-two-params  960 x 470   (new)

const { T, rect, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// Both curves are straight on a log-log axis with the same slope, which is the
// point: the ratio between the two effective rates never changes, and both are
// heading to zero at the same rate.
function perParameterRates() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Every parameter sets its own step size',
    'Divide by the root of what each weight has already seen. Then watch what that does over ten thousand steps.')];

  els.push(...card(M, 86, 880, 66, { stroke: T.primary, strokeWidth: 1.6 }));
  els.push(text(62, 100, 'G  +=  g²', { size: 17, family: 3, stroke: T.ink }));
  els.push(text(230, 100, 'θ  −=  α · g / √(G + ε)', { size: 17, family: 3, stroke: T.ink }));
  els.push(text(560, 96, 'one cache entry per parameter, same shape as the weights',
    { size: 10, stroke: T.inkSubtle }));
  els.push(text(560, 116, 'ε ≈ 1e−7, only to keep the first step finite',
    { size: 10, stroke: T.inkSubtle }));

  els.push(...card(M, 176, 520, 262, { spine: T.accent }));
  els.push(text(62, 188, 'Effective rate  α / √G,  log-log', { size: 14, stroke: T.accent }));
  els.push(text(62, 210, 'two parameters with constant gradient magnitudes', { size: 10, stroke: T.inkSubtle }));

  const X0 = 110, X1 = 520, Y0 = 396, Y1 = 236;
  // x is log10(t) from 0 to 4; y is log10(rate) from -2.5 to 2.
  const px = (lt) => X0 + (lt / 4) * (X1 - X0);
  const py = (lr) => Y0 - ((lr + 2.5) / 4.5) * (Y0 - Y1);
  els.push(line([[X0, Y1 - 6], [X0, Y0 + 6]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0 - 6, Y0], [X1 + 8, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  [[2, '100'], [0, '1'], [-2, '0.01']].forEach(([lr, lab]) => {
    els.push(text(56, py(lr) - 6, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'right', width: 46 }));
  });
  [[0, '1'], [2, '100'], [4, '10 000']].forEach(([lt, lab]) => {
    els.push(text(px(lt) - 30, Y0 + 8, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'center', width: 60 }));
  });

  // For a constant |g|, G_t = t·g², so the rate is α / (|g|·√t): a straight
  // line of slope −1/2 on log-log, offset by log10(1/|g|).
  const CURVES = [
    { g: 2.0, c: T.alert, lab: '|g| = 2.0' },
    { g: 0.02, c: T.success, lab: '|g| = 0.02' },
  ];
  CURVES.forEach((cv) => {
    const pts = [];
    for (let k = 0; k <= 40; k++) {
      const lt = (k / 40) * 4;
      const rate = 1 / (cv.g * Math.sqrt(Math.pow(10, lt)));
      pts.push([px(lt), py(Math.log10(rate))]);
    }
    els.push(line(pts, { stroke: cv.c, strokeWidth: 2.2, roughness: 0.25 }));
    // Labelled at its own right-hand end: a legend block in the upper left sat
    // directly on the upper curve.
    const endRate = 1 / (cv.g * 100);
    els.push(text(px(4) - 100, py(Math.log10(endRate)) - 22, cv.lab,
      { size: 10, family: 3, stroke: cv.c, align: 'right', width: 100 }));
  });
  els.push(text(px(2) - 60, 420, 'iterations', { size: 9, stroke: T.inkSubtle, align: 'center', width: 120 }));

  els.push(...card(580, 176, 340, 250, { spine: T.alert }));
  els.push(text(602, 188, 'The flaw is in the same picture', { size: 14, stroke: T.alert }));
  els.push(rule(596, 904, 216));
  els.push(text(602, 228, 'Both lines have slope −½ and neither', { size: 11, stroke: T.ink }));
  els.push(text(602, 246, 'ever flattens. G only ever grows, so', { size: 11, stroke: T.ink }));
  els.push(text(602, 264, 'every effective rate is on its way to 0.', { size: 11, stroke: T.ink }));
  els.push(rule(596, 904, 292));
  els.push(text(602, 304, 'The small-gradient parameter keeps a', { size: 10, stroke: T.inkMuted }));
  els.push(text(602, 320, 'usable rate for far longer, which is the', { size: 10, stroke: T.inkMuted }));
  els.push(text(602, 336, 'behaviour AdaGrad was designed for.', { size: 10, stroke: T.inkMuted }));
  els.push(text(602, 364, 'But given enough steps, learning stops', { size: 11, stroke: T.alert }));
  els.push(text(602, 382, 'for both. Part 26 is the fix.', { size: 11, stroke: T.alert }));

  els.push(rect(M, 452, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 464, 'AdaGrad · 84.0% after 10 000 epochs, against 64.7% for plain SGD and 95.7% for momentum.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 488, 'Its value here is conceptual: the cleanest illustration of per-parameter scaling, before the moving average makes it opaque.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'AdaGrad: per-parameter effective learning rates, and why they die',
    desc: 'A hand-drawn figure. A band at the top gives the rule: G accumulates the squared gradient '
      + 'and the parameter is moved by alpha times the gradient over the root of G plus epsilon, '
      + 'with one cache entry per parameter and epsilon only there to keep the first step finite. '
      + 'The main chart plots the effective rate alpha over root G on log-log axes over ten thousand '
      + 'iterations for two parameters, one with a constant gradient magnitude of two and one of '
      + 'nought point nought two. Both are straight lines of slope minus a half, separated by a '
      + 'factor of a hundred, and neither flattens. A side card names the flaw visible in the same '
      + 'picture: because G only ever grows, every effective rate is on its way to zero, and while '
      + 'the small-gradient parameter keeps a usable rate far longer, given enough steps learning '
      + 'stops for both. A band records that AdaGrad reaches eighty-four per cent after ten thousand '
      + 'epochs against sixty-four point seven for plain SGD and ninety-five point seven for '
      + 'momentum, and that its value here is conceptual.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 1 is a proof by counterexample: two parameters, gradients a
// hundred apart, and no single alpha that serves both. Working both candidate
// alphas is what makes it a proof rather than an assertion.
function oneRateTwoParams() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('One learning rate cannot serve two parameters',
    'A loss with a hundredfold difference between its two gradients, and both ways of choosing α failing.')];

  els.push(...card(M, 86, 420, 250, { spine: T.primary }));
  els.push(text(62, 98, 'The loss', { size: 15, stroke: T.primary }));
  els.push(text(62, 124, 'L(W₁, W₂)  =  W₁² / 100  +  W₂²', { size: 15, family: 3, stroke: T.ink }));
  els.push(text(62, 152, 'evaluated at W₁ = W₂ = 1', { size: 10, stroke: T.inkSubtle }));
  els.push(rule(56, 444, 176));
  els.push(text(62, 188, 'parameter', { size: 9, stroke: T.inkSubtle }));
  els.push(text(200, 188, 'gradient', { size: 9, stroke: T.inkSubtle }));
  els.push(text(320, 188, 'relative', { size: 9, stroke: T.inkSubtle }));
  [['W₁', '2W₁/100  =  0.02', '1×', T.success], ['W₂', '2W₂  =  2.0', '100×', T.alert]].forEach((r, i) => {
    const y = 212 + i * 34;
    els.push(text(62, y, r[0], { size: 13, family: 3, stroke: r[3] }));
    els.push(text(200, y, r[1], { size: 11, family: 3, stroke: T.ink }));
    els.push(text(320, y, r[2], { size: 12, family: 3, stroke: r[3] }));
  });
  els.push(text(62, 288, 'The same surface is a hundred times steeper in one', { size: 10, stroke: T.inkMuted }));
  els.push(text(62, 304, 'direction than the other, at the very same point.', { size: 10, stroke: T.inkMuted }));

  els.push(...card(500, 86, 420, 250, { spine: T.alert }));
  els.push(text(522, 98, 'Both ways of choosing α', { size: 15, stroke: T.alert }));
  els.push(rule(516, 904, 126));
  const TRIES = [
    {
      a: 'α = 0.1', sub: 'sized for W₂',
      w1: 'W₁ moves 0.002', v1: 'negligible', c1: T.alert,
      w2: 'W₂ moves 0.2', v2: 'about right', c2: T.success,
    },
    {
      a: 'α = 10', sub: 'sized for W₁',
      w1: 'W₁ moves 0.2', v1: 'about right', c1: T.success,
      w2: 'W₂ moves 20', v2: 'diverges', c2: T.alert,
    },
  ];
  TRIES.forEach((t, i) => {
    const y = 142 + i * 84;
    els.push(text(522, y, t.a, { size: 13, family: 3, stroke: T.ink }));
    els.push(text(600, y + 2, t.sub, { size: 9, stroke: T.inkSubtle }));
    els.push(text(522, y + 24, t.w1, { size: 11, family: 3, stroke: T.inkMuted }));
    els.push(text(700, y + 24, t.v1, { size: 11, stroke: t.c1 }));
    els.push(text(522, y + 44, t.w2, { size: 11, family: 3, stroke: T.inkMuted }));
    els.push(text(700, y + 44, t.v2, { size: 11, stroke: t.c2 }));
    if (i === 0) els.push(rule(516, 904, y + 64));
  });
  els.push(text(522, 308, 'There is no third choice. Every α is one of these two.',
    { size: 11, stroke: T.alert }));

  els.push(rect(M, 362, 880, 68, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 374, 'AdaGrad divides each step by the root of that parameter’s own accumulated g².',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 398, 'A gradient 100× larger accumulates a cache 10 000× larger, whose root is 100×, so the two steps come out the same size.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 444, 'The hundredfold gap in the gradients cancels itself. What survives is the sign and a step α sets once for everything.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Why a single learning rate cannot serve parameters with different gradient scales',
    desc: 'A hand-drawn figure in two cards. The left states the loss, W one squared over a hundred '
      + 'plus W two squared, evaluated at both parameters equal to one, and tabulates the two '
      + 'gradients: W one has two W one over a hundred, which is nought point nought two, and W two '
      + 'has two W two, which is two point nought, a hundred times larger. It notes that the same '
      + 'surface is a hundred times steeper in one direction than the other at the very same point. '
      + 'The right card works both candidate learning rates. At alpha of nought point one, sized for '
      + 'W two, W one moves nought point nought nought two which is negligible while W two moves '
      + 'nought point two which is about right. At alpha of ten, sized for W one, W one moves nought '
      + 'point two which is about right while W two moves twenty and diverges. It concludes that '
      + 'there is no third choice. A band explains AdaGrad’s answer: dividing by the root of each '
      + 'parameter’s own accumulated squared gradient means a gradient a hundred times larger '
      + 'accumulates a cache ten thousand times larger, whose root is a hundred times larger, so the '
      + 'two steps come out the same size.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { perParameterRates, oneRateTwoParams };

if (require.main === module) {
  emit('01-per-parameter-rates', perParameterRates());
  emit('02-one-rate-two-params', oneRateTwoParams());
}
