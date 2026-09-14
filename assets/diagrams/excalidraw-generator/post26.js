// The post 26 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 2 explains the fix as a memory horizon — old gradients fade at ρ^k —
// and the existing figure shows the *consequence* (a cache that converges)
// rather than the mechanism (a weight profile that falls off a cliff).
//
//   01-rmsprop-vs-adagrad-cache  960 x 540
//   02-memory-horizon            960 x 470   (new)

const { T, rect, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// Log-log, because AdaGrad's cache is linear in t and RMSProp's is flat: on
// linear axes the flat one is pinned to the floor and the comparison vanishes.
function rmspropVsAdagradCache() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('A cache that forgets',
    'The same constant gradient into both rules. One sum grows without limit; the other settles and stays there.')];

  els.push(...card(M, 86, 420, 70, { spine: T.alert }));
  els.push(text(62, 98, 'AdaGrad', { size: 13, stroke: T.alert }));
  els.push(text(62, 122, 'G  +=  g²', { size: 15, family: 3, stroke: T.ink }));
  els.push(text(250, 124, 'every gradient, forever', { size: 10, stroke: T.inkSubtle }));

  els.push(...card(500, 86, 420, 70, { spine: T.success }));
  els.push(text(522, 98, 'RMSProp', { size: 13, stroke: T.success }));
  els.push(text(522, 122, 'G  =  ρ · G  +  (1 − ρ) · g²', { size: 15, family: 3, stroke: T.ink }));
  els.push(text(760, 124, 'a weighted average', { size: 10, stroke: T.inkSubtle }));

  els.push(...card(M, 182, 580, 250, { spine: T.primary }));
  els.push(text(62, 194, 'Cache value against iteration,  |g| = 0.5,  log-log', { size: 13, stroke: T.primary }));

  const X0 = 116, X1 = 570, Y0 = 396, Y1 = 236;
  const px = (lt) => X0 + (lt / 5) * (X1 - X0);
  const py = (lg) => Y0 - ((lg + 2) / 7) * (Y0 - Y1);
  els.push(line([[X0, Y1 - 6], [X0, Y0 + 6]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0 - 6, Y0], [X1 + 8, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  [[4, '10 000'], [2, '100'], [0, '1'], [-2, '0.01']].forEach(([lg, lab]) => {
    els.push(text(56, py(lg) - 6, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'right', width: 54 }));
  });
  [[0, '1'], [2, '100'], [4, '10 000']].forEach(([lt, lab]) => {
    els.push(text(px(lt) - 32, Y0 + 8, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'center', width: 64 }));
  });

  const ada = [], rms = [];
  for (let k = 0; k <= 50; k++) {
    const lt = (k / 50) * 5, t = Math.pow(10, lt);
    ada.push([px(lt), py(Math.log10(0.25 * t))]);
    rms.push([px(lt), py(Math.log10(0.25 * (1 - Math.pow(0.9, t))))]);
  }
  els.push(line(ada, { stroke: T.alert, strokeWidth: 2.4, roughness: 0.25 }));
  els.push(line(rms, { stroke: T.success, strokeWidth: 2.4, roughness: 0.25 }));
  els.push(text(360, 250, 'AdaGrad  ·  grows as t', { size: 10, family: 3, stroke: T.alert }));
  els.push(text(360, 344, 'RMSProp  ·  settles at 0.25', { size: 10, family: 3, stroke: T.success }));
  els.push(text(px(2) - 60, 420, 'iterations', { size: 9, stroke: T.inkSubtle, align: 'center', width: 120 }));

  els.push(...card(640, 182, 280, 250, { spine: T.success }));
  els.push(text(662, 194, 'The fixed point', { size: 13, stroke: T.success }));
  els.push(rule(656, 904, 218));
  els.push(text(662, 230, 'At steady state G = ρG + (1−ρ)ḡ²,', { size: 10, stroke: T.ink }));
  els.push(text(662, 246, 'which rearranges to G = ḡ².', { size: 10, stroke: T.ink }));
  els.push(text(662, 268, 'The cache converges on the recent', { size: 10, stroke: T.inkMuted }));
  els.push(text(662, 284, 'mean square, not on a running total.', { size: 10, stroke: T.inkMuted }));
  els.push(rule(656, 904, 306));
  els.push(text(662, 318, 'So the effective rate settles at', { size: 10, stroke: T.inkMuted }));
  els.push(text(662, 336, 'α / √0.25  =  2α', { size: 12, family: 3, stroke: T.success }));
  els.push(text(662, 362, 'and stays useful for as long as the', { size: 10, stroke: T.inkMuted }));
  els.push(text(662, 378, 'gradient distribution holds.', { size: 10, stroke: T.inkMuted }));
  els.push(text(662, 404, 'AdaGrad has no fixed point at all.', { size: 10, stroke: T.alert }));

  els.push(rect(M, 456, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 468, 'One symbol changes: a running sum becomes a weighted average. Everything else in the optimiser is AdaGrad.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 492, 'And when the gradient distribution shifts, the cache follows it down. AdaGrad’s cache remembers the turbulence permanently.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'RMSProp against AdaGrad: cache growth under a constant gradient',
    desc: 'A hand-drawn comparison. Two rule cards at the top: AdaGrad accumulates G plus equals g '
      + 'squared, taking every gradient forever; RMSProp uses G equals rho times G plus one minus '
      + 'rho times g squared. The main chart plots cache value against iteration on log-log axes for '
      + 'a constant gradient magnitude of nought point five. AdaGrad’s cache is a straight rising '
      + 'line growing in proportion to t, reaching the thousands; RMSProp’s rises briefly and then '
      + 'flattens at nought point two five and stays there. A side card derives the fixed point: at '
      + 'steady state G equals rho G plus one minus rho times the mean square, which rearranges to G '
      + 'equalling the mean square, so the effective rate settles at alpha over the root of nought '
      + 'point two five, or two alpha, and stays useful for as long as the gradient distribution '
      + 'holds, whereas AdaGrad has no fixed point at all. A band notes that one symbol changes, a '
      + 'running sum becoming a weighted average, and that when the gradient distribution shifts the '
      + 'RMSProp cache follows it down while AdaGrad’s remembers the turbulence permanently.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 2 gives the mechanism as a weight profile: a gradient k steps
// old still counts for ρ^k of its original influence. Drawing that profile puts
// AdaGrad and RMSProp on one axis, where AdaGrad's line simply never falls.
function memoryHorizon() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('How long a gradient keeps counting',
    'The weight a gradient from k steps ago still carries. AdaGrad’s answer is “all of it, forever”.')];

  els.push(...card(M, 86, 560, 280, { spine: T.primary }));
  els.push(text(62, 98, 'Remaining influence  ρᵏ,  against age k', { size: 13, stroke: T.primary }));
  els.push(text(62, 120, 'log scale on age, so each horizon has its own cliff', { size: 10, stroke: T.inkSubtle }));

  const X0 = 116, X1 = 570, Y0 = 320, Y1 = 152;
  const px = (lk) => X0 + (lk / 4) * (X1 - X0);
  const py = (w) => Y0 - w * (Y0 - Y1);
  els.push(line([[X0, Y1 - 6], [X0, Y0 + 6]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0 - 6, Y0], [X1 + 8, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  [[1, '1.0'], [0.5, '0.5'], [0, '0']].forEach(([w, lab]) => {
    els.push(text(60, py(w) - 6, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'right', width: 48 }));
  });
  [[0, '1'], [1, '10'], [2, '100'], [3, '1 000'], [4, '10 000']].forEach(([lk, lab]) => {
    els.push(text(px(lk) - 32, Y0 + 8, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'center', width: 64 }));
  });

  els.push(line([[X0, py(1)], [X1, py(1)]], { stroke: T.alert, strokeWidth: 2.4, roughness: 0.2 }));
  els.push(text(392, py(1) + 8, 'AdaGrad  ·  never falls', { size: 10, family: 3, stroke: T.alert }));

  const RHOS = [
    { r: 0.9, c: T.warn, lab: 'ρ = 0.9', at: 1 },
    { r: 0.99, c: T.accent, lab: 'ρ = 0.99', at: 2 },
    { r: 0.999, c: T.success, lab: 'ρ = 0.999', at: 3 },
  ];
  RHOS.forEach((rh) => {
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const lk = (i / 60) * 4, k = Math.pow(10, lk);
      pts.push([px(lk), py(Math.pow(rh.r, k))]);
    }
    els.push(line(pts, { stroke: rh.c, strokeWidth: 2.2, roughness: 0.25 }));
    els.push(text(px(rh.at) - 44, py(0.37) + 8, rh.lab,
      { size: 10, family: 3, stroke: rh.c, align: 'center', width: 88 }));
  });
  els.push(line([[X0, py(0.37)], [X1, py(0.37)]],
    { stroke: T.border, strokeWidth: 1, strokeStyle: 'dashed', roughness: 0.2 }));
  els.push(text(576, py(0.37) - 6, '1/e', { size: 9, family: 3, stroke: T.inkSubtle }));
  els.push(text(px(2) - 60, 340, 'age of the gradient, in steps', { size: 9, stroke: T.inkSubtle, align: 'center', width: 120 }));

  els.push(...card(620, 86, 300, 280, { spine: T.success }));
  els.push(text(642, 98, 'Horizon  ≈  1 / (1 − ρ)', { size: 13, stroke: T.success }));
  els.push(rule(636, 904, 122));
  const ROWS = [
    ['0.9', '10 steps', 'reacts fast, noisy', T.warn],
    ['0.99', '100 steps', 'the usual default', T.accent],
    ['0.999', '1 000 steps', 'used here; full-batch', T.success],
  ];
  ROWS.forEach((r, i) => {
    const y = 136 + i * 50;
    els.push(text(642, y, `ρ = ${r[0]}`, { size: 12, family: 3, stroke: r[3] }));
    els.push(text(742, y, r[1], { size: 11, family: 3, stroke: T.ink }));
    els.push(text(642, y + 20, r[2], { size: 10, stroke: T.inkSubtle }));
  });
  els.push(rule(636, 904, 292));
  els.push(text(642, 302, 'PyTorch calls it alpha and defaults', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 318, 'to 0.99; TensorFlow calls it rho and', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 334, 'defaults to 0.9. Both are this ρ.', { size: 10, stroke: T.inkMuted }));

  els.push(rect(M, 392, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 404, 'A parameter that was turbulent early and calm later can recover its step size once the turbulence ages out.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 428, 'Under AdaGrad it cannot: the flat red line is the reason its effective rate only ever goes one way.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The memory horizon: how much a past gradient still contributes',
    desc: 'A hand-drawn chart of the weight a gradient still carries against its age in steps, on a '
      + 'logarithmic age axis. AdaGrad is a flat line at one that never falls, labelled as never '
      + 'falling. Three RMSProp decay factors fall away at different ages: rho of nought point nine '
      + 'drops through one over e at about ten steps, nought point nine nine at about a hundred, and '
      + 'nought point nine nine nine at about a thousand, each cliff sitting at one over one minus '
      + 'rho. A dashed line marks the one-over-e level. A side card tabulates the three horizons: '
      + 'ten steps reacts fast but is noisy, a hundred steps is the usual default, and a thousand '
      + 'steps is what this post uses for full-batch training, with a note that PyTorch calls the '
      + 'parameter alpha and defaults to nought point nine nine while TensorFlow calls it rho and '
      + 'defaults to nought point nine. A band notes that a parameter turbulent early and calm later '
      + 'can recover its step size once the turbulence ages out, and that under AdaGrad it cannot.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { rmspropVsAdagradCache, memoryHorizon };

if (require.main === module) {
  emit('01-rmsprop-vs-adagrad-cache', rmspropVsAdagradCache());
  emit('02-memory-horizon', memoryHorizon());
}
