// The post 27 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// bias correction is the one piece of Adam inherited from neither parent, and
// section 2.1 justifies it with an exact derivation and a table of amplifiers.
// The existing figure draws the correction boxes without saying what they do.
//
//   01-adam-pipeline    960 x 540
//   02-bias-correction  960 x 470   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// Two lanes at equal weight, each labelled with the post it came from. Adam is
// a composition rather than an invention, and the lineage labels are what make
// the figure say so.
function adamPipeline() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Adam is momentum over RMSProp',
    'One gradient into two moving averages, each corrected for its cold start, then divided one by the other.')];

  els.push(...card(50, 188, 84, 56, { stroke: T.primary, strokeWidth: 1.6 }));
  els.push(text(50, 204, 'g', { size: 20, family: 3, stroke: T.primary, align: 'center', width: 84 }));
  els.push(text(50, 250, 'the gradient', { size: 9, stroke: T.inkSubtle, align: 'center', width: 84 }));

  const lane = (y, o) => {
    els.push(text(180, y - 24, o.origin, { size: 10, stroke: o.c }));
    els.push(...card(180, y, 180, 62, { spine: o.c }));
    els.push(text(180, y + 12, o.ema, { size: 12, family: 3, stroke: T.ink, align: 'center', width: 180 }));
    els.push(text(180, y + 38, o.emaNote, { size: 9, stroke: T.inkSubtle, align: 'center', width: 180 }));
    els.push(arrow([[364, y + 31], [396, y + 31]], { stroke: o.c, strokeWidth: 1.3, roughness: 0.4 }));
    els.push(...card(400, y, 180, 62, { spine: o.c }));
    els.push(text(400, y + 12, o.corr, { size: 12, family: 3, stroke: T.ink, align: 'center', width: 180 }));
    els.push(text(400, y + 38, 'bias correction', { size: 9, stroke: T.inkSubtle, align: 'center', width: 180 }));
    els.push(arrow([[584, y + 31], [616, y + 31]], { stroke: o.c, strokeWidth: 1.3, roughness: 0.4 }));
    els.push(text(620, y + 18, o.out, { size: 17, family: 3, stroke: o.c }));
    els.push(line([[134, 216], [156, 216], [156, y + 31], [176, y + 31]],
      { stroke: o.c, strokeWidth: 1.3, roughness: 0.4 }));
  };

  lane(122, {
    c: T.accent, origin: 'first moment · from momentum, Part 24',
    ema: 'm = β₁m + (1−β₁)g', emaNote: 'an average of the gradient',
    corr: 'm̂ = m / (1 − β₁ᵗ)', out: 'm̂',
  });
  lane(280, {
    c: T.success, origin: 'second moment · from RMSProp, Part 26',
    ema: 'v = β₂v + (1−β₂)g²', emaNote: 'an average of its square',
    corr: 'v̂ = v / (1 − β₂ᵗ)', out: 'v̂',
  });

  els.push(...card(700, 174, 220, 116, { stroke: T.primary, strokeWidth: 1.8 }));
  els.push(text(700, 190, 'θ  −=', { size: 14, family: 3, stroke: T.ink, align: 'center', width: 220 }));
  els.push(text(700, 216, 'α · m̂ / (√v̂ + ε)', { size: 15, family: 3, stroke: T.primary, align: 'center', width: 220 }));
  els.push(text(700, 252, 'a smoothed direction, divided by', { size: 9, stroke: T.inkSubtle, align: 'center', width: 220 }));
  els.push(text(700, 266, 'a per-parameter scale', { size: 9, stroke: T.inkSubtle, align: 'center', width: 220 }));
  els.push(line([[664, 153], [682, 153], [682, 232]], { stroke: T.accent, strokeWidth: 1.3, roughness: 0.4 }));
  els.push(line([[664, 311], [682, 311], [682, 232]], { stroke: T.success, strokeWidth: 1.3, roughness: 0.4 }));
  els.push(arrow([[682, 232], [696, 232]], { stroke: T.primary, strokeWidth: 1.4, roughness: 0.4 }));

  els.push(...card(M, 366, 880, 74, { spine: T.primary }));
  els.push(text(62, 378, 'Defaults', { size: 12, stroke: T.primary }));
  els.push(text(62, 400, 'α = 0.001 in production, 0.02 here    β₁ = 0.9    β₂ = 0.999    ε = 1e−7',
    { size: 11, family: 3, stroke: T.ink }));
  els.push(text(62, 422, 'β₂ is the longer window because g² is noisier than g: squaring amplifies the extremes.',
    { size: 10, stroke: T.inkSubtle }));

  els.push(text(M, 468, 'Spiral set, 10 000 epochs: 96.3% accuracy at 0.08 loss. The best result in the series, from two ideas already built.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 490, 'Two extra buffers per parameter is the price, which is why very large models sometimes still prefer SGD with momentum.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'The Adam optimiser pipeline: two moments, two corrections, one update',
    desc: 'A hand-drawn left-to-right pipeline. A single gradient g enters on the left and splits into '
      + 'two lanes drawn at equal weight. The upper lane, labelled first moment and credited to '
      + 'momentum from Part 24, computes m as beta one times m plus one minus beta one times g, an '
      + 'average of the gradient, then bias-corrects it by dividing by one minus beta one to the t, '
      + 'giving m-hat. The lower lane, labelled second moment and credited to RMSProp from Part 26, '
      + 'computes v as beta two times v plus one minus beta two times g squared, an average of its '
      + 'square, then bias-corrects to v-hat. The two lanes merge into an update box: theta minus '
      + 'equals alpha times m-hat over the root of v-hat plus epsilon, described as a smoothed '
      + 'direction divided by a per-parameter scale. A card gives the defaults, alpha of nought '
      + 'point nought nought one in production or nought point nought two here, beta one of nought '
      + 'point nine, beta two of nought point nine nine nine and epsilon of one times ten to the '
      + 'minus seven, noting that beta two is the longer window because g squared is noisier. A '
      + 'footer records ninety-six point three per cent accuracy at nought point nought eight loss, '
      + 'the best in the series, at the cost of two extra buffers per parameter.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Bias correction is the only part of Adam that neither parent supplies,
// and section 2.1 proves it exactly rather than motivating it. The amplifier
// curve is the part worth drawing: it shows both that the correction is large
// at the start and that it removes itself.
function biasCorrection() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('The one part Adam does not inherit',
    'Both moving averages start at zero, so both start too small. The fix is exact, and it switches itself off.')];

  els.push(...card(M, 86, 420, 280, { spine: T.alert }));
  els.push(text(62, 98, 'The cold start', { size: 15, stroke: T.alert }));
  els.push(text(62, 120, 'm₀ = 0 and v₀ = 0, so the first step of each EMA is', { size: 10, stroke: T.inkSubtle }));
  els.push(rule(56, 444, 144));
  els.push(text(62, 158, 'm₁  =  (1 − 0.9) · g       =  0.1 g', { size: 12, family: 3, stroke: T.ink }));
  els.push(text(62, 180, 'a tenth of what the gradient asked for', { size: 10, stroke: T.inkSubtle }));
  els.push(text(62, 214, 'v₁  =  (1 − 0.999) · g²  =  0.001 g²', { size: 12, family: 3, stroke: T.ink }));
  els.push(text(62, 236, 'a thousandth of it', { size: 10, stroke: T.inkSubtle }));
  els.push(rule(56, 444, 264));
  els.push(text(62, 276, 'Left alone, the optimiser spends its first dozens', { size: 11, stroke: T.alert }));
  els.push(text(62, 294, 'of iterations crawling towards a steady state it', { size: 11, stroke: T.alert }));
  els.push(text(62, 312, 'has not reached yet.', { size: 11, stroke: T.alert }));
  els.push(text(62, 338, 'And a learning-rate sweep run over that period', { size: 10, stroke: T.inkMuted }));
  els.push(text(62, 354, 'measures the damping, not the rate.', { size: 10, stroke: T.inkMuted }));

  els.push(...card(500, 86, 420, 280, { spine: T.success }));
  els.push(text(522, 98, 'The amplifier  1 / (1 − βᵗ)', { size: 15, stroke: T.success }));
  els.push(text(522, 120, 'log axes; it decays to 1 and stays there', { size: 10, stroke: T.inkSubtle }));

  const X0 = 566, X1 = 890, Y0 = 320, Y1 = 156;
  const px = (lt) => X0 + (lt / 4) * (X1 - X0);
  const py = (la) => Y0 - (la / 3.2) * (Y0 - Y1);
  els.push(line([[X0, Y1 - 6], [X0, Y0 + 6]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0 - 6, Y0], [X1 + 8, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  [[3, '1000×'], [2, '100×'], [1, '10×'], [0, '1×']].forEach(([la, lab]) => {
    els.push(text(514, py(la) - 6, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'right', width: 46 }));
  });
  [[0, '1'], [2, '100'], [4, '10 000']].forEach(([lt, lab]) => {
    els.push(text(px(lt) - 30, Y0 + 8, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'center', width: 60 }));
  });

  [
    { b: 0.9, c: T.accent, lab: 'β₁ = 0.9', at: 0.9 },
    { b: 0.999, c: T.success, lab: 'β₂ = 0.999', at: 2.6 },
  ].forEach((bt) => {
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const lt = (i / 60) * 4, t = Math.pow(10, lt);
      pts.push([px(lt), py(Math.log10(1 / (1 - Math.pow(bt.b, t))))]);
    }
    els.push(line(pts, { stroke: bt.c, strokeWidth: 2.2, roughness: 0.25 }));
    els.push(text(px(bt.at) - 50, py(0.55), bt.lab,
      { size: 10, family: 3, stroke: bt.c, align: 'center', width: 100 }));
  });
  els.push(text(px(2) - 60, 340, 'iteration t', { size: 9, stroke: T.inkSubtle, align: 'center', width: 120 }));

  els.push(rect(M, 392, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 404, 'For a constant gradient, mₜ = (1 − β₁ᵗ) g exactly, so dividing by (1 − β₁ᵗ) recovers g exactly.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 428, 'The correction is not a heuristic and it needs no schedule: as βᵗ → 0 the factor goes to 1 and the term disappears on its own.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Why Adam bias-corrects its moments, and why the correction removes itself',
    desc: 'A hand-drawn figure in two cards. The left states the cold-start problem: both moving '
      + 'averages start at zero, so the first first-moment step is one minus nought point nine times '
      + 'the gradient, a tenth of what was asked for, and the first second-moment step is one minus '
      + 'nought point nine nine nine times the squared gradient, a thousandth of it. Left alone the '
      + 'optimiser spends its first dozens of iterations crawling towards a steady state it has not '
      + 'reached, and a learning-rate sweep over that period measures the damping rather than the '
      + 'rate. The right card plots the correction factor, one over one minus beta to the t, on log '
      + 'axes: the beta-one curve starts at ten times and falls to one within a few dozen '
      + 'iterations, while the beta-two curve starts at a thousand times and takes a few thousand '
      + 'iterations to reach one. A band gives the exactness argument: for a constant gradient the '
      + 'first moment equals one minus beta one to the t times g, so dividing by that factor '
      + 'recovers g exactly, and notes that the correction needs no schedule because as beta to the '
      + 't tends to zero the factor tends to one and the term disappears on its own.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { adamPipeline, biasCorrection };

if (require.main === module) {
  emit('01-adam-pipeline', adamPipeline());
  emit('02-bias-correction', biasCorrection());
}
