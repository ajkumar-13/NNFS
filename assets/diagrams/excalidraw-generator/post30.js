// The post 30 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up, over w in [-2, 2].
// Scene 02 is new: section 2.3's table runs from |w| = 0.01 to 100, five orders
// of magnitude that a [-2, 2] window cannot show, and "constant against
// proportional" is only undeniable across that whole range.
//
//   01-l1-vs-l2-penalty  960 x 540
//   02-gradient-pressure  960 x 470   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// The arrows are the point, not the curves: equal length on the left and
// growing on the right, at the same three weight values in both panels.
function l1VsL2Penalty() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Two penalties, two kinds of pressure',
    'The shape of the penalty decides how hard the optimiser is pushed at each weight magnitude.')];

  const AT = [0.5, 1.0, 1.8];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 330, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 122, o.formula, { size: 13, family: 3, stroke: T.ink }));
    els.push(text(x + 20, 146, o.grad, { size: 11, family: 3, stroke: o.c }));
    els.push(rule(x + 16, x + 404, 172));

    const CX = x + 210, Y0 = 366, XS = 84, YS = 44;
    const px = (w) => CX + w * XS;
    const py = (v) => Y0 - v * YS;
    els.push(line([[px(-2.2), Y0], [px(2.2), Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
    els.push(line([[CX, Y0 + 8], [CX, py(4.4)]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
    els.push(text(px(1.9), Y0 + 10, 'w', { size: 10, stroke: T.inkSubtle }));

    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const w = -2 + (i / 60) * 4;
      pts.push([px(w), py(o.f(w))]);
    }
    els.push(line(pts, { stroke: o.c, strokeWidth: 2.4, roughness: 0.3 }));

    // Equal-length arrows on the left, growing ones on the right.
    AT.forEach((w) => {
      const len = o.arrow(w);
      els.push(arrow([[px(w), py(o.f(w)) - 6 - len], [px(w), py(o.f(w)) - 8]],
        { stroke: T.alert, strokeWidth: 1.6, roughness: 0.35 }));
      els.push(text(px(w) - 20, Y0 + 10, String(w), { size: 9, stroke: T.inkSubtle, align: 'center', width: 40 }));
    });
    els.push(text(x + 20, 386, fit(o.note, 11, 380, 'panel note'), { size: 11, stroke: T.inkMuted }));
  };

  panel(M, {
    c: T.accent, title: 'L1  ·  sum of absolute values',
    formula: 'L_reg  =  λ Σ |w|', grad: '∂L_reg/∂w  =  λ · sign(w)',
    f: (w) => Math.abs(w) * 1.4,
    arrow: () => 26,
    note: 'Equal pressure at every magnitude. Arrows of equal length.',
  });

  panel(500, {
    c: T.primary, title: 'L2  ·  sum of squares',
    formula: 'L_reg  =  λ Σ w²', grad: '∂L_reg/∂w  =  2λ w',
    f: (w) => w * w * 1.1,
    arrow: (w) => 14 + w * 22,
    note: 'Pressure proportional to the weight. Arrows grow with w.',
  });

  els.push(rect(M, 442, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 454, 'L1 drives small weights to exactly zero and leaves a sparse model. L2 shrinks large weights and leaves a dense, small one.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 478, 'Networks default to L2, because most weights contribute something and the aim is to keep them modest rather than to delete them.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'L1 against L2 regularisation: penalty shapes and gradient pressure',
    desc: 'Two hand-drawn panels. The left plots the L1 penalty, lambda times the sum of absolute '
      + 'weights, as a V with a sharp corner at the origin, its gradient being lambda times the sign '
      + 'of w; three downward arrows of equal length sit on the curve at w of nought point five, one '
      + 'and one point eight, showing equal pressure at every magnitude. The right plots the L2 '
      + 'penalty, lambda times the sum of squares, as a parabola, its gradient being two lambda w; '
      + 'the three arrows at the same weight values grow longer with w, showing pressure '
      + 'proportional to the weight. A band states that L1 drives small weights to exactly zero and '
      + 'leaves a sparse model while L2 shrinks large weights and leaves a dense small one, and that '
      + 'networks default to L2 because most weights contribute something and the aim is to keep '
      + 'them modest rather than delete them.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 2.3's table spans five decades of weight magnitude, which is
// exactly the range where "constant" and "proportional" separate. On log-log
// one line is flat and the other has slope 1, and they cross at a readable
// point that the table only implies.
function gradientPressure() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Constant pressure against proportional pressure',
    'The same two penalties, read across five orders of magnitude of weight instead of two units.')];

  els.push(...card(M, 86, 560, 280, { spine: T.primary }));
  els.push(text(62, 98, 'Gradient pressure at λ = 0.01,  log-log', { size: 13, stroke: T.primary }));
  els.push(text(62, 120, 'how hard each penalty pushes a weight of a given size', { size: 10, stroke: T.inkSubtle }));

  const X0 = 122, X1 = 566, Y0 = 320, Y1 = 154;
  const px = (lw) => X0 + ((lw + 2) / 4) * (X1 - X0);
  const py = (lp) => Y0 - ((lp + 4) / 4.6) * (Y0 - Y1);
  els.push(line([[X0, Y1 - 6], [X0, Y0 + 6]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0 - 6, Y0], [X1 + 8, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  [[0, '1'], [-1, '0.1'], [-2, '0.01'], [-4, '0.0001']].forEach(([lp, lab]) => {
    els.push(text(56, py(lp) - 6, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'right', width: 58 }));
  });
  [[-2, '0.01'], [0, '1'], [2, '100']].forEach(([lw, lab]) => {
    els.push(text(px(lw) - 30, Y0 + 8, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'center', width: 60 }));
  });
  els.push(text(px(0) - 60, 336, 'weight magnitude |w|', { size: 9, stroke: T.inkSubtle, align: 'center', width: 120 }));

  // L1 is a flat line at lambda; L2 is 2*lambda*w, slope 1 on log-log.
  els.push(line([[px(-2), py(-2)], [px(2), py(-2)]],
    { stroke: T.accent, strokeWidth: 2.4, roughness: 0.2 }));
  const l2 = [];
  for (let i = 0; i <= 40; i++) {
    const lw = -2 + (i / 40) * 4;
    l2.push([px(lw), py(Math.log10(0.02 * Math.pow(10, lw)))]);
  }
  els.push(line(l2, { stroke: T.primary, strokeWidth: 2.4, roughness: 0.2 }));
  els.push(text(px(1.1), py(-2) - 22, 'L1  ·  always 0.01', { size: 10, family: 3, stroke: T.accent }));
  els.push(text(px(1.0), py(0.3) + 6, 'L2  ·  0.02 w', { size: 10, family: 3, stroke: T.primary }));

  // They cross where lambda = 2*lambda*w, that is at |w| = 0.5.
  els.push(circle(px(Math.log10(0.5)), py(-2), 5, { stroke: T.alert, fill: T.surface, strokeWidth: 1.8 }));
  els.push(line([[px(Math.log10(0.5)), py(-2) + 6], [px(Math.log10(0.5)), Y0]],
    { stroke: T.alert, strokeWidth: 1.2, strokeStyle: 'dashed', roughness: 0.3 }));
  els.push(text(px(Math.log10(0.5)) + 8, 296, '|w| = 0.5', { size: 10, family: 3, stroke: T.alert }));

  els.push(...card(620, 86, 300, 280, { spine: T.accent }));
  els.push(text(642, 98, 'What each side means', { size: 13, stroke: T.accent }));
  els.push(rule(636, 904, 122));
  els.push(text(642, 136, 'Below |w| = 0.5', { size: 12, stroke: T.accent }));
  els.push(text(642, 156, 'L1 pushes harder than L2. Once it', { size: 10, stroke: T.ink }));
  els.push(text(642, 172, 'outweighs the data gradient, the', { size: 10, stroke: T.ink }));
  els.push(text(642, 188, 'weight crosses zero and stays.', { size: 10, stroke: T.ink }));
  els.push(rule(636, 904, 212));
  els.push(text(642, 224, 'Above |w| = 0.5', { size: 12, stroke: T.primary }));
  els.push(text(642, 244, 'L2 pushes harder, and keeps pushing', { size: 10, stroke: T.ink }));
  els.push(text(642, 260, 'harder the larger the weight gets.', { size: 10, stroke: T.ink }));
  els.push(text(642, 276, 'At |w| = 100 it is 200× L1.', { size: 10, stroke: T.primary }));
  els.push(rule(636, 904, 300));
  els.push(text(642, 312, 'The crossing point moves with λ.', { size: 10, stroke: T.inkSubtle }));
  els.push(text(642, 328, 'It is not a universal threshold.', { size: 10, stroke: T.inkSubtle }));

  els.push(rect(M, 392, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 404, 'A flat line and a line of slope 1. That is the whole difference between the two penalties.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 428, 'L1 never eases off on a small weight, which is why it reaches zero. L2 eases off exactly as the weight becomes harmless.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'L1 and L2 gradient pressure across five orders of weight magnitude',
    desc: 'A hand-drawn log-log chart of gradient pressure against weight magnitude at lambda of '
      + 'nought point nought one, from a weight of nought point nought one to one hundred. The L1 '
      + 'pressure is a flat line at nought point nought one for every weight. The L2 pressure is a '
      + 'straight line of slope one, from two ten-thousandths at the smallest weight to two at the '
      + 'largest. They cross at a weight magnitude of nought point five, marked with a dot and a '
      + 'dashed dropline. A side card reads both sides: below nought point five, L1 pushes harder '
      + 'than L2, and once it outweighs the data gradient the weight crosses zero and stays there; '
      + 'above nought point five, L2 pushes harder and keeps pushing harder as the weight grows, '
      + 'reaching two hundred times L1 at a weight of a hundred; and the crossing point moves with '
      + 'lambda rather than being a universal threshold. A band notes that a flat line and a line of '
      + 'slope one are the whole difference between the two penalties, that L1 never eases off on a '
      + 'small weight which is why it reaches zero, and that L2 eases off exactly as the weight '
      + 'becomes harmless.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { l1VsL2Penalty, gradientPressure };

if (require.main === module) {
  emit('01-l1-vs-l2-penalty', l1VsL2Penalty());
  emit('02-gradient-pressure', gradientPressure());
}
