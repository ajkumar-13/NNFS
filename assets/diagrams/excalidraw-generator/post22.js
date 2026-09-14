// The post 22 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 4 reports what ten thousand epochs of plain SGD actually achieve and
// draws the conclusion the next five posts exist for — the optimiser is not
// broken, it is inefficient — and that measured result had no figure.
//
//   01-sgd-update-and-lr  960 x 540
//   02-sgd-is-slow        960 x 470   (new)

const { T, rect, circle, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// A loss curve inside a framed box, on a shared vertical scale so panels can be
// compared against each other rather than each against its own maximum.
function lossPlot(els, x, y, w, h, f, o) {
  const lo = o.lo ?? 0.6, hi = o.hi ?? 1.6;
  const py = (L) => y + h - ((Math.max(lo, Math.min(hi, L)) - lo) / (hi - lo)) * h;
  els.push(line([[x, y], [x, y + h]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[x, y + h], [x + w, y + h]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  const pts = [];
  for (let i = 0; i <= (o.samples ?? 48); i++) {
    const t = i / (o.samples ?? 48);
    pts.push([x + t * w, py(f(t))]);
  }
  els.push(line(pts, { stroke: o.c, strokeWidth: 2.2, roughness: o.rough ?? 0.3 }));
  return { py, px: (t) => x + t * w };
}

// ---------------------------------------------------------------- diagram 1
// The rule above, its one knob below, at three settings on one shared vertical
// scale. Giving each panel its own scale would make the diverging run look as
// controlled as the working one.
function sgdUpdateAndLr() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Vanilla SGD, and its one knob',
    'Subtract a scaled gradient from every parameter. The scale is the only thing there is to choose.')];

  els.push(...card(M, 86, 880, 92, { stroke: T.primary, strokeWidth: 1.6 }));
  els.push(text(62, 104, 'θ_new   =   θ_old   −   α · ∂L/∂θ',
    { size: 20, family: 3, stroke: T.ink }));
  els.push(text(62, 142, 'applied to every weight and every bias, once per step', { size: 10, stroke: T.inkSubtle }));
  els.push(text(560, 100, 'layer.weights -= self.learning_rate * layer.dweights',
    { size: 10, family: 3, stroke: T.inkMuted }));
  els.push(text(560, 118, 'layer.biases  -= self.learning_rate * layer.dbiases',
    { size: 10, family: 3, stroke: T.inkMuted }));
  els.push(text(560, 144, 'The whole optimiser. It holds one number and no state.',
    { size: 10, stroke: T.inkSubtle }));

  const PANELS = [
    {
      c: T.warn, title: 'Too small', sub: 'α ≪ ideal',
      f: (t) => 1.10 - 0.02 * t,
      verdict: 'The loss barely moves. Steps are\ntaken; none of them get anywhere.',
    },
    {
      c: T.success, title: 'Right-sized', sub: 'α = 1.0 here',
      f: (t) => 1.10 - 0.23 * t,
      verdict: 'A smooth descent, and still\ndescending at 10 000 epochs.',
    },
    {
      c: T.alert, title: 'Too large', sub: 'α ≫ ideal',
      f: (t) => 1.05 + 0.42 * t * Math.sin(t * 21),
      verdict: 'The step overshoots the minimum\nevery time, and the loss grows.',
    },
  ];

  const PW = 282, PX = [M, 339, 638];
  PANELS.forEach((p, i) => {
    const x = PX[i], inner = PW - 40;
    els.push(...card(x, 200, PW, 230, { spine: p.c }));
    els.push(text(x + 20, 212, p.title, { size: 14, stroke: p.c }));
    els.push(text(x + 20, 234, p.sub, { size: 10, family: 3, stroke: T.inkSubtle }));
    lossPlot(els, x + 34, 258, 220, 92, p.f, { c: p.c, samples: p.c === T.alert ? 96 : 48 });
    els.push(text(x + 20, 356, 'loss, over epochs', { size: 9, stroke: T.inkSubtle }));
    els.push(text(x + 20, 380, fit(p.verdict, 10, inner, 'panel verdict'), { size: 10, stroke: T.inkMuted }));
  });
  els.push(text(M, 446, 'All three panels share one vertical scale, so the diverging run is drawn as large as it actually is.',
    { size: 10, stroke: T.inkSubtle }));

  els.push(rect(M, 470, 880, 56, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 482, 'Large steps explore and small steps converge, and one constant cannot do both jobs across a whole training run.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 506, 'Every optimiser in Parts 23 to 27 is a way of not having to choose one number and live with it.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'SGD update and the learning-rate trade-off',
    desc: 'A hand-drawn figure in two parts. The upper card gives the update rule, theta new equals '
      + 'theta old minus alpha times the gradient, applied to every weight and bias once per step, '
      + 'beside the two lines of code that implement it and a note that the whole optimiser holds '
      + 'one number and no state. Below, three loss-curve panels on one shared vertical scale. Too '
      + 'small, in warn tan, shows a nearly flat line: steps are taken and none of them get '
      + 'anywhere. Right-sized, in success green at alpha equals one, shows a smooth descent still '
      + 'falling at ten thousand epochs. Too large, in alert red, shows a curve oscillating with '
      + 'growing amplitude as each step overshoots the minimum. A note records that the panels share '
      + 'one scale so the diverging run is drawn as large as it actually is. A band states that '
      + 'large steps explore and small steps converge, that one constant cannot do both across a '
      + 'whole run, and that every optimiser in Parts 23 to 27 is a way of not having to choose.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 4 reports real measurements and draws a conclusion from them
// that the rest of the series depends on. The chart has to make "still
// descending" visible, because the point is not that SGD fails but that it has
// not finished.
function sgdIsSlow() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Not stuck, just slow',
    'Ten thousand epochs of plain SGD on the spiral set. The curve is still falling when the budget runs out.')];

  els.push(...card(M, 86, 520, 280, { spine: T.warn }));
  els.push(text(62, 98, 'Loss over ten thousand epochs', { size: 14, stroke: T.warn }));
  els.push(rule(56, 544, 126));

  // Measured points from the post: (epoch, loss, accuracy).
  const PTS = [[0, 1.10, '36%'], [1000, 1.06, '40%'], [5000, 0.97, '51%'], [10000, 0.87, '65%']];
  const X0 = 96, X1 = 520, Y0 = 300, Y1 = 152;
  const px = (e) => X0 + (e / 10000) * (X1 - X0);
  const py = (L) => Y0 - ((L - 0.82) / (1.14 - 0.82)) * (Y0 - Y1);

  els.push(line([[X0, Y1 - 10], [X0, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0, Y0], [X1 + 10, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  [[1.1, '1.10'], [1.0, '1.00'], [0.9, '0.90']].forEach(([L, lab]) => {
    els.push(text(48, py(L) - 6, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'right', width: 44 }));
    els.push(line([[X0 - 4, py(L)], [X0 + 4, py(L)]], { stroke: T.border, strokeWidth: 1, roughness: 0.2 }));
  });
  els.push(line(PTS.map(([e, L]) => [px(e), py(L)]), { stroke: T.warn, strokeWidth: 2.4, roughness: 0.3 }));
  PTS.forEach(([e, L, acc]) => {
    els.push(circle(px(e), py(L), 4.5, { stroke: T.warn, fill: T.surface, strokeWidth: 1.5 }));
    els.push(text(px(e) - 30, py(L) - 22, acc, { size: 10, stroke: T.inkMuted, align: 'center', width: 60 }));
  });
  els.push(text(X0 - 20, Y0 + 10, '0', { size: 9, stroke: T.inkSubtle, align: 'center', width: 40 }));
  els.push(text(X1 - 30, Y0 + 10, '10 000', { size: 9, stroke: T.inkSubtle, align: 'center', width: 60 }));
  els.push(text(96, 326, 'accuracy is printed above each marked epoch', { size: 10, stroke: T.inkSubtle }));
  els.push(text(96, 344, 'Still descending at the right-hand edge. Run to 50 000 and it reaches 0.29 loss, 91% accuracy.',
    { size: 10, stroke: T.warn }));

  els.push(...card(600, 86, 320, 280, { spine: T.success }));
  els.push(text(622, 98, 'The same budget, spent better', { size: 14, stroke: T.success }));
  els.push(rule(614, 904, 126));
  const BARS = [
    { label: 'SGD · 10 000 epochs', v: 0.65, c: T.warn },
    { label: 'SGD · 50 000 epochs', v: 0.91, c: T.warn },
    { label: 'Adam · 10 000 epochs', v: 0.96, c: T.success },
  ];
  BARS.forEach((b, i) => {
    const y = 150 + i * 62;
    els.push(text(622, y, b.label, { size: 10, stroke: T.inkMuted }));
    els.push(rect(622, y + 18, 220 * b.v, 18, {
      stroke: b.c, fill: b.c, strokeWidth: 1.3, opacity: 34, roundness: null,
    }));
    els.push(text(854, y + 20, `${Math.round(b.v * 100)}%`, { size: 12, family: 3, stroke: b.c }));
  });
  els.push(text(622, 336, 'Adam reaches in ten thousand epochs', { size: 10, stroke: T.inkSubtle }));
  els.push(text(622, 352, 'what SGD does not reach in fifty.', { size: 10, stroke: T.inkSubtle }));

  els.push(rect(M, 392, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 404, 'SGD takes the raw gradient, and the gradient is smallest exactly where the network spends most of its time.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 428, 'Parts 23 to 27 each add one mechanism for taking a larger, better-aimed step in those shallow regions.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Vanilla SGD is inefficient rather than broken',
    desc: 'A hand-drawn figure in two cards. The left plots the loss of plain SGD over ten thousand '
      + 'epochs on the spiral set, from one point one nought at the start through one point nought '
      + 'six at a thousand epochs and nought point nine seven at five thousand to nought point eight '
      + 'seven at ten thousand, with the accuracy printed above each marked point: thirty-six per '
      + 'cent, forty, fifty-one and sixty-five. The curve is still descending at the right-hand '
      + 'edge, and a note records that running to fifty thousand epochs reaches a loss of nought '
      + 'point two nine and ninety-one per cent accuracy. The right card compares three runs as '
      + 'bars: SGD at ten thousand epochs reaching sixty-five per cent, SGD at fifty thousand '
      + 'reaching ninety-one, and Adam at ten thousand reaching ninety-six, with the note that Adam '
      + 'reaches in ten thousand epochs what SGD does not reach in fifty. A band explains that SGD '
      + 'takes the raw gradient and the gradient is smallest exactly where the network spends most '
      + 'of its time, and that Parts 23 to 27 each add one mechanism for a larger, better-aimed step '
      + 'in those shallow regions.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { sgdUpdateAndLr, sgdIsSlow };

if (require.main === module) {
  emit('01-sgd-update-and-lr', sgdUpdateAndLr());
  emit('02-sgd-is-slow', sgdIsSlow());
}
