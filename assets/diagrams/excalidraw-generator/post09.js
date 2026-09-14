// The post 09 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 1 counts the twenty-one parameters and section 4 argues that guessing
// them cannot scale, and neither had a figure.
//
//   01-strategies-compared  960 x 540
//   02-parameter-count      960 x 470   (new)

const { T, rect, circle, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

function grid(els, x, y, cols, rows, cw, ch, o = {}) {
  els.push(rect(x, y, cols * cw, rows * ch, {
    stroke: o.c ?? T.ink, fill: T.surface, strokeWidth: 1.4, roundness: null,
  }));
  for (let i = 1; i < cols; i++) {
    els.push(line([[x + i * cw, y], [x + i * cw, y + rows * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  for (let j = 1; j < rows; j++) {
    els.push(line([[x, y + j * ch], [x + cols * cw, y + j * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  return { w: cols * cw, h: rows * ch, bottom: y + rows * ch };
}

// ---------------------------------------------------------------- diagram 1
// Three panels, each with the same two rows: where the algorithm looks, and
// what its loss curve does. The landscape and the curve share a panel because
// the second is a consequence of the first, and separating them would let a
// reader treat the loss curves as three unrelated results.
function strategiesCompared() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Three ways to move twenty-one numbers',
    'The first two know nothing before they step. The third knows the slope of the loss under every weight.')];

  const PW = 282, PX = [M, 339, 638];

  // A valley cross-section, identical in all three panels.
  const valley = (px) => {
    const pts = [];
    for (let i = 0; i <= 24; i++) {
      const t = i / 24;
      const x = px + 20 + t * 242;
      const y = 186 + 76 * (1 - 4 * (t - 0.5) * (t - 0.5));
      pts.push([x, y]);
    }
    return pts;
  };

  const PANELS = [
    {
      c: T.alert, title: 'Random selection', sub: 'replace every weight, every step',
      // Teleports: scattered, uncorrelated, and kept high on the walls. The
      // floor of the valley is a small region and the point of the panel is
      // that random draws essentially never land in it.
      dots: [[0.05, 194], [0.15, 210], [0.24, 200], [0.33, 224], [0.44, 206],
        [0.53, 218], [0.63, 198], [0.72, 216], [0.83, 196], [0.93, 208]],
      joined: false,
      curve: (t) => 1.099 - 0.004 * t,
      final: 'loss 1.099 → 1.095', note: '100 000 draws. Accuracy stays at 33%.',
    },
    {
      c: T.warn, title: 'Random perturbation', sub: 'nudge the weights, keep what helps',
      // A walk that takes the easy slope and then jitters in place. It has to
      // stop visibly short of the floor, or the panel says the same thing as
      // the gradient-descent one beside it.
      dots: [[0.07, 196], [0.13, 208], [0.18, 218], [0.22, 226], [0.25, 230],
        [0.28, 232], [0.26, 231], [0.29, 233], [0.27, 232]],
      joined: true,
      curve: (t) => 1.099 - 0.059 * Math.min(1, t * 2.4),
      final: 'loss stalls at 1.04', note: 'Works on easy data. Fails on spirals.',
    },
    {
      c: T.success, title: 'Gradient descent', sub: 'step against the slope',
      dots: [[0.10, 198], [0.22, 222], [0.32, 240], [0.40, 252], [0.46, 258],
        [0.50, 261], [0.53, 262]],
      joined: true,
      curve: (t) => 0.3 + 0.799 * Math.exp(-3.2 * t),
      final: 'loss 1.099 → 0.3', note: 'Every step is informed before it is taken.',
    },
  ];

  PANELS.forEach((p, i) => {
    const px = PX[i], inner = PW - 32;
    els.push(...card(px, 86, PW, 330, { spine: p.c }));
    els.push(text(px + 20, 98, fit(p.title, 14, inner, 'panel title'), { size: 14, stroke: p.c }));
    els.push(text(px + 20, 120, fit(p.sub, 10, inner, 'panel sub'), { size: 10, stroke: T.inkSubtle }));
    els.push(rule(px + 16, px + PW - 16, 146));

    els.push(line(valley(px), { stroke: T.border, strokeWidth: 1.4, roughness: 0.3 }));
    const dotXY = p.dots.map(([t, y]) => [px + 20 + t * 242, y]);
    if (p.joined) els.push(line(dotXY, { stroke: p.c, strokeWidth: 1.4, roughness: 0.5 }));
    dotXY.forEach(([dx, dy]) => {
      els.push(circle(dx, dy, 3.6, { stroke: p.c, fill: p.c, strokeWidth: 1, roughness: 0.4 }));
    });

    els.push(text(px + 20, 278, 'loss over iterations', { size: 10, stroke: T.inkSubtle }));
    // 1.2 at the top of the box, 0.2 at the bottom.
    const ly = (L) => 298 + (1.2 - L) / 1.0 * 74;
    els.push(line([[px + 20, 298], [px + 20, 372]], { stroke: T.border, strokeWidth: 1, roughness: 0.2 }));
    els.push(line([[px + 20, 372], [px + 262, 372]], { stroke: T.border, strokeWidth: 1, roughness: 0.2 }));
    const cpts = [];
    for (let k = 0; k <= 30; k++) {
      const t = k / 30;
      cpts.push([px + 20 + t * 242, ly(p.curve(t))]);
    }
    els.push(line(cpts, { stroke: p.c, strokeWidth: 2, roughness: 0.3 }));

    els.push(text(px + 20, 382, p.final, { size: 11, family: 3, stroke: p.c }));
    els.push(text(px + 20, 398, fit(p.note, 10, inner, 'panel note'), { size: 10, stroke: T.inkSubtle }));
  });

  els.push(rect(M, 442, 880, 60, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 454, 'w_new  =  w_old  −  α · ∇L',
    { size: 16, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 480, 'The minus sign is the whole algorithm: ∇L points uphill, so moving against it is the fastest way down.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Three optimisation strategies compared: random search, random perturbation, gradient descent',
    desc: 'Three hand-drawn panels, each showing an algorithm on the same valley cross-section above '
      + 'its own loss curve. The first, random selection in alert red, replaces every weight each '
      + 'step; its samples are scattered and unconnected, almost none near the floor, and its loss '
      + 'curve is flat, moving from one point nought nine nine to one point nought nine five over a '
      + 'hundred thousand draws with accuracy still at thirty-three per cent. The second, random '
      + 'perturbation in warn tan, nudges the weights and keeps what helps; its samples form a short '
      + 'connected walk that descends part-way and stops on a shoulder, and its loss curve flattens '
      + 'at one point nought four, working on easy data and failing on spirals. The third, gradient '
      + 'descent in success green, steps against the slope; its samples walk down into the floor of '
      + 'the valley and its loss curve falls smoothly from one point nought nine nine towards nought '
      + 'point three. A band gives the update rule, w new equals w old minus alpha times the '
      + 'gradient, and notes that the minus sign is the whole algorithm because the gradient points '
      + 'uphill.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 1 counts the parameters and section 4 says guessing them cannot
// scale. Those belong together: the count is small enough to draw in full, and
// drawing it in full is what makes the jump to ten million legible as a jump
// rather than as a bigger number.
function parameterCount() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Twenty-one numbers, and why they cannot be guessed',
    'Small enough to draw every one of them. Already far too many to find by luck.')];

  els.push(...card(M, 86, 440, 288, { spine: T.primary }));
  els.push(text(62, 98, 'Every parameter in the network', { size: 15, stroke: T.primary }));
  els.push(rule(56, 464, 128));

  const ARRAYS = [
    { x: 76, y: 156, cols: 3, rows: 2, label: 'W1  (2, 3)', count: '6 weights', c: T.accent },
    { x: 256, y: 156, cols: 3, rows: 1, label: 'b1  (1, 3)', count: '3 biases', c: T.success },
    { x: 76, y: 248, cols: 3, rows: 3, label: 'W2  (3, 3)', count: '9 weights', c: T.accent },
    { x: 256, y: 248, cols: 3, rows: 1, label: 'b2  (1, 3)', count: '3 biases', c: T.success },
  ];
  ARRAYS.forEach((a) => {
    els.push(text(a.x, a.y - 18, a.label, { size: 11, family: 3, stroke: a.c }));
    grid(els, a.x, a.y, a.cols, a.rows, 26, 22, { c: a.c });
    els.push(text(a.x, a.y + a.rows * 22 + 6, a.count, { size: 10, stroke: T.inkSubtle }));
  });

  els.push(line([[368, 156], [368, 316]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  els.push(text(384, 206, '21', { size: 40, stroke: T.ink }));
  els.push(text(384, 258, 'parameters', { size: 12, stroke: T.inkMuted }));
  els.push(text(384, 276, 'in total', { size: 12, stroke: T.inkMuted }));

  els.push(...card(520, 86, 400, 288, { spine: T.alert }));
  els.push(text(542, 98, 'Why guessing does not scale', { size: 15, stroke: T.alert }));
  els.push(rule(536, 904, 128));

  const SCALES = [
    { label: 'this network', n: '21', note: '100 000 random draws moved the loss by 0.004' },
    { label: 'a small ResNet', n: '~10 million', note: 'the same search, in ten million dimensions' },
    { label: 'GPT-3', n: '175 billion', note: 'and the same update rule still has to work' },
  ];
  SCALES.forEach((s, i) => {
    const y = 148 + i * 62;
    els.push(text(542, y, s.label, { size: 11, stroke: T.inkMuted }));
    els.push(text(542, y + 18, s.n, { size: 17, family: 3, stroke: T.ink }));
    els.push(text(542, y + 42, fit(s.note, 10, 360, 'scale note'), { size: 10, stroke: T.inkSubtle }));
  });

  els.push(rect(M, 400, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 412, 'Random search has to land in a good region by luck, and that region is a vanishing fraction of the space it is drawn from.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 436, 'Doubling the number of draws does not double the odds. Knowing which way is downhill before stepping is what replaces the luck.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The twenty-one parameters, and why random search cannot find them',
    desc: 'A hand-drawn figure in two halves. The left card draws every parameter in the network as a '
      + 'grid of cells: W1 of shape two by three holding six weights, b1 of shape one by three '
      + 'holding three biases, W2 of shape three by three holding nine weights, and b2 of shape one '
      + 'by three holding three biases, totalling twenty-one parameters. The right card gives the '
      + 'scaling argument in three rows: this network at twenty-one parameters, where a hundred '
      + 'thousand random draws moved the loss by four thousandths; a small ResNet at around ten '
      + 'million, where the same search runs in ten million dimensions; and GPT-3 at a hundred and '
      + 'seventy-five billion, where the same update rule still has to work. A band states that '
      + 'random search has to land in a good region by luck and that region is a vanishing fraction '
      + 'of the space, that doubling the draws does not double the odds, and that knowing which way '
      + 'is downhill before stepping is what replaces the luck.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { strategiesCompared, parameterCount };

if (require.main === module) {
  emit('01-strategies-compared', strategiesCompared());
  emit('02-parameter-count', parameterCount());
}
