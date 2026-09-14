// The post 28 diagrams, hand-drawn.
//
// Scenes 01 and 02 mirror the clean vector figures one directory up. Scene 03 is
// new: section 4 classifies a model into four regimes from the size and
// direction of its train-test gap, and both existing figures only ever show one
// of those four.
//
//   01-train-vs-test    960 x 580
//   02-loss-divergence  960 x 540
//   03-four-regimes     960 x 470   (new)

const { T, rect, circle, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// Two classes, with three deliberate outliers sitting on the wrong side of the
// simple boundary. Those three are the whole comparison: the smooth model gets
// them wrong and accepts it, the overfit model bends around each one.
const PTS = [
  // class 0, upper right
  [0.75, 0.55, 0], [0.85, 0.70, 0], [0.65, 0.80, 0], [0.90, 0.45, 0], [0.55, 0.90, 0],
  [0.95, 0.85, 0], [0.70, 0.65, 0], [0.80, 0.90, 0], [0.60, 0.62, 0], [0.88, 0.60, 0],
  // class 1, lower left
  [0.20, 0.30, 1], [0.35, 0.20, 1], [0.15, 0.55, 1], [0.30, 0.45, 1], [0.45, 0.25, 1],
  [0.10, 0.25, 1], [0.25, 0.15, 1], [0.40, 0.35, 1], [0.18, 0.42, 1], [0.22, 0.62, 1],
  // the three the simple boundary cannot get right
  [0.62, 0.52, 1], [0.70, 0.40, 1], [0.35, 0.50, 0],
];
const CLS = [T.primary, T.accent];

// ---------------------------------------------------------------- diagram 1
// Both panels carry the same points. Only the boundary moves, which is what
// makes "the overfit model bent to reach these three" something the eye can
// verify rather than a claim in the caption.
function trainVsTest() {
  resetSeq();
  const W = 960, H = 580;

  const els = [...heading('Two boundaries over one dataset',
    'The same twenty-three points, fitted twice. One model leaves three of them wrong on purpose.')];

  const PX = [40, 500], GX = 30, GW = 360, GTOP = 176, GBOT = 368;
  const ux = (px, u) => px + GX + u * GW;
  const vy = (v) => GBOT - v * (GBOT - GTOP);
  const base = (u) => GTOP + u * (GBOT - GTOP);
  const bump = (u, c, w, a) => a * Math.exp(-Math.pow((u - c) / w, 2));

  const panel = (px, o) => {
    els.push(...card(px, 86, 420, 360, { spine: o.c }));
    els.push(text(px + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(px + 20, 122, fit(o.sub, 11, 380, 'panel sub'), { size: 11, stroke: T.inkSubtle }));
    els.push(rule(px + 16, px + 404, 152));

    // Shaded regions, drawn as vertical strips: the renderer has no filled
    // free-form polygon, and a strip fill is close enough at this scale.
    const STEP = 12;
    for (let x = 0; x < GW; x += STEP) {
      const u = (x + STEP / 2) / GW;
      const yb = Math.max(GTOP, Math.min(GBOT, o.b(u)));
      const sx = ux(px, x / GW);
      if (yb > GTOP) {
        els.push(rect(sx, GTOP, STEP, yb - GTOP, {
          stroke: CLS[0], fill: CLS[0], strokeWidth: 0.5, opacity: 16, roundness: null, roughness: 0.2,
        }));
      }
      if (yb < GBOT) {
        els.push(rect(sx, yb, STEP, GBOT - yb, {
          stroke: CLS[1], fill: CLS[1], strokeWidth: 0.5, opacity: 16, roundness: null, roughness: 0.2,
        }));
      }
    }
    const bpts = [];
    for (let i = 0; i <= 60; i++) {
      const u = i / 60;
      bpts.push([ux(px, u), Math.max(GTOP, Math.min(GBOT, o.b(u)))]);
    }
    els.push(line(bpts, { stroke: o.c, strokeWidth: 2.4, roughness: 0.3 }));

    PTS.forEach(([u, v, c]) => {
      const wrong = o.b(u) > vy(v) ? c !== 0 : c !== 1;
      els.push(circle(ux(px, u), vy(v), wrong ? 5.5 : 4, {
        stroke: wrong ? T.alert : CLS[c], fill: wrong ? T.surface : CLS[c],
        strokeWidth: wrong ? 2 : 1, roughness: 0.4,
      }));
    });

    els.push(text(px + 20, 380, fit(o.note, 11, 380, 'panel note'), { size: 11, stroke: T.inkMuted }));
    els.push(text(px + 20, 416, o.nums, { size: 12, family: 3, stroke: o.c }));
  };

  panel(PX[0], {
    c: T.success, title: 'Smooth boundary', sub: 'low curvature, three training points left wrong',
    b: (u) => base(u),
    note: 'The three ringed points are noise,\nand the model does not chase them.',
    nums: 'train 88%    test 85%    gap 3 points',
  });

  panel(PX[1], {
    c: T.alert, title: 'Jagged boundary', sub: 'high curvature, every training point captured',
    b: (u) => base(u) - bump(u, 0.66, 0.085, 48) + bump(u, 0.35, 0.07, 43),
    note: 'The two detours exist only to reach three points.\nNew points landing inside them are lost.',
    nums: 'train 93%    test 83%    gap 10 points',
  });

  els.push(rect(M, 476, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 488, 'A simpler network is not one with fewer parameters. It is one whose decision boundary has lower curvature.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 512, 'The right-hand model is strictly better on the data it was shown, and strictly worse on everything else.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Good generalization against overfitting: two decision boundaries on one dataset',
    desc: 'Two hand-drawn panels over the identical twenty-three training points, coloured by class, '
      + 'with the model’s predicted class shaded behind them. The left panel, in success green, '
      + 'draws a smooth low-curvature boundary that leaves three points on the wrong side; those '
      + 'three are ringed, and a note records that they are noise the model does not chase, with '
      + 'training accuracy of eighty-eight per cent, test of eighty-five, and a three-point gap. The '
      + 'right panel, in alert red, draws the same boundary with two detours that bend out to '
      + 'capture all three of those points, so no training point is misclassified; a note records '
      + 'that the detours exist only to reach three points and that new points landing inside them '
      + 'are lost, with training accuracy of ninety-three per cent, test of eighty-three, and a '
      + 'ten-point gap. A band states that a simpler network is not one with fewer parameters but '
      + 'one whose decision boundary has lower curvature, and that the right-hand model is strictly '
      + 'better on the data it was shown and strictly worse on everything else.',
  };
}

// ---------------------------------------------------------------- diagram 2
// The stopping point is marked on the validation curve, not between the two.
// The section's claim is that the training curve is not a stopping criterion at
// all, so the marker has to belong to the other one.
function lossDivergence() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('The signal is the divergence, not the descent',
    'Training loss almost always falls. Watching it tells you the optimiser works, and nothing about the model.')];

  const X0 = 110, X1 = 640, Y0 = 400, Y1 = 150;
  const px = (t) => X0 + t * (X1 - X0);
  const py = (l) => Y0 - ((l - 0.05) / 1.15) * (Y0 - Y1);

  els.push(...card(M, 110, 620, 330, { spine: T.primary }));
  els.push(line([[X0, Y1 - 8], [X0, Y0 + 8]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[X0 - 8, Y0], [X1 + 10, Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  [[1.0, '1.0'], [0.5, '0.5'], [0.1, '0.1']].forEach(([l, lab]) => {
    els.push(text(56, py(l) - 6, lab, { size: 9, family: 3, stroke: T.inkSubtle, align: 'right', width: 46 }));
  });
  els.push(text(62, 126, 'loss', { size: 10, stroke: T.inkSubtle }));
  els.push(text(px(0.5) - 60, 412, 'epochs', { size: 10, stroke: T.inkSubtle, align: 'center', width: 120 }));

  const train = [], val = [];
  for (let i = 0; i <= 60; i++) {
    const t = i / 60;
    train.push([px(t), py(0.08 + 1.02 * Math.exp(-3.4 * t))]);
    val.push([px(t), py(0.34 + 0.76 * Math.exp(-4.2 * t) + 0.62 * Math.pow(Math.max(0, t - 0.42), 2))]);
  }
  els.push(line(train, { stroke: T.primary, strokeWidth: 2.4, roughness: 0.3 }));
  els.push(line(val, { stroke: T.accent, strokeWidth: 2.4, roughness: 0.3 }));
  els.push(text(px(0.72), py(0.14) - 4, 'training loss', { size: 11, stroke: T.primary }));
  els.push(text(px(0.72), py(0.66) - 4, 'validation loss', { size: 11, stroke: T.accent }));

  // The minimum of the validation curve, marked on that curve.
  const tStar = 0.42;
  els.push(line([[px(tStar), Y1 - 4], [px(tStar), Y0]],
    { stroke: T.success, strokeWidth: 1.6, strokeStyle: 'dashed', roughness: 0.3 }));
  els.push(circle(px(tStar), py(0.34 + 0.76 * Math.exp(-4.2 * tStar)), 5,
    { stroke: T.success, fill: T.surface, strokeWidth: 1.8 }));
  els.push(text(px(tStar) - 70, Y1 - 24, 'stop here', { size: 12, stroke: T.success, align: 'center', width: 140 }));

  els.push(...card(700, 110, 220, 330, { spine: T.accent }));
  els.push(text(722, 122, 'Three phases', { size: 13, stroke: T.accent }));
  els.push(rule(716, 904, 146));
  const PH = [
    ['1 · together', 'Both fall. Training and\ngeneralising at once.', T.success],
    ['2 · plateau', 'Validation levels off while\ntraining keeps falling.', T.warn],
    ['3 · divergence', 'Validation climbs. Every\nextra epoch makes the\ndeployed model worse.', T.alert],
  ];
  PH.forEach((p, i) => {
    const y = 162 + i * 90;
    els.push(text(722, y, p[0], { size: 11, stroke: p[2] }));
    els.push(text(722, y + 20, fit(p[1], 10, 180, 'phase note'), { size: 10, stroke: T.inkMuted }));
  });

  els.push(rect(M, 464, 880, 56, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 476, 'The right epoch to stop is the minimum of the validation curve, and nothing about the training curve indicates it.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 500, 'Training past that point is not wasted compute. It is compute spent making the model worse.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Train loss against validation loss: divergence is the overfitting signal',
    desc: 'A hand-drawn line chart of loss against epochs. The training loss, in blue, falls smoothly '
      + 'and monotonically across the whole run. The validation loss, in orange, falls with it at '
      + 'first, levels off around the middle, and then climbs through the second half. A dashed '
      + 'green line marks the minimum of the validation curve, with a marker on that curve and the '
      + 'label stop here. A side card names the three phases: first both fall together, training and '
      + 'generalising at once; then validation levels off while training keeps falling; then '
      + 'validation climbs, and every extra epoch makes the deployed model worse. A band states that '
      + 'the right epoch to stop is the minimum of the validation curve and that nothing about the '
      + 'training curve indicates it, and that training past that point is not wasted compute but '
      + 'compute spent making the model worse.',
  };
}

// ---------------------------------------------------------------- diagram 3
// New. Section 4 turns two numbers into a four-way diagnosis with a different
// fix in each case. Drawing them side by side with identical bar scales is what
// shows that the gap, not the training accuracy, is what classifies a model.
function fourRegimes() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Four things a train–test gap can mean',
    'The same two numbers, read four ways. Each regime needs a different fix, and two of them need opposite ones.')];

  const REG = [
    {
      c: T.warn, name: 'Underfitting', train: 55, test: 53,
      diag: 'Not enough capacity, or\nnot enough training time.',
      fix: 'More neurons, more layers,\nmore epochs, a stronger\noptimiser.',
    },
    {
      c: T.success, name: 'Good fit', train: 88, test: 85,
      diag: 'The pattern was learned,\nnot the noise.',
      fix: 'Ship it, then watch for\ndistribution drift in\nproduction.',
    },
    {
      c: T.alert, name: 'Overfitting', train: 93, test: 83,
      diag: 'Noise was memorised along\nwith the pattern.',
      fix: 'Less capacity, fewer epochs,\nregularisation, dropout.\nParts 29 to 31.',
    },
    {
      c: T.alert, name: 'Distribution shift', train: 93, test: 40,
      diag: 'The test set was drawn from\nsomewhere else entirely.',
      fix: 'Re-curate the test set, or\ndomain-adapt. No amount of\nregularisation helps.',
    },
  ];

  const CW = 205, TOP = 90, CH = 272;
  REG.forEach((r, i) => {
    const x = M + i * 225, inner = CW - 32;
    els.push(...card(x, TOP, CW, CH, { spine: r.c }));
    els.push(text(x + 16, TOP + 12, fit(r.name, 13, inner, 'regime name'), { size: 13, stroke: r.c }));

    [['train', r.train, T.inkMuted], ['test', r.test, r.c]].forEach((b, j) => {
      const y = TOP + 44 + j * 36;
      els.push(text(x + 16, y, b[0], { size: 9, stroke: T.inkSubtle }));
      els.push(rect(x + 16, y + 12, 1.45 * b[1], 13, {
        stroke: b[2], fill: b[2], strokeWidth: 1.2, opacity: 34, roundness: null,
      }));
      els.push(text(x + 158, y + 12, `${b[1]}%`, { size: 10, family: 3, stroke: b[2] }));
    });
    els.push(text(x + 16, TOP + 118, `gap  ${r.train - r.test} points`,
      { size: 11, family: 3, stroke: r.c }));

    els.push(rule(x + 14, x + CW - 14, TOP + 142));
    els.push(text(x + 16, TOP + 152, fit(r.diag, 10, inner, 'diagnosis'), { size: 10, stroke: T.ink }));
    els.push(rule(x + 14, x + CW - 14, TOP + 196));
    els.push(text(x + 16, TOP + 206, fit(r.fix, 10, inner, 'fix'), { size: 10, stroke: T.inkSubtle }));
  });

  els.push(rect(M, 388, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 400, 'The gap is the diagnosis. A model at 55% with no gap and a model at 93% with a ten-point gap need opposite treatments.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 424, 'Only the last two are told apart by something outside the numbers: whether the test set came from the same distribution.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Four regimes a train–test gap can indicate, and the fix for each',
    desc: 'Four hand-drawn cards, each pairing a training and a test accuracy as bars on the same '
      + 'scale with the gap between them. Underfitting shows fifty-five per cent training and '
      + 'fifty-three test, a two-point gap, diagnosed as not enough capacity or training time and '
      + 'fixed with more neurons, layers, epochs or a stronger optimiser. Good fit shows '
      + 'eighty-eight and eighty-five, a three-point gap, diagnosed as the pattern having been '
      + 'learned rather than the noise, with the advice to ship it and watch for drift. Overfitting '
      + 'shows ninety-three and eighty-three, a ten-point gap, diagnosed as noise memorised along '
      + 'with the pattern and fixed by less capacity, fewer epochs, regularisation or dropout in '
      + 'Parts 29 to 31. Distribution shift shows ninety-three and forty, a fifty-three-point gap, '
      + 'diagnosed as a test set drawn from somewhere else and fixed only by re-curating or '
      + 'domain-adapting. A band notes that the gap is the diagnosis, that a model at fifty-five per '
      + 'cent with no gap and one at ninety-three with a ten-point gap need opposite treatments, and '
      + 'that only the last two are told apart by something outside the numbers.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { trainVsTest, lossDivergence, fourRegimes };

if (require.main === module) {
  emit('01-train-vs-test', trainVsTest());
  emit('02-loss-divergence', lossDivergence());
  emit('03-four-regimes', fourRegimes());
}
