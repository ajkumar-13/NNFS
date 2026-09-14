// The post 29 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 5 names data leakage as the single most common source of inflated
// benchmarks, and the defence is an ordering rule — split before you fit — which
// is exactly the kind of thing two pipelines side by side can settle.
//
//   01-three-way-split-and-kfold  960 x 540
//   02-data-leakage               960 x 490   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// The split bar and the k-fold grid use the same colours for the same roles, so
// the rotating validation slot on the right is recognisably the same thing as
// the fixed slice on the left.
function threeWaySplitAndKfold() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Two ways to hold data back',
    'One set teaches the weights, one chooses the knobs, and one is opened exactly once.')];

  els.push(...card(M, 86, 420, 330, { spine: T.primary }));
  els.push(text(62, 98, 'Three-way split', { size: 15, stroke: T.primary }));
  els.push(text(62, 120, 'when data is plentiful', { size: 10, stroke: T.inkSubtle }));

  const BX = 62, BW = 380, BY = 152;
  const SLICES = [
    { f: 0.6, c: T.success, lab: 'training', pct: '60%' },
    { f: 0.2, c: T.warn, lab: 'validation', pct: '20%' },
    { f: 0.2, c: T.primary, lab: 'test', pct: '20%' },
  ];
  let cx = BX;
  SLICES.forEach((s) => {
    const w = s.f * BW;
    els.push(rect(cx, BY, w, 44, {
      stroke: s.c, fill: s.c, strokeWidth: 1.4, opacity: 34, roundness: null,
    }));
    els.push(text(cx, BY + 14, s.pct, { size: 12, family: 3, stroke: s.c, align: 'center', width: w }));
    els.push(text(cx, BY + 50, s.lab, { size: 10, stroke: s.c, align: 'center', width: w }));
    cx += w;
  });

  els.push(rule(56, 476, 226));
  const ROLES = [
    ['training', 'weights are learned from it', 'every pass', T.success],
    ['validation', 'hyperparameters are chosen by it', 'freely, between runs', T.warn],
    ['test', 'the final number, reported once', 'once, at the very end', T.primary],
  ];
  ROLES.forEach((r, i) => {
    const y = 240 + i * 52;
    els.push(text(62, y, r[0], { size: 12, stroke: r[3] }));
    els.push(text(62, y + 18, fit(r[1], 10, 380, 'role'), { size: 10, stroke: T.ink }));
    els.push(text(62, y + 34, fit(r[2], 9, 380, 'when'), { size: 9, stroke: T.inkSubtle }));
  });

  els.push(...card(500, 86, 420, 330, { spine: T.warn }));
  els.push(text(522, 98, '5-fold cross-validation', { size: 15, stroke: T.warn }));
  els.push(text(522, 120, 'when a 20% slice would be too few examples', { size: 10, stroke: T.inkSubtle }));

  const FX = 560, FW = 320, SW = FW / 5;
  ['A', 'B', 'C', 'D', 'E'].forEach((lab, i) => {
    els.push(text(FX + i * SW, 140, lab, { size: 10, stroke: T.inkSubtle, align: 'center', width: SW }));
  });
  for (let f = 0; f < 5; f++) {
    const y = 162 + f * 38;
    els.push(text(522, y + 8, `${f + 1}`, { size: 10, family: 3, stroke: T.inkSubtle }));
    for (let s = 0; s < 5; s++) {
      const isVal = s === f;
      els.push(rect(FX + s * SW, y, SW, 28, {
        stroke: isVal ? T.warn : T.success, fill: isVal ? T.warn : T.success,
        strokeWidth: 1.3, opacity: isVal ? 42 : 16, roundness: null,
      }));
    }
  }
  els.push(text(522, 360, 'Every example takes a turn in the validation slot,', { size: 10, stroke: T.inkMuted }));
  els.push(text(522, 376, 'and the score is the mean of the five.', { size: 10, stroke: T.inkMuted }));
  els.push(text(522, 398, 'Averaging shrinks the standard error, which is what\nresolves two candidates a single slice cannot.',
    { size: 10, stroke: T.inkSubtle }));

  els.push(rect(M, 442, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 454, 'Both patterns exist to keep one slice of data unseen until every design decision has been frozen.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 478, 'K-fold rotates the validation slot; it does not touch the test set, which sits outside both diagrams.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Three-way split and 5-fold cross-validation, side by side',
    desc: 'Two hand-drawn panels. The left shows the three-way split used when data is plentiful: one '
      + 'horizontal bar divided sixty per cent green for training, twenty per cent tan for '
      + 'validation and twenty per cent blue for test, with a table beneath giving each slice its '
      + 'role and when it is touched — training every pass, validation freely between runs, and test '
      + 'once at the very end. The right shows five-fold cross-validation as five rows of five '
      + 'segments labelled A to E, with the validation slot in a different position on each row and '
      + 'the remaining four segments used for training. A note records that every example takes a '
      + 'turn in the validation slot and the score is the mean of the five, which shrinks the '
      + 'standard error. A band states that both patterns exist to keep one slice unseen until every '
      + 'design decision is frozen, and that k-fold rotates the validation slot without touching the '
      + 'test set.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. The defence against leakage is an ordering rule, and an ordering rule is
// what two pipelines drawn in parallel can settle in a way prose cannot: the
// same three boxes, in a different sequence, with one arrow pointing the wrong
// way.
function dataLeakage() {
  resetSeq();
  const W = 960, H = 490;

  const els = [...heading('Leakage is an ordering mistake',
    'The same three operations. Doing them in the wrong sequence puts the held-out data inside the model.')];

  const pipeline = (x, o) => {
    els.push(...card(x, 86, 420, 190, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 120, o.sub, { size: 10, stroke: T.inkSubtle }));
    o.steps.forEach((s, i) => {
      const y = 148 + i * 40;
      els.push(rect(x + 20, y, 300, 30, { stroke: o.c, fill: T.surface, strokeWidth: 1.3 }));
      els.push(text(x + 20, y + 8, fit(s, 11, 280 * 0.94, 'step'),
        { size: 11, family: 3, stroke: T.ink, align: 'center', width: 300 }));
      if (i < o.steps.length - 1) {
        els.push(arrow([[x + 170, y + 32], [x + 170, y + 38]],
          { stroke: o.c, strokeWidth: 1.2, roughness: 0.3 }));
      }
    });
    els.push(text(x + 336, 160, o.verdictHead, { size: 12, stroke: o.c }));
    els.push(text(x + 336, 182, fit(o.verdict, 9, 82, 'verdict'), { size: 9, stroke: T.inkMuted }));
  };

  pipeline(M, {
    c: T.success, title: 'Split first', sub: 'the only safe order',
    steps: ['shuffle and split', 'fit the scaler on train', 'apply it to val and test'],
    verdictHead: 'clean',
    verdict: 'No held-out\nstatistic ever\nreaches the\ntraining set.',
  });

  pipeline(500, {
    c: T.alert, title: 'Preprocess first', sub: 'the usual mistake',
    steps: ['fit the scaler on everything', 'shuffle and split', 'train'],
    verdictHead: 'leaked',
    verdict: 'Training data\nnow carries the\ntest set’s mean\nand variance.',
  });

  const PAT = [
    {
      c: T.alert, name: 'Global preprocessing',
      body: 'Standardising with the full-dataset\nmean, then splitting. Same for PCA,\nencoders and vocabularies.',
    },
    {
      c: T.alert, name: 'Look-ahead features',
      body: 'A feature that quietly depends on a\nvalue from after the prediction point.\nSplit sequential data by time.',
    },
    {
      c: T.alert, name: 'Label-correlated metadata',
      body: '“Did the customer call to cancel?”\nonly happens after a non-purchase,\nso the feature encodes the label.',
    },
  ];
  els.push(text(M, 296, 'Three ways it gets in', { size: 13, stroke: T.ink }));
  PAT.forEach((p, i) => {
    const x = M + i * 300;
    els.push(...card(x, 318, 280, 96, { spine: p.c }));
    els.push(text(x + 18, 328, fit(p.name, 12, 244, 'pattern name'), { size: 12, stroke: p.c }));
    els.push(text(x + 18, 350, fit(p.body, 9, 244, 'pattern body'), { size: 9, stroke: T.inkMuted }));
  });

  els.push(rect(M, 430, 880, 46, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 444, 'The test set is invisible until the final report. For k-fold, refit every preprocessor inside the loop, on that fold’s training portion.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Data leakage: how held-out information reaches the model',
    desc: 'A hand-drawn figure. Two pipelines run in parallel over the same three operations. The '
      + 'left, in success green and labelled split first, shuffles and splits, fits the scaler on '
      + 'the training portion, then applies it to validation and test; its verdict is clean, because '
      + 'no held-out statistic ever reaches the training set. The right, in alert red and labelled '
      + 'preprocess first, fits the scaler on everything, then shuffles and splits, then trains; its '
      + 'verdict is leaked, because the training data now carries the test set’s mean and variance. '
      + 'Three cards beneath name the common leak patterns: global preprocessing, standardising with '
      + 'the full-dataset mean before splitting and likewise for PCA, encoders and vocabularies; '
      + 'look-ahead features, where a feature quietly depends on a value from after the prediction '
      + 'point; and label-correlated metadata, such as asking whether a customer called to cancel, '
      + 'which only happens after a non-purchase and so encodes the label. A band states that the '
      + 'test set is invisible until the final report, and that k-fold must refit every preprocessor '
      + 'inside the loop on that fold’s training portion.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { threeWaySplitAndKfold, dataLeakage };

if (require.main === module) {
  emit('01-three-way-split-and-kfold', threeWaySplitAndKfold());
  emit('02-data-leakage', dataLeakage());
}
