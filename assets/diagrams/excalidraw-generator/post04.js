// The post 04 diagrams, hand-drawn.
//
// Scenes 01 to 03 mirror the clean vector figures one directory up. Scene 04 is
// new: section 2.1 argues for the spiral set over MNIST on the grounds of scale,
// and that argument had no figure.
//
//   01-spiral-data             960 x 540
//   02-weight-convention       960 x 460
//   03-dense-layer-class       960 x 540
//   04-why-spirals-not-mnist   960 x 440   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

const CLASS_C = [T.primary, T.accent, T.success];

// nnfs.datasets.spiral_data, reimplemented so the sketch plots the same curve
// the reader's own `spiral_data(samples=100, classes=3)` call produces:
//
//   r = linspace(0, 1, n)
//   t = linspace(c*4, (c+1)*4, n) + randn(n)*0.2
//   x, y = r*sin(t*2.5), r*cos(t*2.5)
//
// The noise term is the one thing that cannot be copied: nnfs draws it from a
// seeded normal. A hashed sine stands in for it, which keeps the scatter looking
// sampled rather than swept while staying byte-identical across rebuilds — an
// actual random draw would put a diff in every regeneration.
function spiralPoints(nPerClass, classes = 3) {
  const jitter = (i) => {
    const s = Math.sin(i * 12.9898) * 43758.5453;
    return (s - Math.floor(s)) * 2 - 1;
  };
  const out = [];
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < nPerClass; i++) {
      const f = i / (nPerClass - 1);
      const r = f;
      // nnfs uses randn * 0.2 here. A uniform hash with the same half-width
      // scatters harder, because a normal draw spends most of its mass near
      // zero and this does not, so the arms stop reading as arms. 0.12 puts the
      // visible spread back where the seeded original has it.
      const t = c * 4 + f * 4 + jitter(c * nPerClass + i) * 0.12;
      out.push({ x: r * Math.sin(t * 2.5), y: r * Math.cos(t * 2.5), c });
    }
  }
  return out;
}

function scatter(els, cx, cy, scale, pts, dotR) {
  for (const p of pts) {
    els.push(circle(cx + p.x * scale, cy - p.y * scale, dotR, {
      stroke: CLASS_C[p.c], fill: CLASS_C[p.c], strokeWidth: 0.8, roughness: 0.4,
    }));
  }
}

// ---------------------------------------------------------------- diagram 1
// The hero. The dashed line is the whole argument: it is placed to be the most
// generous straight cut available, and it still leaves every class on both
// sides of itself.
function spiralData() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('The spiral dataset',
    'Three classes wound through each other. No straight line separates even two of them.')];

  const CX = 280, CY = 276, S = 150;

  // Axes first, so 300 dots land on top of them rather than under.
  els.push(line([[CX - S - 16, CY], [CX + S + 16, CY]],
    { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  els.push(line([[CX, CY - S - 16], [CX, CY + S + 16]],
    { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  els.push(text(CX + S - 26, CY + 10, 'X₁', { size: 11, stroke: T.inkSubtle }));
  els.push(text(CX + 8, CY - S - 12, 'X₂', { size: 11, stroke: T.inkSubtle }));

  scatter(els, CX, CY, S, spiralPoints(100), 2.6);

  // The best straight cut available, and it still fails.
  els.push(line([[CX - S - 10, CY + 86], [CX + S + 10, CY - 94]],
    { stroke: T.alert, strokeWidth: 2, strokeStyle: 'dashed', roughness: 0.4 }));
  els.push(text(CX - 190, CY + S + 22, 'any straight line leaves all three classes on both sides',
    { size: 11, stroke: T.alert, align: 'center', width: 380 }));

  els.push(...card(560, 110, 360, 300, { spine: T.primary }));
  els.push(text(584, 124, 'The dataset', { size: 16, stroke: T.primary }));
  els.push(rule(576, 904, 154));
  const ROWS = [
    ['samples', '300', '100 per class'],
    ['features', '2', 'X₁ and X₂, plottable'],
    ['classes', '3', 'labels 0, 1, 2'],
    ['shape of X', '(300, 2)', 'one row per point'],
    ['shape of y', '(300,)', 'one label per row'],
  ];
  ROWS.forEach((r, i) => {
    const y = 166 + i * 34;
    els.push(text(584, y, r[0], { size: 11, stroke: T.inkMuted }));
    els.push(text(692, y, r[1], { size: 12, family: 3, stroke: T.ink }));
    els.push(text(772, y, fit(r[2], 10, 140, 'stat note'), { size: 10, stroke: T.inkSubtle }));
  });
  els.push(rule(576, 904, 348));
  els.push(text(584, 360, 'Non-linear, and not linearly separable.', { size: 12, stroke: T.alert }));
  els.push(text(584, 382, 'Chance for three classes is 33%.', { size: 11, stroke: T.inkMuted }));

  els.push(text(M, 478, 'A linear model cannot beat chance here, whatever its weights, because one straight boundary cannot carve three spiral regions.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 500, 'This is the benchmark the rest of the series trains against, and Part 06 is where a network first becomes able to solve it.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'The spiral dataset: three intertwined classes',
    desc: 'A hand-drawn scatter plot of three hundred points in two dimensions, one hundred per '
      + 'class, forming three spirals wound through each other from the origin and coloured blue, '
      + 'orange and green. A dashed red line cuts across the plot to show the failure mode of any '
      + 'linear classifier: whichever way the line is drawn, all three classes fall on both sides of '
      + 'it. A side panel lists the dataset properties: three hundred samples at one hundred per '
      + 'class, two features called X1 and X2, three classes labelled zero to two, an X of shape '
      + 'three hundred by two and a y of shape three hundred, and the note that the data is '
      + 'non-linear and not linearly separable with chance for three classes at thirty-three per '
      + 'cent. A footer explains that a linear model cannot beat chance here whatever its weights, '
      + 'because a single straight boundary cannot carve three spiral regions.',
  };
}

// ---------------------------------------------------------------- diagram 2
// Both panels hold the same nine weights. What moves is which axis a neuron
// runs along, so each neuron is drawn as one outlined block rather than as a
// tint on a shared grid: the block is the neuron, and its orientation is the
// entire difference between the two conventions.
function weightConvention() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('Two weight conventions, one arithmetic',
    'The same numbers laid out two ways. Which axis a neuron runs along decides whether the call needs a transpose.')];

  const CELL = 44;

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 234, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 120, o.sub, { size: 11, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + 404, 144));
    els.push(text(x, 268, o.shape, { size: 13, family: 3, stroke: T.ink, align: 'center', width: 420 }));
    els.push(text(x, 292, o.call, { size: 13, family: 3, stroke: o.c, align: 'center', width: 420 }));
  };

  // --- old: one row per neuron, W is (m, n) = (3, 2)
  panel(M, {
    c: T.inkMuted, title: 'Old · Parts 01 to 03', sub: 'one row per neuron',
    shape: 'W  (3, 2)', call: 'np.dot(X, W.T) + b',
  });
  const OX = 206, OY = 156;
  [0, 1, 2].forEach((k) => {
    els.push(rect(OX, OY + k * 36, CELL * 2, 36, {
      stroke: CLASS_C[k], fill: T.surface, strokeWidth: 1.5, roundness: null,
    }));
    els.push(line([[OX + CELL, OY + k * 36], [OX + CELL, OY + k * 36 + 36]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
    els.push(text(OX - 92, OY + k * 36 + 10, `neuron ${k + 1}`, { size: 11, stroke: CLASS_C[k] }));
  });

  // --- new: one column per neuron, W is (n, m) = (2, 3)
  panel(500, {
    c: T.primary, title: 'New · Part 04 onward', sub: 'one column per neuron',
    shape: 'W  (2, 3)', call: 'np.dot(X, W) + b',
  });
  const NX = 644, NY = 166;
  [0, 1, 2].forEach((k) => {
    els.push(rect(NX + k * CELL, NY, CELL, 72, {
      stroke: CLASS_C[k], fill: T.surface, strokeWidth: 1.5, roundness: null,
    }));
    els.push(line([[NX + k * CELL, NY + 36], [NX + k * CELL + CELL, NY + 36]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
    els.push(text(NX + k * CELL, NY + 78, `n${k + 1}`, { size: 11, stroke: CLASS_C[k], align: 'center', width: CELL }));
  });

  els.push(rect(M, 348, 880, 64, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 362, 'The arithmetic is identical and the output numbers are the same.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 386, 'Only the layout of W in memory and the shape of the call site change.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 432, 'The new layout is also the shape the weights are allocated in, so initialisation and the dot product stay consistent.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Two weight conventions: rows per neuron versus columns per neuron',
    desc: 'Two hand-drawn panels comparing the two weight-matrix layouts for a dense layer with two '
      + 'inputs and three neurons. The left panel, labelled old and covering Parts 01 to 03, stores '
      + 'one row per neuron: W has shape three by two, each neuron is drawn as a horizontal block '
      + 'running across the columns, and the forward pass is np.dot of X and W transposed plus b. '
      + 'The right panel, labelled new and covering Part 04 onward, stores one column per neuron: W '
      + 'has shape two by three, each neuron is drawn as a vertical block running down the rows, '
      + 'and the forward pass is np.dot of X and W with no transpose. Each of the three neurons is '
      + 'coloured consistently across both panels. A band states that the arithmetic is identical '
      + 'and the output numbers are the same, and that only the layout of W in memory and the shape '
      + 'of the call site change, with a footer noting that the new layout is also the shape the '
      + 'weights are allocated in.',
  };
}

// ---------------------------------------------------------------- diagram 3
// The blueprint above, two instances below. The instances are drawn as separate
// cards with their own arrays spelled out, because the thing that confuses
// first readers is not what the class does but that each instance owns a
// private copy of everything in it.
function denseLayerClass() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Layer_Dense: one blueprint, independent instances',
    'Two methods. One allocates the state that travels with the layer, the other runs the call.')];

  els.push(...card(M, 86, 880, 180, { stroke: T.primary, strokeWidth: 1.6 }));
  els.push(text(62, 98, 'class Layer_Dense', { size: 16, family: 3, stroke: T.primary }));

  const method = (x, o) => {
    els.push(...card(x, 126, 406, 124, { spine: o.c, fill: T.neutral1 }));
    els.push(text(x + 20, 136, o.sig, { size: 12, family: 3, stroke: o.c }));
    els.push(text(x + 20, 160, fit(o.body, 10, 370 * 0.94, 'method body'),
      { size: 10, family: 3, stroke: T.ink }));
    els.push(text(x + 20, 216, fit(o.note, 10, 370, 'method note'), { size: 10, stroke: T.inkSubtle }));
  };
  method(62, {
    c: T.accent, sig: '__init__(self, n_inputs, n_neurons)',
    body: 'self.weights = 0.01 * np.random.randn(\n                   n_inputs, n_neurons)\nself.biases  = np.zeros((1, n_neurons))',
    note: 'allocates the state · runs once, at construction',
  });
  method(492, {
    c: T.success, sig: 'forward(self, inputs)',
    body: 'self.output = np.dot(inputs, self.weights)\n              + self.biases',
    note: 'stored on self, not returned:\nthe backward pass needs it again',
  });

  els.push(text(M, 288, 'Two instances, sharing the class and nothing else',
    { size: 14, stroke: T.ink }));

  const inst = (x, o) => {
    els.push(...card(x, 312, 300, 142, { spine: o.c }));
    els.push(text(x + 20, 324, o.ctor, { size: 12, family: 3, stroke: o.c }));
    els.push(rule(x + 16, x + 284, 350));
    els.push(text(x + 20, 360, o.w, { size: 11, family: 3, stroke: T.ink }));
    els.push(text(x + 20, 382, o.b, { size: 11, family: 3, stroke: T.ink }));
    els.push(text(x + 20, 412, fit(o.note, 10, 264, 'instance note'), { size: 10, stroke: T.inkSubtle }));
  };
  inst(100, {
    c: T.accent, ctor: 'dense1 = Layer_Dense(2, 3)',
    w: 'weights  (2, 3)', b: 'biases   (1, 3)',
    note: '2 spiral features in, 3 neurons out',
  });
  inst(560, {
    c: T.success, ctor: 'dense2 = Layer_Dense(3, 3)',
    w: 'weights  (3, 3)', b: 'biases   (1, 3)',
    note: '3 inputs, because that is what dense1 emits',
  });
  els.push(arrow([[400, 372], [560, 372]], { stroke: T.inkMuted, strokeWidth: 1.4, roughness: 0.5 }));
  els.push(text(400, 344, 'dense1.output', { size: 11, family: 3, stroke: T.inkMuted, align: 'center', width: 160 }));

  els.push(text(M, 478, 'dense1.weights and dense2.weights are different arrays. Nothing about one instance reaches the other.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 500, 'That encapsulation is what the class buys: state travels with the layer instead of being smeared across the script.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'The Layer_Dense class: anatomy and lifecycle',
    desc: 'A hand-drawn schematic of the Layer_Dense class. The upper card holds the blueprint with '
      + 'its two methods. The constructor, taking n_inputs and n_neurons, allocates self.weights as '
      + 'zero-point-zero-one times a standard normal draw of shape n_inputs by n_neurons and '
      + 'self.biases as zeros of shape one by n_neurons; it is annotated as allocating the state and '
      + 'running once at construction. The forward method computes self.output as np.dot of inputs '
      + 'and self.weights plus self.biases, annotated as stored on self rather than returned because '
      + 'the backward pass needs it again. Below, two instance cards: dense1 built as Layer_Dense of '
      + 'two and three, holding weights of shape two by three and biases of shape one by three for '
      + 'the two spiral features; and dense2 built as Layer_Dense of three and three, taking three '
      + 'inputs because that is what dense1 emits. An arrow labelled dense1.output runs from the '
      + 'first instance to the second. A footer notes that the two weight arrays are entirely '
      + 'separate and that this encapsulation is what the class buys.',
  };
}

// ---------------------------------------------------------------- diagram 4
// New. Section 2.1 defends the dataset choice on scale, and the defence is
// numeric: two features against seven hundred and eighty-four is the whole
// argument, so the figure puts the two counts side by side at the same scale.
function whySpiralsNotMnist() {
  resetSeq();
  const W = 960, H = 440;

  const els = [...heading('Why spirals and not MNIST, for now',
    'The choice is about how much of the computation stays inspectable, not about which dataset is more realistic.')];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 234, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 16, stroke: o.c }));
    els.push(text(x + 20, 122, o.sub, { size: 11, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + 404, 148));
    o.rows.forEach((r, i) => {
      const y = 162 + i * 28;
      els.push(text(x + 216, y, r[0], { size: 11, stroke: T.inkMuted }));
      els.push(text(x + 306, y, r[1], { size: 12, family: 3, stroke: T.ink }));
    });
    els.push(text(x + 20, 272, fit(o.verdict, 11, 384, 'verdict'), { size: 11, stroke: o.c }));
  };

  panel(M, {
    c: T.success, title: 'Spiral', sub: 'this series, Parts 04 to 34',
    rows: [['samples', '300'], ['features', '2'], ['classes', '3']],
    verdict: 'Every intermediate array plots on one chart\nand can be checked by eye.',
  });
  // 40 per class rather than 24: below about thirty the arms stop resolving and
  // the panel argues for a blob instead of for a plottable structure.
  scatter(els, 128, 206, 56, spiralPoints(40), 1.9);

  panel(500, {
    c: T.inkMuted, title: 'MNIST', sub: 'the project, once the network works',
    rows: [['samples', '60 000'], ['features', '784'], ['classes', '10']],
    verdict: '784 dimensions, and nothing that can be\nplotted directly against a decision boundary.',
  });
  // 28 x 28, drawn as a coarse grid: the point is the count, not the pixels.
  const GX = 566, GY = 162, GS = 84;
  els.push(rect(GX, GY, GS, GS, { stroke: T.inkMuted, fill: T.surface, strokeWidth: 1.4, roundness: null }));
  for (let i = 1; i < 7; i++) {
    els.push(line([[GX + i * (GS / 7), GY], [GX + i * (GS / 7), GY + GS]],
      { stroke: T.border, strokeWidth: 0.8, roughness: 0.2 }));
    els.push(line([[GX, GY + i * (GS / 7)], [GX + GS, GY + i * (GS / 7)]],
      { stroke: T.border, strokeWidth: 0.8, roughness: 0.2 }));
  }
  els.push(text(GX - 28, GY + GS + 6, '28 × 28 = 784', { size: 10, stroke: T.inkSubtle, align: 'center', width: 140 }));

  els.push(rect(M, 348, 880, 60, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 362, 'Two features is what makes the next thirty posts checkable by hand.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 386, 'Small enough to inspect beats realistic, until the thing being inspected is known to work.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Why the series starts on spiral data rather than MNIST',
    desc: 'Two hand-drawn panels comparing the two datasets on scale. The left panel, spiral, used '
      + 'through Parts 04 to 34, lists three hundred samples, two features and three classes beside '
      + 'a small scatter of the three spirals, and concludes that every intermediate array plots on '
      + 'one chart and can be checked by eye. The right panel, MNIST, held back until the project '
      + 'once the network works, lists sixty thousand samples, seven hundred and eighty-four '
      + 'features and ten classes beside a coarse twenty-eight by twenty-eight grid, and concludes '
      + 'that seven hundred and eighty-four dimensions leave nothing that can be plotted directly '
      + 'against a decision boundary. A band underneath states that two features is what makes the '
      + 'next thirty posts checkable by hand, and that small enough to inspect beats realistic until '
      + 'the thing being inspected is known to work.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { spiralData, weightConvention, denseLayerClass, whySpiralsNotMnist };

if (require.main === module) {
  emit('01-spiral-data', spiralData());
  emit('02-weight-convention', weightConvention());
  emit('03-dense-layer-class', denseLayerClass());
  emit('04-why-spirals-not-mnist', whySpiralsNotMnist());
}
