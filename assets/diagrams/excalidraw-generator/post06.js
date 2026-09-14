// The post 06 diagrams, hand-drawn.
//
// Scenes 01 to 03 mirror the clean vector figures one directory up. Scene 04 is
// new: section 2 builds its whole argument on "that kink is everything" and the
// post never draws the kink, while section 2.2 lists four boundaries that had
// nowhere to sit.
//
//   01-why-nonlinearity       960 x 480
//   02-softmax-stability      960 x 480
//   03-forward-pass-pipeline  960 x 420
//   04-relu-anatomy           960 x 460   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// A row of value boxes, used for every stage of the softmax walk.
function valueRow(els, x, y, vals, o = {}) {
  const bw = o.bw ?? 76, bh = o.bh ?? 34, size = o.size ?? 12;
  vals.forEach((v, i) => {
    els.push(rect(x + i * bw, y, bw, bh, {
      stroke: o.c ?? T.ink, fill: o.fill ?? T.surface, strokeWidth: 1.4, roundness: null,
    }));
    els.push(text(x + i * bw, y + (bh - size) / 2 - 1, String(v), {
      size, family: 3, stroke: o.ink ?? T.ink, align: 'center', width: bw,
    }));
  });
  return { end: x + vals.length * bw, bottom: y + bh };
}

// ---------------------------------------------------------------- diagram 1
// The two panels share an axis frame and a scale, so the only difference the
// eye can find between them is the shape of the curve.
function whyNonlinearity() {
  resetSeq();
  const W = 960, H = 480;

  const els = [...heading('What an activation buys',
    'Two layers, drawn twice. The only change between the panels is one call between them.')];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 300, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 122, fit(o.sub, 11, 380 * 0.94, 'panel sub'),
      { size: 11, family: 3, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + 404, 148));
    // A shared frame: same origin, same extent, in both panels.
    els.push(line([[x + 46, 336], [x + 392, 336]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
    els.push(line([[x + 46, 170], [x + 46, 336]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
    els.push(text(x + 20, 348, fit(o.note, 11, 380, 'panel note'), { size: 11, stroke: T.inkMuted }));
  };

  panel(M, {
    c: T.alert, title: 'No activation', sub: 'Dense → Dense',
    note: 'One straight line, whatever the weights are set to.\nPart 03 showed the two layers collapse into one.',
  });
  els.push(line([[62, 322], [426, 188]], { stroke: T.alert, strokeWidth: 2.4, roughness: 0.5 }));
  els.push(text(240, 214, 'the only shape available', { size: 11, stroke: T.alert }));

  panel(500, {
    c: T.success, title: 'With ReLU between', sub: 'Dense → ReLU → Dense',
    note: 'A bend for every hidden neuron that crosses zero.\nEnough of them approximate any continuous shape.',
  });
  // Each vertex is one hidden unit switching on or off.
  const BENDS = [[522, 322], [590, 296], [652, 212], [714, 250], [782, 186], [886, 204]];
  els.push(line(BENDS, { stroke: T.success, strokeWidth: 2.4, roughness: 0.5 }));
  BENDS.slice(1, -1).forEach(([bx, by]) => {
    els.push(circle(bx, by, 4.5, { stroke: T.success, fill: T.surface, strokeWidth: 1.4 }));
  });
  els.push(text(700, 320, 'one kink per neuron', { size: 11, stroke: T.success }));

  els.push(rect(M, 412, 880, 60, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 424, 'The depth was already there. What the activation adds is the ability to bend.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 448, 'Without it the substitution from Part 03 still goes through, and a fifty-layer stack is still one line.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Why activations matter: linear stacks stay linear',
    desc: 'Two hand-drawn panels sharing the same axis frame and scale. The left panel, in alert red '
      + 'and labelled no activation, shows a Dense to Dense stack producing a single straight '
      + 'diagonal line, annotated as the only shape available whatever the weights are set to, with '
      + 'a note that Part 03 showed the two layers collapse into one. The right panel, in success '
      + 'green and labelled with ReLU between, shows a Dense to ReLU to Dense stack producing a '
      + 'piecewise-linear curve with four marked bends, annotated as one kink per neuron, with a '
      + 'note that enough of them approximate any continuous shape. A band underneath states that '
      + 'the depth was already there and what the activation adds is the ability to bend, and that '
      + 'without it a fifty-layer stack is still one line.',
  };
}

// ---------------------------------------------------------------- diagram 2
// Both panels run the same four steps in the same positions, so the row where
// they diverge is found by scanning down rather than by reading captions.
function softmaxStability() {
  resetSeq();
  const W = 960, H = 480;

  const els = [...heading('Softmax, naively and safely',
    'The same three logits down both columns. Subtracting the row maximum changes nothing except whether it works.')];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 300, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 122, fit(o.call, 11, 380 * 0.94, 'call'),
      { size: 11, family: 3, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + 404, 146));
    o.steps.forEach((s, i) => {
      const y = 158 + i * 48;
      els.push(text(x + 18, y + 10, fit(s.label, 10, 84, 'step label'), { size: 10, stroke: T.inkSubtle }));
      valueRow(els, x + 108, y, s.vals, { c: s.c ?? T.ink, ink: s.ink ?? T.ink, bw: 96, size: 12 });
    });
    els.push(text(x + 20, 350, fit(o.verdict, 12, 380, 'verdict'), { size: 12, stroke: o.c }));
  };

  panel(M, {
    c: T.alert, title: 'Naive', call: 'np.exp(logits)',
    steps: [
      { label: 'logits', vals: [1000, 1001, 999] },
      { label: 'shifted', vals: ['—', '—', '—'], c: T.border, ink: T.inkSubtle },
      { label: 'exp', vals: ['inf', 'inf', 'inf'], c: T.alert, ink: T.alert },
      { label: 'divide', vals: ['nan', 'nan', 'nan'], c: T.alert, ink: T.alert },
    ],
    verdict: 'e¹⁰⁰⁰ exceeds float range.\ninf / inf is nan, and nan spreads.',
  });

  panel(500, {
    c: T.success, title: 'Stable', call: 'np.exp(logits - np.max(logits, axis=1, keepdims=True))',
    steps: [
      { label: 'logits', vals: [1000, 1001, 999] },
      { label: 'minus max', vals: [-1, 0, -2], c: T.success },
      { label: 'exp', vals: ['0.368', '1.000', '0.135'], c: T.success },
      { label: 'divide', vals: ['0.245', '0.665', '0.090'], c: T.success },
    ],
    verdict: 'Largest exponent is exactly 0,\nso the largest e^x is exactly 1.',
  });

  els.push(rect(M, 410, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 422, 'exp(o_i - c) / Σ exp(o_j - c)   =   exp(o_i) / Σ exp(o_j)      for any constant c',
    { size: 13, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 448, 'Subtracting the same constant from every logit scales numerator and denominator by the same factor, which cancels.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Softmax stability: subtracting the per-row max keeps every exponent safe',
    desc: 'Two hand-drawn panels running the same four steps on the same three logits, one thousand, '
      + 'one thousand and one, and nine hundred and ninety-nine. The left panel, in alert red and '
      + 'labelled naive, skips the shift, exponentiates directly to infinity in all three slots, and '
      + 'divides to give nan in all three; its verdict is that e to the thousand exceeds the '
      + 'floating-point range and infinity over infinity is nan, which then spreads. The right '
      + 'panel, in success green and labelled stable, subtracts the row maximum to give minus one, '
      + 'nought and minus two, exponentiates to nought point three six eight, one, and nought point '
      + 'one three five, and divides to give the correct probabilities nought point two four five, '
      + 'nought point six six five and nought point nought nine nought; its verdict is that the '
      + 'largest exponent is exactly zero so the largest exponential is exactly one. A band gives '
      + 'the identity showing the two formulas are equal for any constant, because the shift scales '
      + 'numerator and denominator by the same factor.',
  };
}

// ---------------------------------------------------------------- diagram 3
// Four objects in a row, alternating dense and activation, with the shape under
// each. The shape stays (N, 3) from the first dense onward, which is the thing
// worth being able to see: the activations change values, never shapes.
function forwardPassPipeline() {
  resetSeq();
  const W = 960, H = 420;

  const els = [...heading('The complete forward pass',
    'Four objects in a row. Only the dense layers change the shape; the activations change the values.')];

  const X = [41, 223, 405, 587, 769];
  const CW = 150, CY = 130, CH = 92;

  const STAGES = [
    { c: T.primary, name: 'X', detail: 'spiral data', shape: '(N, 2)' },
    { c: T.accent, name: 'Dense', detail: '2 in · 3 out', shape: '(N, 3)' },
    { c: T.success, name: 'ReLU', detail: 'max(0, x)', shape: '(N, 3)' },
    { c: T.accent, name: 'Dense', detail: '3 in · 3 out', shape: '(N, 3)' },
    { c: T.primary, name: 'Softmax', detail: 'rows sum to 1', shape: '(N, 3)' },
  ];

  STAGES.forEach((s, i) => {
    const x = X[i];
    els.push(...card(x, CY, CW, CH, { stroke: s.c, spine: s.c }));
    els.push(text(x, CY + 20, s.name, { size: 17, stroke: s.c, align: 'center', width: CW }));
    els.push(text(x, CY + 50, fit(s.detail, 11, CW - 24, 'stage detail'),
      { size: 11, stroke: T.inkMuted, align: 'center', width: CW }));
    els.push(text(x, CY + 104, s.shape, { size: 13, family: 3, stroke: T.ink, align: 'center', width: CW }));
    if (i < 4) {
      els.push(arrow([[x + CW + 2, CY + CH / 2], [X[i + 1] - 2, CY + CH / 2]],
        { stroke: T.inkMuted, strokeWidth: 1.4, roughness: 0.5 }));
    }
  });

  els.push(text(223, 270, 'shape changes here', { size: 10, stroke: T.accent, align: 'center', width: 150 }));
  els.push(text(405, 270, 'values only', { size: 10, stroke: T.success, align: 'center', width: 150 }));
  els.push(text(769, 270, 'values only', { size: 10, stroke: T.primary, align: 'center', width: 150 }));

  els.push(rect(M, 300, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 312, 'Every object writes its result to self.output, and the next object reads it.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 336, 'dense1.forward(X) · relu.forward(dense1.output) · dense2.forward(relu.output) · softmax.forward(dense2.output)',
    { size: 10, family: 3, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 384, 'Only the two dense layers hold parameters. The two activations have no weights and nothing to learn.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'End-to-end forward pass: spiral data through Dense, ReLU, Dense, Softmax',
    desc: 'A hand-drawn left-to-right pipeline of five stages. Spiral data X of shape N by two enters '
      + 'a Dense layer of two in and three out, giving shape N by three; that passes through a ReLU '
      + 'activation computing max of nought and x, which leaves the shape unchanged; then a second '
      + 'Dense layer of three in and three out; then Softmax, whose rows sum to one, again leaving '
      + 'the shape at N by three. Annotations mark the first dense layer as the only place the shape '
      + 'changes and both activations as changing values only. A band records the calling '
      + 'convention, that every object writes its result to self.output and the next object reads '
      + 'it, and spells out the four calls in order. A footer notes that only the two dense layers '
      + 'hold parameters, while the two activations have no weights and nothing to learn.',
  };
}

// ---------------------------------------------------------------- diagram 4
// New. Section 2 rests on "that kink is everything" and never draws it, and
// section 2.2 lists four boundaries with nowhere to sit. The plot and the
// boundaries belong on one canvas, because each boundary is a property of the
// shape drawn beside it.
function reluAnatomy() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('ReLU, and what it does not do',
    'Two straight lines meeting at the origin. Every bend a network can make traces back to this one.')];

  const OX = 200, OY = 300;
  els.push(line([[70, OY], [430, OY]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  els.push(line([[OX, 120], [OX, 336]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  els.push(text(410, OY + 10, 'x', { size: 11, stroke: T.inkSubtle }));
  els.push(text(OX + 8, 118, 'f(x)', { size: 11, stroke: T.inkSubtle }));

  els.push(line([[74, OY], [OX, OY]], { stroke: T.success, strokeWidth: 2.6, roughness: 0.4 }));
  els.push(line([[OX, OY], [400, 152]], { stroke: T.success, strokeWidth: 2.6, roughness: 0.4 }));
  els.push(circle(OX, OY, 5.5, { stroke: T.success, fill: T.surface, strokeWidth: 1.6 }));

  // Above the flat arm rather than below it: the kink's leader comes up from
  // underneath, and below the arm the two collide.
  els.push(text(80, OY - 32, 'negative in → 0 out', { size: 11, stroke: T.inkMuted }));
  els.push(text(292, 196, 'positive in → unchanged', { size: 11, stroke: T.inkMuted }));
  els.push(arrow([[OX - 62, 344], [OX - 6, 310]], { stroke: T.accent, strokeWidth: 1.3, roughness: 0.5 }));
  els.push(text(70, 350, 'the kink, at x = 0', { size: 12, stroke: T.accent }));
  els.push(text(70, 376, 'f(x) = max(0, x)', { size: 14, family: 3, stroke: T.ink }));

  const BOUNDS = [
    {
      c: T.warn, head: 'Not smooth',
      body: 'The derivative jumps at x = 0. In practice an input is\nnever exactly zero, so implementations pick a side.',
    },
    {
      c: T.warn, head: 'Not zero-centred',
      body: 'Every output is non-negative, which biases the inputs\nof the layer after it.',
    },
    {
      c: T.alert, head: 'Not safe from dying',
      body: 'Weights that drive every input negative leave a neuron\nat zero with zero gradient, and it never updates again.',
    },
    {
      c: T.alert, head: 'Not a probability',
      body: 'Outputs can be 0, 5, or 10000. Turning them into a\ndistribution is what the output layer needs softmax for.',
    },
  ];
  BOUNDS.forEach((b, i) => {
    const y = 100 + i * 78;
    els.push(...card(500, y, 420, 70, { spine: b.c }));
    els.push(text(522, y + 10, b.head, { size: 13, stroke: b.c }));
    els.push(text(522, y + 32, fit(b.body, 10, 384, 'bound body'), { size: 10, stroke: T.inkMuted }));
  });

  els.push(text(M, 420, 'One kink is not much. A layer of them, shifted and scaled by their own weights, is enough to bend any continuous shape.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'ReLU: the kink at zero, and the four things it does not do',
    desc: 'A hand-drawn plot of the rectified linear unit on the left: a flat line along the x axis '
      + 'for negative inputs, a sharp kink at the origin marked with a circle and an arrow, and a '
      + 'straight diagonal rising for positive inputs, labelled f of x equals max of nought and x, '
      + 'with negative in giving nought out and positive in passing through unchanged. Four boundary '
      + 'cards on the right list what ReLU does not do: it is not smooth, since the derivative jumps '
      + 'at zero, though an input is never exactly zero in practice; it is not zero-centred, since '
      + 'every output is non-negative and that biases the next layer; it is not safe from dying, '
      + 'since weights that drive every input negative leave a neuron at zero with zero gradient, '
      + 'never updating again; and it is not a probability, since outputs can be nought, five or ten '
      + 'thousand, which is what the output layer needs softmax for. A footer notes that one kink is '
      + 'not much, but a layer of them shifted and scaled by their own weights can bend any '
      + 'continuous shape.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { whyNonlinearity, softmaxStability, forwardPassPipeline, reluAnatomy };

if (require.main === module) {
  emit('01-why-nonlinearity', whyNonlinearity());
  emit('02-softmax-stability', softmaxStability());
  emit('03-forward-pass-pipeline', forwardPassPipeline());
  emit('04-relu-anatomy', reluAnatomy());
}
