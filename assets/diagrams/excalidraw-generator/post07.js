// The post 07 diagrams, hand-drawn.
//
// Scenes 01 and 02 mirror the clean vector figures one directory up. Scene 03 is
// new: section 5 reads the uniform output as two separate results — the pipeline
// is correct, and it knows nothing — and neither had a figure.
//
//   01-pipeline-anatomy  960 x 540
//   02-shape-audit       960 x 420
//   03-uniform-baseline  960 x 470   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// A shape drawn as its two axes, one box each, so a batch axis that never moves
// can be coloured differently from a feature axis that does.
function shapeBoxes(els, x, y, vals, colours, o = {}) {
  const bw = o.bw ?? 50, bh = o.bh ?? 44, size = o.size ?? 15;
  vals.forEach((v, i) => {
    els.push(rect(x + i * bw, y, bw, bh, {
      stroke: colours[i], fill: T.surface, strokeWidth: 1.5, roundness: null,
    }));
    els.push(text(x + i * bw, y + (bh - size) / 2 - 1, String(v), {
      size, family: 3, stroke: T.ink, align: 'center', width: bw,
    }));
  });
  return { end: x + vals.length * bw, cx: x + (vals.length * bw) / 2 };
}

// ---------------------------------------------------------------- diagram 1
// The pipeline above, the code that builds it below, in the same left-to-right
// order. The two halves line up column for column so an object in the diagram
// and the line that creates it are found at the same place on the page.
function pipelineAnatomy() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('The complete forward pass, as four objects',
    'Everything from Parts 04 and 06, wired together. Nothing here is new; the wiring is the post.')];

  const STAGES = [
    { x: 50, w: 100, c: T.primary, name: 'X', sub: 'spiral data' },
    { x: 170, w: 170, c: T.accent, name: 'dense1', sub: 'Layer_Dense(2, 3)' },
    { x: 360, w: 170, c: T.success, name: 'activation1', sub: 'Activation_ReLU()' },
    { x: 550, w: 170, c: T.accent, name: 'dense2', sub: 'Layer_Dense(3, 3)' },
    { x: 740, w: 170, c: T.primary, name: 'activation2', sub: 'Activation_Softmax()' },
  ];
  STAGES.forEach((s, i) => {
    els.push(...card(s.x, 100, s.w, 86, { stroke: s.c, spine: s.c }));
    els.push(text(s.x, 116, fit(s.name, 15, s.w - 24, 'stage name'),
      { size: 15, stroke: s.c, align: 'center', width: s.w }));
    els.push(text(s.x, 146, fit(s.sub, 10, (s.w - 24) * 0.94, 'stage sub'),
      { size: 10, family: 3, stroke: T.inkMuted, align: 'center', width: s.w }));
    if (i < STAGES.length - 1) {
      els.push(arrow([[s.x + s.w + 2, 143], [STAGES[i + 1].x - 2, 143]],
        { stroke: T.inkMuted, strokeWidth: 1.4, roughness: 0.5 }));
    }
  });
  els.push(text(740, 196, 'per-row probability distributions',
    { size: 11, stroke: T.primary, align: 'center', width: 170 }));

  const CODE = [
    {
      x: M, c: T.inkMuted, title: 'Classes',
      body: 'class Layer_Dense:\n    __init__, forward\n\nclass Activation_ReLU:\n    forward\n\nclass Activation_Softmax:\n    forward',
      note: 'Reused unchanged from\nParts 04 and 06.',
    },
    {
      x: 340, c: T.accent, title: 'Build',
      body: 'dense1 = Layer_Dense(2, 3)\nactivation1 = Activation_ReLU()\n\ndense2 = Layer_Dense(3, 3)\nactivation2 = \\\n    Activation_Softmax()',
      note: 'Four instances, each with\nits own arrays.',
    },
    {
      x: 640, c: T.success, title: 'Forward',
      body: 'dense1.forward(X)\nactivation1.forward(\n    dense1.output)\ndense2.forward(\n    activation1.output)\nactivation2.forward(\n    dense2.output)',
      note: 'Each call reads the\nprevious object’s output.',
    },
  ];
  CODE.forEach((c) => {
    els.push(...card(c.x, 240, 280, 250, { spine: c.c }));
    els.push(text(c.x + 20, 252, c.title, { size: 14, stroke: c.c }));
    els.push(rule(c.x + 16, c.x + 264, 276));
    els.push(text(c.x + 20, 286, fit(c.body, 10, 244 * 0.94, 'code body'),
      { size: 10, family: 3, stroke: T.ink }));
    els.push(text(c.x + 20, 442, fit(c.note, 10, 244, 'code note'), { size: 10, stroke: T.inkSubtle }));
  });

  els.push(text(M, 510, 'This is the inference path in full. Nothing in it learns; Part 08 adds the loss and Part 09 onward adds the loop that changes the weights.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'The complete forward pass: four-object inference pipeline',
    desc: 'A hand-drawn diagram in two halves. The upper half runs left to right through five stages: '
      + 'the spiral input X, a dense layer built as Layer_Dense of two and three, a ReLU activation, '
      + 'a second dense layer of three and three, and a softmax activation producing per-row '
      + 'probability distributions. The lower half shows the code in three cards aligned under the '
      + 'pipeline: Classes, holding the three class definitions reused unchanged from Parts 04 and '
      + '06; Build, instantiating the four objects each with its own arrays; and Forward, calling '
      + 'forward on each in order, where every call reads the previous object’s output. A footer '
      + 'notes that this is the inference path in full, that nothing in it learns, and that Part 08 '
      + 'adds the loss while Part 09 onward adds the loop that changes the weights.',
  };
}

// ---------------------------------------------------------------- diagram 2
// Every shape is drawn as two boxes so the batch axis and the feature axis can
// carry different colours. The batch column is the same colour and the same
// number across all five shapes, which is the observation the section makes.
function shapeAudit() {
  resetSeq();
  const W = 960, H = 420;

  const els = [...heading('Shape audit through the whole pipeline',
    'Five shapes, three hundred rows throughout. Only the feature axis moves, and only at a dense layer.')];

  const XS = [50, 240, 430, 620, 810];
  const SHAPES = [[300, 2], [300, 3], [300, 3], [300, 3], [300, 3]];
  const OPS = [
    { name: 'dense1', note: '2 → 3', c: T.accent },
    { name: 'activation1', note: 'unchanged', c: T.success },
    { name: 'dense2', note: '3 → 3', c: T.accent },
    { name: 'activation2', note: 'unchanged', c: T.primary },
  ];

  SHAPES.forEach((s, i) => {
    shapeBoxes(els, XS[i], 150, s, [T.primary, T.accent]);
  });
  OPS.forEach((op, i) => {
    const a = XS[i] + 100, b = XS[i + 1];
    els.push(arrow([[a + 6, 172], [b - 6, 172]], { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.5 }));
    els.push(text((a + b) / 2 - 60, 128, op.name, { size: 11, stroke: op.c, align: 'center', width: 120 }));
    els.push(text((a + b) / 2 - 60, 186, op.note, { size: 10, stroke: T.inkSubtle, align: 'center', width: 120 }));
  });


  // The batch axis, tied across every shape it appears in.
  els.push(line([[62, 240], [872, 240]],
    { stroke: T.primary, strokeWidth: 1.3, strokeStyle: 'dashed', roughness: 0.3 }));
  XS.forEach((x) => els.push(line([[x + 25, 196], [x + 25, 240]],
    { stroke: T.primary, strokeWidth: 1, strokeStyle: 'dotted', roughness: 0.3 })));
  els.push(text(M, 248, 'the batch axis never changes · 300 rows in, 300 rows out',
    { size: 11, stroke: T.primary, align: 'center', width: 880 }));
  // Below the tie, where the dotted droppers have already stopped.
  els.push(text(50, 264, 'X', { size: 11, stroke: T.inkSubtle, align: 'center', width: 100 }));
  els.push(text(810, 264, 'probabilities', { size: 11, stroke: T.inkSubtle, align: 'center', width: 100 }));

  els.push(rect(M, 288, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 300, 'Activations never change the shape. Dense layers are the only place the feature axis moves.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 324, 'ReLU is pointwise and softmax normalises within a row; neither has a neuron count to impose.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 376, 'Checking that the row count survives the pipeline is the fastest way to catch a transposed array.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Shape audit through the complete forward pass',
    desc: 'A hand-drawn horizontal shape audit for three hundred spiral samples through four stages. '
      + 'Each shape is drawn as two boxes, a batch axis and a feature axis in different colours. X '
      + 'starts at three hundred by two; dense1 takes the feature axis from two to three giving '
      + 'three hundred by three; activation1 leaves it unchanged; dense2 leaves it at three by '
      + 'three; and activation2 leaves it unchanged again, ending at three hundred by three '
      + 'probabilities. A dashed tie runs under the batch axis of all five shapes, labelled as never '
      + 'changing, three hundred rows in and three hundred rows out. A band states that activations '
      + 'never change the shape and dense layers are the only place the feature axis moves, because '
      + 'ReLU is pointwise and softmax normalises within a row, so neither has a neuron count to '
      + 'impose. A footer notes that checking the row count survives the pipeline is the fastest way '
      + 'to catch a transposed array.',
  };
}

// ---------------------------------------------------------------- diagram 3
// New. Section 5 pulls two separate readings out of one printed row: the
// pipeline is correct, and it knows nothing. Those are different claims and the
// figure keeps them apart, because conflating them is how a working forward
// pass gets mistaken for a working model.
function uniformBaseline() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('What a uniform output does and does not tell you',
    'The first row the network ever prints. It is evidence the pipeline works, and no evidence at all that it has learned.')];

  const BASE = 298, SCALE = 130, BW = 60, GAP = 30;

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 270, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 120, fit(o.sub, 11, 380, 'panel sub'), { size: 11, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + 404, 144));

    const gx = x + 90;
    els.push(line([[gx - 10, BASE], [gx + 3 * BW + 2 * GAP + 10, BASE]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
    o.vals.forEach((v, i) => {
      const bx = gx + i * (BW + GAP), h = Math.max(3, v * SCALE);
      els.push(rect(bx, BASE - h, BW, h, {
        stroke: o.c, fill: o.c, strokeWidth: 1.4, opacity: 34, roundness: null,
      }));
      els.push(text(bx, BASE - h - 20, v.toFixed(3),
        { size: 11, family: 3, stroke: T.ink, align: 'center', width: BW }));
      els.push(text(bx, BASE + 6, `class ${i}`,
        { size: 10, stroke: T.inkSubtle, align: 'center', width: BW }));
    });
    els.push(text(x + 20, 326, fit(o.verdict, 11, 380, 'verdict'), { size: 11, stroke: T.inkMuted }));
  };

  panel(M, {
    c: T.warn, title: 'At initialisation', sub: 'the row this post actually prints',
    vals: [0.333, 0.333, 0.334],
    verdict: 'Sums to 1.0, so softmax is implemented correctly.\nArgmax is right about a third of the time, which is chance.',
  });

  panel(500, {
    c: T.success, title: 'After training', sub: 'what Parts 09 to 27 are for',
    vals: [0.95, 0.03, 0.02],
    verdict: 'Also sums to 1.0. The difference is not well-formedness,\nit is that one class now carries most of the mass.',
  });

  els.push(rect(M, 384, 880, 68, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 396, 'weights ≈ 0.01   ×   features ≈ 0.5   →   logits ≈ 0.01   →   softmax of near-equal logits ≈ 1/C',
    { size: 12, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 422, 'The uniform row is what small random weights are supposed to produce. Anything else at initialisation would be the bug.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The uniform output at initialisation: well-formed, and uninformative',
    desc: 'Two hand-drawn bar charts of one output row each. The left, labelled at initialisation, is '
      + 'the row this post actually prints: three near-equal bars at roughly nought point three '
      + 'three three across classes nought, one and two. It sums to one, which shows softmax is '
      + 'implemented correctly, but its argmax is right about a third of the time, which is chance. '
      + 'The right, labelled after training, shows nought point nine five, nought point nought '
      + 'three and nought point nought two: also summing to one, with the difference being not '
      + 'well-formedness but that one class now carries most of the mass. A band traces the cause: '
      + 'weights of about nought point nought one times features of about nought point five give '
      + 'logits of about nought point nought one, and softmax of near-equal logits is about one over '
      + 'the class count. A footer notes that the uniform row is exactly what small random weights '
      + 'should produce, and that anything else at initialisation would be the bug.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { pipelineAnatomy, shapeAudit, uniformBaseline };

if (require.main === module) {
  emit('01-pipeline-anatomy', pipelineAnatomy());
  emit('02-shape-audit', shapeAudit());
  emit('03-uniform-baseline', uniformBaseline());
}
