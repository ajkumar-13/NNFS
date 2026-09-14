// The post 20 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 1 opens by counting what ten posts of backprop actually produced —
// three classes, eight methods, and only one of the three holding parameters —
// and the existing figure is about the wiring rather than about the inventory.
//
//   01-full-backprop-pipeline  960 x 540
//   02-the-toolkit             960 x 470   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// Forward and backward in the same five columns, with the handoff named on
// every arrow. The claim of section 5 is that one variable is the glue, so the
// figure writes that variable on each link rather than labelling the row once.
function fullBackpropPipeline() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Forward, then backward, through the same four objects',
    'Each backward call is handed the previous one’s dinputs. Nothing else passes between them.')];

  const NODES = [
    { x: 40, w: 70, t: 'X', s: '(N, 2)', c: T.primary },
    { x: 132, w: 150, t: 'layer1', s: 'Layer_Dense(2, 3)', c: T.accent },
    { x: 304, w: 150, t: 'activation1', s: 'ReLU', c: T.success },
    { x: 476, w: 150, t: 'layer2', s: 'Layer_Dense(3, 3)', c: T.accent },
    { x: 648, w: 180, t: 'loss_activation', s: 'Softmax + CE', c: T.primary },
    { x: 850, w: 70, t: 'L', s: 'scalar', c: T.alert },
  ];
  NODES.forEach((n, i) => {
    els.push(...card(n.x, 110, n.w, 70, { stroke: n.c, spine: n.c }));
    els.push(text(n.x, 124, fit(n.t, 13, n.w - 24, 'node title'),
      { size: 13, stroke: n.c, align: 'center', width: n.w }));
    els.push(text(n.x, 148, fit(n.s, 9, (n.w - 20) * 0.94, 'node sub'),
      { size: 9, family: 3, stroke: T.inkMuted, align: 'center', width: n.w }));
    if (i < NODES.length - 1) {
      els.push(arrow([[n.x + n.w + 2, 145], [NODES[i + 1].x - 2, 145]],
        { stroke: T.primary, strokeWidth: 1.3, roughness: 0.4 }));
    }
  });
  els.push(text(40, 90, 'forward', { size: 10, stroke: T.primary }));
  [[121, 'Z₁'], [293, 'A₁'], [465, 'Z₂'], [637, 'ŷ']].forEach(([cx, lab]) => {
    els.push(text(cx - 30, 188, lab, { size: 10, family: 3, stroke: T.inkSubtle, align: 'center', width: 60 }));
  });

  const BACK = [
    { i: 1, t: 'layer1.backward', out: 'dweights · dbiases\ndinputs' },
    { i: 2, t: 'activation1.backward', out: 'dinputs only' },
    { i: 3, t: 'layer2.backward', out: 'dweights · dbiases\ndinputs' },
    { i: 4, t: 'loss_activation.backward', out: '(ŷ − y) / N' },
  ];
  BACK.forEach((b) => {
    const n = NODES[b.i];
    els.push(...card(n.x, 250, n.w, 70, { spine: T.alert }));
    els.push(text(n.x, 262, fit(b.t, 10, n.w - 20, 'back title'),
      { size: 10, family: 3, stroke: T.alert, align: 'center', width: n.w }));
    els.push(text(n.x, 290, fit(b.out, 9, n.w - 18, 'back out'),
      { size: 9, stroke: T.inkMuted, align: 'center', width: n.w }));
    els.push(line([[n.x + n.w / 2, 184], [n.x + n.w / 2, 246]],
      { stroke: T.border, strokeWidth: 1, strokeStyle: 'dotted', roughness: 0.3 }));
  });
  [[648, 626], [476, 454], [304, 282]].forEach(([a, b]) => {
    els.push(arrow([[a - 2, 285], [b + 2, 285]], { stroke: T.alert, strokeWidth: 1.3, roughness: 0.4 }));
  });
  els.push(text(828, 230, 'backward', { size: 10, stroke: T.alert }));
  [[637, 'dinputs'], [465, 'dinputs'], [293, 'dinputs']].forEach(([cx, lab]) => {
    els.push(text(cx - 40, 328, lab, { size: 9, family: 3, stroke: T.alert, align: 'center', width: 80 }));
  });

  els.push(...card(M, 360, 880, 92, { spine: T.accent }));
  els.push(text(62, 372, 'Four trainable arrays, four subtractions', { size: 13, stroke: T.accent }));
  els.push(rule(56, 904, 394));
  const UPD = [
    'layer1.weights -= lr * layer1.dweights',
    'layer1.biases  -= lr * layer1.dbiases',
    'layer2.weights -= lr * layer2.dweights',
    'layer2.biases  -= lr * layer2.dbiases',
  ];
  UPD.forEach((u, i) => els.push(text(62 + (i % 2) * 440, 406 + Math.floor(i / 2) * 20, u,
    { size: 10, family: 3, stroke: T.ink })));

  els.push(text(M, 478, 'The two activations hold no parameters, so nothing is subtracted from them. They exist to route gradients, not to learn.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 500, 'Repeating forward, backward and update is what training is. Part 21 puts the three inside a loop.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Full backpropagation pipeline: forward and backward through three classes',
    desc: 'A hand-drawn two-row pipeline. The forward row runs left to right in blue: the input X of '
      + 'shape N by two enters layer1, a Layer_Dense of two and three, producing Z1; then '
      + 'activation1, a ReLU, producing A1; then layer2, a Layer_Dense of three and three, producing '
      + 'Z2; then loss_activation, the fused softmax and cross-entropy, producing y-hat and the '
      + 'scalar loss. The backward row sits in the same columns, joined to the forward row by dotted '
      + 'droppers and read right to left in red: loss_activation.backward emits y-hat minus y over '
      + 'N, layer2.backward emits dweights, dbiases and dinputs, activation1.backward emits dinputs '
      + 'only, and layer1.backward emits its own three. Each link between them is labelled dinputs. '
      + 'A card underneath lists the four parameter updates, one subtraction per trainable array, '
      + 'and a footer notes that the two activations hold no parameters so nothing is subtracted '
      + 'from them, and that repeating forward, backward and update is what training is.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 1 counts the toolkit: three classes, eight methods, and only one
// of the three carrying anything to learn. Sorting the cards by that last
// property is what makes the inventory an argument rather than a list.
function theToolkit() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('What ten posts of backpropagation produced',
    'Three classes and eight methods. Two of the three have nothing to learn, and never will.')];

  const CLASSES = [
    {
      c: T.accent, name: 'Layer_Dense', trainable: true, badge: 'trainable',
      fwd: 'Z = X @ W + b', caches: 'self.inputs',
      bwd: 'dweights\ndbiases\ndinputs',
      foot: 'Two instances in this network,\nholding 21 parameters between them.',
    },
    {
      c: T.success, name: 'Activation_ReLU', trainable: false, badge: 'no parameters',
      fwd: 'max(0, Z)', caches: 'self.inputs',
      bwd: 'dinputs',
      foot: 'A gate. It decides which gradients\nsurvive and changes none of them.',
    },
    {
      c: T.primary, name: 'Softmax + CE', trainable: false, badge: 'no parameters',
      fwd: 'softmax, then the loss', caches: 'self.output',
      bwd: 'dinputs',
      foot: 'The fused class from Part 19.\nStarts the backward pass.',
    },
  ];

  const PW = 282, PX = [M, 339, 638], TOP = 90, PH = 282;
  CLASSES.forEach((k, i) => {
    const x = PX[i], inner = PW - 40;
    els.push(...card(x, TOP, PW, PH, { spine: k.c }));
    els.push(text(x + 20, TOP + 12, fit(k.name, 14, inner, 'class name'), { size: 14, stroke: k.c }));
    els.push(text(x + 20, TOP + 36, k.badge, { size: 10, stroke: k.trainable ? T.accent : T.inkSubtle }));
    els.push(rule(x + 16, x + PW - 16, TOP + 60));

    els.push(text(x + 20, TOP + 70, 'forward', { size: 10, stroke: T.inkSubtle }));
    els.push(text(x + 20, TOP + 86, fit(k.fwd, 11, inner * 0.94, 'fwd'),
      { size: 11, family: 3, stroke: T.ink }));
    els.push(text(x + 20, TOP + 108, `caches  ${k.caches}`, { size: 9, family: 3, stroke: T.inkMuted }));
    els.push(rule(x + 16, x + PW - 16, TOP + 130));

    els.push(text(x + 20, TOP + 140, 'backward writes', { size: 10, stroke: T.inkSubtle }));
    els.push(text(x + 20, TOP + 158, fit(k.bwd, 11, inner * 0.94, 'bwd'),
      { size: 11, family: 3, stroke: k.c }));
    els.push(rule(x + 16, x + PW - 16, TOP + 216));
    els.push(text(x + 20, TOP + 226, fit(k.foot, 10, inner, 'class foot'), { size: 10, stroke: T.inkSubtle }));
  });

  els.push(rect(M, 398, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 410, 'Every one of the network’s 21 parameters lives in the two Layer_Dense instances.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 434, 'The other two classes exist to move gradients through, which is why the optimiser in Part 22 only ever touches Layer_Dense.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The three classes and eight methods a classification network needs',
    desc: 'Three hand-drawn class cards, sorted by whether they hold parameters. Layer_Dense is '
      + 'marked trainable: its forward computes Z as X matrix-multiplied by W plus b and caches '
      + 'self.inputs, its backward writes dweights, dbiases and dinputs, and it has two instances in '
      + 'this network holding twenty-one parameters between them. Activation_ReLU is marked as '
      + 'having no parameters: its forward is max of nought and Z and caches self.inputs, its '
      + 'backward writes dinputs alone, and it is described as a gate that decides which gradients '
      + 'survive and changes none of them. The fused softmax and cross-entropy class is likewise '
      + 'parameterless: its forward runs softmax then the loss and caches self.output, its backward '
      + 'writes dinputs, and it starts the backward pass. A band states that every one of the '
      + 'network’s twenty-one parameters lives in the two Layer_Dense instances, and that the other '
      + 'two classes exist to move gradients through, which is why the optimiser in Part 22 only '
      + 'ever touches Layer_Dense.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { fullBackpropPipeline, theToolkit };

if (require.main === module) {
  emit('01-full-backprop-pipeline', fullBackpropPipeline());
  emit('02-the-toolkit', theToolkit());
}
