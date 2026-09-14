// The post 21 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 4 checks the loss against −log(1/3) and section 6 checks every
// gradient's shape against its parameter. Both run before any training happens,
// which is what makes them worth a figure: they catch a broken backward pass on
// iteration zero rather than after an hour of a loss that will not move.
//
//   01-forward-backward-script  960 x 540
//   02-two-sanity-checks        960 x 460   (new)

const { T, rect, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// The script on the left, what it leaves behind on the right. The three blocks
// are kept in the order they run, and the gradient panel is aligned so the
// arrays line up beside the calls that wrote them.
function forwardBackwardScript() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('The whole training step, in fifteen lines',
    'Instantiate four objects, walk forward, walk back. Every gradient the optimiser needs is then sitting on an instance.')];

  els.push(...card(M, 86, 520, 330, { stroke: T.primary, strokeWidth: 1.6 }));

  const BLOCKS = [
    {
      y: 100, c: T.inkMuted, title: '# Network.',
      lines: [
        'dense1 = Layer_Dense(2, 3)',
        'activation1 = Activation_ReLU()',
        'dense2 = Layer_Dense(3, 3)',
        'loss_activation = \\',
        '    Activation_Softmax_Loss_CategoricalCrossentropy()',
      ],
    },
    {
      y: 208, c: T.primary, title: '# Forward.',
      lines: [
        'dense1.forward(X)',
        'activation1.forward(dense1.output)',
        'dense2.forward(activation1.output)',
        'loss = loss_activation.forward(dense2.output, y)',
      ],
    },
    {
      y: 302, c: T.alert, title: '# Backward.',
      lines: [
        'loss_activation.backward(loss_activation.output, y)',
        'dense2.backward(loss_activation.dinputs)',
        'activation1.backward(dense2.dinputs)',
        'dense1.backward(activation1.dinputs)',
      ],
    },
  ];
  BLOCKS.forEach((b) => {
    els.push(line([[54, b.y + 2], [54, b.y + 16 + b.lines.length * 17]],
      { stroke: b.c, strokeWidth: 3.5, roughness: 0.5 }));
    els.push(text(68, b.y, b.title, { size: 11, family: 3, stroke: b.c }));
    b.lines.forEach((ln, i) => els.push(text(68, b.y + 20 + i * 17,
      fit(ln, 9, 470 * 0.94, 'script line'), { size: 9, family: 3, stroke: T.ink })));
  });

  els.push(...card(580, 86, 340, 330, { spine: T.accent }));
  els.push(text(602, 98, 'What is left on the instances', { size: 14, stroke: T.accent }));
  els.push(rule(596, 904, 126));
  const GRADS = [
    ['dense1.dweights', '(2, 3)', '6'],
    ['dense1.dbiases', '(1, 3)', '3'],
    ['dense2.dweights', '(3, 3)', '9'],
    ['dense2.dbiases', '(1, 3)', '3'],
  ];
  els.push(text(602, 140, 'array', { size: 9, stroke: T.inkSubtle }));
  els.push(text(756, 140, 'shape', { size: 9, stroke: T.inkSubtle }));
  els.push(text(852, 140, 'count', { size: 9, stroke: T.inkSubtle }));
  GRADS.forEach((g, i) => {
    const y = 164 + i * 36;
    els.push(text(602, y, g[0], { size: 10, family: 3, stroke: T.ink }));
    els.push(text(756, y, g[1], { size: 10, family: 3, stroke: T.inkMuted }));
    els.push(text(852, y, g[2], { size: 11, family: 3, stroke: T.accent }));
  });
  els.push(rule(596, 904, 316));
  els.push(text(602, 328, '21 gradients', { size: 17, stroke: T.accent }));
  els.push(text(602, 356, 'Exactly the 21 parameters Part 09', { size: 10, stroke: T.inkMuted }));
  els.push(text(602, 372, 'said random search could never find.', { size: 10, stroke: T.inkMuted }));

  els.push(rect(M, 442, 880, 60, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 454, 'This is one step. Nothing has been updated yet, and the loss is still at its starting value.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 478, 'Part 22 wraps the same fifteen lines in a loop and hands the four gradient arrays to an optimiser.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The fifteen-line full forward and backward script',
    desc: 'A hand-drawn code-and-shapes layout. The left card holds the script in three spined '
      + 'blocks. Network instantiates dense1 as Layer_Dense of two and three, an ReLU activation, '
      + 'dense2 as Layer_Dense of three and three, and the fused softmax cross-entropy object. '
      + 'Forward runs dense1, the activation, dense2 and the loss in order. Backward runs the loss, '
      + 'dense2, the activation and dense1 in reverse, each reading the previous call’s dinputs. The '
      + 'right card lists what the script leaves on the instances: dense1.dweights of shape two by '
      + 'three holding six numbers, dense1.dbiases of shape one by three holding three, '
      + 'dense2.dweights of shape three by three holding nine, and dense2.dbiases holding three, '
      + 'totalling twenty-one gradients — exactly the twenty-one parameters Part 09 said random '
      + 'search could never find. A band notes that this is one step, nothing has been updated yet, '
      + 'and Part 22 wraps the same lines in a loop with an optimiser.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Both checks in this figure run on iteration zero, before a single
// parameter moves. That is the property worth drawing: neither of them needs
// the loss to go down, so a broken backward pass is caught immediately instead
// of being mistaken for a bad learning rate an hour later.
function twoSanityChecks() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('Two checks that run before any training',
    'Neither needs the loss to fall. Both fail loudly on a broken implementation, on the very first step.')];

  els.push(...card(M, 86, 420, 280, { spine: T.primary }));
  els.push(text(62, 98, 'Check 1 · the loss on step zero', { size: 14, stroke: T.primary }));
  els.push(rule(56, 444, 126));
  els.push(text(62, 142, 'the script prints', { size: 10, stroke: T.inkSubtle }));
  els.push(text(62, 160, '1.0986', { size: 26, family: 3, stroke: T.ink }));
  els.push(text(200, 168, 'and chance for three classes is', { size: 10, stroke: T.inkSubtle }));
  els.push(text(200, 186, '−log(1/3)  =  1.0986', { size: 13, family: 3, stroke: T.primary }));
  els.push(rule(56, 444, 216));
  els.push(text(62, 228, 'A match means softmax and cross-entropy are', { size: 11, stroke: T.inkMuted }));
  els.push(text(62, 246, 'wired together correctly and the forward pass', { size: 11, stroke: T.inkMuted }));
  els.push(text(62, 264, 'is sound.', { size: 11, stroke: T.inkMuted }));
  els.push(text(62, 292, 'Anything far from 1.0986 on iteration zero is a', { size: 11, stroke: T.alert }));
  els.push(text(62, 310, 'bug that no amount of training will fix.', { size: 11, stroke: T.alert }));

  els.push(...card(500, 86, 420, 280, { spine: T.accent }));
  els.push(text(522, 98, 'Check 2 · every gradient matches its parameter', { size: 14, stroke: T.accent }));
  els.push(rule(516, 904, 126));
  els.push(text(522, 138, 'parameter', { size: 9, stroke: T.inkSubtle }));
  els.push(text(680, 138, 'gradient', { size: 9, stroke: T.inkSubtle }));
  els.push(text(838, 138, 'shape', { size: 9, stroke: T.inkSubtle }));
  const PAIRS = [
    ['dense1.weights', 'dweights', '(2, 3)'],
    ['dense1.biases', 'dbiases', '(1, 3)'],
    ['dense2.weights', 'dweights', '(3, 3)'],
    ['dense2.biases', 'dbiases', '(1, 3)'],
  ];
  PAIRS.forEach((p, i) => {
    const y = 160 + i * 28;
    els.push(text(522, y, p[0], { size: 10, family: 3, stroke: T.ink }));
    els.push(text(680, y, p[1], { size: 10, family: 3, stroke: T.accent }));
    els.push(text(838, y, p[2], { size: 10, family: 3, stroke: T.inkMuted }));
  });
  els.push(rule(516, 904, 284));
  els.push(text(522, 296, '21 parameters, 21 gradients, shape for shape.', { size: 11, stroke: T.inkMuted }));
  els.push(text(522, 320, 'Magnitudes of order 1e−3 to 1e−4. Anything much', { size: 11, stroke: T.alert }));
  els.push(text(522, 338, 'larger on step zero means the initialisation is wrong.', { size: 11, stroke: T.alert }));

  els.push(rect(M, 392, 880, 60, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 404, 'A backward pass can be wrong in a way that still runs, still returns arrays, and still lets the loss drift downwards.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 428, 'These two checks cost one print each and rule that out before the training loop is ever written.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Two checks that confirm the backward pass before training starts',
    desc: 'Two hand-drawn cards. The first, check one, compares the loss the script prints on step '
      + 'zero, one point nought nine eight six, against chance for three classes, minus the log of '
      + 'one third, which is also one point nought nine eight six. A match means softmax and '
      + 'cross-entropy are wired together correctly and the forward pass is sound; anything far from '
      + 'it on iteration zero is a bug no amount of training will fix. The second, check two, pairs '
      + 'each parameter with its gradient and its shape: dense1.weights with dweights at two by '
      + 'three, dense1.biases with dbiases at one by three, dense2.weights with dweights at three by '
      + 'three, and dense2.biases with dbiases at one by three, for twenty-one parameters and '
      + 'twenty-one gradients, shape for shape, with magnitudes of order one in a thousand to one in '
      + 'ten thousand. A band notes that a backward pass can be wrong in a way that still runs, '
      + 'still returns arrays and still lets the loss drift downwards, and that these two checks '
      + 'cost one print each and rule that out before the training loop is written.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { forwardBackwardScript, twoSanityChecks };

if (require.main === module) {
  emit('01-forward-backward-script', forwardBackwardScript());
  emit('02-two-sanity-checks', twoSanityChecks());
}
