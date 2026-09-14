// The post 16 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 4.1 warns that dropping .copy() silently mutates the caller's array,
// which is a memory-aliasing fault rather than an arithmetic one, and nothing in
// the series draws what aliasing actually looks like.
//
//   01-dense-backward-class  960 x 500
//   02-copy-not-alias        960 x 470   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// A row of values as one boxed array.
function arr(els, x, y, vals, o = {}) {
  const cw = o.cw ?? 44, ch = o.ch ?? 32, size = o.size ?? 12;
  els.push(rect(x, y, vals.length * cw, ch, {
    stroke: o.c ?? T.ink, fill: T.surface, strokeWidth: 1.5, roundness: null,
  }));
  for (let i = 1; i < vals.length; i++) {
    els.push(line([[x + i * cw, y], [x + i * cw, y + ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  vals.forEach((v, i) => els.push(text(x + i * cw, y + (ch - size) / 2 - 1, String(v), {
    size, family: 3, stroke: o.ink ?? T.ink, align: 'center', width: cw,
  })));
  return { w: vals.length * cw, cx: x + (vals.length * cw) / 2, bottom: y + ch };
}

// ---------------------------------------------------------------- diagram 1
// The two methods side by side at the same size, with the cache line marked in
// forward and the line that consumes it marked in backward. The pairing is the
// contract; separating the methods into two figures would hide it.
function denseBackwardClass() {
  resetSeq();
  const W = 960, H = 500;

  const els = [...heading('Layer_Dense, with a backward method',
    'One cached array in forward, three gradients out of backward. That is the whole contract of a layer.')];

  els.push(...card(M, 86, 560, 300, { stroke: T.primary, strokeWidth: 1.6 }));
  els.push(text(62, 98, 'class Layer_Dense', { size: 15, family: 3, stroke: T.primary }));

  els.push(...card(56, 126, 264, 244, { spine: T.success, fill: T.neutral1 }));
  els.push(text(76, 136, 'forward(self, inputs)', { size: 11, family: 3, stroke: T.success }));
  els.push(rule(70, 306, 160));
  els.push(text(76, 170, 'self.inputs = inputs', { size: 10, family: 3, stroke: T.accent }));
  els.push(text(76, 190, 'self.output = (', { size: 10, family: 3, stroke: T.ink }));
  els.push(text(76, 206, '    inputs @ self.weights', { size: 10, family: 3, stroke: T.ink }));
  els.push(text(76, 222, '    + self.biases)', { size: 10, family: 3, stroke: T.ink }));
  els.push(line([[68, 168], [68, 184]], { stroke: T.accent, strokeWidth: 3.5, roughness: 0.5 }));
  els.push(text(76, 262, 'The cache line is the only', { size: 10, stroke: T.inkSubtle }));
  els.push(text(76, 278, 'addition to the Part 04 class.', { size: 10, stroke: T.inkSubtle }));
  els.push(text(76, 306, 'Without it, backward has', { size: 10, stroke: T.inkSubtle }));
  els.push(text(76, 322, 'nothing to multiply by.', { size: 10, stroke: T.inkSubtle }));

  els.push(...card(332, 126, 264, 244, { spine: T.alert, fill: T.neutral1 }));
  els.push(text(352, 136, 'backward(self, dvalues)', { size: 11, family: 3, stroke: T.alert }));
  els.push(rule(346, 582, 160));
  const CODE = [
    'self.dweights = (',
    '    self.inputs.T @ dvalues)',
    'self.dbiases = np.sum(',
    '    dvalues, axis=0,',
    '    keepdims=True)',
    'self.dinputs = (',
    '    dvalues @ self.weights.T)',
  ];
  CODE.forEach((ln, i) => els.push(text(352, 170 + i * 17, fit(ln, 9, 228 * 0.94, 'code line'),
    { size: 9, family: 3, stroke: T.ink })));
  els.push(text(352, 306, 'Reads the cache on line 1.', { size: 10, stroke: T.inkSubtle }));
  els.push(text(352, 322, 'Writes three arrays, no returns.', { size: 10, stroke: T.inkSubtle }));

  const OUT = [
    { y: 110, c: T.accent, t: 'self.dweights', shape: '(n_inputs, n_neurons)', to: 'the optimiser' },
    { y: 200, c: T.accent, t: 'self.dbiases', shape: '(1, n_neurons)', to: 'the optimiser' },
    { y: 290, c: T.success, t: 'self.dinputs', shape: '(N, n_inputs)', to: 'the previous layer' },
  ];
  OUT.forEach((o) => {
    els.push(...card(620, o.y, 300, 76, { spine: o.c }));
    els.push(text(642, o.y + 12, o.t, { size: 13, family: 3, stroke: T.ink }));
    els.push(text(642, o.y + 34, o.shape, { size: 10, family: 3, stroke: T.inkMuted }));
    els.push(text(642, o.y + 52, `→  ${o.to}`, { size: 10, stroke: o.c }));
    els.push(arrow([[600, o.y + 38], [616, o.y + 38]],
      { stroke: o.c, strokeWidth: 1.3, roughness: 0.4 }));
  });

  els.push(rect(M, 412, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 424, 'Every class with a backward caches something in its forward. Layer_Dense caches its inputs; ReLU caches its inputs too.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 448, 'The cost is one array reference per layer, held until the backward pass has used it.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Layer_Dense with forward and backward methods, and what each one caches',
    desc: 'A hand-drawn class blueprint. On the left, a card for class Layer_Dense holds its two '
      + 'methods side by side at equal size. The forward method caches self.inputs, marked with a '
      + 'coloured spine, then computes self.output as inputs matrix-multiplied by the weights plus '
      + 'the biases; a note records that the cache line is the only addition to the Part 04 class '
      + 'and that without it backward has nothing to multiply by. The backward method computes '
      + 'self.dweights from the transposed cached inputs times dvalues, self.dbiases from a sum '
      + 'along the batch axis with keepdims, and self.dinputs from dvalues times the transposed '
      + 'weights; a note records that it reads the cache on its first line and writes three arrays '
      + 'without returning anything. Three cards on the right receive them: dweights of shape '
      + 'n_inputs by n_neurons and dbiases of shape one by n_neurons both go to the optimiser, and '
      + 'dinputs of shape N by n_inputs goes to the previous layer.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 4.1 warns that omitting .copy() mutates the caller's array. That
// is a fault in what the names point at rather than in the arithmetic, so the
// figure draws the names and the arrays as separate things and lets the arrows
// carry the difference.
function copyNotAlias() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('One array, or two',
    'ReLU’s backward writes zeros into a gradient. Whether that damages the caller depends on one method call.')];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 300, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 122, fit(o.code, 11, 380 * 0.94, 'code'),
      { size: 11, family: 3, stroke: T.ink }));
    els.push(rule(x + 16, x + 404, 150));
    els.push(text(x + 20, 160, 'after the assignment', { size: 10, stroke: T.inkSubtle }));
    els.push(text(x + 20, 268, 'after dinputs[inputs <= 0] = 0', { size: 10, family: 3, stroke: T.inkSubtle }));
    els.push(text(x + 20, 356, fit(o.verdict, 11, 380, 'verdict'), { size: 11, stroke: o.c }));
  };

  // --- aliased: two names, one array
  panel(M, {
    c: T.alert, title: 'Without .copy()', code: 'self.dinputs = dvalues',
    verdict: 'The caller’s dvalues is now [5, 0, 7] as well.',
  });
  arr(els, 200, 186, [5, 6, 7], { c: T.alert });
  els.push(text(62, 184, 'dvalues', { size: 11, family: 3, stroke: T.inkMuted }));
  els.push(text(62, 206, 'self.dinputs', { size: 11, family: 3, stroke: T.inkMuted }));
  els.push(line([[126, 190], [180, 198]], { stroke: T.alert, strokeWidth: 1.2, roughness: 0.4 }));
  els.push(line([[142, 212], [180, 204]], { stroke: T.alert, strokeWidth: 1.2, roughness: 0.4 }));
  els.push(text(200, 226, 'one array, two names', { size: 10, stroke: T.alert, align: 'center', width: 132 }));
  arr(els, 200, 292, [5, 0, 7], { c: T.alert, ink: T.alert });
  els.push(text(62, 300, 'both names', { size: 11, family: 3, stroke: T.inkMuted }));
  els.push(text(62, 318, 'see the zero', { size: 11, family: 3, stroke: T.alert }));

  // --- copied: two names, two arrays
  panel(500, {
    c: T.success, title: 'With .copy()', code: 'self.dinputs = dvalues.copy()',
    verdict: 'The caller’s dvalues is untouched. One extra allocation.',
  });
  arr(els, 560, 194, [5, 6, 7], { c: T.inkMuted });
  els.push(text(560, 176, 'dvalues', { size: 10, family: 3, stroke: T.inkMuted, align: 'center', width: 132 }));
  arr(els, 760, 194, [5, 6, 7], { c: T.success });
  els.push(text(760, 176, 'self.dinputs', { size: 10, family: 3, stroke: T.success, align: 'center', width: 132 }));
  els.push(text(560, 232, 'two arrays, two names', { size: 10, stroke: T.success, align: 'center', width: 332 }));
  arr(els, 560, 292, [5, 6, 7], { c: T.inkMuted });
  arr(els, 760, 292, [5, 0, 7], { c: T.success, ink: T.success });
  els.push(text(560, 330, 'unchanged', { size: 10, stroke: T.inkMuted, align: 'center', width: 132 }));
  els.push(text(760, 330, 'masked', { size: 10, stroke: T.success, align: 'center', width: 132 }));

  els.push(rect(M, 406, 880, 50, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 420, 'Both versions compute the right answer for this layer. Only one of them leaves the layer behind it able to compute anything.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Why ReLU’s backward copies dvalues instead of aliasing it',
    desc: 'Two hand-drawn panels tracing what the same masking operation does to memory. The left '
      + 'panel, in alert red and labelled without copy, assigns self.dinputs to dvalues directly: a '
      + 'single array holding five, six, seven has two names pointing at it. After writing a zero at '
      + 'the masked position, the one array reads five, nought, seven and both names see the zero, '
      + 'so the caller’s dvalues has been changed as well. The right panel, in success green and '
      + 'labelled with copy, assigns self.dinputs to a copy: two separate arrays each hold five, '
      + 'six, seven. After the same masking, dvalues is unchanged at five, six, seven while '
      + 'self.dinputs reads five, nought, seven. A band notes that both versions compute the right '
      + 'answer for this layer, and only one of them leaves the layer behind it able to compute '
      + 'anything.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { denseBackwardClass, copyNotAlias };

if (require.main === module) {
  emit('01-dense-backward-class', denseBackwardClass());
  emit('02-copy-not-alias', copyNotAlias());
}
