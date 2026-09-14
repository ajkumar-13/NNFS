// The post 15 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 1 justifies the whole post in one sentence — a layer's input gradient
// is the next-earlier layer's upstream gradient — and section 6 collects the
// three gradients a dense layer emits. Neither had a figure, and the existing
// one is about the sum inside a single layer rather than the handoff between two.
//
//   01-input-gradients    960 x 460
//   02-gradient-handoff   960 x 470   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// Both panels draw the same three neurons and the same nodes. Only the number
// of lit paths changes, which is the entire difference between a gradient that
// is one product and one that is a sum.
function inputGradients() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('One path, or every path',
    'A weight touches one neuron. An input touches all of them, and the chain rule sums over every route it takes.')];

  const ZY = [160, 210, 260];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 250, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 120, fit(o.sub, 11, 380, 'panel sub'), { size: 11, stroke: T.inkSubtle }));

    els.push(circle(x + 60, 210, 20, { stroke: T.primary, fill: T.surface, strokeWidth: 1.5 }));
    els.push(text(x + 40, 202, 'X₁', { size: 12, family: 3, stroke: T.ink, align: 'center', width: 40 }));
    ZY.forEach((zy, k) => {
      const lit = o.lit.includes(k);
      els.push(line([[x + 80, 210], [x + 210, zy]],
        { stroke: lit ? o.c : T.border, strokeWidth: lit ? 1.8 : 1, roughness: 0.4 }));
      if (lit) {
        els.push(text(x + 106, (210 + zy) / 2 - 20, `W${k + 1}₁`,
          { size: 10, family: 3, stroke: o.c }));
      }
      els.push(circle(x + 230, zy, 20, { stroke: lit ? o.c : T.border, fill: T.surface, strokeWidth: 1.5 }));
      els.push(text(x + 210, zy - 8, `Z${k + 1}`, { size: 12, family: 3, stroke: lit ? T.ink : T.inkSubtle, align: 'center', width: 40 }));
      els.push(line([[x + 250, zy], [x + 330, 210]],
        { stroke: lit ? o.c : T.border, strokeWidth: lit ? 1.8 : 1, roughness: 0.4 }));
    });
    els.push(circle(x + 350, 210, 20, { stroke: T.alert, fill: T.surface, strokeWidth: 1.5 }));
    els.push(text(x + 330, 202, 'L', { size: 13, stroke: T.alert, align: 'center', width: 40 }));

    els.push(text(x + 20, 292, fit(o.formula, 12, 380 * 0.94, 'formula'),
      { size: 12, family: 3, stroke: T.ink }));
    els.push(text(x + 20, 312, fit(o.note, 10, 380, 'panel note'), { size: 10, stroke: T.inkSubtle }));
  };

  panel(M, {
    c: T.accent, title: 'A weight: one path', sub: 'W₁₁ reaches the loss through Z₁ and nothing else',
    lit: [0],
    formula: '∂L/∂W₁₁  =  ∂L/∂Z₁ · X₁',
    note: 'One term, so the chain rule reads as a plain product.',
  });

  panel(500, {
    c: T.success, title: 'An input: every path', sub: 'X₁ reaches the loss through all three neurons',
    lit: [0, 1, 2],
    formula: '∂L/∂X₁  =  Σₖ ∂L/∂Zₖ · Wₖ₁',
    note: 'Three terms, one per neuron. The sum is not optional.',
  });

  els.push(rect(M, 360, 880, 68, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 372, '∂L/∂X  =  ∂L/∂Z · W                (1, m) · (m, n)  =  (1, n)',
    { size: 13, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 400, 'The contracted dimension is m, the neuron count: exactly the paths the sum has to run over.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The input gradient sums contributions from every neuron the input feeds',
    desc: 'Two hand-drawn panels over the same three-neuron layer, differing only in how many paths '
      + 'are lit. In the left panel a weight has one path: W one one reaches the loss through Z one '
      + 'and nothing else, so its gradient is the plain product of the upstream at Z one and the '
      + 'input, and the other two routes are drawn faint. In the right panel an input has every '
      + 'path: X one reaches the loss through all three neurons via W one one, W two one and W three '
      + 'one, so its gradient is a sum of three terms, one per neuron. A band gives the matrix form, '
      + 'the input gradient equals the upstream gradient dotted with W, with shapes one by m dotted '
      + 'with m by n giving one by n, and notes that the contracted dimension m is the neuron count, '
      + 'exactly the paths the sum has to run over.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 1 gives the one-sentence reason this gradient exists and section
// 6 collects the three a dense layer emits. Sorting them by where they go —
// two stay, one leaves — is what makes the sentence structural rather than a
// definition to memorise.
function gradientHandoff() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Two gradients stay, one travels',
    'A dense layer’s backward call emits three things. Only one of them is any use to the layer before it.')];

  els.push(...card(720, 168, 200, 66, { stroke: T.alert, strokeWidth: 1.6 }));
  els.push(text(720, 182, '∂L/∂Z', { size: 16, family: 3, stroke: T.alert, align: 'center', width: 200 }));
  els.push(text(720, 208, 'upstream · (N, m)', { size: 10, stroke: T.inkMuted, align: 'center', width: 200 }));
  els.push(text(720, 148, 'arrives from the right', { size: 10, stroke: T.inkSubtle, align: 'center', width: 200 }));

  els.push(...card(360, 112, 300, 190, { stroke: T.primary, strokeWidth: 1.6 }));
  els.push(text(360, 126, 'Layer_Dense.backward()', { size: 13, family: 3, stroke: T.primary, align: 'center', width: 300 }));
  els.push(rule(376, 644, 152));
  els.push(text(382, 164, 'dW = self.inputs.T @ dZ', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(382, 190, 'db = np.sum(dZ, axis=0,', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(382, 206, '            keepdims=True)', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(382, 232, 'dX = dZ @ self.weights.T', { size: 11, family: 3, stroke: T.ink }));
  els.push(text(360, 266, 'three lines, three gradients', { size: 10, stroke: T.inkSubtle, align: 'center', width: 300 }));
  els.push(arrow([[716, 201], [664, 201]], { stroke: T.alert, strokeWidth: 1.4, roughness: 0.4 }));

  const OUT = [
    {
      y: 112, c: T.accent, t: '∂L/∂W', shape: '(n, m)',
      fate: 'stays here', note: 'the optimiser consumes it and it is gone',
    },
    {
      y: 184, c: T.accent, t: '∂L/∂b', shape: '(1, m)',
      fate: 'stays here', note: 'same: consumed, never passed on',
    },
    {
      y: 256, c: T.success, t: '∂L/∂X', shape: '(N, n)',
      fate: 'travels left', note: 'this is the previous layer’s ∂L/∂Z',
    },
  ];
  OUT.forEach((o) => {
    els.push(...card(M, o.y, 280, 64, { spine: o.c }));
    els.push(text(62, o.y + 10, o.t, { size: 15, family: 3, stroke: T.ink }));
    els.push(text(140, o.y + 13, o.shape, { size: 11, family: 3, stroke: T.inkMuted }));
    els.push(text(214, o.y + 13, o.fate, { size: 10, stroke: o.c }));
    els.push(text(62, o.y + 40, fit(o.note, 10, 240, 'fate note'), { size: 10, stroke: T.inkSubtle }));
    els.push(arrow([[356, o.y + 32], [324, o.y + 32]],
      { stroke: o.c, strokeWidth: 1.3, roughness: 0.4 }));
  });

  els.push(arrow([[36, 288], [36, 330], [200, 330]],
    { stroke: T.success, strokeWidth: 1.4, roughness: 0.4 }));
  els.push(text(210, 322, 'on to layer L − 1, where it arrives as that layer’s upstream',
    { size: 11, stroke: T.success }));

  els.push(rect(M, 358, 880, 66, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 370, 'A layer’s input gradient is the next-earlier layer’s upstream gradient.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 394, 'Drop it and backpropagation stops at the first layer boundary it meets, and every earlier weight stays at its initial value.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 444, 'This is why a single-layer example never needs ∂L/∂X, and why a two-layer one cannot work without it.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Where each of a dense layer’s three gradients goes',
    desc: 'A hand-drawn diagram of one layer’s backward call. The upstream gradient, of shape N by m, '
      + 'arrives from the right into a card holding the three lines of Layer_Dense.backward: the '
      + 'weight gradient from the transposed inputs dotted with the upstream, the bias gradient from '
      + 'a sum along the batch axis, and the input gradient from the upstream dotted with the '
      + 'transposed weights. Three cards on the left receive them, sorted by where they go: the '
      + 'weight gradient of shape n by m stays here and is consumed by the optimiser, the bias '
      + 'gradient of shape one by m likewise, and the input gradient of shape N by n travels left, '
      + 'annotated as being the previous layer’s upstream gradient, with an arrow carrying it on to '
      + 'layer L minus one. A band states that a layer’s input gradient is the next-earlier layer’s '
      + 'upstream gradient, and that dropping it stops backpropagation at the first layer boundary '
      + 'it meets, leaving every earlier weight at its initial value.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { inputGradients, gradientHandoff };

if (require.main === module) {
  emit('01-input-gradients', inputGradients());
  emit('02-gradient-handoff', gradientHandoff());
}
