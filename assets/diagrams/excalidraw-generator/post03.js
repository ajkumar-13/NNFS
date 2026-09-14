// The post 03 diagrams, hand-drawn.
//
// Scenes 01 and 02 mirror the clean vector figures one directory up. Scene 03 is
// new: section 4 works out that two linear layers collapse into one, which is
// the claim the whole post turns on and the reason Part 06 exists, and the post
// had no figure for it.
//
//   01-multi-layer-anatomy  960 x 540
//   02-dimension-flow       960 x 420
//   03-linear-collapse      960 x 460   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// A stack of cells standing for one array, drawn as an outer rectangle plus
// interior rules rather than one rectangle per cell.
function stack(els, x, y, rows, w, h, o = {}) {
  els.push(rect(x, y, w, rows * h, {
    stroke: o.c ?? T.ink, fill: T.surface, strokeWidth: 1.4, roundness: null,
  }));
  for (let j = 1; j < rows; j++) {
    els.push(line([[x, y + j * h], [x + w, y + j * h]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  return { cx: x + w / 2, bottom: y + rows * h };
}

// A labelled chip: a name over a shape. Used for every stage of a shape chain.
function chip(els, x, y, w, h, name, shape, o = {}) {
  els.push(...card(x, y, w, h, { stroke: o.c ?? T.ink, spine: o.spine }));
  els.push(text(x, y + 12, name, { size: o.size ?? 16, stroke: o.c ?? T.ink, align: 'center', width: w }));
  els.push(text(x, y + 38, shape, { size: 12, family: 3, stroke: T.inkMuted, align: 'center', width: w }));
  return { cx: x + w / 2, right: x + w, bottom: y + h };
}

// ---------------------------------------------------------------- diagram 1
// One sample through two layers, with the layer cards carrying their own
// neurons. The two cards are drawn identically on purpose: the argument of the
// post is that the second layer is not a new kind of object.
function multiLayerAnatomy() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Two layers, one repeated operation',
    'The second layer is the first layer again, with the previous output standing in for the input.')];

  const MID = 210;

  // X, then layer 1, then Z1, then layer 2, then Z2.
  const xs = stack(els, 85, MID - 52, 4, 70, 26, { c: T.primary });
  els.push(text(85, MID - 76, 'X', { size: 16, stroke: T.primary, align: 'center', width: 70 }));
  els.push(text(85, MID + 60, '(4,)', { size: 12, family: 3, stroke: T.inkMuted, align: 'center', width: 70 }));

  const layer = (x, o) => {
    els.push(...card(x, 130, 190, 160, { spine: o.c }));
    els.push(text(x + 20, 142, o.name, { size: 15, stroke: o.c }));
    [0, 1, 2].forEach((k) => {
      els.push(circle(x + 55 + k * 40, 196, 15, { stroke: o.c, fill: T.surface, strokeWidth: 1.5 }));
    });
    els.push(text(x, 220, '3 neurons', { size: 10, stroke: T.inkSubtle, align: 'center', width: 190 }));
    els.push(rule(x + 16, x + 174, 240));
    els.push(text(x + 20, 248, o.w, { size: 11, family: 3, stroke: T.ink }));
    els.push(text(x + 20, 268, o.b, { size: 11, family: 3, stroke: T.ink }));
    els.push(text(x, 306, o.formula, { size: 12, family: 3, stroke: T.inkMuted, align: 'center', width: 190 }));
  };

  const arr = (x1, x2) => els.push(arrow([[x1, MID], [x2, MID]],
    { stroke: T.inkMuted, strokeWidth: 1.4, roughness: 0.5 }));

  arr(163, 201);
  layer(205, { c: T.accent, name: 'Layer 1', w: 'W₁  (3, 4)', b: 'b₁  (3,)', formula: 'Z₁ = X · W₁ᵀ + b₁' });
  arr(399, 439);
  stack(els, 445, MID - 39, 3, 70, 26, { c: T.ink });
  els.push(text(445, MID - 63, 'Z₁', { size: 16, stroke: T.ink, align: 'center', width: 70 }));
  els.push(text(445, MID + 47, '(3,)', { size: 12, family: 3, stroke: T.inkMuted, align: 'center', width: 70 }));
  arr(521, 561);
  layer(565, { c: T.success, name: 'Layer 2', w: 'W₂  (3, 3)', b: 'b₂  (3,)', formula: 'Z₂ = Z₁ · W₂ᵀ + b₂' });
  arr(759, 799);
  stack(els, 805, MID - 39, 3, 70, 26, { c: T.success });
  els.push(text(805, MID - 63, 'Z₂', { size: 16, stroke: T.success, align: 'center', width: 70 }));
  els.push(text(805, MID + 47, '(3,)', { size: 12, family: 3, stroke: T.inkMuted, align: 'center', width: 70 }));

  els.push(rect(M, 344, 880, 76, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(56, 358, 'W₁ is (3, 4) because layer 1 has 3 neurons and each one consumes the 4 input features.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(56, 382, 'W₂ is (3, 3) because layer 2 has 3 neurons and each one consumes the 3 outputs of layer 1.',
    { size: 12, stroke: T.inkMuted }));

  els.push(text(M, 446, 'The weights per neuron in a layer equal the number of neurons in the layer before it.',
    { size: 13, stroke: T.ink }));
  els.push(text(M, 470, 'Both cards above are the same call with different indices. Depth comes from chaining the call,',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 490, 'not from introducing new mathematics.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'A two-layer network: forward pass with shapes labelled',
    desc: 'A hand-drawn two-layer dense network running left to right. A four-cell input X of shape '
      + 'four enters the first layer card, which holds three neurons, a weight matrix W1 of shape '
      + 'three by four, a bias b1 of shape three, and the formula Z1 equals X dotted with W1 '
      + 'transposed plus b1. Its three-cell output Z1 of shape three feeds the second layer card, '
      + 'drawn identically, holding three neurons, a weight matrix W2 of shape three by three, a '
      + 'bias b2 of shape three, and the formula Z2 equals Z1 dotted with W2 transposed plus b2, '
      + 'producing the three-cell output Z2. A band explains the weight shapes: W1 is three by four '
      + 'because layer one has three neurons each consuming the four input features, and W2 is three '
      + 'by three because layer two has three neurons each consuming the three outputs of layer one. '
      + 'A footer states that the weights per neuron in a layer equal the number of neurons in the '
      + 'layer before it, and that depth comes from chaining the same call rather than from new '
      + 'mathematics.',
  };
}

// ---------------------------------------------------------------- diagram 2
// The batch case as a chain of shapes, with the two inner-dimension checks
// called out between the operands they belong to. Shape continuity is the only
// thing stacking actually adds, so it is the only thing this figure tracks.
function dimensionFlow() {
  resetSeq();
  const W = 960, H = 420;

  const els = [...heading('Shape continuity across two layers',
    'A batch of N samples, and the two places where an inner dimension has to agree.')];

  const CW = 150, CY = 120, CH = 70;
  const X = [41, 223, 405, 587, 769];

  chip(els, X[0], CY, CW, CH, 'X', '(N, 4)', { c: T.primary });
  chip(els, X[1], CY, CW, CH, 'W₁ᵀ', '(4, 3)', { c: T.accent });
  chip(els, X[2], CY, CW, CH, 'Z₁', '(N, 3)', { c: T.ink });
  chip(els, X[3], CY, CW, CH, 'W₂ᵀ', '(3, 3)', { c: T.success });
  chip(els, X[4], CY, CW, CH, 'Z₂', '(N, 3)', { c: T.success });

  const op = (cx, s) => els.push(text(cx - 12, CY + 22, s,
    { size: 18, stroke: T.inkMuted, align: 'center', width: 24 }));
  op(207, '·'); op(389, '='); op(571, '·'); op(753, '=');

  // The checks sit centred under the pair they concern, not under the gap, so
  // it is clear which two shapes are being compared.
  const check = (a, b, label) => {
    const cx = (X[a] + X[b] + CW) / 2;
    els.push(line([[X[a] + CW - 20, CY + CH], [cx, CY + CH + 24], [X[b] + 20, CY + CH]],
      { stroke: T.accent, strokeWidth: 1.3, strokeStyle: 'dashed', roughness: 0.5 }));
    els.push(text(cx - 110, CY + CH + 30, label,
      { size: 11, stroke: T.accent, align: 'center', width: 220 }));
  };
  check(0, 1, 'inner 4 = 4');
  check(2, 3, 'inner 3 = 3');

  els.push(text(390, 252, '+ b₁ (3,) broadcast over N rows',
    { size: 10, stroke: T.inkSubtle, align: 'center', width: 180 }));
  els.push(text(754, 252, '+ b₂ (3,)',
    { size: 10, stroke: T.inkSubtle, align: 'center', width: 180 }));

  els.push(rect(M, 288, 880, 64, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 302, 'Shape continuity is the one thing stacking adds.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 326, 'A layer’s weight matrix needs one column per neuron in the layer before it.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 374, 'Print Z₁.shape after layer 1 and check it against (N, m₁). A mismatch surfaces here rather than deep in training.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Dimension flow across two stacked layers',
    desc: 'A hand-drawn left-to-right shape chain for a batch of N samples through two dense layers. '
      + 'The input X of shape N by four is dotted with W1 transposed of shape four by three, giving '
      + 'Z1 of shape N by three; Z1 is then dotted with W2 transposed of shape three by three, '
      + 'giving Z2 of shape N by three. Two dashed checks are drawn under the pairs they concern: '
      + 'the inner four against four between X and W1 transposed, and the inner three against three '
      + 'between Z1 and W2 transposed. Notes record that b1 of shape three is broadcast over all N '
      + 'rows, and likewise b2. A band states that shape continuity is the one thing stacking adds '
      + 'and that a layer’s weight matrix needs one column per neuron in the layer before it, '
      + 'and a footer recommends printing the shape of Z1 after layer one and checking it against N '
      + 'by m1, so a mismatch surfaces there rather than deep in training.',
  };
}

// ---------------------------------------------------------------- diagram 3
// New. Section 4 substitutes Z1 away and lands on a single equivalent layer.
// That collapse is the reason Part 06 exists, and drawing the two chains at the
// same scale is what makes "these are the same function" a thing you can see
// rather than a line of algebra you have to trust.
function linearCollapse() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('Two linear layers are one linear layer',
    'Substitute Z₁ away and the stack collapses. Nothing above an activation function survives depth alone.')];

  const CW = 140, CH = 64;

  const TOP = [70, 240, 410, 580, 750];
  chip(els, TOP[0], 96, CW, CH, 'X', '(N, 4)', { c: T.primary });
  chip(els, TOP[1], 96, CW, CH, 'W₁ᵀ, b₁', 'layer 1', { c: T.accent });
  chip(els, TOP[2], 96, CW, CH, 'Z₁', '(N, 3)', { c: T.ink });
  chip(els, TOP[3], 96, CW, CH, 'W₂ᵀ, b₂', 'layer 2', { c: T.accent });
  chip(els, TOP[4], 96, CW, CH, 'Z₂', '(N, 3)', { c: T.success });
  [[210, 240], [380, 410], [550, 580], [720, 750]].forEach(([a, b]) => {
    els.push(arrow([[a, 128], [b, 128]], { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.5 }));
  });

  els.push(arrow([[480, 176], [480, 240]], { stroke: T.primary, strokeWidth: 1.6, roughness: 0.4 }));
  els.push(text(496, 196, 'collapses to', { size: 12, stroke: T.primary }));

  const BOT = [240, 410, 580];
  chip(els, BOT[0], 248, CW, CH, 'X', '(N, 4)', { c: T.primary });
  chip(els, BOT[1], 248, CW, CH, 'W*, b*', 'one layer', { c: T.primary, spine: T.primary });
  chip(els, BOT[2], 248, CW, CH, 'Z₂', '(N, 3)', { c: T.success });
  [[380, 410], [550, 580]].forEach(([a, b]) => {
    els.push(arrow([[a, 280], [b, 280]], { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.5 }));
  });

  els.push(rect(M, 340, 880, 70, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 354, 'W* = W₁ᵀ W₂ᵀ            b* = b₁ W₂ᵀ + b₂',
    { size: 14, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 382, 'Same input shape, same output shape, and a single layer’s parameters doing the work of two.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 428, 'Part 06 puts an activation between the layers, and Z₁ can no longer be substituted away. That is where depth starts paying.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Without an activation, two stacked linear layers collapse into one',
    desc: 'A hand-drawn before-and-after. The upper chain shows the full two-layer forward pass: an '
      + 'input X of shape N by four through layer one, carrying W1 transposed and b1, producing Z1 '
      + 'of shape N by three, then through layer two, carrying W2 transposed and b2, producing Z2 of '
      + 'shape N by three. An arrow labelled collapses to leads down to a shorter chain: the same X '
      + 'through a single layer carrying W star and b star, producing the same Z2. A band gives the '
      + 'substitution: W star is W1 transposed times W2 transposed, and b star is b1 times W2 '
      + 'transposed plus b2, so the collapsed form has the same input shape, the same output shape, '
      + 'and one layer’s worth of parameters doing the work of two. A footer notes that Part 06 '
      + 'puts an activation between the layers, after which Z1 can no longer be substituted away, '
      + 'and that this is where depth starts paying.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { multiLayerAnatomy, dimensionFlow, linearCollapse };

if (require.main === module) {
  emit('01-multi-layer-anatomy', multiLayerAnatomy());
  emit('02-dimension-flow', dimensionFlow());
  emit('03-linear-collapse', linearCollapse());
}
