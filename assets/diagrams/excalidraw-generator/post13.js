// The post 13 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 7.1 calls one line the structural heart of the post and names it an
// outer product, and that line is the bridge into Part 14, but nothing drew it.
//
//   01-layer-backprop  960 x 540
//   02-outer-product   960 x 450   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

const INPUTS = [1, 2, 3, 4];
const UPSTREAM = 43.2;

function grid(els, x, y, vals, cw, ch, o = {}) {
  const rows = vals.length, cols = vals[0].length;
  els.push(rect(x, y, cols * cw, rows * ch, {
    stroke: o.c ?? T.ink, fill: T.surface, strokeWidth: o.strokeWidth ?? 1.4, roundness: null,
  }));
  for (let i = 1; i < cols; i++) {
    els.push(line([[x + i * cw, y], [x + i * cw, y + rows * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  for (let j = 1; j < rows; j++) {
    els.push(line([[x, y + j * ch], [x + cols * cw, y + j * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  const size = o.size ?? 12;
  vals.forEach((row, j) => row.forEach((v, i) => {
    els.push(text(x + i * cw, y + j * ch + (ch - size) / 2 - 1, String(v), {
      size, family: 3, stroke: o.ink ?? T.ink, align: 'center', width: cw,
    }));
  }));
  return { w: cols * cw, h: rows * ch, bottom: y + rows * ch };
}

// ---------------------------------------------------------------- diagram 1
// One upstream number entering three neurons at once. The broadcast is drawn as
// three arrows leaving a single chip rather than as three separate chains,
// because "computing it once at the loss is enough" is the section's claim.
function layerBackprop() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('Backpropagation through a layer of three neurons',
    'Fifteen parameters, one upstream gradient. The structure of Part 12, run three times over.')];

  const IY = [118, 162, 206, 250];
  INPUTS.forEach((v, j) => {
    els.push(...card(50, IY[j], 76, 34, { stroke: T.primary }));
    els.push(text(50, IY[j] + 9, `X${j + 1} = ${v}`,
      { size: 12, family: 3, stroke: T.primary, align: 'center', width: 76 }));
  });

  const NY = [140, 196, 252];
  const ZV = ['3.1', '7.2', '11.3'];
  NY.forEach((cy, k) => {
    IY.forEach((iy) => els.push(line([[128, iy + 17], [222, cy]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.4 })));
    els.push(circle(250, cy, 28, { stroke: T.accent, fill: T.surface, strokeWidth: 1.6 }));
    els.push(text(222, cy - 8, `Z${k + 1}`, { size: 12, family: 3, stroke: T.ink, align: 'center', width: 56 }));
    els.push(text(286, cy - 7, `A${k + 1} = ${ZV[k]}`, { size: 11, family: 3, stroke: T.inkMuted }));
    els.push(line([[278, cy], [372, 196]], { stroke: T.border, strokeWidth: 1, roughness: 0.4 }));
  });

  els.push(circle(400, 196, 26, { stroke: T.success, fill: T.surface, strokeWidth: 1.6 }));
  els.push(text(374, 188, 'Σ', { size: 17, stroke: T.success, align: 'center', width: 52 }));
  els.push(arrow([[426, 196], [452, 196]], { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
  els.push(...card(456, 178, 76, 36, { stroke: T.success }));
  els.push(text(456, 188, 'Y = 21.6', { size: 12, family: 3, stroke: T.success, align: 'center', width: 76 }));
  els.push(arrow([[534, 196], [556, 196]], { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
  els.push(...card(560, 178, 40, 36, { stroke: T.alert }));
  els.push(text(560, 188, 'Y²', { size: 13, family: 3, stroke: T.alert, align: 'center', width: 40 }));
  els.push(text(456, 222, 'L = 466.56', { size: 11, family: 3, stroke: T.inkSubtle, align: 'center', width: 144 }));

  const BACK = [
    { x: 40, w: 140, t: '∂L/∂Y = 2Y', v: '43.2', c: T.alert },
    { x: 200, w: 130, t: '∂Y/∂A', v: '1', c: T.alert },
    { x: 350, w: 130, t: 'ReLU′(Z)', v: '1', c: T.alert },
    { x: 500, w: 100, t: '× Xⱼ', v: 'per weight', c: T.accent },
  ];
  BACK.forEach((b, i) => {
    els.push(...card(b.x, 324, b.w, 60, { spine: b.c }));
    els.push(text(b.x, 336, b.t, { size: 11, family: 3, stroke: T.inkMuted, align: 'center', width: b.w }));
    els.push(text(b.x, 356, b.v, { size: 14, family: 3, stroke: b.c, align: 'center', width: b.w }));
    if (i > 0) {
      els.push(arrow([[b.x - 4, 354], [BACK[i - 1].x + BACK[i - 1].w + 4, 354]],
        { stroke: T.alert, strokeWidth: 1.3, roughness: 0.4 }));
    }
  });
  els.push(text(40, 302, 'backward · one upstream number, shared by all fifteen parameters',
    { size: 10, stroke: T.alert }));

  els.push(...card(620, 110, 300, 274, { spine: T.accent }));
  els.push(text(642, 122, 'The fifteen gradients', { size: 15, stroke: T.accent }));
  els.push(rule(634, 906, 150));
  els.push(text(642, 158, '∂L/∂W', { size: 11, family: 3, stroke: T.inkSubtle }));
  INPUTS.forEach((v, j) => els.push(text(700 + j * 52, 158, `X${j + 1}`,
    { size: 10, stroke: T.inkSubtle, align: 'center', width: 52 })));
  const GW = [1, 2, 3].map(() => INPUTS.map((v) => (UPSTREAM * v).toFixed(1)));
  grid(els, 700, 176, GW, 52, 30, { c: T.accent, size: 10 });
  [1, 2, 3].forEach((k, i) => els.push(text(642, 176 + i * 30 + 9, `n${k}`,
    { size: 10, stroke: T.inkSubtle })));
  els.push(rule(634, 906, 282));
  els.push(text(642, 292, '∂L/∂b', { size: 11, family: 3, stroke: T.inkSubtle }));
  grid(els, 700, 288, [['43.2', '43.2', '43.2']], 52, 30, { c: T.success, size: 10 });
  els.push(text(642, 336, 'Every weight on the same input gets', { size: 10, stroke: T.inkMuted }));
  els.push(text(642, 352, 'the same gradient. Only the column moves.', { size: 10, stroke: T.inkMuted }));

  els.push(rect(M, 414, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 426, '∂L/∂Wₖⱼ  =  2Y · 1 · ReLU′(Zₖ) · Xⱼ',
    { size: 14, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 452, 'Two factors are shared by the whole layer, one by each neuron’s four weights, and only the last varies within a neuron.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 500, 'The gradient grows with the input magnitude, which is why X₄ = 4 produces four times the gradient of X₁ = 1.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Backpropagation through a layer of three neurons sharing four inputs',
    desc: 'A hand-drawn diagram of one layer. The forward part shows four inputs, X1 equals one '
      + 'through X4 equals four, feeding all three neurons, whose pre-activations are three point '
      + 'one, seven point two and eleven point three; ReLU leaves each unchanged, the three '
      + 'activations are summed to Y equals twenty-one point six, and squaring gives a loss of four '
      + 'hundred and sixty-six point five six. The backward row is a chain of four chips read right '
      + 'to left: the loss derivative two Y equals forty-three point two, the sum derivative of one, '
      + 'the ReLU derivative of one, and finally multiplication by each input, annotated as one '
      + 'upstream number shared by all fifteen parameters. A side panel tabulates the results: a '
      + 'three-by-four grid of weight gradients where every row reads forty-three point two, '
      + 'eighty-six point four, one hundred and twenty-nine point six and one hundred and '
      + 'seventy-two point eight, and three bias gradients all equal to forty-three point two. A '
      + 'band gives the general formula and notes that two factors are shared by the whole layer, '
      + 'one by each neuron’s four weights, and only the last varies within a neuron.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 7.1 names one line the structural heart of the post and calls it
// an outer product. Drawing the column against the row, with one cell traced
// back to the two numbers that produced it, is what makes the reshape in that
// line stop looking like a NumPy trick.
function outerProduct() {
  resetSeq();
  const W = 960, H = 450;

  const els = [...heading('The weight gradients are an outer product',
    'A column of upstream gradients against a row of inputs. Every cell is one product of one pair.')];

  const CW = 70, CH = 40;
  const COL = [[UPSTREAM.toFixed(1)], [UPSTREAM.toFixed(1)], [UPSTREAM.toFixed(1)]];
  const ROW = [INPUTS];
  const OUT = [1, 2, 3].map(() => INPUTS.map((v) => (UPSTREAM * v).toFixed(1)));

  els.push(text(120, 124, 'dL_dZ   (3, 1)', { size: 11, family: 3, stroke: T.alert, align: 'center', width: 70 }));
  grid(els, 120, 150, COL, CW, CH, { c: T.alert, ink: T.alert, size: 11 });
  els.push(text(210, 196, '×', { size: 18, stroke: T.inkMuted, align: 'center', width: 30 }));
  els.push(text(260, 234, 'inputs   (4,)', { size: 11, family: 3, stroke: T.primary, align: 'center', width: 280 }));
  grid(els, 260, 190, ROW, CW, CH, { c: T.primary, ink: T.primary, size: 12 });
  els.push(text(556, 196, '=', { size: 18, stroke: T.inkMuted, align: 'center', width: 30 }));
  els.push(text(610, 124, 'dL_dW   (3, 4)', { size: 11, family: 3, stroke: T.accent, align: 'center', width: 280 }));
  grid(els, 610, 150, OUT, CW, CH, { c: T.accent, size: 11 });

  // One cell traced back to the pair that made it.
  els.push(rect(120, 150, CW, CH, { stroke: T.ink, strokeWidth: 2, roundness: null }));
  els.push(rect(470, 190, CW, CH, { stroke: T.ink, strokeWidth: 2, roundness: null }));
  els.push(rect(820, 150, CW, CH, { stroke: T.ink, strokeWidth: 2, roundness: null }));
  els.push(line([[190, 170], [608, 170]],
    { stroke: T.inkSubtle, strokeWidth: 1, strokeStyle: 'dotted', roughness: 0.3 }));
  els.push(line([[505, 188], [505, 148], [818, 148]],
    { stroke: T.inkSubtle, strokeWidth: 1, strokeStyle: 'dotted', roughness: 0.3 }));
  els.push(text(300, 296, '43.2  ×  4  =  172.8',
    { size: 13, family: 3, stroke: T.ink, align: 'center', width: 360 }));
  els.push(text(300, 318, 'row 1 of the column, column 4 of the row',
    { size: 10, stroke: T.inkSubtle, align: 'center', width: 360 }));

  els.push(rect(M, 350, 880, 76, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 362, 'dL_dW = dL_dZ.reshape(-1, 1) * inputs',
    { size: 14, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 388, 'The reshape makes a (3, 1) column; broadcasting stretches it against the (4,) row to fill a (3, 4) result.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));
  els.push(text(M, 408, 'Part 14 replaces the reshape and the broadcast with one np.dot that does the same thing for a whole batch.',
    { size: 11, stroke: T.inkSubtle, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The weight-gradient matrix as an outer product of upstream gradient and input',
    desc: 'A hand-drawn outer product. On the left, a three-by-one column of upstream gradients, each '
      + 'forty-three point two, labelled dL_dZ. In the middle, a one-by-four row of the inputs one, '
      + 'two, three and four. On the right, the resulting three-by-four matrix of weight gradients, '
      + 'every row reading forty-three point two, eighty-six point four, one hundred and twenty-nine '
      + 'point six and one hundred and seventy-two point eight. One cell of the result is outlined '
      + 'and traced by dotted leaders back to the two numbers that produced it, with the arithmetic '
      + 'written out: forty-three point two times four equals one hundred and seventy-two point '
      + 'eight, from row one of the column and column four of the row. A band gives the line of code '
      + 'that computes it, explains that the reshape makes a three-by-one column which broadcasting '
      + 'stretches against the four-element row to fill a three-by-four result, and notes that Part '
      + '14 replaces both with a single np.dot that does the same thing for a whole batch.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { layerBackprop, outerProduct };

if (require.main === module) {
  emit('01-layer-backprop', layerBackprop());
  emit('02-outer-product', outerProduct());
}
