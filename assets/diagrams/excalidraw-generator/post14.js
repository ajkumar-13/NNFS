// The post 14 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. It covers similar
// ground to post 13's `02-outer-product`, which is deliberate on the posts' part
// — 13 reaches the matrix by broadcasting and 14 by a matrix product — so this
// alternate leans on what is actually different: the shape arithmetic, and the
// fact that the result already has the shape of the stored weights.
//
// Scene 02 is new: section 6 observes that the gradient shapes do not depend on
// the batch size, and section 9 explains why the sum over samples needs no code.
// That is the deepest claim in the post and it had no figure.
//
//   01-matrix-weight-gradient  960 x 480
//   02-batch-axis-contracts    960 x 480   (new)

const { T, rect, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

function grid(els, x, y, vals, cw, ch, o = {}) {
  const rows = vals.length, cols = vals[0].length;
  els.push(rect(x, y, cols * cw, rows * ch, {
    stroke: o.c ?? T.ink, fill: T.surface, strokeWidth: 1.4, roundness: null,
  }));
  for (let i = 1; i < cols; i++) {
    els.push(line([[x + i * cw, y], [x + i * cw, y + rows * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  for (let j = 1; j < rows; j++) {
    els.push(line([[x, y + j * ch], [x + cols * cw, y + j * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  const size = o.size ?? 11;
  vals.forEach((row, j) => row.forEach((v, i) => {
    els.push(text(x + i * cw, y + j * ch + (ch - size) / 2 - 1, String(v), {
      size, family: 3, stroke: o.ink ?? T.ink, align: 'center', width: cw,
    }));
  }));
  return { bottom: y + rows * ch };
}

// A shape as its two axes, one box each, so the contracted axis can be marked.
function shapeBoxes(els, x, y, vals, colours, o = {}) {
  const bw = o.bw ?? 46, bh = o.bh ?? 42, size = o.size ?? 16;
  const centres = [];
  vals.forEach((v, i) => {
    els.push(rect(x + i * bw, y, bw, bh, {
      stroke: colours[i], fill: T.surface, strokeWidth: 1.5, roundness: null,
    }));
    els.push(text(x + i * bw, y + (bh - size) / 2 - 1, String(v), {
      size, family: 3, stroke: T.ink, align: 'center', width: bw,
    }));
    centres.push(x + i * bw + bw / 2);
  });
  return { centres, end: x + vals.length * bw, bottom: y + bh };
}

// ---------------------------------------------------------------- diagram 1
// The shape arithmetic is the point here, so it is drawn at full size rather
// than written as a caption: the contracted 1 disappears and the two outer
// axes survive into exactly the shape the weights are already stored in.
function matrixWeightGradient() {
  resetSeq();
  const W = 960, H = 480;

  const els = [...heading('The weight gradient as one matrix product',
    'The shapes decide the operation. Contract the length-1 axis and what is left is already the weights’ shape.')];

  const BY = 108;
  const A = shapeBoxes(els, 250, BY, ['m', 1], [T.primary, T.accent]);
  const B = shapeBoxes(els, 430, BY, [1, 'n'], [T.accent, T.primary]);
  const C = shapeBoxes(els, 610, BY, ['m', 'n'], [T.primary, T.primary]);
  els.push(text(346, BY + 10, '·', { size: 20, stroke: T.inkMuted, align: 'center', width: 24 }));
  els.push(text(526, BY + 10, '=', { size: 20, stroke: T.inkMuted, align: 'center', width: 24 }));
  els.push(text(250, BY - 24, '(∂L/∂Z)ᵀ', { size: 12, family: 3, stroke: T.alert, align: 'center', width: 92 }));
  els.push(text(430, BY - 24, 'X', { size: 12, family: 3, stroke: T.primary, align: 'center', width: 92 }));
  els.push(text(610, BY - 24, '∂L/∂W', { size: 12, family: 3, stroke: T.accent, align: 'center', width: 92 }));

  els.push(line([[A.centres[1], A.bottom], [(A.centres[1] + B.centres[0]) / 2, A.bottom + 26], [B.centres[0], A.bottom]],
    { stroke: T.accent, strokeWidth: 1.4, strokeStyle: 'dashed', roughness: 0.5 }));
  els.push(text(288, A.bottom + 32, 'contracted away',
    { size: 11, stroke: T.accent, align: 'center', width: 254 }));
  els.push(text(610, C.bottom + 10, 'the shape the weights already have',
    { size: 11, stroke: T.primary, align: 'center', width: 92 + 200 }));

  els.push(...card(M, 232, 880, 160, { spine: T.accent }));
  els.push(text(62, 244, 'The same twelve numbers as Part 13, now in one call',
    { size: 13, stroke: T.ink }));
  els.push(text(62, 266, 'dL_dW = dL_dZ.T @ X', { size: 12, family: 3, stroke: T.accent }));
  els.push(rule(56, 904, 292));

  els.push(text(78, 302, '(∂L/∂Z)ᵀ', { size: 10, family: 3, stroke: T.alert, align: 'center', width: 60 }));
  grid(els, 78, 318, [['43.2'], ['43.2'], ['43.2']], 60, 22, { c: T.alert, ink: T.alert, size: 10 });
  els.push(text(148, 342, '·', { size: 16, stroke: T.inkMuted, align: 'center', width: 20 }));
  els.push(text(176, 302, 'X', { size: 10, family: 3, stroke: T.primary, align: 'center', width: 224 }));
  grid(els, 176, 340, [[1, 2, 3, 4]], 56, 22, { c: T.primary, ink: T.primary, size: 11 });
  els.push(text(410, 342, '=', { size: 16, stroke: T.inkMuted, align: 'center', width: 20 }));
  els.push(text(440, 302, '∂L/∂W', { size: 10, family: 3, stroke: T.accent, align: 'center', width: 224 }));
  grid(els, 440, 318, [
    ['43.2', '86.4', '129.6', '172.8'],
    ['43.2', '86.4', '129.6', '172.8'],
    ['43.2', '86.4', '129.6', '172.8'],
  ], 56, 22, { c: T.accent, size: 10 });
  els.push(text(690, 318, '(3, 1) · (1, 4) = (3, 4)', { size: 12, family: 3, stroke: T.ink }));
  els.push(text(690, 344, 'One row per neuron,', { size: 10, stroke: T.inkMuted }));
  els.push(text(690, 360, 'one column per input.', { size: 10, stroke: T.inkMuted }));

  els.push(text(M, 414, 'The matrix product is not an approximation of the chain rule. It is the chain-rule formula, packaged so one call fills every entry.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 436, 'Nothing is traded for the speed: the twelve numbers are the twelve numbers Part 13 wrote out by hand.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'The weight-gradient matrix as an outer product of upstream gradient and input',
    desc: 'A hand-drawn schematic of the matrix-form weight gradient. Along the top the shapes are '
      + 'drawn as axis boxes: the transposed upstream gradient of shape m by one, dotted with the '
      + 'input of shape one by n, giving m by n. The two inner length-one axes are joined by a '
      + 'dashed tie labelled contracted away, and the result is annotated as the shape the weights '
      + 'already have. Beneath, the same product is worked with numbers: a column of three copies '
      + 'of forty-three point two dotted with the row one, two, three, four, giving a three-by-four '
      + 'grid whose every row reads forty-three point two, eighty-six point four, one hundred and '
      + 'twenty-nine point six and one hundred and seventy-two point eight, with the shape '
      + 'arithmetic three by one dot one by four equals three by four written beside it. A footer '
      + 'notes that the matrix product is not an approximation of the chain rule but the chain-rule '
      + 'formula packaged so one call fills every entry, and that the twelve numbers are the same '
      + 'twelve Part 13 wrote out by hand.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 6 makes the observation the whole implementation rests on — the
// gradient shape is independent of the batch size — and section 9 explains why
// the sum over samples costs no code. Both come from the same fact: the batch
// axis is the axis the product contracts.
function batchAxisContracts() {
  resetSeq();
  const W = 960, H = 480;

  const els = [...heading('The batch axis is the axis that gets contracted',
    'Which is why the gradient keeps its shape whatever N is, and why nothing in the code sums over samples.')];

  const row = (y, o) => {
    els.push(...card(M, y, 880, 124, { spine: o.c }));
    els.push(text(62, y + 12, o.title, { size: 14, stroke: o.c }));
    els.push(text(62, y + 34, fit(o.sub, 10, 200, 'row sub'), { size: 10, stroke: T.inkSubtle }));

    const BY = y + 38;
    const A = shapeBoxes(els, 300, BY, o.a, [T.primary, o.c], { bw: 44, bh: 40, size: 15 });
    const B = shapeBoxes(els, 434, BY, o.b, [o.c, T.primary], { bw: 44, bh: 40, size: 15 });
    const C = shapeBoxes(els, 590, BY, o.r, [T.primary, T.primary], { bw: 44, bh: 40, size: 15 });
    els.push(text(392, BY + 8, '·', { size: 18, stroke: T.inkMuted, align: 'center', width: 24 }));
    els.push(text(538, BY + 8, '=', { size: 18, stroke: T.inkMuted, align: 'center', width: 24 }));
    els.push(line([[A.centres[1], A.bottom], [(A.centres[1] + B.centres[0]) / 2, A.bottom + 22], [B.centres[0], A.bottom]],
      { stroke: o.c, strokeWidth: 1.4, strokeStyle: 'dashed', roughness: 0.5 }));
    els.push(text(300, A.bottom + 28, o.tie, { size: 10, stroke: o.c, align: 'center', width: 220 }));
    els.push(text(700, BY + 2, fit(o.note, 11, 200, 'row note'), { size: 11, stroke: T.inkMuted }));
  };

  row(104, {
    c: T.inkMuted, title: 'One sample', sub: 'N = 1',
    a: ['m', 1], b: [1, 'n'], r: ['m', 'n'],
    tie: 'a length-1 axis, contracted',
    note: 'Nothing to sum: there is\nonly one sample to add up.',
  });

  row(248, {
    c: T.success, title: 'A batch of N', sub: 'N = 3, or 32, or 512',
    a: ['m', 'N'], b: ['N', 'n'], r: ['m', 'n'],
    tie: 'the batch axis, contracted',
    note: 'The contraction sums over\nthe batch. No loop, no np.sum.',
  });

  els.push(rect(M, 392, 880, 68, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(56, 404, 'dL_dW = dL_dZ.T @ X        →  (m, n) whatever N is',
    { size: 12, family: 3, stroke: T.ink }));
  els.push(text(56, 428, 'dL_dB = np.sum(dL_dZ, axis=0)   →  the bias has no input to scale it, so its sum has nowhere to hide and is written out',
    { size: 11, family: 3, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Why the weight-gradient shape does not depend on the batch size',
    desc: 'Two hand-drawn rows comparing shape arithmetic for one sample and for a batch. The upper '
      + 'row, one sample, dots an m by one with a one by n to give m by n; the two inner length-one '
      + 'axes are tied and labelled as contracted, with a note that there is nothing to sum because '
      + 'there is only one sample. The lower row, a batch of N, dots an m by N with an N by n and '
      + 'again gives m by n; the two inner N axes are tied and labelled as the batch axis being '
      + 'contracted, with a note that the contraction sums over the batch with no loop and no call '
      + 'to np.sum. A band gives both lines of the backward pass: the weight gradient from a single '
      + 'matrix product, which comes out m by n whatever N is, and the bias gradient from an '
      + 'explicit sum along axis nought, because the bias has no input to scale it so its sum has '
      + 'nowhere to hide and has to be written out.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { matrixWeightGradient, batchAxisContracts };

if (require.main === module) {
  emit('01-matrix-weight-gradient', matrixWeightGradient());
  emit('02-batch-axis-contracts', batchAxisContracts());
}
