// The post 18 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 5 flags that `dvalues` means something different in the loss class
// than it does everywhere else in the series, because the loss sits at the top
// of the chain and its upstream is the scalar 1. That is a naming trap, not an
// arithmetic one, and nothing drew it.
//
//   01-cross-entropy-backward  960 x 480
//   02-where-backprop-starts   960 x 460   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

const Y_TRUE = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const Y_PRED = [['0.7', '0.2', '0.1'], ['0.1', '0.6', '0.3'], ['0.0', '0.0', '1.0']];
const RAW = [['−1.43', '0', '0'], ['0', '−1.67', '0'], ['0', '0', '−1.0']];
const SCALED = [['−0.476', '0', '0'], ['0', '−0.556', '0'], ['0', '0', '−0.333']];

function grid(els, x, y, vals, cw, ch, o = {}) {
  const rows = vals.length, cols = vals[0].length;
  els.push(rect(x, y, cols * cw, rows * ch, {
    stroke: o.c ?? T.ink, fill: T.surface, strokeWidth: 1.5, roundness: null,
  }));
  for (let i = 1; i < cols; i++) {
    els.push(line([[x + i * cw, y], [x + i * cw, y + rows * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  for (let j = 1; j < rows; j++) {
    els.push(line([[x, y + j * ch], [x + cols * cw, y + j * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  const size = o.size ?? 9;
  vals.forEach((row, j) => row.forEach((v, i) => {
    const live = o.live ? o.live(j, i) : true;
    els.push(text(x + i * cw, y + j * ch + (ch - size) / 2 - 1, String(v), {
      size, family: 3, align: 'center', width: cw,
      stroke: live ? (o.ink ?? T.ink) : T.inkSubtle,
    }));
  }));
}

// ---------------------------------------------------------------- diagram 1
// Four grids in a row, with the one-hot mask carried through as colour: the
// entries the mask kills are drawn subtle in every stage, so the fact that only
// one number per row survives is visible before any arithmetic is read.
function crossEntropyBackward() {
  resetSeq();
  const W = 960, H = 480;

  const els = [...heading('The first gradient in the whole backward pass',
    'One-hot labels divided into the predictions, then scaled by the batch size. One live number per row.')];

  const CW = 52, CH = 32, GY = 176;
  const XS = [108, 304, 500, 696];
  const live = (j, i) => Y_TRUE[j][i] === 1;

  const STAGES = [
    { x: XS[0], label: 'y_true', vals: Y_TRUE, c: T.primary, note: 'one-hot' },
    { x: XS[1], label: 'ŷ  (predictions)', vals: Y_PRED, c: T.inkMuted, note: 'softmax output' },
    { x: XS[2], label: '−y / ŷ', vals: RAW, c: T.accent, note: 'element-wise' },
    { x: XS[3], label: '÷ N = 3', vals: SCALED, c: T.success, note: 'batch-normalised' },
  ];
  STAGES.forEach((s, i) => {
    els.push(text(s.x, 148, s.label, { size: 12, family: 3, stroke: s.c, align: 'center', width: 156 }));
    grid(els, s.x, GY, s.vals, CW, CH, { c: s.c, live, ink: i >= 2 ? s.c : T.ink });
    els.push(text(s.x, GY + 106, s.note, { size: 10, stroke: T.inkSubtle, align: 'center', width: 156 }));
    if (i < 3) {
      els.push(arrow([[s.x + 160, GY + 48], [XS[i + 1] - 4, GY + 48]],
        { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
    }
  });
  els.push(text(108, 322, 'Everywhere the label is 0, the gradient is 0 whatever the prediction was.',
    { size: 11, stroke: T.inkSubtle }));

  els.push(rect(M, 352, 880, 68, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 364, '∂L/∂ŷᵢⱼ  =  − yᵢⱼ / (ŷᵢⱼ · N)',
    { size: 14, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 392, 'The smaller the probability put on the right class, the larger the pull. A confident wrong answer generates the biggest gradient there is.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  els.push(text(M, 444, 'The two 0.0 entries in row three would make −y/ŷ read 0/0. That is why forward clips ŷ into [1e−7, 1 − 1e−7] before backward ever sees it.',
    { size: 11, stroke: T.alert }));

  return {
    W, H, els,
    title: 'Cross-entropy backward: element-wise division by the prediction, normalised by batch size',
    desc: 'A hand-drawn four-stage worked example for a batch of three samples. The first grid is the '
      + 'one-hot label matrix with a single one per row. The second is the predicted probabilities, '
      + 'nought point seven, nought point two, nought point one on the first row, nought point one, '
      + 'nought point six, nought point three on the second, and nought, nought, one on the third. '
      + 'The third grid is minus y divided by y-hat element-wise, giving minus one point four three, '
      + 'minus one point six seven and minus one on the live positions. The fourth divides by the '
      + 'batch size of three, giving minus nought point four seven six, minus nought point five five '
      + 'six and minus nought point three three three. In every grid the entries the one-hot mask '
      + 'kills are drawn faint, so it is visible that only one number per row survives. A band gives '
      + 'the formula and notes that the smaller the probability put on the right class, the larger '
      + 'the pull. A footer warns that the two nought entries in row three would make the division '
      + 'read nought over nought, which is why forward clips the predictions before backward sees '
      + 'them.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 5 names the trap: `dvalues` is the incoming gradient in every
// other class and the predictions themselves in this one. Drawing what is to
// the right of each layer is what explains the difference, since the loss is
// the only class with nothing there.
function whereBackpropStarts() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('Why the loss class reads its argument differently',
    'Every backward takes something called dvalues. In one of them it is not a gradient at all.')];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 272, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 122, fit(o.sig, 11, 380 * 0.94, 'signature'),
      { size: 11, family: 3, stroke: T.ink }));
    els.push(rule(x + 16, x + 404, 150));

    // What sits to the right of this layer.
    o.chain.forEach((c, i) => {
      const cx = x + 24 + i * 128;
      if (c.ghost) {
        els.push(rect(cx, 170, 116, 52, {
          stroke: T.border, strokeWidth: 1.4, strokeStyle: 'dashed',
        }));
        els.push(text(cx, 186, c.t, { size: 11, stroke: T.inkSubtle, align: 'center', width: 116 }));
      } else {
        els.push(...card(cx, 170, 116, 52, { stroke: c.c ?? T.ink }));
        els.push(text(cx, 186, c.t, { size: 12, family: 3, stroke: c.c ?? T.ink, align: 'center', width: 116 }));
      }
      if (i > 0) {
        els.push(arrow([[cx - 6, 196], [cx - 118 + 116 + 2, 196]],
          { stroke: o.c, strokeWidth: 1.3, roughness: 0.4 }));
      }
    });
    els.push(text(x + 24, 238, o.arrives, { size: 10, stroke: o.c }));
    els.push(rule(x + 16, x + 404, 264));
    els.push(text(x + 24, 274, 'so dvalues holds', { size: 10, stroke: T.inkSubtle }));
    els.push(text(x + 24, 292, o.holds, { size: 14, family: 3, stroke: o.c }));
    els.push(text(x + 24, 322, fit(o.note, 10, 380, 'panel note'), { size: 10, stroke: T.inkMuted }));
  };

  panel(M, {
    c: T.primary, title: 'Every other class', sig: 'backward(self, dvalues)',
    chain: [
      { t: 'this layer', c: T.primary },
      { t: 'the next one', c: T.inkMuted },
      { t: '…', c: T.inkMuted },
    ],
    arrives: 'a real gradient arrives from the right',
    holds: '∂L / ∂(this output)',
    note: 'The class multiplies it by its own local\nderivative and passes the product on.',
  });

  panel(500, {
    c: T.alert, title: 'The loss class', sig: 'backward(self, dvalues, y_true)',
    chain: [
      { t: 'the loss', c: T.alert },
      { t: 'nothing', ghost: true },
    ],
    arrives: 'there is no layer to the right · ∂L/∂L = 1',
    holds: 'ŷ, the predictions',
    note: 'The chain-rule multiply by the upstream is\nskipped, because the upstream is exactly 1.',
  });

  els.push(rect(M, 384, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 396, 'The loss is where the backward pass starts, so it has nothing upstream to be handed.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 420, 'It computes the first gradient outright, and every class to its left receives one because this one did not need to.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Why dvalues means the predictions in the loss class and a gradient everywhere else',
    desc: 'Two hand-drawn panels comparing what sits to the right of a layer. The left panel, every '
      + 'other class, has a signature taking dvalues alone and shows this layer with further layers '
      + 'to its right; a real gradient arrives from the right, so dvalues holds the partial '
      + 'derivative of the loss with respect to this layer’s output, and the class multiplies it by '
      + 'its own local derivative and passes the product on. The right panel, the loss class, has a '
      + 'signature taking dvalues and y_true and shows the loss with a dashed empty box beside it '
      + 'labelled nothing; there is no layer to the right and the derivative of the loss with '
      + 'respect to itself is one, so dvalues instead holds y-hat, the predictions, and the '
      + 'chain-rule multiplication by the upstream is skipped because the upstream is exactly one. A '
      + 'band notes that the loss is where the backward pass starts so it has nothing upstream to be '
      + 'handed, and that every class to its left receives a gradient because this one did not need '
      + 'to.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { crossEntropyBackward, whereBackpropStarts };

if (require.main === module) {
  emit('01-cross-entropy-backward', crossEntropyBackward());
  emit('02-where-backprop-starts', whereBackpropStarts());
}
