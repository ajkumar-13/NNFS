// The post 17 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 3 generalises ReLU into a family that all share one backward line and
// differ only in f′, and a table of three formulas is exactly the thing a figure
// can say better than prose.
//
//   01-elementwise-vs-coupled  960 x 480
//   02-elementwise-family      960 x 480   (new)

const { T, rect, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

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
  const size = o.size ?? 10;
  vals.forEach((row, j) => row.forEach((v, i) => {
    els.push(text(x + i * cw, y + j * ch + (ch - size) / 2 - 1, String(v), {
      size, family: 3, align: 'center', width: cw,
      stroke: (o.dim && o.dim(j, i)) ? T.inkSubtle : (o.ink ?? T.ink),
    }));
  }));
}

// A small function plot with its own y-scale, so a derivative that peaks at
// 0.25 is not drawn as tall as one that reaches 1.
function plot(els, x, y, w, h, f, o) {
  const zMin = o.zMin ?? -4, zMax = o.zMax ?? 4;
  const vMin = o.vMin, vMax = o.vMax;
  const px = (z) => x + ((z - zMin) / (zMax - zMin)) * w;
  const py = (v) => y + h - ((v - vMin) / (vMax - vMin)) * h;
  els.push(line([[x - 4, py(0)], [x + w + 4, py(0)]],
    { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  els.push(line([[px(0), y - 4], [px(0), y + h + 4]],
    { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
  const pts = [];
  for (let i = 0; i <= 48; i++) {
    const z = zMin + (i / 48) * (zMax - zMin);
    pts.push([px(z), py(Math.max(vMin, Math.min(vMax, f(z))))]);
  }
  els.push(line(pts, { stroke: o.c, strokeWidth: 2.2, roughness: o.rough ?? 0.3 }));
  els.push(text(x + w - 40, y - 4, o.top, { size: 9, stroke: T.inkSubtle, align: 'right', width: 40 }));
}

// ---------------------------------------------------------------- diagram 1
// The two Jacobians at the same size with the same labels. The only difference
// the figure has to carry is how many cells are non-zero, so the zeros are kept
// on the canvas rather than left blank.
function elementwiseVsCoupled() {
  resetSeq();
  const W = 960, H = 480;

  const els = [...heading('Diagonal, or dense',
    'Whether an activation’s backward is one multiply or one matrix product is decided by its Jacobian.')];

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 290, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 122, fit(o.sub, 11, 380, 'panel sub'), { size: 11, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + 404, 148));
    grid(els, x + 132, 164, o.J, 52, 40, { c: o.c, size: 9, dim: o.dim });
    els.push(text(x + 132, 292, '∂A / ∂Z', { size: 10, stroke: T.inkSubtle, align: 'center', width: 156 }));
    els.push(text(x + 20, 320, fit(o.formula, 12, 380 * 0.94, 'formula'),
      { size: 12, family: 3, stroke: o.c }));
    els.push(text(x + 20, 344, fit(o.note, 10, 380, 'panel note'), { size: 10, stroke: T.inkMuted }));
  };

  panel(M, {
    c: T.success, title: 'Element-wise', sub: 'aₖ depends on zₖ and nothing else',
    J: [
      ['f′(z₁)', '0', '0'],
      ['0', 'f′(z₂)', '0'],
      ['0', '0', 'f′(z₃)'],
    ],
    dim: (j, i) => j !== i,
    formula: 'dinputs = dvalues * f_prime(Z)',
    note: 'Multiplying by a diagonal matrix is just\nmultiplying each entry by its own diagonal value.',
  });

  panel(500, {
    c: T.alert, title: 'Coupled', sub: 'aₖ depends on every zⱼ, through the denominator',
    J: [
      ['a₁(1−a₁)', '−a₁a₂', '−a₁a₃'],
      ['−a₂a₁', 'a₂(1−a₂)', '−a₂a₃'],
      ['−a₃a₁', '−a₃a₂', 'a₃(1−a₃)'],
    ],
    formula: 'dinputs = dvalues @ J',
    note: 'A real matrix product, per sample,\nwith the full Jacobian built first.',
  });

  els.push(rect(M, 402, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 414, 'ReLU, sigmoid and tanh are diagonal. Softmax is dense, and it is the only activation in this series that is.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 438, 'Part 19 shows that pairing softmax with cross-entropy cancels the dense Jacobian away before it ever has to be built.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Element-wise versus coupled activation backwards: the Jacobian decides the code',
    desc: 'Two hand-drawn panels holding two three-by-three Jacobians at the same size. The left, in '
      + 'success green and labelled element-wise, has f prime of z on its diagonal and zeros '
      + 'everywhere else, drawn faint; its backward is the single line dinputs equals dvalues times '
      + 'f_prime of Z, because multiplying by a diagonal matrix is multiplying each entry by its own '
      + 'diagonal value. The right, in alert red and labelled coupled, is fully populated: a k times '
      + 'one minus a k on the diagonal and minus a k a j off it; its backward is a real matrix '
      + 'product per sample with the full Jacobian built first. A band notes that ReLU, sigmoid and '
      + 'tanh are diagonal while softmax is dense and is the only such activation in the series, and '
      + 'that Part 19 cancels the dense Jacobian away before it ever has to be built.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. Section 3 makes ReLU one member of a family whose backward is a single
// shared line. Plotting each f above its own f′ is what shows why: the shape of
// the derivative is the only thing that changes between them.
function elementwiseFamily() {
  resetSeq();
  const W = 960, H = 480;

  const els = [...heading('One backward line, three derivatives',
    'ReLU is not special. Every element-wise activation runs the same code and differs only in what f′ returns.')];

  const FAM = [
    {
      c: T.success, name: 'ReLU', f: 'max(0, z)', d: '1 if z > 0, else 0',
      fn: (z) => Math.max(0, z), fMin: -0.4, fMax: 4, fTop: '4',
      dfn: (z) => (z > 0 ? 1 : 0), dMin: -0.15, dMax: 1.15, dTop: '1',
      note: 'A binary derivative, so the\nbackward is a mask and nothing else.',
    },
    {
      c: T.primary, name: 'Sigmoid', f: '1 / (1 + e⁻ᶻ)', d: 'σ(z) · (1 − σ(z))',
      fn: (z) => 1 / (1 + Math.exp(-z)), fMin: -0.15, fMax: 1.15, fTop: '1',
      dfn: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); },
      dMin: -0.04, dMax: 0.30, dTop: '0.25',
      note: 'Peaks at 0.25 and vanishes at\nboth ends: the vanishing gradient.',
    },
    {
      c: T.accent, name: 'Tanh', f: '(eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)', d: '1 − tanh²(z)',
      fn: (z) => Math.tanh(z), fMin: -1.2, fMax: 1.2, fTop: '1',
      dfn: (z) => 1 - Math.tanh(z) ** 2, dMin: -0.15, dMax: 1.15, dTop: '1',
      note: 'Zero-centred, and peaks at 1,\nbut saturates the same way.',
    },
  ];

  const PW = 282, PX = [M, 339, 638];
  FAM.forEach((a, i) => {
    const x = PX[i], inner = PW - 40;
    els.push(...card(x, 86, PW, 300, { spine: a.c }));
    els.push(text(x + 20, 98, a.name, { size: 15, stroke: a.c }));
    els.push(text(x + 20, 122, fit(a.f, 10, inner * 0.94, 'f formula'),
      { size: 10, family: 3, stroke: T.inkMuted }));
    els.push(rule(x + 16, x + PW - 16, 146));

    els.push(text(x + 20, 154, 'f(z)', { size: 9, stroke: T.inkSubtle }));
    plot(els, x + 34, 168, 214, 62, a.fn, { c: a.c, vMin: a.fMin, vMax: a.fMax, top: a.fTop });

    els.push(text(x + 20, 244, 'f′(z)', { size: 9, stroke: T.inkSubtle }));
    plot(els, x + 34, 258, 214, 54, a.dfn, {
      c: a.c, vMin: a.dMin, vMax: a.dMax, top: a.dTop, rough: 0.2,
    });

    els.push(text(x + 20, 326, fit(a.d, 11, inner * 0.94, 'derivative'),
      { size: 11, family: 3, stroke: a.c }));
    els.push(text(x + 20, 348, fit(a.note, 9, inner, 'family note'), { size: 9, stroke: T.inkSubtle }));
  });

  els.push(rect(M, 410, 880, 58, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 422, 'dinputs  =  dvalues  *  f_prime(Z)',
    { size: 14, family: 3, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 448, 'The same line for all three. Only the second operand changes, which is why swapping an activation costs one function.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Every element-wise activation shares one backward line',
    desc: 'Three hand-drawn panels, one per activation, each plotting the function above its own '
      + 'derivative on separate vertical scales. ReLU is max of nought and z with a derivative of '
      + 'one where z is positive and nought otherwise, noted as a binary derivative that makes the '
      + 'backward a mask and nothing else. Sigmoid is one over one plus e to the minus z, with a '
      + 'derivative of sigma times one minus sigma that peaks at nought point two five and vanishes '
      + 'at both ends, noted as the vanishing gradient. Tanh is the usual exponential ratio with a '
      + 'derivative of one minus tanh squared, noted as zero-centred and peaking at one but '
      + 'saturating the same way. Each derivative is plotted on its own scale so the sigmoid '
      + 'derivative is not drawn as tall as the other two. A band gives the shared backward line, '
      + 'dinputs equals dvalues times f_prime of Z, and notes that only the second operand changes, '
      + 'which is why swapping an activation costs one function.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { elementwiseVsCoupled, elementwiseFamily };

if (require.main === module) {
  emit('01-elementwise-vs-coupled', elementwiseVsCoupled());
  emit('02-elementwise-family', elementwiseFamily());
}
